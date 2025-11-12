"""
Utilities for handling long protein sequences.

ESM2 has a maximum context length (~1024 tokens). For longer sequences,
we need to chunk them and aggregate embeddings.
"""

import math
import torch
import torch.nn.functional as F


def chunk_sequence(seq, max_len=1000, overlap=50):
    """
    Split a long sequence into overlapping chunks.

    The overlap helps maintain context at chunk boundaries.

    Args:
        seq: Amino acid sequence string
        max_len: Maximum chunk length
        overlap: Number of overlapping amino acids between chunks

    Returns:
        List of sequence chunks

    Example:
        >>> seq = "ABCDEFGHIJKLMNOP"
        >>> chunks = chunk_sequence(seq, max_len=10, overlap=3)
        >>> # ['ABCDEFGHIJ', 'HIJKLMNOP']
    """
    if len(seq) <= max_len:
        return [seq]

    chunks = []
    i = 0

    while i < len(seq):
        chunk = seq[i:i + max_len]
        chunks.append(chunk)

        # If this chunk reaches the end, stop
        if i + max_len >= len(seq):
            break

        # Move forward by (max_len - overlap)
        i += max_len - overlap

    return chunks


@torch.no_grad()
def encode_long_sequence(model, seq, device="cuda", max_len=1000, overlap=50):
    """
    Encode a long protein sequence by chunking and averaging embeddings.

    Strategy:
    1. Split sequence into overlapping chunks
    2. Encode each chunk independently
    3. Average the chunk embeddings
    4. Re-normalize to unit length

    Args:
        model: ProteinEncoderESM2 instance
        seq: Long amino acid sequence string
        device: Device for computation
        max_len: Maximum chunk length
        overlap: Overlap between chunks

    Returns:
        L2-normalized embedding tensor (1, embedding_dim)
    """
    # Split into chunks
    chunks = chunk_sequence(seq, max_len=max_len, overlap=overlap)

    if len(chunks) == 1:
        # Sequence is short enough, encode normally
        batch = model.tokenize([seq])
        batch = {k: v.to(device) for k, v in batch.items()}
        return model(batch)

    # Encode each chunk
    embeddings = []
    for chunk in chunks:
        batch = model.tokenize([chunk])
        batch = {k: v.to(device) for k, v in batch.items()}
        z = model(batch)  # (1, embedding_dim)
        embeddings.append(z)

    # Average chunk embeddings
    z_avg = torch.stack(embeddings, dim=0).mean(dim=0)  # (1, embedding_dim)

    # Re-normalize
    z_norm = F.normalize(z_avg, p=2, dim=-1)

    return z_norm


def get_sequence_stats(sequences):
    """
    Get statistics about sequence lengths in a dataset.

    Useful for determining appropriate max_len and chunking parameters.

    Args:
        sequences: List of amino acid sequence strings

    Returns:
        Dictionary with statistics
    """
    lengths = [len(s) for s in sequences]

    if not lengths:
        return {}

    return {
        'count': len(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'mean': sum(lengths) / len(lengths),
        'median': sorted(lengths)[len(lengths) // 2],
        'num_over_1024': sum(1 for l in lengths if l > 1024),
        'num_over_2048': sum(1 for l in lengths if l > 2048)
    }


def truncate_or_chunk(seq, max_len=1024, strategy='truncate'):
    """
    Handle sequences that exceed max_len.

    Args:
        seq: Amino acid sequence string
        max_len: Maximum length
        strategy: 'truncate' (keep first max_len) or 'chunk' (split into pieces)

    Returns:
        If truncate: truncated sequence string
        If chunk: list of sequence chunks
    """
    if len(seq) <= max_len:
        return seq if strategy == 'truncate' else [seq]

    if strategy == 'truncate':
        return seq[:max_len]
    elif strategy == 'chunk':
        return chunk_sequence(seq, max_len=max_len, overlap=50)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def validate_sequence(seq):
    """
    Validate that a sequence contains only valid amino acids.

    Args:
        seq: Amino acid sequence string

    Returns:
        (is_valid, error_message)
    """
    # Standard amino acids + some special characters ESM2 can handle
    valid_chars = set('ACDEFGHIKLMNPQRSTVWYXBZJUO*')

    invalid_chars = set(seq.upper()) - valid_chars

    if invalid_chars:
        return False, f"Invalid characters found: {invalid_chars}"

    if len(seq) == 0:
        return False, "Empty sequence"

    return True, None


def batch_encode_with_chunking(model, sequences, device="cuda", max_len=1000, batch_size=1, show_progress=False):
    """
    Encode multiple sequences, automatically handling long sequences with chunking.

    Args:
        model: ProteinEncoderESM2 instance
        sequences: List of amino acid sequence strings
        device: Device for computation
        max_len: Maximum sequence length before chunking
        batch_size: Batch size (for short sequences)
        show_progress: Show progress bar (requires tqdm)

    Returns:
        Tensor of embeddings (N, embedding_dim)
    """
    model.eval()
    embeddings = []

    # Separate short and long sequences
    short_seqs = []
    short_indices = []
    long_seqs = []
    long_indices = []

    for i, seq in enumerate(sequences):
        if len(seq) <= max_len:
            short_seqs.append(seq)
            short_indices.append(i)
        else:
            long_seqs.append(seq)
            long_indices.append(i)

    # Create placeholder for results
    results = [None] * len(sequences)

    # Encode short sequences in batches
    if short_seqs:
        for i in range(0, len(short_seqs), batch_size):
            batch_seqs = short_seqs[i:i + batch_size]
            batch = model.tokenize(batch_seqs, max_len=max_len)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                z = model(batch)

            for j, idx in enumerate(short_indices[i:i + batch_size]):
                results[idx] = z[j:j + 1]

    # Encode long sequences with chunking
    if long_seqs:
        if show_progress:
            try:
                from tqdm import tqdm
                long_seqs = tqdm(long_seqs, desc="Encoding long sequences")
            except ImportError:
                pass

        for seq, idx in zip(long_seqs, long_indices):
            z = encode_long_sequence(model, seq, device=device, max_len=max_len)
            results[idx] = z

    # Concatenate all results
    return torch.cat(results, dim=0)
