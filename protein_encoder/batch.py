"""
Batch processing utilities for protein sequences.

Provides collate functions for creating batches with proper padding and masking.
"""

import torch


def collate_sequences(tokenizer, seqs, max_len=None):
    """
    Collate a list of protein sequences into a batched input.

    Args:
        tokenizer: ESM2 tokenizer from transformers
        seqs: List of amino acid sequence strings
        max_len: Maximum sequence length (truncate if longer)

    Returns:
        Dictionary with:
        - input_ids: Tensor of token IDs (B, L)
        - attention_mask: Tensor of attention masks (B, L)
    """
    if max_len is not None:
        seqs = [s[:max_len] for s in seqs]

    batch = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    )

    return {k: v for k, v in batch.items()}


class ProteinDataset(torch.utils.data.Dataset):
    """
    Simple dataset wrapper for protein sequences.

    Useful for creating DataLoaders with batching.
    """

    def __init__(self, sequences, labels=None):
        """
        Args:
            sequences: List of amino acid sequence strings
            labels: Optional list of labels (for supervised tasks)
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


def create_protein_dataloader(sequences, tokenizer, batch_size=8, max_len=None, shuffle=False):
    """
    Create a DataLoader for protein sequences.

    Args:
        sequences: List of amino acid sequence strings
        tokenizer: ESM2 tokenizer
        batch_size: Batch size
        max_len: Maximum sequence length
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader that yields batched, tokenized sequences
    """
    dataset = ProteinDataset(sequences)

    def collate_fn(batch):
        # batch is a list of sequences
        return collate_sequences(tokenizer, batch, max_len=max_len)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
