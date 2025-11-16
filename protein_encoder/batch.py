"""
Batch processing utilities for protein sequences.

This module provides utilities for batching protein sequences with proper
tokenization, padding, and masking for transformer models like ESM2.
"""

import torch
from typing import List, Optional, Dict, Any


def collate_sequences(
    tokenizer,
    sequences: List[str],
    max_length: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Collate a list of protein sequences into a batched, tokenized input.

    This function prepares protein sequences for batch processing by tokenizing
    them and applying padding/truncation to create uniform-length tensors.

    Args:
        tokenizer: ESM2 tokenizer from transformers library
        sequences: List of amino acid sequence strings
                  Example: ["MSKGEEL...", "MAHHHHH..."]
        max_length: Maximum sequence length. Sequences longer than this will
                   be truncated. If None, no truncation is applied.

    Returns:
        Dictionary containing tokenized batch:
            - input_ids: Tensor of token IDs, shape (batch_size, seq_length)
            - attention_mask: Tensor of attention masks, shape (batch_size, seq_length)
                            1 for real tokens, 0 for padding

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        >>> seqs = ["MSKGEEL", "MAHHHHH"]
        >>> batch = collate_sequences(tokenizer, seqs, max_length=10)
        >>> batch['input_ids'].shape
        torch.Size([2, 9])  # Includes special tokens
    """
    # Apply truncation if max_length is specified
    if max_length is not None:
        sequences = [sequence[:max_length] for sequence in sequences]

    # Tokenize all sequences with padding and special tokens
    batch = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    )

    # Return as dictionary (input_ids, attention_mask, etc.)
    return {key: value for key, value in batch.items()}


class ProteinDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for protein sequences.

    This simple dataset stores protein sequences and optional labels,
    useful for supervised learning tasks like function prediction,
    localization classification, etc.

    Args:
        sequences: List of amino acid sequence strings
        labels: Optional list of labels for supervised tasks.
               Can be class indices, one-hot vectors, or any other label format.

    Attributes:
        sequences: Stored protein sequences
        labels: Stored labels (None if not provided)

    Example:
        >>> sequences = ["MSKGEEL", "MAHHHHH"]
        >>> labels = [0, 1]  # Binary classification
        >>> dataset = ProteinDataset(sequences, labels)
        >>> seq, label = dataset[0]
    """

    def __init__(
        self,
        sequences: List[str],
        labels: Optional[List[Any]] = None
    ):
        """Initialize protein dataset with sequences and optional labels."""
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        """Return the total number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, index: int):
        """
        Get a single sequence (and label if available) by index.

        Args:
            index: Index of the sequence to retrieve

        Returns:
            If labels provided: Tuple of (sequence, label)
            If no labels: Just the sequence string
        """
        if self.labels is not None:
            return self.sequences[index], self.labels[index]
        return self.sequences[index]


def create_protein_dataloader(
    sequences: List[str],
    tokenizer,
    batch_size: int = 8,
    max_length: Optional[int] = None,
    shuffle: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for batching and tokenizing protein sequences.

    This convenience function creates a ProteinDataset and wraps it in a
    DataLoader with a custom collate function that handles tokenization.

    Args:
        sequences: List of amino acid sequence strings
        tokenizer: ESM2 tokenizer from transformers
        batch_size: Number of sequences per batch (default: 8)
        max_length: Maximum sequence length for truncation (default: None)
        shuffle: Whether to shuffle sequences each epoch (default: False)

    Returns:
        DataLoader that yields batched, tokenized protein sequences

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        >>> sequences = ["MSKGEEL", "MAHHHHH", "MVKVYAPASS"]
        >>> loader = create_protein_dataloader(sequences, tokenizer, batch_size=2)
        >>> for batch in loader:
        ...     print(batch['input_ids'].shape)  # (2, seq_len)
    """
    # Create dataset from sequences
    dataset = ProteinDataset(sequences)

    # Define collate function that tokenizes sequences
    def collate_fn(batch):
        """Tokenize a batch of sequences."""
        # batch is a list of sequences (strings)
        return collate_sequences(tokenizer, batch, max_length=max_length)

    # Create and return DataLoader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
