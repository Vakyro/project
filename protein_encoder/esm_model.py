"""
ESM2-based protein encoder with attention pooling and projection.

This module wraps HuggingFace's ESM2 model and adds:
- Flexible pooling strategies (attention/mean/cls)
- Projection head for embedding space alignment
- L2 normalization for cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .pooling import AttentionPool, mean_pool, cls_pool


class ProjectionHead(nn.Module):
    """
    Projection head with Dropout and LayerNorm.

    Maps from PLM hidden dimension to a lower-dimensional embedding space
    suitable for contrastive learning with reactions.
    """

    def __init__(self, in_dim, proj_dim=256, hidden=None, dropout=0.1):
        """
        Args:
            in_dim: Input dimension (ESM2 hidden size)
            proj_dim: Output embedding dimension
            hidden: Hidden layer size (defaults to in_dim)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        h = hidden or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(h),
            nn.Linear(h, proj_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Pooled protein embedding (B, in_dim)

        Returns:
            Projected and L2-normalized embedding (B, proj_dim)
        """
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class ProteinEncoderESM2(nn.Module):
    """
    Complete protein encoder pipeline:
    1. Tokenize amino acid sequences
    2. Pass through ESM2 transformer
    3. Apply pooling (attention/mean/cls)
    4. Project to embedding space
    5. L2 normalize for cosine similarity

    This encoder is designed to be used alongside the reaction encoder
    in a CLIP-style contrastive learning setup.
    """

    def __init__(
        self,
        plm_name="facebook/esm2_t33_650M_UR50D",
        pooling="attention",
        proj_dim=256,
        dropout=0.1,
        gradient_checkpointing=False
    ):
        """
        Args:
            plm_name: HuggingFace model identifier for ESM2
                      Options: facebook/esm2_t33_650M_UR50D (650M params)
                               facebook/esm2_t30_150M_UR50D (150M params)
                               facebook/esm2_t12_35M_UR50D (35M params)
            pooling: Pooling strategy - "attention", "mean", or "cls"
            proj_dim: Output embedding dimension (should match reaction encoder)
            dropout: Dropout probability in projection head
            gradient_checkpointing: Enable to reduce memory (slower training)
        """
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(plm_name, do_lower_case=False)
        self.plm = AutoModel.from_pretrained(plm_name)

        if gradient_checkpointing and hasattr(self.plm, "gradient_checkpointing_enable"):
            self.plm.gradient_checkpointing_enable()

        self.hidden_size = self.plm.config.hidden_size

        if pooling == "attention":
            self.pool = AttentionPool(self.hidden_size)
        elif pooling == "mean":
            self.pool = None
        elif pooling == "cls":
            self.pool = None
        else:
            raise ValueError(f"pooling must be 'attention', 'mean', or 'cls', got: {pooling}")

        self.pooling_type = pooling
        self.proj = ProjectionHead(self.hidden_size, proj_dim=proj_dim, dropout=dropout)

    @torch.no_grad()
    def tokenize(self, sequences, max_length=None):
        """
        Tokenize protein sequences for ESM2 input.

        Converts amino acid sequences into token IDs suitable for the ESM2 model.
        Automatically adds special tokens (<cls> at start, <eos> at end) and handles
        padding to create uniform-length batches.

        Args:
            sequences: List of amino acid sequence strings (without spaces)
                      Example: ["MSKGEELFTGVVPILVELDGDV...", "MAHHHHH..."]
            max_length: Maximum sequence length. Longer sequences will be truncated.
                       If None, no truncation is applied.

        Returns:
            Dictionary with tokenized inputs:
                - input_ids: Tensor of token IDs, shape (batch_size, seq_length)
                - attention_mask: Tensor of attention masks, shape (batch_size, seq_length)

        Example:
            >>> encoder = ProteinEncoderESM2()
            >>> seqs = ["MSKGEEL", "MAHHHHH"]
            >>> batch = encoder.tokenize(seqs, max_length=10)
            >>> batch['input_ids'].shape
            torch.Size([2, 9])
        """
        # Truncate sequences if max_length is specified
        if max_length is not None:
            sequences = [sequence[:max_length] for sequence in sequences]

        # Tokenize with ESM2 tokenizer
        # Automatically adds <cls> at start and <eos> at end
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )

        return batch

    def forward(self, batch_inputs):
        """
        Forward pass through the protein encoder.

        Args:
            batch_inputs: Dictionary with:
                - input_ids: Token IDs (B, L)
                - attention_mask: Mask for valid tokens (B, L)

        Returns:
            L2-normalized protein embeddings (B, proj_dim)
        """
        # Pass through ESM2
        out = self.plm(**batch_inputs)
        h = out.last_hidden_state  # (B, L, hidden_size)

        # Get attention mask
        mask = batch_inputs["attention_mask"]

        # Apply pooling
        if self.pooling_type == "attention":
            g = self.pool(h, mask)
        elif self.pooling_type == "mean":
            g = mean_pool(h, mask)
        else:  # cls
            g = cls_pool(h, mask)

        # Project and normalize
        z = self.proj(g)  # (B, proj_dim), L2-normalized

        return z

    def encode(
        self,
        sequences,
        device="cpu",
        max_length=None,
        batch_size=1
    ):
        """
        Encode protein sequences to L2-normalized embeddings end-to-end.

        This is a convenience method that handles the full encoding pipeline:
        tokenization, forward pass, and batching for memory efficiency.

        Args:
            sequences: List of amino acid sequence strings
                      Example: ["MSKGEEL...", "MAHHHHH..."]
            device: Device to use for computation ('cpu' or 'cuda')
            max_length: Maximum sequence length for truncation (None = no truncation)
            batch_size: Number of sequences to process at once.
                       If None, process all sequences in one batch.
                       Use smaller batches to reduce memory usage.

        Returns:
            Tensor of L2-normalized protein embeddings, shape (num_sequences, proj_dim)

        Example:
            >>> encoder = ProteinEncoderESM2()
            >>> seqs = ["MSKGEEL", "MAHHHHH", "MVKVYAPASS"]
            >>> embeddings = encoder.encode(seqs, device='cuda', batch_size=2)
            >>> embeddings.shape
            torch.Size([3, 256])
            >>> torch.norm(embeddings[0])  # Check L2 normalization
            tensor(1.0000)
        """
        self.eval()
        all_embeddings = []

        # If batch_size is None, process all sequences at once
        if batch_size is None:
            batch_size = len(sequences)

        # Process sequences in batches for memory efficiency
        for batch_start in range(0, len(sequences), batch_size):
            # Get current batch of sequences
            batch_sequences = sequences[batch_start:batch_start + batch_size]

            # Tokenize the batch
            batch_inputs = self.tokenize(batch_sequences, max_length=max_length)

            # Move tensors to target device
            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}

            # Encode without gradient computation
            with torch.no_grad():
                batch_embeddings = self(batch_inputs)
                # Move back to CPU to save GPU memory
                all_embeddings.append(batch_embeddings.cpu())

        # Concatenate all batch embeddings
        return torch.cat(all_embeddings, dim=0)

    def get_embedding_dim(self) -> int:
        """
        Get the output embedding dimension.

        Returns:
            Output dimension of the projection head
        """
        # Get output dim from the last linear layer in projection head
        return self.proj.net[-1].out_features
