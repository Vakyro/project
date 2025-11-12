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

        print(f"Loading ESM2 model: {plm_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(plm_name, do_lower_case=False)
        self.plm = AutoModel.from_pretrained(plm_name)

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing and hasattr(self.plm, "gradient_checkpointing_enable"):
            self.plm.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        self.hidden_size = self.plm.config.hidden_size
        print(f"ESM2 hidden size: {self.hidden_size}")

        # Setup pooling
        if pooling == "attention":
            self.pool = AttentionPool(self.hidden_size)
            print("Using attention pooling")
        elif pooling == "mean":
            self.pool = None  # Will use mean_pool function
            print("Using mean pooling")
        elif pooling == "cls":
            self.pool = None  # Will use cls_pool function
            print("Using CLS pooling")
        else:
            raise ValueError(f"pooling must be 'attention', 'mean', or 'cls', got: {pooling}")

        self.pooling_type = pooling

        # Projection head
        self.proj = ProjectionHead(self.hidden_size, proj_dim=proj_dim, dropout=dropout)
        print(f"Projection: {self.hidden_size} -> {proj_dim}")

    @torch.no_grad()
    def tokenize(self, seqs, max_len=None):
        """
        Tokenize protein sequences.

        Args:
            seqs: List of amino acid sequences (strings without spaces)
                  Example: ["MSKGEELFTGVVPILVELDGDV...", "MAHHHHH..."]
            max_len: Maximum sequence length (truncate if longer)

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Truncate sequences if needed
        if max_len is not None:
            seqs = [s[:max_len] for s in seqs]

        # Tokenize with ESM2 tokenizer
        # Automatically adds <cls> at start and <eos> at end
        batch = self.tokenizer(
            seqs,
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

    def encode(self, seqs, device="cpu", max_len=None, batch_size=1):
        """
        Convenience method to encode sequences end-to-end.

        Args:
            seqs: List of amino acid sequence strings
            device: Device to use for computation
            max_len: Maximum sequence length
            batch_size: Batch size for processing (None = process all at once)

        Returns:
            Tensor of L2-normalized embeddings (N, proj_dim)
        """
        self.eval()
        embeddings = []

        # Handle None batch_size
        if batch_size is None:
            batch_size = len(seqs)

        # Process in batches
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i + batch_size]

            # Tokenize
            batch = self.tokenize(batch_seqs, max_len=max_len)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Encode
            with torch.no_grad():
                z = self(batch)
                embeddings.append(z.cpu())

        return torch.cat(embeddings, dim=0)

    def get_embedding_dim(self) -> int:
        """
        Get the output embedding dimension.

        Returns:
            Output dimension of the projection head
        """
        # Get output dim from the last linear layer in projection head
        return self.proj.net[-1].out_features
