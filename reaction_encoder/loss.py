"""
Loss functions for contrastive learning (CLIPZyme-style).
"""

import torch
import torch.nn.functional as F


def clip_loss(z_prot: torch.Tensor, z_rxn: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Compute CLIP-style contrastive loss between protein and reaction embeddings.

    This is a symmetric cross-entropy loss that encourages matching pairs
    (protein, reaction) to have high cosine similarity while non-matching
    pairs have low similarity.

    Args:
        z_prot: Protein embeddings [batch_size, embedding_dim]
        z_rxn: Reaction embeddings [batch_size, embedding_dim]
        temperature: Temperature scaling parameter (lower = sharper distributions)

    Returns:
        Scalar loss value
    """
    # Normalize embeddings
    z_prot = F.normalize(z_prot, dim=-1)
    z_rxn = F.normalize(z_rxn, dim=-1)

    # Compute similarity matrix
    logits = z_prot @ z_rxn.t() / temperature

    # Labels are diagonal (matching pairs)
    labels = torch.arange(z_prot.size(0), device=z_prot.device)

    # Symmetric loss (protein->reaction and reaction->protein)
    loss_p2r = F.cross_entropy(logits, labels)
    loss_r2p = F.cross_entropy(logits.t(), labels)

    return (loss_p2r + loss_r2p) / 2


def infonce_loss(embeddings: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE loss for self-supervised contrastive learning.

    Args:
        embeddings: Embeddings [batch_size, embedding_dim]
        temperature: Temperature parameter

    Returns:
        Scalar loss value
    """
    embeddings = F.normalize(embeddings, dim=-1)
    similarity_matrix = embeddings @ embeddings.t() / temperature

    batch_size = embeddings.size(0)
    labels = torch.arange(batch_size, device=embeddings.device)

    # Mask out diagonal (self-similarity)
    mask = torch.eye(batch_size, device=embeddings.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
