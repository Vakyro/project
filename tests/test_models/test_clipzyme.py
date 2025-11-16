"""
Tests for CLIPZyme model.
"""

import pytest
import torch
from models.clipzyme import CLIPZymeModel
from config.config import CLIPZymeConfig


class TestCLIPZymeModel:
    """Test suite for CLIPZyme model."""

    @pytest.fixture
    def model(self, sample_config, device):
        """Create CLIPZyme model for testing."""
        model = CLIPZymeModel(sample_config)
        return model.to(device)

    def test_model_creation(self, sample_config):
        """Test model can be created."""
        model = CLIPZymeModel(sample_config)
        assert model is not None
        assert hasattr(model, 'protein_encoder')
        assert hasattr(model, 'reaction_encoder')

    def test_encode_protein(self, model, sample_protein_sequence, device):
        """Test protein encoding."""
        sequences = [sample_protein_sequence]
        embeddings = model.encode_proteins(sequences)

        assert embeddings is not None
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == model.config.protein_encoder.proj_dim

    def test_encode_proteins_batch(self, model, sample_protein_sequences, device):
        """Test batch protein encoding."""
        embeddings = model.encode_proteins(sample_protein_sequences)

        assert embeddings.shape[0] == len(sample_protein_sequences)
        assert embeddings.shape[1] == model.config.protein_encoder.proj_dim

    def test_encode_reaction(self, model, sample_reaction_smiles, device):
        """Test reaction encoding."""
        reactions = [sample_reaction_smiles]
        embeddings = model.encode_reactions(reactions)

        assert embeddings is not None
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == model.config.reaction_encoder.proj_dim

    def test_encode_reactions_batch(self, model, sample_reactions, device):
        """Test batch reaction encoding."""
        embeddings = model.encode_reactions(sample_reactions)

        assert embeddings.shape[0] == len(sample_reactions)
        assert embeddings.shape[1] == model.config.reaction_encoder.proj_dim

    def test_embeddings_normalized(self, model, sample_protein_sequence, sample_reaction_smiles):
        """Test that embeddings are L2-normalized."""
        protein_emb = model.encode_proteins([sample_protein_sequence])
        reaction_emb = model.encode_reactions([sample_reaction_smiles])

        # Check L2 norm is approximately 1
        protein_norm = torch.norm(protein_emb, p=2, dim=1)
        reaction_norm = torch.norm(reaction_emb, p=2, dim=1)

        assert torch.allclose(protein_norm, torch.ones_like(protein_norm), atol=1e-5)
        assert torch.allclose(reaction_norm, torch.ones_like(reaction_norm), atol=1e-5)

    def test_model_eval_mode(self, model):
        """Test model can be set to eval mode."""
        model.eval()
        assert not model.training
        assert not model.protein_encoder.training
        assert not model.reaction_encoder.training

    def test_model_train_mode(self, model):
        """Test model can be set to train mode."""
        model.train()
        assert model.training
        assert model.protein_encoder.training
        assert model.reaction_encoder.training

    @pytest.mark.gpu
    def test_model_gpu(self, sample_config):
        """Test model on GPU."""
        model = CLIPZymeModel(sample_config).to('cuda')
        sequences = ["MSKQLIVNLLK"]

        embeddings = model.encode_proteins(sequences)

        assert embeddings.device.type == 'cuda'

    def test_model_parameters_count(self, model):
        """Test model has parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_model_gradient_flow(self, model, sample_protein_sequence, sample_reaction_smiles):
        """Test gradients can flow through model."""
        model.train()

        protein_emb = model.encode_proteins([sample_protein_sequence])
        reaction_emb = model.encode_reactions([sample_reaction_smiles])

        # Compute simple loss
        loss = -torch.sum(protein_emb * reaction_emb)
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients found in model"
