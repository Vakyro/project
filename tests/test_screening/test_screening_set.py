"""
Tests for ScreeningSet.
"""

import pytest
import torch
from screening.screening_set import ScreeningSet


class TestScreeningSet:
    """Test suite for ScreeningSet."""

    def test_create_screening_set(self):
        """Test creating a screening set."""
        names = ["prot1", "prot2", "prot3"]
        embeddings = torch.randn(3, 128)

        screening_set = ScreeningSet(names, embeddings)

        assert len(screening_set) == 3
        assert screening_set.protein_names == names
        assert torch.equal(screening_set.embeddings, embeddings)

    def test_save_and_load(self, temp_dir):
        """Test saving and loading screening set."""
        names = ["prot1", "prot2", "prot3"]
        embeddings = torch.randn(3, 128)

        screening_set = ScreeningSet(names, embeddings)

        # Save
        save_path = temp_dir / "test_set.pkl"
        screening_set.save(save_path)

        assert save_path.exists()

        # Load
        loaded_set = ScreeningSet.load(save_path)

        assert len(loaded_set) == len(screening_set)
        assert loaded_set.protein_names == screening_set.protein_names
        assert torch.equal(loaded_set.embeddings, screening_set.embeddings)

    def test_get_protein_by_index(self):
        """Test getting protein by index."""
        names = ["prot1", "prot2", "prot3"]
        embeddings = torch.randn(3, 128)

        screening_set = ScreeningSet(names, embeddings)

        name, emb = screening_set[1]

        assert name == "prot2"
        assert torch.equal(emb, embeddings[1])

    def test_iteration(self):
        """Test iterating over screening set."""
        names = ["prot1", "prot2", "prot3"]
        embeddings = torch.randn(3, 128)

        screening_set = ScreeningSet(names, embeddings)

        collected_names = []
        for name, emb in screening_set:
            collected_names.append(name)

        assert collected_names == names

    def test_empty_screening_set(self):
        """Test empty screening set."""
        screening_set = ScreeningSet([], torch.empty(0, 128))

        assert len(screening_set) == 0

    def test_embedding_dimension(self):
        """Test embedding dimension property."""
        embeddings = torch.randn(5, 256)
        screening_set = ScreeningSet(["p1", "p2", "p3", "p4", "p5"], embeddings)

        assert screening_set.embedding_dim == 256

    @pytest.mark.gpu
    def test_to_device(self):
        """Test moving screening set to device."""
        embeddings = torch.randn(3, 128)
        screening_set = ScreeningSet(["p1", "p2", "p3"], embeddings)

        screening_set_gpu = screening_set.to('cuda')

        assert screening_set_gpu.embeddings.device.type == 'cuda'
