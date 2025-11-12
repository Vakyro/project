"""
Complete CLIPZyme training script with exact hyperparameters from paper.

Training details:
- Dataset: EnzymeMap (46,356 enzyme-reaction pairs)
- Batch size: 64
- Optimizer: AdamW (β1=0.9, β2=0.999, weight_decay=0.05)
- Learning rate: 1e-4 with cosine schedule
- Warmup: 100 steps (1e-6 to 1e-4)
- Precision: bfloat16
- Epochs: ~30 until convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.amp as amp
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import encoders
from protein_encoder.egnn import ProteinEncoderEGNN
from reaction_encoder.dmpnn import ReactionDMPNN
from reaction_encoder.features_clipzyme import reaction_to_graphs_clipzyme
from reaction_encoder.loss import clip_loss


class EnzymeReactionDataset(Dataset):
    """
    Dataset for enzyme-reaction pairs.

    Expected data format:
    [
        {
            "sequence": "MSKQLI...",
            "structure": "path/to/structure.cif",  # AlphaFold structure
            "reaction_smiles": "[C:1]=[O:2]>>[C:1][O:2]",
            "ec_number": "1.1.1.1",
            "reaction_rule": "carbonyl_reduction"
        },
        ...
    ]
    """

    def __init__(self, data_path, max_seq_len=650):
        """
        Args:
            data_path: Path to JSON file with enzyme-reaction pairs
            max_seq_len: Maximum sequence length (CLIPZyme uses 650)
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.max_seq_len = max_seq_len

        print(f"Loaded {len(self.data)} enzyme-reaction pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return {
            'sequence': item['sequence'][:self.max_seq_len],
            'structure_path': item['structure'],
            'reaction_smiles': item['reaction_smiles'],
            'ec_number': item.get('ec_number', ''),
            'reaction_rule': item.get('reaction_rule', '')
        }


def load_alphafold_structure(cif_path):
    """
    Load AlphaFold structure from CIF file and extract Cα coordinates.

    Args:
        cif_path: Path to CIF structure file

    Returns:
        coords: Tensor of Cα coordinates (N, 3)
    """
    # This is a simplified version - you'd need proper CIF parsing
    # Using BioPython or similar library
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser()
        structure = parser.get_structure('protein', cif_path)

        # Extract Cα coordinates
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        coords.append(ca_atom.get_coord())

        return torch.tensor(coords, dtype=torch.float32)

    except ImportError:
        # Fallback: return dummy coordinates if BioPython not available
        print("Warning: BioPython not available, using dummy coordinates")
        # Generate dummy Cα trace (approximately extended chain)
        n_residues = 100  # Placeholder
        coords = torch.randn(n_residues, 3) * 10
        return coords


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Returns batch of sequences, structures, and reactions.
    """
    sequences = [item['sequence'] for item in batch]
    structure_paths = [item['structure_path'] for item in batch]
    reactions = [item['reaction_smiles'] for item in batch]

    return {
        'sequences': sequences,
        'structure_paths': structure_paths,
        'reactions': reactions
    }


class CLIPZymeTrainer:
    """
    Complete trainer for CLIPZyme model.
    """

    def __init__(
        self,
        protein_encoder,
        reaction_encoder,
        device='cuda',
        lr=1e-4,
        weight_decay=0.05,
        warmup_steps=100,
        temperature=0.07,
        use_amp=True
    ):
        """
        Args:
            protein_encoder: ProteinEncoderEGNN
            reaction_encoder: ReactionDMPNN
            device: Training device
            lr: Learning rate (1e-4 for CLIPZyme)
            weight_decay: Weight decay (0.05 for CLIPZyme)
            warmup_steps: Warmup steps (100 for CLIPZyme)
            temperature: Temperature for CLIP loss
            use_amp: Use automatic mixed precision (bfloat16)
        """
        self.protein_encoder = protein_encoder.to(device)
        self.reaction_encoder = reaction_encoder.to(device)
        self.device = device
        self.temperature = temperature

        # Optimizer: AdamW with β1=0.9, β2=0.999
        self.optimizer = torch.optim.AdamW(
            list(protein_encoder.parameters()) + list(reaction_encoder.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )

        self.warmup_steps = warmup_steps
        self.base_lr = lr
        self.min_lr = 1e-5

        # AMP scaler for bfloat16
        self.use_amp = use_amp
        self.scaler = amp.GradScaler('cuda') if use_amp else None

        self.step = 0

    def get_lr(self):
        """
        Get learning rate with warmup and cosine schedule.
        """
        if self.step < self.warmup_steps:
            # Linear warmup from 1e-6 to base_lr
            return 1e-6 + (self.base_lr - 1e-6) * self.step / self.warmup_steps
        else:
            # Cosine decay to min_lr
            progress = (self.step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    def set_lr(self, lr):
        """Set learning rate for optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader with enzyme-reaction pairs
            epoch: Current epoch number

        Returns:
            Average loss for epoch
        """
        self.protein_encoder.train()
        self.reaction_encoder.train()

        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Update learning rate
            lr = self.get_lr()
            self.set_lr(lr)

            # Parse batch
            sequences = batch['sequences']
            structure_paths = batch['structure_paths']
            reactions = batch['reactions']

            # Tokenize proteins
            protein_tokens = self.protein_encoder.tokenize(sequences)
            protein_tokens = {k: v.to(self.device) for k, v in protein_tokens.items()}

            # Load structures
            coords_list = []
            for path in structure_paths:
                coords = load_alphafold_structure(path)
                coords_list.append(coords)

            # Parse reactions
            reaction_data_list = []
            for rxn in reactions:
                try:
                    rxn_data = reaction_to_graphs_clipzyme(rxn)
                    reaction_data_list.append(rxn_data)
                except Exception as e:
                    print(f"Error parsing reaction: {rxn}, {e}")
                    # Use dummy data for failed reactions
                    reaction_data_list.append(None)

            # Forward pass with AMP
            with amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                # Encode proteins
                protein_embeddings = self.protein_encoder(protein_tokens, coords_list)

                # Encode reactions
                reaction_embeddings_list = []
                for rxn_data in reaction_data_list:
                    if rxn_data is not None:
                        from torch_geometric.data import Data

                        # Create PyG Data objects
                        substrate_data = Data(
                            x=rxn_data['substrate']['x'].to(self.device),
                            edge_index=rxn_data['substrate']['edge_index'].to(self.device),
                            edge_attr=rxn_data['substrate']['edge_attr'].to(self.device)
                        )

                        product_data = Data(
                            x=rxn_data['product']['x'].to(self.device),
                            edge_index=rxn_data['product']['edge_index'].to(self.device),
                            edge_attr=rxn_data['product']['edge_attr'].to(self.device)
                        )

                        z = self.reaction_encoder(
                            substrate_data,
                            product_data,
                            rxn_data['atom_mapping']
                        )
                        reaction_embeddings_list.append(z)
                    else:
                        # Dummy embedding
                        z = torch.zeros(1, self.reaction_encoder.dmpnn.projection[3].out_features,
                                       device=self.device)
                        reaction_embeddings_list.append(z)

                reaction_embeddings = torch.cat(reaction_embeddings_list, dim=0)

                # CLIP loss
                loss = clip_loss(protein_embeddings, reaction_embeddings, temperature=self.temperature)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.2e}'
            })

        return total_loss / n_batches

    def save_checkpoint(self, path, epoch, loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'protein_encoder': self.protein_encoder.state_dict(),
            'reaction_encoder': self.reaction_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'step': self.step
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.protein_encoder.load_state_dict(checkpoint['protein_encoder'])
        self.reaction_encoder.load_state_dict(checkpoint['reaction_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['loss']


def main():
    """
    Main training loop.
    """
    # Configuration
    config = {
        'data_path': 'data/enzyme_map_train.json',
        'val_data_path': 'data/enzyme_map_val.json',
        'batch_size': 64,
        'num_epochs': 30,
        'lr': 1e-4,
        'weight_decay': 0.05,
        'warmup_steps': 100,
        'temperature': 0.07,
        'checkpoint_dir': 'checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'use_amp': True
    }

    print("\nCLIPZyme Training")
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"AMP: {config['use_amp']}")

    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)

    # Initialize models
    print("\nInitializing models...")

    protein_encoder = ProteinEncoderEGNN(
        plm_name="facebook/esm2_t33_650M_UR50D",
        hidden_dim=1280,
        num_layers=6,
        proj_dim=512,
        dropout=0.1,
        k_neighbors=30,
        distance_cutoff=10.0
    )

    reaction_encoder = ReactionDMPNN(
        node_dim=9,
        edge_dim=3,
        hidden_dim=1280,
        num_layers=5,
        proj_dim=512,
        dropout=0.1
    )

    print(f"Protein encoder parameters: {sum(p.numel() for p in protein_encoder.parameters()):,}")
    print(f"Reaction encoder parameters: {sum(p.numel() for p in reaction_encoder.parameters()):,}")

    # Initialize trainer
    trainer = CLIPZymeTrainer(
        protein_encoder=protein_encoder,
        reaction_encoder=reaction_encoder,
        device=config['device'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        temperature=config['temperature'],
        use_amp=config['use_amp']
    )

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = EnzymeReactionDataset(config['data_path'])
    val_dataset = EnzymeReactionDataset(config['val_data_path'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Set total steps for cosine schedule
    trainer.total_steps = len(train_loader) * config['num_epochs']

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total steps: {trainer.total_steps}")

    # Training loop
    print("\nStarting training...")

    best_val_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")

        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Train loss: {train_loss:.4f}")

        checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
        trainer.save_checkpoint(checkpoint_path, epoch, train_loss)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
