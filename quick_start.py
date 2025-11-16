#!/usr/bin/env python3
"""
CLIPZyme Quick Start Script

This script helps you get started with CLIPZyme in 3 simple steps:
1. Download the pretrained checkpoint (if needed)
2. Set up the screening system
3. Run your first virtual screening

Usage:
    python quick_start.py
"""

import sys
from pathlib import Path
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🧬 CLIPZyme Quick Start")
print("=" * 80)
print("\nThis script will help you:")
print("  1. Download the pretrained CLIPZyme model (~2.4 GB)")
print("  2. Set up the virtual screening system")
print("  3. Run your first enzyme-reaction screening")
print()

# Step 1: Download checkpoint
print("-" * 80)
print("STEP 1: Download Pretrained Model")
print("-" * 80)
print()

checkpoint_dir = Path("data/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print(f"Checkpoint directory: {checkpoint_dir.absolute()}")
print()

# Check if checkpoint already exists
checkpoint_file = checkpoint_dir / "clipzyme_model.ckpt"
if checkpoint_file.exists():
    print(f"✓ Checkpoint already exists: {checkpoint_file}")
    print(f"  Size: {checkpoint_file.stat().st_size / (1024**3):.2f} GB")
    use_existing = input("\nUse existing checkpoint? (y/n): ").lower().strip()
    if use_existing == 'n':
        print("Downloading new checkpoint...")
        from checkpoints.downloader import ZenodoDownloader
        downloader = ZenodoDownloader(output_dir=str(checkpoint_dir))
        try:
            downloader.download_file('clipzyme_model.zip', extract=True)
            print("✓ Download complete!")
        except Exception as e:
            print(f"⚠ Download failed: {e}")
            print("\nYou can manually download from:")
            print("https://zenodo.org/records/15161343")
            print(f"Save to: {checkpoint_dir.absolute()}")
            sys.exit(1)
else:
    print("⚠ Checkpoint not found locally.")
    print()
    print("CLIPZyme official checkpoint is available at:")
    print("https://zenodo.org/records/15161343")
    print()
    print("File to download: clipzyme_model.zip (2.4 GB)")
    print()

    download_now = input("Download now? (y/n): ").lower().strip()

    if download_now == 'y':
        print("\nDownloading from Zenodo...")
        from checkpoints.downloader import ZenodoDownloader

        downloader = ZenodoDownloader(output_dir=str(checkpoint_dir))

        try:
            # Try to download
            print("Fetching file list from Zenodo...")
            files = downloader.list_files()

            print(f"\nAvailable files:")
            for f in files:
                size_mb = f['size'] / (1024**2)
                print(f"  - {f['key']}: {size_mb:.1f} MB")

            print("\nDownloading clipzyme_model.zip...")
            downloader.download_file('clipzyme_model.zip', extract=True)

            print("✓ Download complete!")

        except Exception as e:
            print(f"\n⚠ Automatic download failed: {e}")
            print("\nPlease download manually:")
            print("1. Go to: https://zenodo.org/records/15161343")
            print("2. Download: clipzyme_model.zip")
            print(f"3. Extract to: {checkpoint_dir.absolute()}")
            print()
            sys.exit(1)
    else:
        print("\n📥 Manual Download Instructions:")
        print("-" * 80)
        print("1. Visit: https://zenodo.org/records/15161343")
        print("2. Download: clipzyme_model.zip (2.4 GB)")
        print(f"3. Extract the ZIP file to: {checkpoint_dir.absolute()}")
        print("4. Run this script again")
        print()
        sys.exit(0)

# Step 2: Load the model
print()
print("-" * 80)
print("STEP 2: Load CLIPZyme Model")
print("-" * 80)
print()

try:
    from models import load_checkpoint

    print(f"Loading checkpoint: {checkpoint_file}")
    model = load_checkpoint(str(checkpoint_file), device="cpu")

    print("✓ Model loaded successfully!")
    print(f"  Device: cpu")
    print(f"  Parameters: ~660M (650M ESM2 + 10M DMPNN)")

except Exception as e:
    print(f"⚠ Failed to load checkpoint: {e}")
    print("\nPossible issues:")
    print("  - Checkpoint file is corrupted")
    print("  - Checkpoint format is incompatible")
    print("  - Missing dependencies")
    print()
    sys.exit(1)

# Step 3: Run a simple screening demo
print()
print("-" * 80)
print("STEP 3: Run Virtual Screening Demo")
print("-" * 80)
print()

print("Example: Finding enzymes that catalyze ethanol oxidation")
print()

# Example reaction: Ethanol -> Acetaldehyde
reaction_smiles = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
print(f"Reaction SMILES: {reaction_smiles}")
print(f"Chemical reaction: Ethanol → Acetaldehyde")
print()

# Load example proteins
import pandas as pd

proteins_file = Path("data/proteins.csv")
if proteins_file.exists():
    proteins_df = pd.read_csv(proteins_file)
    print(f"✓ Loaded {len(proteins_df)} example proteins")

    # Encode reaction
    print("\nEncoding reaction...")
    model.eval()
    with torch.no_grad():
        reaction_embedding = model.encode_reactions([reaction_smiles], device="cpu")

    print(f"✓ Reaction embedding shape: {reaction_embedding.shape}")
    print(f"  Embedding norm: {reaction_embedding.norm().item():.4f}")

    # Encode proteins (using sequences from CSV)
    if 'sequence' in proteins_df.columns:
        print(f"\nEncoding {len(proteins_df)} proteins...")
        sequences = proteins_df['sequence'].tolist()

        with torch.no_grad():
            protein_embeddings = model.encode_proteins(sequences[:5], device="cpu")  # First 5

        print(f"✓ Protein embeddings shape: {protein_embeddings.shape}")

        # Compute similarities
        print("\nComputing similarities...")
        similarities = torch.cosine_similarity(
            reaction_embedding.expand_as(protein_embeddings),
            protein_embeddings,
            dim=1
        )

        # Rank results
        sorted_indices = torch.argsort(similarities, descending=True)

        print("\n" + "=" * 80)
        print("🏆 SCREENING RESULTS")
        print("=" * 80)
        print()
        print("Top enzymes for ethanol oxidation:")
        print()

        for rank, idx in enumerate(sorted_indices[:5], 1):
            protein_name = proteins_df.iloc[idx.item()]['name']
            score = similarities[idx].item()
            print(f"{rank}. {protein_name}")
            print(f"   Similarity score: {score:.4f}")
            print()

    else:
        print("⚠ No sequences found in proteins.csv")
        print("Creating dummy demonstration...")

        print("\n✓ Model is ready for screening!")
        print("\nTo run full screening, you need:")
        print("  - Protein sequences or pre-embedded proteins")
        print("  - Reaction SMILES with atom mapping")

else:
    print("⚠ proteins.csv not found")
    print("\nModel loaded successfully! You can now:")
    print("  1. Encode reactions: model.encode_reactions([smiles])")
    print("  2. Encode proteins: model.encode_proteins([sequences])")
    print("  3. Run virtual screening with your own data")

# Final instructions
print()
print("=" * 80)
print("✅ SETUP COMPLETE!")
print("=" * 80)
print()
print("Next steps:")
print("  1. See example scripts in scripts/ directory")
print("  2. Run: python scripts/demo_screening.py")
print("  3. Run: python scripts/demo_inference.py")
print()
print("For full documentation, see:")
print("  - README.md")
print("  - docs/SCREENING_SYSTEM.md")
print("  - docs/CHECKPOINTS_INTEGRATION.md")
print()
print("Happy enzyme screening! 🧬")
print()
