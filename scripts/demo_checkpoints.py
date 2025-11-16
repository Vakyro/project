#!/usr/bin/env python3
"""
CLIPZyme Checkpoint Management Demo

Demonstrates all checkpoint management features:
1. Downloading from Zenodo
2. Loading checkpoints
3. Inspecting checkpoints
4. Converting formats
5. Validating checkpoints
6. Comparing checkpoints

Run all demos:
    python scripts/demo_checkpoints.py --demo all

Run specific demo:
    python scripts/demo_checkpoints.py --demo download
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("CLIPZyme Checkpoint Management Demo")
print("=" * 80)


def demo_1_load_pretrained():
    """Demo 1: Load Pretrained Model (Simplest)"""
    print("\n" + "=" * 80)
    print("Demo 1: Load Pretrained Model")
    print("=" * 80)

    print("\nThis is the SIMPLEST way to get started with CLIPZyme!")
    print("The model will be automatically downloaded if not already cached.\n")

    from models import load_pretrained

    print("Loading pretrained CLIPZyme model...")
    print("Note: First run will download ~2.4 GB from Zenodo")

    try:
        # This will download from Zenodo if not cached
        model = load_pretrained(
            model_name="clipzyme",
            cache_dir="data/checkpoints",
            device="cpu",
            download_if_missing=False  # Set to True to auto-download
        )

        print("✓ Model loaded successfully!")
        print(f"Model config: {model.get_config()}")

        # Test inference
        print("\nTesting inference...")
        reaction = ["[C:1]=[O:2]>>[C:1]-[O:2]"]
        embeddings = model.encode_reactions(reaction, device="cpu")

        print(f"✓ Inference test passed!")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding norm: {embeddings.norm().item():.4f}")

    except FileNotFoundError:
        print("⚠ Checkpoint not found locally.")
        print("To download automatically, set download_if_missing=True")
        print("Or run: python scripts/manage_checkpoints.py download")


def demo_2_download_zenodo():
    """Demo 2: Download from Zenodo"""
    print("\n" + "=" * 80)
    print("Demo 2: Download from Zenodo")
    print("=" * 80)

    from checkpoints.downloader import ZenodoDownloader

    print("\nCLIPZyme checkpoints are hosted on Zenodo:")
    print("https://zenodo.org/records/15161343\n")

    downloader = ZenodoDownloader(output_dir="data/checkpoints")

    print("1. Listing available files...")
    try:
        files = downloader.list_files()

        print(f"\n✓ Found {len(files)} files:")
        for file_info in files:
            size_mb = file_info['size'] / (1024 ** 2)
            print(f"  - {file_info['key']}: {size_mb:.1f} MB")

    except Exception as e:
        print(f"⚠ Could not fetch metadata: {e}")
        print("(This is normal if offline or Zenodo is unreachable)")

    print("\n2. To download checkpoint:")
    print("   python scripts/manage_checkpoints.py download --output data/checkpoints")

    print("\n3. To download all files:")
    print("   python scripts/manage_checkpoints.py download --all")


def demo_3_inspect_checkpoint():
    """Demo 3: Inspect Checkpoint"""
    print("\n" + "=" * 80)
    print("Demo 3: Inspect Checkpoint")
    print("=" * 80)

    print("\nInspecting a checkpoint shows detailed information without loading the full model.\n")

    # Create a mock checkpoint for demonstration
    print("Creating mock checkpoint for demonstration...")

    mock_checkpoint = {
        'epoch': 29,
        'global_step': 45600,
        'state_dict': {
            'protein_encoder.weight': torch.randn(512, 1280),
            'reaction_encoder.weight': torch.randn(512, 256),
            'temperature': torch.tensor(0.07)
        },
        'hyper_parameters': {
            'learning_rate': 1e-4,
            'batch_size': 64,
            'temperature': 0.07
        }
    }

    temp_path = "temp_checkpoint.ckpt"
    torch.save(mock_checkpoint, temp_path)

    # Inspect it
    from checkpoints.validator import inspect_checkpoint

    info = inspect_checkpoint(temp_path)

    print("\n✓ Checkpoint Information:")
    print(f"  Format: {info.get('format', 'unknown')}")
    print(f"  Parameters: {info.get('parameter_count', 0):,}")
    print(f"  Epoch: {info.get('epoch', 'N/A')}")
    print(f"  Global Step: {info.get('global_step', 'N/A')}")

    if 'largest_parameters' in info:
        print("\n  Largest Parameters:")
        for name, size in info['largest_parameters']:
            print(f"    - {name}: {size:,}")

    # Cleanup
    Path(temp_path).unlink()
    print(f"\n✓ Cleaned up demo checkpoint")

    print("\nFor real checkpoints, use:")
    print("  python scripts/manage_checkpoints.py inspect --checkpoint path/to/checkpoint.ckpt")


def demo_4_load_checkpoint():
    """Demo 4: Load Checkpoint File"""
    print("\n" + "=" * 80)
    print("Demo 4: Load Checkpoint File")
    print("=" * 80)

    print("\nLoading a checkpoint directly from a file.\n")

    from checkpoints.loader import CheckpointLoader
    from config import CLIPZymeConfig

    # Create mock checkpoint
    print("Creating mock checkpoint...")

    # Build a small model
    config = CLIPZymeConfig.get_preset('default')
    config.protein_encoder.projection_dim = 128
    config.reaction_encoder.projection_dim = 128

    from models import CLIPZymeBuilder
    model = CLIPZymeBuilder().with_config(config).build()

    # Save it
    temp_path = "temp_model.ckpt"
    checkpoint = {
        'epoch': 10,
        'state_dict': model.state_dict(),
        'hyper_parameters': {'lr': 1e-4}
    }
    torch.save(checkpoint, temp_path)

    print("✓ Mock checkpoint created\n")

    # Load it
    print("Loading checkpoint with CheckpointLoader...")

    loader = CheckpointLoader(device="cpu")
    loaded_model = loader.load(temp_path, strict=False)

    print("✓ Checkpoint loaded successfully!")
    print(f"Model config: {loaded_model.get_config()}")

    # Cleanup
    Path(temp_path).unlink()
    print("\n✓ Cleaned up demo checkpoint")

    print("\nFor real checkpoints, use:")
    print("  from models import load_checkpoint")
    print("  model = load_checkpoint('path/to/checkpoint.ckpt', device='cuda')")


def demo_5_convert_checkpoint():
    """Demo 5: Convert Checkpoint Format"""
    print("\n" + "=" * 80)
    print("Demo 5: Convert Checkpoint Format")
    print("=" * 80)

    print("\nConverting between checkpoint formats.\n")

    from checkpoints.converter import StateDictConverter
    import torch

    # Create mock official checkpoint (Lightning format)
    print("1. Creating mock PyTorch Lightning checkpoint...")

    official_state_dict = {
        'model.protein_encoder.esm_model.weight': torch.randn(100, 100),
        'model.reaction_encoder.dmpnn.weight': torch.randn(100, 100),
        'model.temperature': torch.tensor(0.07)
    }

    print("   Keys (official format):")
    for key in list(official_state_dict.keys())[:3]:
        print(f"     - {key}")

    # Convert to local format
    print("\n2. Converting to local format...")

    converter = StateDictConverter()
    local_state_dict = converter.convert(
        official_state_dict,
        source_format="official",
        target_format="local"
    )

    print("   Keys (local format):")
    for key in list(local_state_dict.keys())[:3]:
        print(f"     - {key}")

    print("\n✓ Conversion complete!")

    # Analyze differences
    print("\n3. Analyzing differences...")

    differences = converter.analyze_differences(
        official_state_dict,
        local_state_dict,
        verbose=False
    )

    print(f"   Original parameters: {differences['total_params_1']}")
    print(f"   Converted parameters: {differences['total_params_2']}")
    print(f"   Common parameters: {differences['common_params']}")

    print("\nFor real conversions, use:")
    print("  python scripts/manage_checkpoints.py convert \\")
    print("    --input official.ckpt --output local.pt")


def demo_6_validate_checkpoint():
    """Demo 6: Validate Checkpoint"""
    print("\n" + "=" * 80)
    print("Demo 6: Validate Checkpoint")
    print("=" * 80)

    print("\nValidating checkpoint integrity.\n")

    from checkpoints.validator import CheckpointValidator
    from config import CLIPZymeConfig
    from models import CLIPZymeBuilder

    # Build model
    print("1. Building model...")
    config = CLIPZymeConfig.get_preset('default')
    model = CLIPZymeBuilder().with_config(config).build()

    # Validate
    print("\n2. Validating model...")

    validator = CheckpointValidator()
    results = validator.validate(
        model=model,
        test_forward=True
    )

    print(f"\n✓ Validation Results:")
    print(f"  Valid: {results['valid']}")
    print(f"  Total parameters: {results['parameter_count']:,}")
    print(f"  Trainable parameters: {results['trainable_params']:,}")

    if results['forward_pass']:
        fp = results['forward_pass']
        print(f"  Forward pass: {'✓ Success' if fp['success'] else '✗ Failed'}")
        if fp['success']:
            print(f"  Output shape: {fp.get('output_shape', 'N/A')}")

    if results['errors']:
        print(f"\n  Errors:")
        for error in results['errors']:
            print(f"    - {error}")

    if results['warnings']:
        print(f"\n  Warnings:")
        for warning in results['warnings']:
            print(f"    - {warning}")

    print("\nFor checkpoint files, use:")
    print("  python scripts/manage_checkpoints.py validate \\")
    print("    --checkpoint path/to/checkpoint.ckpt --test-forward")


def demo_7_compare_checkpoints():
    """Demo 7: Compare Checkpoints"""
    print("\n" + "=" * 80)
    print("Demo 7: Compare Checkpoints")
    print("=" * 80)

    print("\nComparing two checkpoints.\n")

    # Create two mock checkpoints
    print("1. Creating two mock checkpoints...")

    state_dict1 = {
        'layer1.weight': torch.randn(100, 100),
        'layer2.weight': torch.randn(50, 100),
    }

    state_dict2 = {
        'layer1.weight': torch.randn(100, 100),
        'layer2.weight': torch.randn(50, 100),
        'layer3.weight': torch.randn(25, 50),  # Extra layer
    }

    temp_path1 = "temp_checkpoint1.pt"
    temp_path2 = "temp_checkpoint2.pt"

    torch.save(state_dict1, temp_path1)
    torch.save(state_dict2, temp_path2)

    # Compare
    print("\n2. Comparing checkpoints...")

    from checkpoints.validator import compare_checkpoints

    results = compare_checkpoints(
        temp_path1,
        temp_path2,
        compare_weights=True
    )

    print(f"\n✓ Comparison Results:")
    print(f"  Keys match: {results['keys_match']}")
    print(f"  Shapes match: {results['shapes_match']}")

    if 'values_match' in results:
        print(f"  Values match: {results['values_match']}")
        print(f"  Max difference: {results.get('max_weight_diff', 0):.2e}")

    if results['differences']:
        print(f"\n  Differences found:")
        for diff in results['differences'][:3]:
            print(f"    - {diff}")

    # Cleanup
    Path(temp_path1).unlink()
    Path(temp_path2).unlink()
    print("\n✓ Cleaned up demo checkpoints")

    print("\nFor real comparisons, use:")
    print("  python scripts/manage_checkpoints.py compare \\")
    print("    --checkpoint1 model1.ckpt --checkpoint2 model2.ckpt")


def demo_8_complete_workflow():
    """Demo 8: Complete Workflow"""
    print("\n" + "=" * 80)
    print("Demo 8: Complete Workflow - Download, Load, and Use")
    print("=" * 80)

    print("\nComplete workflow for using official CLIPZyme checkpoint.\n")

    print("Step 1: Download checkpoint (if needed)")
    print("  python scripts/manage_checkpoints.py download")

    print("\nStep 2: Load pretrained model")
    print("  from models import load_pretrained")
    print("  model = load_pretrained('clipzyme', device='cuda')")

    print("\nStep 3: Use for encoding")
    print("  reactions = ['[C:1]=[O:2]>>[C:1]-[O:2]']")
    print("  embeddings = model.encode_reactions(reactions)")

    print("\nStep 4: Build screening set")
    print("  from screening import build_screening_set_from_model")
    print("  screening_set = build_screening_set_from_model(model, protein_db)")

    print("\nStep 5: Screen reactions")
    print("  from screening import InteractiveScreener")
    print("  screener = InteractiveScreener(model, screening_set)")
    print("  results = screener.screen_reaction(reaction_smiles)")

    print("\n✓ That's the complete workflow!")

    print("\nQuick start (one-liner):")
    print(">>> from models import load_pretrained")
    print(">>> model = load_pretrained('clipzyme', device='cuda', download_if_missing=True)")


def main():
    parser = argparse.ArgumentParser(description="CLIPZyme Checkpoint Demo")
    parser.add_argument(
        '--demo',
        type=str,
        default='all',
        choices=['all', '1', '2', '3', '4', '5', '6', '7', '8',
                 'pretrained', 'download', 'inspect', 'load',
                 'convert', 'validate', 'compare', 'workflow'],
        help='Which demo to run'
    )

    args = parser.parse_args()

    demos = {
        '1': ('pretrained', demo_1_load_pretrained),
        '2': ('download', demo_2_download_zenodo),
        '3': ('inspect', demo_3_inspect_checkpoint),
        '4': ('load', demo_4_load_checkpoint),
        '5': ('convert', demo_5_convert_checkpoint),
        '6': ('validate', demo_6_validate_checkpoint),
        '7': ('compare', demo_7_compare_checkpoints),
        '8': ('workflow', demo_8_complete_workflow),
    }

    if args.demo == 'all':
        for num, (name, func) in demos.items():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Demo {num} failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Find demo by number or name
        demo_func = None
        if args.demo in demos:
            demo_func = demos[args.demo][1]
        else:
            for num, (name, func) in demos.items():
                if name == args.demo:
                    demo_func = func
                    break

        if demo_func:
            demo_func()
        else:
            print(f"Unknown demo: {args.demo}")

    print("\n" + "=" * 80)
    print("✓ Demo complete!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Download official checkpoint:")
    print("     python scripts/manage_checkpoints.py download")
    print("\n  2. Load and use:")
    print("     from models import load_pretrained")
    print("     model = load_pretrained('clipzyme', device='cuda')")
    print("\n  3. See full documentation:")
    print("     checkpoints/README.md")


if __name__ == '__main__':
    main()
