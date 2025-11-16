#!/usr/bin/env python3
"""
Checkpoint Management CLI

Manage CLIPZyme model checkpoints:
- Download from Zenodo
- Inspect checkpoint files
- Validate checkpoints
- Convert between formats
- Compare checkpoints

Usage:
    # Download official checkpoint
    python scripts/manage_checkpoints.py download --output data/checkpoints

    # Inspect checkpoint
    python scripts/manage_checkpoints.py inspect --checkpoint data/checkpoints/clipzyme_model.ckpt

    # Validate checkpoint
    python scripts/manage_checkpoints.py validate --checkpoint data/checkpoints/clipzyme_model.ckpt

    # Convert checkpoint
    python scripts/manage_checkpoints.py convert \
        --input data/checkpoints/official.ckpt \
        --output data/checkpoints/local.pt

    # Compare checkpoints
    python scripts/manage_checkpoints.py compare \
        --checkpoint1 model1.ckpt \
        --checkpoint2 model2.ckpt
"""

import argparse
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from checkpoints.downloader import ZenodoDownloader, download_clipzyme_checkpoint
from checkpoints.loader import load_official_checkpoint, load_pretrained
from checkpoints.validator import (
    validate_checkpoint,
    compare_checkpoints,
    inspect_checkpoint
)
from checkpoints.converter import (
    convert_official_to_local,
    extract_pytorch_lightning_state_dict
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_download(args):
    """Download checkpoint from Zenodo."""
    logger.info("=== Downloading CLIPZyme Checkpoint from Zenodo ===")

    downloader = ZenodoDownloader(output_dir=args.output)

    if args.all:
        # Download all files
        logger.info("Downloading all CLIPZyme files...")
        results = downloader.download_all(extract=args.extract)

        for file_key, path in results.items():
            if path:
                logger.info(f"✓ {file_key}: {path}")
            else:
                logger.error(f"✗ {file_key}: download failed")

    else:
        # Download specific file
        file_key = args.file or 'model'
        logger.info(f"Downloading {file_key}...")

        path = downloader.download_clipzyme_file(file_key, extract=args.extract)
        logger.info(f"✓ Downloaded to: {path}")

    logger.info("Download complete!")


def cmd_inspect(args):
    """Inspect checkpoint file."""
    logger.info("=== Inspecting Checkpoint ===")

    info = inspect_checkpoint(args.checkpoint)

    print("\n" + "=" * 60)
    print("CHECKPOINT INFORMATION")
    print("=" * 60)

    print(f"\nFile: {info['file_path']}")
    print(f"Size: {info['file_size_mb']:.2f} MB")
    print(f"Format: {info.get('format', 'unknown')}")

    if 'parameter_count' in info:
        print(f"Total Parameters: {info['parameter_count']:,}")

    if 'num_parameters' in info:
        print(f"Parameter Tensors: {info['num_parameters']}")

    if 'epoch' in info:
        print(f"Epoch: {info['epoch']}")

    if 'global_step' in info:
        print(f"Global Step: {info['global_step']}")

    if 'largest_parameters' in info:
        print("\nLargest Parameters:")
        for name, size in info['largest_parameters']:
            print(f"  - {name}: {size:,}")

    if 'hyper_parameters' in info:
        print("\nHyperparameters:")
        for key, value in list(info['hyper_parameters'].items())[:10]:
            print(f"  - {key}: {value}")

    print("=" * 60)


def cmd_validate(args):
    """Validate checkpoint."""
    logger.info("=== Validating Checkpoint ===")

    is_valid = validate_checkpoint(
        checkpoint_path=args.checkpoint,
        expected_params=args.expected_params,
        test_forward=args.test_forward
    )

    if is_valid:
        print("\n✓ Checkpoint is valid!")
        return 0
    else:
        print("\n✗ Checkpoint validation failed!")
        return 1


def cmd_convert(args):
    """Convert checkpoint format."""
    logger.info("=== Converting Checkpoint ===")

    if args.extract_lightning:
        # Extract state_dict from Lightning
        output_path = extract_pytorch_lightning_state_dict(
            lightning_checkpoint_path=args.input,
            output_path=args.output
        )
    else:
        # Convert official to local
        output_path = convert_official_to_local(
            official_checkpoint_path=args.input,
            output_path=args.output,
            verify=args.verify
        )

    logger.info(f"✓ Conversion complete: {output_path}")


def cmd_compare(args):
    """Compare two checkpoints."""
    logger.info("=== Comparing Checkpoints ===")

    results = compare_checkpoints(
        checkpoint_path1=args.checkpoint1,
        checkpoint_path2=args.checkpoint2,
        compare_weights=args.compare_weights
    )

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\nKeys match: {'✓' if results['keys_match'] else '✗'}")
    print(f"Shapes match: {'✓' if results['shapes_match'] else '✗'}")

    if 'values_match' in results:
        print(f"Values match: {'✓' if results['values_match'] else '✗'}")
        print(f"Max difference: {results.get('max_weight_diff', 0):.2e}")

    if results['differences']:
        print("\nDifferences:")
        for diff in results['differences']:
            print(f"  - {diff}")

    print("=" * 60)


def cmd_load(args):
    """Test loading checkpoint."""
    logger.info("=== Loading Checkpoint ===")

    if args.pretrained:
        # Load pretrained model
        logger.info(f"Loading pretrained model: {args.pretrained}")
        model = load_pretrained(
            model_name=args.pretrained,
            cache_dir=args.cache_dir,
            device=args.device,
            download_if_missing=args.download
        )
    else:
        # Load from file
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        model = load_official_checkpoint(
            checkpoint_path=args.checkpoint,
            device=args.device
        )

    logger.info(f"✓ Model loaded successfully!")
    logger.info(f"Model config: {model.get_config()}")

    # Test inference if requested
    if args.test_inference:
        logger.info("\nTesting inference...")

        test_reaction = "[C:1]=[O:2]>>[C:1]-[O:2]"
        embeddings = model.encode_reactions([test_reaction], device=args.device)

        logger.info(f"✓ Inference test passed!")
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Embedding norm: {embeddings.norm().item():.4f}")


def cmd_list(args):
    """List available checkpoints."""
    logger.info("=== Available Checkpoints ===")

    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return

    # Find all checkpoint files
    checkpoint_patterns = ['*.ckpt', '*.pt', '*.pth', '*.pkl']
    checkpoints = []

    for pattern in checkpoint_patterns:
        checkpoints.extend(cache_dir.rglob(pattern))

    if not checkpoints:
        print(f"No checkpoints found in {cache_dir}")
        return

    print(f"\nFound {len(checkpoints)} checkpoint(s) in {cache_dir}:\n")

    for ckpt in sorted(checkpoints):
        size_mb = ckpt.stat().st_size / (1024 ** 2)
        rel_path = ckpt.relative_to(cache_dir)
        print(f"  - {rel_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="CLIPZyme Checkpoint Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download checkpoint from Zenodo')
    download_parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/checkpoints',
        help='Output directory (default: data/checkpoints)'
    )
    download_parser.add_argument(
        '--file', '-f',
        type=str,
        choices=['model', 'data', 'splits'],
        help='Specific file to download (default: model)'
    )
    download_parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Download all CLIPZyme files'
    )
    download_parser.add_argument(
        '--no-extract',
        dest='extract',
        action='store_false',
        help='Do not extract zip files'
    )

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect checkpoint file')
    inspect_parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate checkpoint')
    validate_parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    validate_parser.add_argument(
        '--expected-params',
        type=int,
        help='Expected number of parameters'
    )
    validate_parser.add_argument(
        '--no-test-forward',
        dest='test_forward',
        action='store_false',
        help='Skip forward pass test'
    )

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert checkpoint format')
    convert_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input checkpoint path'
    )
    convert_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output checkpoint path'
    )
    convert_parser.add_argument(
        '--extract-lightning',
        action='store_true',
        help='Extract state_dict from PyTorch Lightning checkpoint'
    )
    convert_parser.add_argument(
        '--no-verify',
        dest='verify',
        action='store_false',
        help='Skip verification after conversion'
    )

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two checkpoints')
    compare_parser.add_argument(
        '--checkpoint1', '-c1',
        type=str,
        required=True,
        help='First checkpoint path'
    )
    compare_parser.add_argument(
        '--checkpoint2', '-c2',
        type=str,
        required=True,
        help='Second checkpoint path'
    )
    compare_parser.add_argument(
        '--compare-weights',
        action='store_true',
        help='Compare actual weight values (slower)'
    )

    # Load command
    load_parser = subparsers.add_parser('load', help='Test loading checkpoint')
    load_parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        help='Path to checkpoint file'
    )
    load_parser.add_argument(
        '--pretrained', '-p',
        type=str,
        help='Load pretrained model by name (e.g., "clipzyme")'
    )
    load_parser.add_argument(
        '--device', '-d',
        type=str,
        default='cpu',
        help='Device to load to (default: cpu)'
    )
    load_parser.add_argument(
        '--test-inference',
        action='store_true',
        help='Run inference test'
    )
    load_parser.add_argument(
        '--download',
        action='store_true',
        help='Download if missing (for pretrained)'
    )
    load_parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/checkpoints',
        help='Cache directory for pretrained models'
    )

    # List command
    list_parser = subparsers.add_parser('list', help='List available checkpoints')
    list_parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/checkpoints',
        help='Cache directory to search (default: data/checkpoints)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == 'download':
        cmd_download(args)
    elif args.command == 'inspect':
        cmd_inspect(args)
    elif args.command == 'validate':
        return cmd_validate(args)
    elif args.command == 'convert':
        cmd_convert(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'load':
        cmd_load(args)
    elif args.command == 'list':
        cmd_list(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
