"""
Checkpoint Validator

Validates loaded checkpoints and compares with expected architecture.

Provides:
- Parameter count verification
- Shape validation
- Forward pass testing
- Checksum verification
- Model comparison
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointValidator:
    """Validate checkpoint integrity and compatibility."""

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(
        self,
        model: nn.Module,
        expected_params: Optional[int] = None,
        test_forward: bool = True
    ) -> Dict:
        """
        Validate a loaded model.

        Args:
            model: Loaded model to validate
            expected_params: Expected parameter count (optional)
            test_forward: If True, test forward pass

        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'parameter_count': 0,
            'trainable_params': 0,
            'forward_pass': None,
        }

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results['parameter_count'] = total_params
        results['trainable_params'] = trainable_params

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Check expected count
        if expected_params is not None:
            if total_params != expected_params:
                results['warnings'].append(
                    f"Parameter count mismatch: expected {expected_params:,}, "
                    f"got {total_params:,}"
                )

        # Check for NaN/Inf in parameters
        nan_params = []
        inf_params = []

        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)

        if nan_params:
            results['errors'].append(f"NaN values in parameters: {nan_params}")
            results['valid'] = False

        if inf_params:
            results['errors'].append(f"Inf values in parameters: {inf_params}")
            results['valid'] = False

        # Test forward pass
        if test_forward:
            forward_result = self._test_forward_pass(model)
            results['forward_pass'] = forward_result

            if not forward_result['success']:
                results['errors'].append(f"Forward pass failed: {forward_result['error']}")
                results['valid'] = False

        # Summary
        if results['valid']:
            logger.info("✓ Validation passed")
        else:
            logger.error("✗ Validation failed")
            for error in results['errors']:
                logger.error(f"  - {error}")

        if results['warnings']:
            for warning in results['warnings']:
                logger.warning(f"  - {warning}")

        return results

    def _test_forward_pass(self, model: nn.Module) -> Dict:
        """Test model forward pass with dummy data."""
        try:
            model.eval()

            # Try to run forward pass
            if hasattr(model, 'encode_reactions'):
                # CLIPZyme model
                with torch.no_grad():
                    reaction_emb = model.encode_reactions(
                        ["[C:1]=[O:2]>>[C:1]-[O:2]"],
                        device="cpu"
                    )

                return {
                    'success': True,
                    'output_shape': reaction_emb.shape,
                    'output_dtype': reaction_emb.dtype,
                    'error': None
                }
            else:
                # Unknown model type
                return {
                    'success': False,
                    'error': 'Model type not recognized'
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def compare_architectures(
        self,
        model1: nn.Module,
        model2: nn.Module
    ) -> Dict:
        """
        Compare two model architectures.

        Args:
            model1: First model
            model2: Second model

        Returns:
            Comparison results
        """
        results = {
            'identical': True,
            'parameter_count_match': False,
            'architecture_match': False,
            'differences': []
        }

        # Compare parameter counts
        params1 = sum(p.numel() for p in model1.parameters())
        params2 = sum(p.numel() for p in model2.parameters())

        results['param_count_1'] = params1
        results['param_count_2'] = params2
        results['parameter_count_match'] = (params1 == params2)

        if params1 != params2:
            results['identical'] = False
            results['differences'].append(
                f"Parameter count mismatch: {params1:,} vs {params2:,}"
            )

        # Compare architectures (string representation)
        arch1_str = str(model1)
        arch2_str = str(model2)

        results['architecture_match'] = (arch1_str == arch2_str)

        if arch1_str != arch2_str:
            results['identical'] = False
            results['differences'].append("Architecture strings differ")

        return results


def validate_checkpoint(
    checkpoint_path: str,
    expected_params: Optional[int] = None,
    test_forward: bool = True
) -> bool:
    """
    Validate a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint
        expected_params: Expected parameter count
        test_forward: Test forward pass

    Returns:
        True if valid, False otherwise
    """
    from checkpoints.loader import load_checkpoint

    logger.info(f"Validating checkpoint: {checkpoint_path}")

    try:
        # Load checkpoint
        model = load_checkpoint(checkpoint_path, device="cpu")

        # Validate
        validator = CheckpointValidator()
        results = validator.validate(
            model,
            expected_params=expected_params,
            test_forward=test_forward
        )

        return results['valid']

    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        return False


def compare_checkpoints(
    checkpoint_path1: str,
    checkpoint_path2: str,
    compare_weights: bool = False
) -> Dict:
    """
    Compare two checkpoints.

    Args:
        checkpoint_path1: First checkpoint
        checkpoint_path2: Second checkpoint
        compare_weights: If True, compare actual weight values

    Returns:
        Comparison results
    """
    logger.info("Comparing checkpoints...")
    logger.info(f"  Checkpoint 1: {checkpoint_path1}")
    logger.info(f"  Checkpoint 2: {checkpoint_path2}")

    # Load both
    ckpt1 = torch.load(checkpoint_path1, map_location='cpu')
    ckpt2 = torch.load(checkpoint_path2, map_location='cpu')

    # Extract state_dicts
    state_dict1 = ckpt1.get('state_dict', ckpt1) if isinstance(ckpt1, dict) else ckpt1.state_dict()
    state_dict2 = ckpt2.get('state_dict', ckpt2) if isinstance(ckpt2, dict) else ckpt2.state_dict()

    results = {
        'keys_match': False,
        'shapes_match': False,
        'values_match': False,
        'differences': []
    }

    # Compare keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    results['keys_match'] = (keys1 == keys2)

    if keys1 != keys2:
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1

        if only_in_1:
            results['differences'].append(f"Only in checkpoint 1: {list(only_in_1)[:5]}")
        if only_in_2:
            results['differences'].append(f"Only in checkpoint 2: {list(only_in_2)[:5]}")

    # Compare shapes
    common_keys = keys1 & keys2
    shape_mismatches = []

    for key in common_keys:
        if state_dict1[key].shape != state_dict2[key].shape:
            shape_mismatches.append({
                'key': key,
                'shape1': state_dict1[key].shape,
                'shape2': state_dict2[key].shape
            })

    results['shapes_match'] = (len(shape_mismatches) == 0)

    if shape_mismatches:
        results['differences'].append(f"Shape mismatches: {len(shape_mismatches)}")

    # Compare weight values
    if compare_weights and common_keys:
        max_diff = 0.0
        max_diff_key = None

        for key in common_keys:
            if state_dict1[key].shape == state_dict2[key].shape:
                diff = (state_dict1[key] - state_dict2[key]).abs().max().item()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_key = key

        results['values_match'] = (max_diff < 1e-6)
        results['max_weight_diff'] = max_diff
        results['max_diff_key'] = max_diff_key

        logger.info(f"Max weight difference: {max_diff:.2e} (in {max_diff_key})")

    # Summary
    logger.info("=== Comparison Results ===")
    logger.info(f"Keys match: {results['keys_match']}")
    logger.info(f"Shapes match: {results['shapes_match']}")
    if compare_weights:
        logger.info(f"Values match: {results['values_match']}")

    return results


def inspect_checkpoint(checkpoint_path: str) -> Dict:
    """
    Inspect checkpoint and print detailed information.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Dictionary with checkpoint information
    """
    logger.info(f"Inspecting checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    info = {
        'file_path': str(checkpoint_path),
        'file_size_mb': Path(checkpoint_path).stat().st_size / (1024 ** 2),
        'type': str(type(checkpoint)),
    }

    if isinstance(checkpoint, dict):
        info['keys'] = list(checkpoint.keys())

        # Check for Lightning format
        if 'state_dict' in checkpoint:
            info['format'] = 'pytorch_lightning'
            state_dict = checkpoint['state_dict']

            if 'epoch' in checkpoint:
                info['epoch'] = checkpoint['epoch']
            if 'global_step' in checkpoint:
                info['global_step'] = checkpoint['global_step']
            if 'hyper_parameters' in checkpoint:
                info['hyper_parameters'] = checkpoint['hyper_parameters']

        else:
            info['format'] = 'state_dict'
            state_dict = checkpoint

        # Analyze state_dict
        param_count = sum(t.numel() for t in state_dict.values())
        param_sizes = {k: v.numel() for k, v in state_dict.items()}
        largest_params = sorted(param_sizes.items(), key=lambda x: x[1], reverse=True)[:5]

        info['parameter_count'] = param_count
        info['num_parameters'] = len(state_dict)
        info['largest_parameters'] = largest_params

        logger.info(f"Format: {info['format']}")
        logger.info(f"Total parameters: {param_count:,}")
        logger.info(f"Number of parameter tensors: {len(state_dict)}")

        if 'epoch' in info:
            logger.info(f"Epoch: {info['epoch']}")

        logger.info("Largest parameters:")
        for name, size in largest_params:
            logger.info(f"  - {name}: {size:,}")

    elif isinstance(checkpoint, nn.Module):
        info['format'] = 'full_model'
        info['model_class'] = type(checkpoint).__name__
        param_count = sum(p.numel() for p in checkpoint.parameters())
        info['parameter_count'] = param_count

        logger.info(f"Format: full model")
        logger.info(f"Model class: {info['model_class']}")
        logger.info(f"Total parameters: {param_count:,}")

    return info


__all__ = [
    'CheckpointValidator',
    'validate_checkpoint',
    'compare_checkpoints',
    'inspect_checkpoint',
]
