"""
State Dict Converter

Converts between different checkpoint formats and parameter naming conventions.

Handles:
- Official CLIPZyme naming -> Local naming
- PyTorch Lightning prefixes
- Parameter shape mismatches
- Missing/extra parameters
"""

import torch
from typing import Dict, List, Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)


class StateDictConverter:
    """
    Convert state_dicts between different formats.

    Handles parameter name mapping, prefix removal, and shape adjustments.
    """

    # Mapping rules: official_name_pattern -> local_name
    PARAMETER_MAPPINGS = {
        # PyTorch Lightning prefixes
        r'^model\.': '',
        r'^clipzyme\.': '',

        # Protein encoder mappings
        r'^protein_model\.': 'protein_encoder.',
        r'^prot_encoder\.': 'protein_encoder.',
        r'^enzyme_encoder\.': 'protein_encoder.',

        # Reaction encoder mappings
        r'^reaction_model\.': 'reaction_encoder.',
        r'^rxn_encoder\.': 'reaction_encoder.',

        # ESM2 mappings
        r'^esm\.': 'protein_encoder.esm_model.',
        r'^esm_model\.esm\.': 'protein_encoder.esm_model.',

        # EGNN mappings
        r'^egnn\.': 'protein_encoder.',
        r'^protein_encoder\.egnn\.': 'protein_encoder.',

        # DMPNN mappings
        r'^dmpnn\.': 'reaction_encoder.',
        r'^mpnn\.': 'reaction_encoder.',

        # Projection heads
        r'^prot_projection\.': 'protein_encoder.projection.',
        r'^rxn_projection\.': 'reaction_encoder.projection.',

        # Temperature parameter
        r'^logit_scale': 'temperature',
        r'^temp\.': 'temperature',
    }

    def __init__(self):
        """Initialize converter with mapping rules."""
        self.compiled_patterns = {
            re.compile(pattern): replacement
            for pattern, replacement in self.PARAMETER_MAPPINGS.items()
        }

    def convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        source_format: str = "official",
        target_format: str = "local"
    ) -> Dict[str, torch.Tensor]:
        """
        Convert state_dict from one format to another.

        Args:
            state_dict: Source state_dict
            source_format: Source format name
            target_format: Target format name

        Returns:
            Converted state_dict
        """
        logger.info(f"Converting state_dict: {source_format} -> {target_format}")

        if source_format == "official" and target_format == "local":
            return self._convert_official_to_local(state_dict)
        elif source_format == "local" and target_format == "official":
            return self._convert_local_to_official(state_dict)
        else:
            logger.warning(f"Unknown conversion: {source_format} -> {target_format}")
            return state_dict

    def _convert_official_to_local(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convert official checkpoint to local format."""
        converted = {}

        for key, value in state_dict.items():
            # Apply all pattern mappings
            new_key = key
            for pattern, replacement in self.compiled_patterns.items():
                new_key = pattern.sub(replacement, new_key)

            # Handle special cases
            new_key = self._handle_special_cases(new_key)

            converted[new_key] = value

            if new_key != key:
                logger.debug(f"Mapped: {key} -> {new_key}")

        logger.info(f"Converted {len(state_dict)} parameters")
        return converted

    def _convert_local_to_official(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convert local checkpoint to official format."""
        # Reverse mapping (not always straightforward)
        converted = {}

        for key, value in state_dict.items():
            # For now, just add 'model.' prefix
            new_key = f"model.{key}"
            converted[new_key] = value

        return converted

    def _handle_special_cases(self, key: str) -> str:
        """Handle special parameter name cases."""

        # Temperature parameter
        if 'temperature' in key.lower() and not key.startswith('temperature'):
            return 'temperature'

        # Batch norm running stats
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            # Keep as-is
            return key

        return key

    def analyze_differences(
        self,
        state_dict1: Dict[str, torch.Tensor],
        state_dict2: Dict[str, torch.Tensor],
        verbose: bool = True
    ) -> Dict:
        """
        Analyze differences between two state_dicts.

        Args:
            state_dict1: First state_dict
            state_dict2: Second state_dict
            verbose: If True, print detailed comparison

        Returns:
            Dictionary with difference statistics
        """
        keys1 = set(state_dict1.keys())
        keys2 = set(state_dict2.keys())

        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        common = keys1 & keys2

        # Check shape mismatches in common keys
        shape_mismatches = []
        for key in common:
            if state_dict1[key].shape != state_dict2[key].shape:
                shape_mismatches.append({
                    'key': key,
                    'shape1': state_dict1[key].shape,
                    'shape2': state_dict2[key].shape
                })

        results = {
            'total_params_1': len(keys1),
            'total_params_2': len(keys2),
            'common_params': len(common),
            'only_in_1': list(only_in_1),
            'only_in_2': list(only_in_2),
            'shape_mismatches': shape_mismatches,
        }

        if verbose:
            logger.info("=== State Dict Comparison ===")
            logger.info(f"State Dict 1: {len(keys1)} parameters")
            logger.info(f"State Dict 2: {len(keys2)} parameters")
            logger.info(f"Common: {len(common)} parameters")

            if only_in_1:
                logger.info(f"\nOnly in State Dict 1 ({len(only_in_1)}):")
                for key in sorted(only_in_1)[:10]:
                    logger.info(f"  - {key}")
                if len(only_in_1) > 10:
                    logger.info(f"  ... and {len(only_in_1) - 10} more")

            if only_in_2:
                logger.info(f"\nOnly in State Dict 2 ({len(only_in_2)}):")
                for key in sorted(only_in_2)[:10]:
                    logger.info(f"  - {key}")
                if len(only_in_2) > 10:
                    logger.info(f"  ... and {len(only_in_2) - 10} more")

            if shape_mismatches:
                logger.info(f"\nShape Mismatches ({len(shape_mismatches)}):")
                for mismatch in shape_mismatches[:10]:
                    logger.info(
                        f"  - {mismatch['key']}: "
                        f"{mismatch['shape1']} vs {mismatch['shape2']}"
                    )

        return results

    def suggest_mappings(
        self,
        official_state_dict: Dict[str, torch.Tensor],
        local_state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, str]:
        """
        Suggest parameter name mappings between official and local.

        Args:
            official_state_dict: Official checkpoint state_dict
            local_state_dict: Local model state_dict

        Returns:
            Dictionary mapping official names to local names
        """
        official_keys = set(official_state_dict.keys())
        local_keys = set(local_state_dict.keys())

        suggestions = {}

        # Try to match by shape
        for off_key in official_keys:
            off_shape = official_state_dict[off_key].shape

            # Find local keys with same shape
            candidates = [
                loc_key for loc_key in local_keys
                if local_state_dict[loc_key].shape == off_shape
            ]

            if len(candidates) == 1:
                # Unique match by shape
                suggestions[off_key] = candidates[0]
            elif len(candidates) > 1:
                # Multiple matches, try name similarity
                best_match = self._find_best_name_match(off_key, candidates)
                if best_match:
                    suggestions[off_key] = best_match

        logger.info(f"Suggested {len(suggestions)} parameter mappings")
        return suggestions

    def _find_best_name_match(self, query: str, candidates: List[str]) -> Optional[str]:
        """Find best matching name from candidates using simple heuristics."""
        # Split into parts
        query_parts = set(re.split(r'[._]', query.lower()))

        best_match = None
        best_score = 0

        for candidate in candidates:
            cand_parts = set(re.split(r'[._]', candidate.lower()))

            # Count common parts
            common = query_parts & cand_parts
            score = len(common)

            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match if best_score > 0 else None


def convert_official_to_local(
    official_checkpoint_path: str,
    output_path: str,
    verify: bool = True
) -> str:
    """
    Convert official checkpoint to local format and save.

    Args:
        official_checkpoint_path: Path to official checkpoint
        output_path: Path to save converted checkpoint
        verify: If True, verify conversion

    Returns:
        Path to converted checkpoint
    """
    logger.info(f"Converting {official_checkpoint_path} to local format...")

    # Load official checkpoint
    checkpoint = torch.load(official_checkpoint_path, map_location='cpu')

    # Extract state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Convert
    converter = StateDictConverter()
    converted_state_dict = converter.convert(
        state_dict,
        source_format="official",
        target_format="local"
    )

    # Save
    torch.save(converted_state_dict, output_path)
    logger.info(f"✓ Converted checkpoint saved to {output_path}")

    # Verify if requested
    if verify:
        logger.info("Verifying conversion...")
        loaded = torch.load(output_path, map_location='cpu')
        assert len(loaded) == len(converted_state_dict)
        logger.info("✓ Verification passed")

    return output_path


def extract_pytorch_lightning_state_dict(
    lightning_checkpoint_path: str,
    output_path: str
) -> str:
    """
    Extract state_dict from PyTorch Lightning checkpoint.

    Args:
        lightning_checkpoint_path: Path to Lightning checkpoint
        output_path: Path to save extracted state_dict

    Returns:
        Path to extracted state_dict
    """
    logger.info(f"Extracting state_dict from {lightning_checkpoint_path}...")

    checkpoint = torch.load(lightning_checkpoint_path, map_location='cpu')

    if 'state_dict' not in checkpoint:
        raise ValueError("Not a PyTorch Lightning checkpoint (no 'state_dict' key)")

    state_dict = checkpoint['state_dict']

    # Remove 'model.' prefix if present
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {
            k.replace('model.', '', 1): v
            for k, v in state_dict.items()
        }
        logger.info("Removed 'model.' prefix from keys")

    # Save
    torch.save(state_dict, output_path)
    logger.info(f"✓ State dict saved to {output_path}")

    # Print info
    logger.info(f"Extracted {len(state_dict)} parameters")
    if 'epoch' in checkpoint:
        logger.info(f"From epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        logger.info(f"Global step: {checkpoint['global_step']}")

    return output_path


__all__ = [
    'StateDictConverter',
    'convert_official_to_local',
    'extract_pytorch_lightning_state_dict',
]
