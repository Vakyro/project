"""
Screening-related tasks for dispatcher.

Provides tasks for building screening sets and executing virtual screening.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from ..core.task import Task, TaskConfig, TaskContext, TaskResult, TaskStatus
from ..monitoring.progress import ProgressTracker


logger = logging.getLogger(__name__)


class BuildScreeningSetTask(Task):
    """Build screening set by embedding proteins."""

    def __init__(
        self,
        config: TaskConfig,
        proteins_csv: Optional[str] = None,
        output_path: Optional[Path] = None,
        device: str = "cuda",
        batch_size: int = 8
    ):
        """
        Initialize build screening set task.

        Args:
            config: Task configuration
            proteins_csv: Path to proteins CSV
            output_path: Output path for screening set
            device: Device for encoding
            batch_size: Batch size for encoding
        """
        super().__init__(config)
        self.proteins_csv = proteins_csv
        self.output_path = output_path
        self.device = device
        self.batch_size = batch_size

    def execute(self, context: TaskContext) -> TaskResult:
        """Build screening set."""
        from datetime import datetime
        start_time = datetime.now()

        try:
            # Get model from shared state
            model = context.shared_state.get('model')
            if model is None:
                raise ValueError("No model found in shared state. Run LoadCheckpointTask first.")

            # Use config from task params or defaults
            proteins_csv = self.proteins_csv or context.config.get('proteins_csv')
            output_path = self.output_path or context.config.get('screening_set_path')

            if proteins_csv is None:
                raise ValueError("No proteins CSV provided")

            if output_path is None:
                output_path = Path("data/screening_set.pkl")
            else:
                output_path = Path(output_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build screening set
            from screening.screening_set import ScreeningSetBuilder

            builder = ScreeningSetBuilder(model=model, device=self.device)

            # Progress callback
            def progress_callback(current, total):
                if context.progress_callback:
                    percentage = (current / total * 100) if total > 0 else 0
                    message = f"Embedding proteins: {current}/{total}"
                    context.progress_callback(percentage, message)

            screening_set = builder.build_from_csv(
                proteins_csv=proteins_csv,
                batch_size=self.batch_size,
                progress_callback=progress_callback
            )

            # Save screening set
            screening_set.save(output_path)

            # Store in shared state
            context.shared_state['screening_set'] = screening_set
            context.shared_state['screening_set_path'] = str(output_path)

            logger.info(f"Screening set built: {len(screening_set)} proteins")

            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=str(output_path),
                metadata={
                    'num_proteins': len(screening_set),
                    'output_path': str(output_path)
                },
                metrics={
                    'num_proteins': len(screening_set)
                },
                start_time=start_time,
                end_time=datetime.now()
            )

        except Exception as e:
            logger.error(f"Screening set build failed: {str(e)}")
            import traceback
            return TaskResult(
                status=TaskStatus.FAILED,
                error=e,
                error_traceback=traceback.format_exc(),
                start_time=start_time,
                end_time=datetime.now()
            )


class RunScreeningTask(Task):
    """Run virtual screening against screening set."""

    def __init__(
        self,
        config: TaskConfig,
        reactions: Optional[List[str]] = None,
        screening_set_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        top_k: int = 100,
        mode: str = "interactive"
    ):
        """
        Initialize screening task.

        Args:
            config: Task configuration
            reactions: List of reaction SMILES to screen
            screening_set_path: Path to screening set
            output_dir: Output directory for results
            top_k: Number of top proteins to return
            mode: Screening mode (interactive or batched)
        """
        super().__init__(config)
        self.reactions = reactions
        self.screening_set_path = screening_set_path
        self.output_dir = output_dir or Path("results/screening")
        self.top_k = top_k
        self.mode = mode

    def execute(self, context: TaskContext) -> TaskResult:
        """Run screening."""
        from datetime import datetime
        start_time = datetime.now()

        try:
            # Get model and screening set from shared state
            model = context.shared_state.get('model')
            screening_set = context.shared_state.get('screening_set')

            if model is None:
                raise ValueError("No model found in shared state")

            # Load screening set if not in shared state
            if screening_set is None:
                screening_set_path = self.screening_set_path or context.shared_state.get('screening_set_path')

                if screening_set_path is None:
                    raise ValueError("No screening set provided")

                from screening.screening_set import ScreeningSet
                screening_set = ScreeningSet.load(screening_set_path)

            # Get reactions
            reactions = self.reactions or context.config.get('reactions')

            if reactions is None or len(reactions) == 0:
                raise ValueError("No reactions provided")

            # Create screener
            if self.mode == "interactive":
                from screening.interactive_mode import InteractiveScreener
                screener = InteractiveScreener(
                    model=model,
                    screening_set=screening_set
                )
            elif self.mode == "batched":
                from screening.batched_mode import BatchedScreener
                screener = BatchedScreener(
                    model=model,
                    screening_set=screening_set
                )
            else:
                raise ValueError(f"Unknown screening mode: {self.mode}")

            # Run screening
            results = []

            def progress_callback(current, total):
                if context.progress_callback:
                    percentage = (current / total * 100) if total > 0 else 0
                    message = f"Screening reactions: {current}/{total}"
                    context.progress_callback(percentage, message)

            for i, reaction_smiles in enumerate(reactions):
                result = screener.screen(
                    reaction_smiles=reaction_smiles,
                    top_k=self.top_k
                )
                results.append(result)

                progress_callback(i + 1, len(reactions))

            # Save results
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / "screening_results.pkl"

            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)

            # Store in shared state
            context.shared_state['screening_results'] = results

            logger.info(f"Screening completed: {len(reactions)} reactions")

            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=results,
                metadata={
                    'num_reactions': len(reactions),
                    'top_k': self.top_k,
                    'mode': self.mode,
                    'output_path': str(output_path)
                },
                metrics={
                    'num_reactions': len(reactions),
                    'total_hits': sum(len(r.top_proteins) for r in results)
                },
                start_time=start_time,
                end_time=datetime.now()
            )

        except Exception as e:
            logger.error(f"Screening failed: {str(e)}")
            import traceback
            return TaskResult(
                status=TaskStatus.FAILED,
                error=e,
                error_traceback=traceback.format_exc(),
                start_time=start_time,
                end_time=datetime.now()
            )


# Export
__all__ = [
    'BuildScreeningSetTask',
    'RunScreeningTask',
]
