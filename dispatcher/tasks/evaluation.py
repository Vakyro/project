"""
Evaluation-related tasks for dispatcher.

Provides tasks for evaluating screening results and computing metrics.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from ..core.task import Task, TaskConfig, TaskContext, TaskResult, TaskStatus


logger = logging.getLogger(__name__)


class EvaluateScreeningTask(Task):
    """Evaluate screening results with ground truth."""

    def __init__(
        self,
        config: TaskConfig,
        ground_truth_csv: Optional[str] = None,
        output_dir: Optional[Path] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize evaluation task.

        Args:
            config: Task configuration
            ground_truth_csv: Path to ground truth CSV
            output_dir: Output directory for results
            metrics: List of metrics to compute
        """
        super().__init__(config)
        self.ground_truth_csv = ground_truth_csv
        self.output_dir = output_dir or Path("results/evaluation")
        self.metrics = metrics or ['bedroc', 'topk_accuracy', 'enrichment_factor']

    def execute(self, context: TaskContext) -> TaskResult:
        """Evaluate screening results."""
        from datetime import datetime
        start_time = datetime.now()

        try:
            # Get screening results from shared state
            screening_results = context.shared_state.get('screening_results')

            if screening_results is None:
                raise ValueError("No screening results found in shared state")

            # Get ground truth
            ground_truth_csv = self.ground_truth_csv or context.config.get('ground_truth_csv')

            if ground_truth_csv is None:
                raise ValueError("No ground truth CSV provided")

            # Run evaluation
            from evaluation.benchmark import BenchmarkEvaluator

            evaluator = BenchmarkEvaluator()

            # Compute metrics
            eval_results = evaluator.evaluate(
                screening_results=screening_results,
                ground_truth_csv=ground_truth_csv,
                metrics=self.metrics
            )

            # Save results
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / "evaluation_results.json"

            import json
            with open(output_path, 'w') as f:
                json.dump(eval_results, f, indent=2)

            # Store in shared state
            context.shared_state['evaluation_results'] = eval_results

            logger.info(f"Evaluation completed: {output_path}")

            # Extract key metrics for reporting
            metrics_summary = {}
            if 'bedroc' in eval_results:
                metrics_summary['bedroc_85'] = eval_results['bedroc'].get('bedroc_85', 0)

            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=eval_results,
                metadata={
                    'output_path': str(output_path),
                    'metrics': self.metrics
                },
                metrics=metrics_summary,
                start_time=start_time,
                end_time=datetime.now()
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            import traceback
            return TaskResult(
                status=TaskStatus.FAILED,
                error=e,
                error_traceback=traceback.format_exc(),
                start_time=start_time,
                end_time=datetime.now()
            )


class GenerateReportTask(Task):
    """Generate evaluation report with visualizations."""

    def __init__(
        self,
        config: TaskConfig,
        output_dir: Optional[Path] = None,
        include_plots: bool = True
    ):
        """
        Initialize report generation task.

        Args:
            config: Task configuration
            output_dir: Output directory for report
            include_plots: Whether to include plots
        """
        super().__init__(config)
        self.output_dir = output_dir or Path("results/report")
        self.include_plots = include_plots

    def execute(self, context: TaskContext) -> TaskResult:
        """Generate report."""
        from datetime import datetime
        start_time = datetime.now()

        try:
            # Get evaluation results from shared state
            eval_results = context.shared_state.get('evaluation_results')

            if eval_results is None:
                raise ValueError("No evaluation results found in shared state")

            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Generate text report
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("CLIPZYME SCREENING EVALUATION REPORT")
            report_lines.append("=" * 60)
            report_lines.append("")

            # Add metrics
            for metric_name, metric_data in eval_results.items():
                report_lines.append(f"{metric_name.upper()}:")
                if isinstance(metric_data, dict):
                    for key, value in metric_data.items():
                        report_lines.append(f"  {key}: {value}")
                else:
                    report_lines.append(f"  {metric_data}")
                report_lines.append("")

            report_text = "\n".join(report_lines)

            # Save text report
            report_path = self.output_dir / "report.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)

            # Generate plots if requested
            if self.include_plots:
                try:
                    from evaluation.visualization import EvaluationVisualizer

                    visualizer = EvaluationVisualizer()
                    visualizer.plot_metrics(
                        eval_results,
                        output_dir=self.output_dir
                    )
                except Exception as e:
                    logger.warning(f"Plot generation failed: {str(e)}")

            logger.info(f"Report generated: {report_path}")

            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=str(report_path),
                metadata={
                    'report_path': str(report_path),
                    'include_plots': self.include_plots
                },
                start_time=start_time,
                end_time=datetime.now()
            )

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
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
    'EvaluateScreeningTask',
    'GenerateReportTask',
]
