"""
Unified training display callback using Rich and Plotext.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import plotext as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


class TrainingDisplayCallback(Callback):
    """
    Unified callback for training visualization in the terminal.

    Combines:
    1. Real-time chart of metrics (e.g., reward/loss)
    2. Grid of current metric values
    3. Progress bars for Epochs and Steps
    4. Status header
    """

    def __init__(
        self,
        metric_keys: str | List[str] = "train/reward",
        chart_title: str = "Training Progress",
        refresh_rate: int = 4,
        history_length: Optional[int] = None,
        theme: str = "dark",
    ):
        """
        Initialize the training display dashboard.

        Args:
            metric_keys: Keys of metrics to track in the chart.
            chart_title: Title of the progress chart.
            refresh_rate: Terminal refresh rate in seconds.
            history_length: Maximum number of history points to keep.
            theme: Plotext theme for the chart.
        """
        super().__init__()
        # Handle single string or list of strings
        if isinstance(metric_keys, str):
            self.metric_keys = [k.strip() for k in metric_keys.split(",")]
        else:
            self.metric_keys = metric_keys

        self.chart_title = chart_title
        self.refresh_rate = refresh_rate
        self.history_length = history_length
        self.theme = theme

        # Chart Data: Dict mapping metric_key -> List of values/steps
        self.history: Dict[str, List[float]] = {k: [] for k in self.metric_keys}
        self.steps: Dict[str, List[int]] = {k: [] for k in self.metric_keys}

        # Colors for multiple lines
        self.colors = ["cyan", "magenta", "green", "yellow", "red", "blue", "white"]

        # Rich Components
        self.console = Console()
        self.layout = Layout()
        self.live: Optional[Live] = None
        self.progress: Optional[Progress] = None

        # Task IDs
        self.epoch_task_id: Optional[TaskID] = None
        self.step_task_id: Optional[TaskID] = None
        self.val_task_id: Optional[TaskID] = None

        # State
        self.current_metrics: Dict[str, Any] = {}
        self.trainer: Optional[pl.Trainer] = None
        self.start_time = 0.0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize the display when training starts."""
        if not trainer.is_global_zero:
            return

        self.trainer = trainer
        self.start_time = time.time()
        self._init_layout()
        self._start_live_display()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stop the display when training ends."""
        if self.live:
            self.live.stop()

    def _init_layout(self) -> None:
        """Initialize the Rich layout."""
        self.layout.split(
            Layout(name="header", size=1),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=5),
        )
        self.layout["main"].split_row(
            Layout(name="chart", ratio=2),
            Layout(name="metrics", ratio=1),
        )

        # Initialize Progress
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, complete_style="blue", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        # Create Tasks
        if self.trainer:
            self.epoch_task_id = self.progress.add_task("Epochs", total=self.trainer.max_epochs, start=False)
            self.step_task_id = self.progress.add_task("Steps", total=0, start=False)
            self.val_task_id = self.progress.add_task("Validation", total=0, visible=False)

    def _start_live_display(self) -> None:
        """Start the live display."""
        self.live = Live(
            self.layout,  # Use layout directly
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=False,
            auto_refresh=True,
        )
        self.live.start()

    def _render_layout(self) -> Layout:
        """Render the current state of the layout."""
        # Header (No Panel for simple single-line header)
        elapsed = time.time() - self.start_time
        header_text = Text(
            f" WSmart-Route Training • {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ",
            style="bold white on blue",
            justify="center",
        )
        self.layout["header"].update(header_text)

        # Chart
        self.layout["chart"].update(self._generate_chart())

        # Metrics
        self.layout["metrics"].update(self._generate_metrics_table())

        # Footer (Progress)
        if self.progress:
            # Borderless footer to save space
            self.layout["footer"].update(self.progress)
        else:
            self.layout["footer"].update(Text("Initializing progress...", justify="center"))

        return self.layout

    def _generate_chart(self) -> Panel:
        """Generate the plotext chart."""
        plt.clf()
        plt.theme(self.theme)

        # Get actual layout width if possible, or use sensible defaults
        width = (self.console.width * 2 // 3) - 10
        height = self.layout["main"].size or 15
        if height <= 0:
            height = 15

        has_data = False
        all_histories = []

        for i, key in enumerate(self.metric_keys):
            if len(self.history[key]) > 1:
                color = self.colors[i % len(self.colors)]
                plt.plot(self.steps[key], self.history[key], color=color, marker="dot", label=key)
                all_histories.extend(self.history[key])
                has_data = True

        if has_data:
            plt.title(self.chart_title)
            plt.xlabel("Step")
            # plt.ylabel("Value") # Combined Y label is generic

            # Ensure X-axis shows the full range if history_length is None
            if self.history_length is None:
                max_step = max(max(s for s in steps if s) for steps in self.steps.values() if any(steps))
                plt.xlim(0, max_step)

            # Smart Y-axis limits across all metrics
            ymin, ymax = min(all_histories), max(all_histories)
            yrange = ymax - ymin
            if yrange > 0:
                plt.ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        else:
            plt.text("Waiting for data...", x=0.5, y=0.5, alignment="center")

        plt.plotsize(max(width, 40), max(height, 10))

        # Wrap the plotext output in a Text object that correctly handles ANSI codes
        chart_text = Text.from_ansi(plt.build())
        return Panel(chart_text, title="📈 Training Chart", border_style="blue", expand=True)

    def _generate_metrics_table(self) -> Panel:
        """Generate a table of current metrics."""
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        # Sort metrics for consistent display
        for key in sorted(self.current_metrics.keys()):
            if key in ["epoch", "v_num"]:
                continue
            value = self.current_metrics[key]
            val_str = f"{value:.4e}" if isinstance(value, float) else str(value)
            table.add_row(key, val_str)

        return Panel(table, title="📊 Metrics", border_style="blue")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update epoch task."""
        if self.progress and self.epoch_task_id is not None:
            self.progress.start_task(self.epoch_task_id)
            self.progress.update(
                self.epoch_task_id,
                completed=trainer.current_epoch,
                total=trainer.max_epochs,
                description=f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
            )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update epoch task and live display."""
        if self.progress and self.epoch_task_id is not None:
            self.progress.update(
                self.epoch_task_id,
                completed=trainer.current_epoch + 1,
            )
        if self.live:
            self.live.update(self._render_layout())

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Update step task and metrics."""
        # Update Steps
        if self.progress and self.step_task_id is not None:
            # Determine total steps
            total = trainer.num_training_batches
            if total == float("inf"):
                total = 0  # Unknown

            self.progress.start_task(self.step_task_id)
            self.progress.update(
                self.step_task_id,
                completed=batch_idx + 1,
                total=total,
                description=f"Step {batch_idx + 1}/{total}",
            )

        # Update Metrics
        metrics = trainer.callback_metrics
        for k, v in metrics.items():
            if hasattr(v, "item"):
                self.current_metrics[k] = v.item()
            else:
                self.current_metrics[k] = v

        # Update History for chart (per batch/step)
        for key in self.metric_keys:
            if key in metrics:
                val = metrics[key]
                if hasattr(val, "item"):
                    val = val.item()  # type: ignore[assignment]

                self.history[key].append(float(val))
                self.steps[key].append(trainer.global_step)

                # Trim history if history_length is set
                if self.history_length is not None and len(self.history[key]) > self.history_length:
                    self.history[key] = self.history[key][-self.history_length :]
                    self.steps[key] = self.steps[key][-self.history_length :]

        if self.live:
            self.live.update(self._render_layout())

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update metrics display with latest validation scores."""
        metrics = trainer.callback_metrics

        # Record metrics for chart
        for key in self.metric_keys:
            if key in metrics:
                val = metrics[key]
                if hasattr(val, "item"):
                    val = val.item()  # type: ignore[assignment]

                self.history[key].append(float(val))
                self.steps[key].append(trainer.global_step)

                # Trim history if history_length is set
                if self.history_length is not None and len(self.history[key]) > self.history_length:
                    self.history[key] = self.history[key][-self.history_length :]
                    self.steps[key] = self.steps[key][-self.history_length :]

        # Update metrics table with latest validation scores
        for k, v in metrics.items():
            if hasattr(v, "item"):
                self.current_metrics[k] = v.item()
            else:
                self.current_metrics[k] = v
