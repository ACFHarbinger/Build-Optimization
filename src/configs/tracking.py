"""
Tracking configuration for experiment logging and monitoring.
"""

from dataclasses import dataclass


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking, logging, and monitoring.

    Attributes:
        enabled: Whether tracking is enabled.
        log_dir: Directory for log files.
        wandb_mode: WandB mode ('online', 'offline', 'disabled').
        no_tensorboard: Whether to disable TensorBoard logging.
        log_gradients: Whether to log gradient statistics.
        log_activations: Whether to log activation statistics.
        log_weights: Whether to log weight distributions.
        nan_guard: Whether to enable NaN detection in gradients.
        log_attention_heatmaps: Whether to log attention heatmaps.
        log_embeddings: Whether to log weight distributions and embeddings.
        log_loss_landscape: Whether to log loss landscape visualizations.
        log_every_n_steps: How often to log metrics (in steps).
        checkpoint_every_n_epochs: How often to save checkpoints.
    """

    enabled: bool = True
    log_dir: str = "logs"
    wandb_mode: str = "disabled"
    no_tensorboard: bool = True
    log_gradients: bool = False
    log_activations: bool = False
    log_weights: bool = False
    nan_guard: bool = False
    log_attention_heatmaps: bool = False
    log_embeddings: bool = False
    log_loss_landscape: bool = False
    log_every_n_steps: int = 50
    checkpoint_every_n_epochs: int = 5
