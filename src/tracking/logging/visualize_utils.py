"""
Visualization utilities for the routing problems.

This module provides functions for:
- Visualizing routing solutions and graphs.
- Plotting loss landscapes (if used).
- Creating PCA visualizations of embeddings.
- Interfacing with TensorBoard for visual logging.

DEPRECATED: Use utils.logging.visualization package instead.
"""

from tracking.logging.visualization import (
    MyModelWrapper,
    get_batch,
    imitation_loss_fn,
    load_model_instance,
    log_weight_distributions,
    main,
    plot_attention_heatmaps,
    plot_logit_lens,
    plot_loss_landscape,
    plot_weight_trajectories,
    project_node_embeddings,
    rl_loss_fn,
    visualize_epoch,
)

__all__ = [
    "get_batch",
    "MyModelWrapper",
    "load_model_instance",
    "plot_weight_trajectories",
    "log_weight_distributions",
    "project_node_embeddings",
    "plot_attention_heatmaps",
    "plot_logit_lens",
    "imitation_loss_fn",
    "rl_loss_fn",
    "plot_loss_landscape",
    "visualize_epoch",
    "main",
]

if __name__ == "__main__":
    main()
