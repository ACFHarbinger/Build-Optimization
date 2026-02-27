"""
Visualization package for WSmart-Route.
"""

import argparse
import os
from typing import Any, List, Optional

import torch

from configs import Config
from tracking.logging.visualization.embeddings import (
    log_weight_distributions,
    plot_weight_trajectories,
    project_node_embeddings,
)
from tracking.logging.visualization.heatmaps import plot_attention_heatmaps, plot_logit_lens
from tracking.logging.visualization.helpers import MyModelWrapper, get_batch, load_model_instance
from tracking.logging.visualization.landscape import (
    imitation_loss_fn,
    plot_loss_landscape,
    rl_loss_fn,
)


def visualize_epoch(model: Any, problem: Any, cfg: Config, epoch: int, tb_logger: Any = None) -> None:
    """
    Main entry point for visualization during training.

    Args:
        model: The neural model.
        problem: The problem instance.
        cfg: Root Hydra configuration.
        epoch: Current epoch number.
        tb_logger: Optional TensorBoard logger.
    """
    rl = cfg.rl
    model_cfg = cfg.model

    viz_modes: List[str] = getattr(rl, "viz_modes", [])
    if not viz_modes:
        return

    log_dir: str = getattr(rl, "log_dir", "logs")
    run_name: str = getattr(rl, "run_name", "run")
    save_dir: str = getattr(rl, "save_dir", "outputs")
    graph_size: int = cfg.train.graph.num_loc
    temporal_horizon: int = model_cfg.temporal_horizon

    viz_output_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)

    print(f"\n--- Visualizing Epoch {epoch} ---")

    # Move model to CPU for visualization to avoid device mismatch issues
    orig_device = next(model.parameters()).device
    model.cpu()

    # Move cost weights to CPU if they exist
    if hasattr(model, "cost_weights"):
        model.cost_weights = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.cost_weights.items()}

    try:
        if "distributions" in viz_modes or "both" in viz_modes:
            writer = tb_logger
            log_weight_distributions(
                model,
                epoch,
                log_dir=os.path.join(log_dir, run_name),
                writer=writer,
            )

        if "embeddings" in viz_modes:
            x_batch = get_batch(
                torch.device("cpu"),
                size=graph_size,
                batch_size=1,
                temporal_horizon=temporal_horizon,
            )
            writer = tb_logger
            project_node_embeddings(
                model,
                x_batch,
                log_dir=os.path.join(log_dir, run_name),
                writer=writer,
                epoch=epoch,
            )

        if "heatmaps" in viz_modes:
            plot_attention_heatmaps(model, viz_output_dir, epoch=epoch)

        if "logit_lens" in viz_modes:
            x_batch = get_batch(
                torch.device("cpu"),
                size=graph_size,
                batch_size=1,
                temporal_horizon=temporal_horizon,
            )
            plot_logit_lens(
                model,
                x_batch,
                os.path.join(viz_output_dir, f"logit_lens_ep{epoch}.png"),
                epoch=epoch,
            )

        if "loss" in viz_modes or "both" in viz_modes:
            plot_loss_landscape(
                model,
                cfg,
                viz_output_dir,
                epoch=epoch,
                size=graph_size,
                batch_size=4,
                resolution=10,
            )

        if "trajectory" in viz_modes:
            checkpoint_dir = save_dir
            plot_weight_trajectories(checkpoint_dir, os.path.join(viz_output_dir, "trajectory.png"))

    finally:
        # Restore model to original device
        model.to(orig_device)
        if hasattr(model, "cost_weights"):
            model.cost_weights = {
                k: v.to(orig_device) if isinstance(v, torch.Tensor) else v for k, v in model.cost_weights.items()
            }

    print("Visualization complete.\n")


def main():
    """Main execution entry point for visualization debugging."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory with multiple checkpoints")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard log directory")

    parser.add_argument("--size", type=int, default=100, help="Problem size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--resolution", type=int, default=10, help="Resolution for landscapes")
    parser.add_argument("--span", type=float, default=1.0, help="Span for landscapes")
    parser.add_argument("--problem", type=str, default="wcvrp", help="Problem type")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "trajectory",
            "distributions",
            "embeddings",
            "heatmaps",
            "logit_lens",
            "loss",
            "both",
        ],
        help="Visualization mode",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "trajectory":
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir required for trajectory")
        plot_weight_trajectories(args.checkpoint_dir, os.path.join(args.output_dir, "trajectory.png"))
        return  # Trajectory doesn't need model loading

    # Load model
    if not args.model_path:
        raise ValueError("--model_path required for this mode")
    model = load_model_instance(args.model_path, device, size=args.size, problem_name=args.problem)

    if args.mode == "distributions":
        log_weight_distributions(model, 0, args.log_dir)

    elif args.mode == "embeddings":
        x_batch = get_batch(device, size=args.size, batch_size=1)
        project_node_embeddings(model, x_batch, args.log_dir)

    elif args.mode == "heatmaps":
        plot_attention_heatmaps(model, args.output_dir)

    elif args.mode == "logit_lens":
        x_batch = get_batch(device, size=args.size, batch_size=1)
        plot_logit_lens(model, x_batch, os.path.join(args.output_dir, "logit_lens.png"))

    elif args.mode in {"loss", "both"}:
        # Standalone mode creates a minimal Config
        standalone_cfg = Config()
        standalone_cfg.model.temporal_horizon = 0
        plot_loss_landscape(
            model,
            standalone_cfg,
            args.output_dir,
            size=args.size,
            batch_size=args.batch_size,
            resolution=args.resolution,
            span=args.span,
        )


if __name__ == "__main__":
    main()
