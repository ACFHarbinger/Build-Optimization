"""
Enhanced model summary callback for WSmart-Route.
"""

from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from rich.console import Console
from rich.table import Table


class ModelSummaryCallback(Callback):
    """
    Callback to print a detailed summary of the model architecture,
    including encoder, decoder, and must-go selection details.
    """

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print summary when training starts."""
        if not trainer.is_global_zero:
            return

        self._print_summary(pl_module)

    def _print_summary(self, model: pl.LightningModule) -> None:
        """Extract information and print the table."""
        from rich import box

        console = Console()
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.HEAVY_EDGE,
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Idx", justify="right", style="dim")
        table.add_column("Name", style="bold cyan")
        table.add_column("Type", style="green")
        table.add_column("Params", justify="right")
        table.add_column("Mode", justify="center")
        table.add_column("FLOPs", justify="right")

        def fmt_params(module):
            """
            Format parameter count for display.

            Args:
                module: PyTorch module or LightningModule.

            Returns:
                str: Human-readable parameter count (e.g., '1.2 M', '450.0 K').
            """
            count = sum(p.numel() for p in module.parameters())
            if count >= 1e6:
                return f"{count / 1e6:.1f} M"
            if count >= 1e3:
                return f"{count / 1e3:.1f} K"
            return str(count)

        # 0. Environment
        env_name = model.env.__class__.__name__
        table.add_row("0", "env", env_name, "0", "train", "0")

        # 1. Algorithm
        algo_name = model.__class__.__name__
        table.add_row("1", "algo", algo_name, "0", "N/A", "0")

        # 2. Policy
        policy = model.policy
        policy_name = policy.__class__.__name__
        table.add_row("2", "policy", policy_name, fmt_params(policy), "train", "0")

        # 2a. Encoder
        if hasattr(policy, "encoder"):
            enc = policy.encoder
            enc_name = enc.__class__.__name__
            if hasattr(policy, "encoder_config"):
                enc_name = f"{enc_name} ({policy.encoder_config.type})"
            table.add_row("2a", "encoder", enc_name, fmt_params(enc), "train", "0")

        # 2b. Decoder
        if hasattr(policy, "decoder"):
            dec = policy.decoder
            dec_name = dec.__class__.__name__
            if hasattr(policy, "decoder_config"):
                dec_name = f"{dec_name} ({policy.decoder_config.type})"
            table.add_row("2b", "decoder", dec_name, fmt_params(dec), "train", "0")

        # 3. Baseline
        if hasattr(model, "baseline") and model.baseline is not None:
            bl_name = model.baseline.__class__.__name__
            table.add_row("3", "baseline", bl_name, fmt_params(model.baseline), "train", "0")

        # 4. Expert Policy (for Imitation/Adaptive Imitation)
        if hasattr(model, "expert_policy") and model.expert_policy is not None:
            expert = model.expert_policy
            expert_name = getattr(model, "expert_name", expert.__class__.__name__)
            # Clean up names like VectorizedHGS -> HGS
            if expert_name and "Vectorized" in expert_name:
                expert_name = expert_name.replace("Vectorized", "")
            table.add_row("5", "expert", expert_name, fmt_params(expert), "N/A", "0")

        console.print("\n")
        console.print(table)
        console.print(f"Total Trainable Params: {fmt_params(model)}")
        console.print("\n")
