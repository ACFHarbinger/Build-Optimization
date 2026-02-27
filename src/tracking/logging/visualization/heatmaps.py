"""
Heatmap visualization tools.

Functions for plotting attention weights and logit lens analysis.
"""

import os

import matplotlib
import numpy as np
import seaborn as sns
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_attention_heatmaps(model, output_dir, epoch=0):
    """
    Plots heatmaps of attention weights (Query, Key, Value) for all layers.

    Args:
        model (nn.Module): The model.
        output_dir (str): Directory to save images.
        epoch (int, optional): Current epoch. Defaults to 0.
    """
    print("Plotting Attention Heatmaps...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    # Encoder attention weights
    # Structure: model.embedder.layers[i].att.module.W_query.weight etc.
    # Looking at gat_encoder.py: self.layers[i].att.module is MultiHeadAttention

    for i, layer in enumerate(model.embedder.layers):
        # att is SkipConnection, att.module is MultiHeadAttention
        mha = layer.att.module

        # W_query, W_key, W_val are typically the names in MultiHeadAttention
        for name in ["W_query", "W_key", "W_val"]:
            if hasattr(mha, name):
                param = getattr(mha, name)
                weight = param.weight.data.cpu().numpy() if hasattr(param, "weight") else param.data.cpu().numpy()

                # Handle 3D weights (Multi-Head Attention)
                if weight.ndim == 3:
                    # (Heads, Input, Output) -> (Input, Heads * Output)
                    h, input_dim, o = weight.shape
                    weight = weight.transpose(1, 0, 2).reshape(input_dim, h * o)
                elif weight.ndim > 2:
                    # Generic flatten to 2D
                    weight = weight.reshape(weight.shape[0], -1)

                plt.figure(figsize=(12, 10))
                sns.heatmap(weight, cmap="RdBu_r", center=0)
                plt.title(f"Layer {i} - {name} Weights (Epoch {epoch})")
                plt.savefig(os.path.join(output_dir, f"layer_{i}_{name}_ep{epoch}.png"))
                plt.close()

    print(f"Heatmaps saved to {output_dir}")


def plot_logit_lens(model, x_batch, output_file, epoch=0):
    """
    Implements the 'Logit Lens' technique for Attention Models.
    Project intermediate encoder layer outputs through the decoder's
    attention mechanism to see what the 'best' node is at each layer.

    Args:
        model (nn.Module): The Attention Model.
        x_batch (dict): Input batch.
        output_file (str): Filename to save the heatmap.
        epoch (int): Current epoch.
    """
    print("Computing Logit Lens...")
    model.eval()

    # Ensure compatible devices
    dev = next(model.parameters()).device
    x_batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in x_batch.items()}

    with torch.no_grad():
        h = model._get_initial_embeddings(x_batch)
        edges = x_batch.get("edges")

        all_probs = []

        # Helper to get first step probs
        def get_probs(embeddings):
            """Compute probability distribution for a given set of embeddings."""
            fixed = model.decoder._precompute(embeddings)
            # Correctly instantiate state with all required arguments
            state = model.problem.make_state(
                x_batch,
                edges=edges,
                cost_weights=getattr(model, "cost_weights", None),
                dist_matrix=x_batch.get("dist"),
            )
            log_p, _ = model.decoder._get_log_p(fixed, state)
            return log_p.exp()  # (Batch, 1, Nodes)

        # 0. Initial Embeddings (Layer 0)
        all_probs.append(get_probs(h))

        # 1. Intermediate Layers
        curr = h
        if hasattr(model.embedder, "layers"):
            for _i, layer in enumerate(model.embedder.layers):
                curr = layer(curr, mask=edges)
                all_probs.append(get_probs(curr))

        # Final dropout/projection if any
        if hasattr(model.embedder, "dropout"):
            curr = model.embedder.dropout(curr)
            # Only add if it changed something or if we want to see the final output
            # Usually redundant if dropout is 0 during eval, but good for completeness
            # all_probs.append(get_probs(curr))

        # Shape: (Batch, NumLayers, Nodes)
        probs_tensor = torch.cat(all_probs, dim=1).cpu().numpy()

        # Visualize first sample in batch
        sample_probs = probs_tensor[0]  # (L, N)

        plt.figure(figsize=(12, 8))
        sns.heatmap(sample_probs, cmap="viridis", annot=False)
        plt.title(f"Logit Lens - Probability Distribution per Layer (Epoch {epoch})")
        plt.xlabel("Node Index")
        plt.ylabel("Encoder Layer Index")

        # Highlight top prediction per layer
        top_indices = np.argmax(sample_probs, axis=1)
        for layer_idx, node_idx in enumerate(top_indices):
            plt.text(
                node_idx + 0.5,
                layer_idx + 0.5,
                f"{node_idx}",
                color="white",
                ha="center",
                va="center",
                weight="bold",
                fontsize=8,
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()

    print(f"Logit Lens saved to {output_file}")
