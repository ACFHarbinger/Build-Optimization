"""
Attention map visualization.
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import seaborn as sns
from utils.io.files import compose_dirpath


@compose_dirpath
def plot_attention_maps_wrapper(
    dir_path,
    attention_dict,
    model_name,
    execution_function,
    layer_idx=0,
    sample_idx=0,
    head_idx=0,
    batch_idx=0,
    x_labels=None,
    y_labels=None,
    **execution_kwargs,
):
    """
    Plot attention maps as heatmaps for a given layer, head, batch, and simulation sample.

    Args:
        dir_path (str): Directory path to save the heatmap image.
        attention_dict (dict): Dictionary where:
                              - Keys are model names (str);
                              - Values are lists of attention data for each sample, where each
                                element is a dictionary containing: 'attention_weights' tensor of
                                shape [num_layers, n_heads, batch_size, graph_size, graph_size].
        model_name (str): Name of the model to extract attention maps for.
        execution_function (function): Function that handles the plotting/saving
        layer_idx (int): Index of the layer to visualize.
        sample_idx (int): Index of the simulation sample to visualize.
        head_idx (int): Index of the head to visualize (-1 for average over all heads).
        batch_idx (int): Index of the data batch to visualize (-1 for average over all batches).
        x_labels (list, optional): Custom labels for x-axis vertices.
        y_labels (list, optional): Custom labels for y-axis vertices.
        **execution_kwargs: Additional arguments to pass to the execution function.

    Returns:
        attn_map (np.ndarray): The attention map as a Numpy array.
    """
    assert sample_idx >= 0, f"sample_idx {sample_idx} must be a non-negative integer"

    attention_weights = attention_dict[model_name][sample_idx]["attention_weights"]
    assert layer_idx < attention_weights.shape[0], (
        f"layer_idx {layer_idx} exceeds number of layers {attention_weights.shape[0]}"
    )
    assert head_idx < attention_weights.shape[1], (
        f"head_idx {head_idx} exceeds number of heads {attention_weights.shape[1]}"
    )
    assert batch_idx < attention_weights.shape[2], (
        f"layer_idx {batch_idx} exceeds batch size {attention_weights.shape[2]}"
    )

    # Extract attention map
    if head_idx >= 0:
        if batch_idx >= 0:
            attn_map = attention_weights[layer_idx, head_idx, batch_idx].cpu().numpy()
            title = "Attention Map (Layer {}, Head {}, Batch {})".format(layer_idx, head_idx, batch_idx)
            attention_filename = os.path.join(
                dir_path,
                "attention_maps",
                model_name,
                f"layer{layer_idx}_head{head_idx}_map{sample_idx}.png",
            )
        else:
            attn_map = attention_weights[layer_idx, head_idx, :].mean(dim=0).cpu().numpy()  # Average over batches
            title = "Attention Map Average Over All Batches (Layer {}, Head {})".format(layer_idx, head_idx)
            attention_filename = os.path.join(
                dir_path,
                "attention_maps",
                model_name,
                f"layer{layer_idx}_head{head_idx}_map{sample_idx}.png",
            )
    elif batch_idx >= 0:
        attn_map = attention_weights[layer_idx, :, batch_idx].mean(dim=0).cpu().numpy()  # Average over heads
        title = "Attention Map Average Over All Heads (Layer {}, Batch {})".format(layer_idx, batch_idx)
        attention_filename = os.path.join(
            dir_path,
            "attention_maps",
            model_name,
            f"layer{layer_idx}_headavg_map{sample_idx}.png",
        )
    else:
        attn_map = attention_weights[layer_idx, :, :].mean(dim=(0, 1)).cpu().numpy()  # Average over heads and batches
        title = "Attention Map Average Over All Heads and Batches (Layer {})".format(layer_idx)
        attention_filename = os.path.join(
            dir_path,
            "attention_maps",
            model_name,
            f"layer{layer_idx}_headavg_map{sample_idx}.png",
        )

    try:
        os.makedirs(os.path.dirname(attention_filename), exist_ok=True)
    except Exception as e:
        raise Exception("directories to save attention maps do not exist and could not be created") from e

    # Dynamically set figure size based on map_size
    base_vertexsize = 0.5
    map_size = math.isqrt(attn_map.shape[0] * attn_map.shape[1])
    min_figsize = 6.0
    max_figsize = 30.0
    figsize = min(max(min_figsize, base_vertexsize * map_size), max_figsize)
    fig = plt.figure(figsize=(figsize, figsize))

    # Adjust annotations and font sizes to scale inversely with map_size
    max_ticsize = 8
    max_annotsize = 8
    annot = map_size <= 55  # Disable annotations for large graphs to avoid clutter
    tick_fontsize = max(max_ticsize, 14 - map_size // 10)
    annot_fontsize = max(max_annotsize, 12 - map_size // 10)

    # Plot and/or log attention heatmap
    plt.title(title)
    sns.heatmap(
        attn_map,
        annot=annot,
        cmap="viridis",
        fmt=".2f",
        cbar=True,
        annot_kws={"fontsize": annot_fontsize},
    )
    plt.xlabel("Key Vertices")
    plt.ylabel("Query Vertices")
    if x_labels is None:
        x_labels = [f"Vertex {i}" for i in range(attn_map.shape[0])]
    if y_labels is None:
        y_labels = [f"Vertex {i}" for i in range(attn_map.shape[1])]
    plt.xticks(
        ticks=range(attn_map.shape[0]),
        labels=x_labels,
        rotation=45,
        fontsize=tick_fontsize,
    )
    plt.yticks(
        ticks=range(attn_map.shape[1]),
        labels=y_labels,
        rotation=0,
        fontsize=tick_fontsize,
    )
    plt.tight_layout()
    execution_function(
        plot_target=attn_map,
        fig=fig,
        title=title,
        figsize=figsize,
        x_labels=x_labels,
        y_labels=y_labels,
        fig_filename=attention_filename,
        **execution_kwargs,
    )
    return attn_map
