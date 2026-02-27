"""
TSP and VRP route visualization.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def draw_graph(distance_matrix):
    """
    Draws a networkx graph from a distance matrix using spring layout.

    Args:
        distance_matrix (np.ndarray): The adjacency/distance matrix.
    """
    # 1. Cast nx to Any to bypass the missing-attribute checks
    nx_any = cast(Any, nx)

    G = nx.from_numpy_array(distance_matrix)

    # 2. Use the casted object for the layout and drawing
    pos = nx_any.spring_layout(G)
    nx_any.draw(G, pos, with_labels=True)

    labels = {(u, v): str(attr["weight"]) for u, v, attr in G.edges(data=True)}
    print(G.edges(data=True))

    nx_any.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py
def plot_tsp(xy, tour, ax1):
    """
    Plot the TSP tour on matplotlib axis ax1.

    Args:
        xy (np.ndarray): Node coordinates [N, 2].
        tour (np.ndarray): Tour indices [N+1] (including return to start).
        ax1 (matplotlib.axes.Axes): The axis to plot on.
    """
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    xs, ys = xy[tour].transpose()
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()

    # Scatter nodes
    ax1.scatter(xs, ys, s=40, color="blue")
    # Starting node
    ax1.plot(xy[0][0], xy[0][1], "sk", markersize=15, zorder=5)
    ax1.scatter([xs[0]], [ys[0]], s=100, color="red", zorder=4)

    # Arcs
    ax1.quiver(
        xs,
        ys,
        dx,
        dy,
        scale_units="xy",
        angles="xy",
        scale=1,
    )

    ax1.set_title("{} nodes, total length {:.2f}".format(len(tour), lengths[-1]))


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map.

    Args:
        N (int): Number of bins.
        base_cmap (str or Colormap, optional): Base colormap name.

    Returns:
        Colormap: Discretized colormap.
    """
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    # Use the more modern and robust colormaps registry
    if base_cmap is None:
        base_cmap = "viridis"
    base = colormaps.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = f"{getattr(base, 'name', 'custom')}_{N}"
    return LinearSegmentedColormap.from_list(cmap_name, color_list, N)


def plot_vehicle_routes(
    data,
    route,
    ax1,
    markersize=5,
    visualize_waste=False,
    waste_scale=1,
    round_waste=False,
):
    """
    Plot the vehicle routes on matplotlib axis ax1.

    Args:
        data (dict): Dictionary with 'depot', 'locs', 'waste'.
        route (Tensor): Route indices (single sequence with delimiters).
        ax1 (matplotlib.axes.Axes): Axis to plot on.
        markersize (int, optional): Size of markers. Defaults to 5.
        visualize_waste (bool, optional): Visualize waste levels as bars. Defaults to False.
        waste_scale (float, optional): Scaling factor for waste values. Defaults to 1.
        round_waste (bool, optional): Round waste values in labels. Defaults to False.
    """
    # Route is one sequence, separating different routes with 0 (depot)
    routes = [r[r != 0] for r in np.split(route.cpu().numpy(), np.where(route == 0)[0]) if (r != 0).any()]
    depot = data["depot"].cpu().numpy()
    locs_tensor = data.get("locs") if "locs" in data.keys() else data.get("loc")
    locs = locs_tensor.cpu().numpy()
    wastes = data.get("waste", data.get("waste")).cpu().numpy() * waste_scale
    capacity = waste_scale  # capacity is always 1

    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, "sk", markersize=markersize * 4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper center")

    cmap = discrete_cmap(len(routes) + 2, "nipy_spectral")
    waste_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number)  # invert to have in rainbow order

        route_waste = wastes[r - 1]
        coords = locs[r - 1, :]
        xs, ys = coords.transpose()

        total_route_waste = sum(route_waste)
        assert total_route_waste <= capacity
        if not visualize_waste:
            ax1.plot(xs, ys, "o", mfc=color, markersize=markersize, markeredgewidth=0.0)

        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_waste = 0
        for (x, y), d in zip(coords, route_waste):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_waste / capacity))
            waste_rects.append(Rectangle((x, y + 0.1 * cum_waste / capacity), 0.01, 0.1 * d / capacity))

            x_prev, y_prev = x, y
            cum_waste += d

        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color=color,
            label="R{}, # {}, c {} / {}, d {:.2f}".format(
                veh_number,
                len(r),
                int(total_route_waste) if round_waste else total_route_waste,
                int(capacity) if round_waste else capacity,
                dist,
            ),
        )

        qvs.append(qv)

    ax1.set_title("{} routes, total distance {:.2f}".format(len(routes), total_dist))
    ax1.legend(handles=qvs)

    pc_cap = PatchCollection(cap_rects, facecolor="whitesmoke", alpha=1.0, edgecolor="lightgray")
    pc_used = PatchCollection(used_rects, facecolor="lightgray", alpha=1.0, edgecolor="lightgray")
    pc_waste = PatchCollection(waste_rects, facecolor="black", alpha=1.0, edgecolor="black")

    if visualize_waste:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_waste)
