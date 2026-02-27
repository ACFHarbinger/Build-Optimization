"""
Line chart plotting utilities.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def plot_linechart(
    output_dest,
    graph_log,
    plot_func,
    policies,
    x_label=None,
    y_label=None,
    title=None,
    fsave=True,
    scale="linear",
    x_values=None,
    linestyles=None,
    markers=None,
    annotate=True,
    pareto_front=False,
):
    """
    Plots a generic line chart, optionally handling multiple policies and Pareto fronts.

    Args:
        output_dest (str): File path to save the plot.
        graph_log (np.ndarray): Data to plot.
        plot_func (callable): The pyplot plotting function (e.g., plt.plot, plt.scatter).
        policies (list): List of policy names for the legend.
        x_label (str, optional): Label for x-axis.
        y_label (str, optional): Label for y-axis.
        title (str, optional): Chart title.
        fsave (bool, optional): Whether to save figure to file. Defaults to True.
        scale (str, optional): Axis scale ('linear', 'log'). Defaults to "linear".
        x_values (list, optional): Custom x-values.
        linestyles (list, optional): List of linestyles.
        markers (list, optional): List of markers.
        annotate (bool, optional): Whether to annotate points. Defaults to True.
        pareto_front (bool, optional): Whether to calculate and overlay Pareto front. Defaults to False.

    Returns:
        list or None: Pareto dominants if pareto_front is True, else None.
    """

    def plot_graphs_out(plot_func, graph_log, x_values, linestyles, markers):
        """
        Helper to plot graphs for different policies.

        Args:
            plot_func: Function to plot a single line.
            graph_log: Log data.
            x_values: X-axis values.
            linestyles: Line styles.
            markers: Markers.
        """
        points_by_nbins = {}  # type: ignore[var-annotated]
        for id, lg in enumerate(zip(*graph_log)):
            to_plot = (*lg,) if x_values is None else (x_values, *lg)

            line = linestyles[id % len(linestyles)] if linestyles is not None else False

            mark = markers[id % len(markers)] if markers is not None else False

            if not line and not mark:
                plot_func(*to_plot)
            elif not mark:
                plot_func(*to_plot, linestyle=line)
            elif not line:
                plot_func(*to_plot, marker=mark)
            else:
                plot_func(*to_plot, linestyle=line, marker=mark)

            for id, (x, y) in enumerate(zip(list(zip(*lg))[0], list(zip(*lg))[5])):
                if id not in points_by_nbins:
                    points_by_nbins[id] = []
                points_by_nbins[id].append((x, y))
        return points_by_nbins

    points_by_nbins: Dict[int, List[Tuple[float, float]]] = {}
    plt.figure(dpi=200)
    if title is not None:
        plt.title(title)

    if len(graph_log.shape) == 2:
        points_by_nbins = _plot_2d_graph(plot_func, graph_log, markers)
    elif len(graph_log.shape) == 3:
        points_by_nbins = plot_graphs_out(plot_func, graph_log, x_values, linestyles, markers)
        if annotate:
            _annotate_plot(graph_log)

    # Pareto front for minimizing x and maximizing y
    pareto_dominants = []
    if pareto_front:
        for id_nbins, points in points_by_nbins.items():
            dominance_ls = _calculate_dominance(points)
            pareto_dominants.append(dominance_ls)
            _plot_pareto_front(points, dominance_ls, id_nbins)

    _set_plot_attributes(scale, x_label, y_label, policies)

    if fsave:
        _save_plot(output_dest, x_values)
    plt.show()
    return pareto_dominants if pareto_front else None


def _plot_2d_graph(plot_func, graph_log, markers) -> Dict[int, List[Tuple[float, float]]]:
    """Helper to plot 2D graph log data."""
    points_by_nbins: Dict[int, List[Tuple[float, float]]] = {0: []}
    for id_val, ll in enumerate(graph_log):
        if markers is not None:
            mark = markers[id_val % len(markers)]
            plot_func(ll, mark)
        else:
            plot_func(ll)
        points_by_nbins[0].append((ll[0], ll[5]))
    return points_by_nbins


def _annotate_plot(graph_log) -> None:
    """Helper to annotate plot points."""
    for lg in zip(*graph_log):
        for id_val, xy in enumerate(zip(list(zip(*lg))[0], list(zip(*lg))[5])):
            if id_val == graph_log.shape[0] - 1:
                _add_scatter_marker(xy)


def _set_plot_attributes(scale, x_label, y_label, policies) -> None:
    """Helper to set plot scales, labels and legend."""
    if scale != "linear":
        plt.yscale(scale)
        plt.xscale(scale)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.legend(policies)


def _save_plot(output_dest, x_values) -> None:
    """Helper to save plot with appropriate parameters."""
    if x_values is not None:
        plt.savefig(output_dest)
    else:
        plt.savefig(output_dest, bbox_inches="tight")


def _add_scatter_marker(xy: Tuple[float, float]) -> None:
    """Helper to add a specific scatter marker to the plot."""
    plt.scatter(
        xy[0],
        xy[1],
        s=200,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1,
        zorder=10,
    )


def _plot_pareto_front(points: List[Tuple[float, float]], dominance_ls: List[int], id_nbins: int) -> None:
    """Helper to calculate and plot Pareto front for a set of points."""
    pareto_points = [p for i, p in enumerate(points) if dominance_ls[i] == 1]
    pareto_points.sort(key=lambda p: p[0])

    pareto_x = [p[0] for p in pareto_points]
    pareto_y = [p[1] for p in pareto_points]

    label = f"Pareto Front (ID {id_nbins})"
    plt.plot(
        pareto_x,
        pareto_y,
        "--",
        color="black",
        linewidth=2,
        label=label,
        zorder=5,
    )


def _calculate_dominance(points: List[Tuple[float, float]]) -> List[int]:
    """Calculate Pareto dominance for a list of points (min x, max y)."""
    dominance_ls = [0] * len(points)
    for id_point, point in enumerate(points):
        dominated = False
        for other_point in points:
            # Check if other_point dominates point
            if (
                other_point[0] <= point[0]
                and other_point[1] >= point[1]
                and (other_point[0] < point[0] or other_point[1] > point[1])
            ):
                dominated = True
                break
        if not dominated:
            dominance_ls[id_point] = 1
    return dominance_ls
