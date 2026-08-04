"""Reproducible figures for the exploratory KernelBench agent pilot."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from abstrak.evaluation.agent_analysis import load_agent_metrics

_TARGET_COLORS = ("#0072B2", "#D55E00", "#009E73")
_MARKERS = ("o", "s", "^")
_LINE_STYLES = ("-", "--", ":")
_MISSING_COLOR = "#D9D9D9"


class AgentFigureError(ValueError):
    """Raised when derived pilot metrics cannot be plotted."""


def _display_target(target: str) -> str:
    return {"triton": "Triton", "tilelang": "TileLang", "cute": "CuTe"}.get(
        target, target
    )


def _style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }


def _target_styles(targets: Sequence[str]) -> dict[str, tuple[str, str, str]]:
    if len(targets) > len(_TARGET_COLORS):
        raise AgentFigureError("pilot renderer supports at most three target series")
    return {
        target: (
            _TARGET_COLORS[index],
            _MARKERS[index],
            _LINE_STYLES[index],
        )
        for index, target in enumerate(targets)
    }


def _task_title(task: Mapping[str, str]) -> str:
    name = task["name"]
    if name == task["ref"]:
        return task["ref"]
    return f"{task['ref']}\n{name.replace('_', ' ')}"


def _profile_figure(metrics: Mapping[str, Any]) -> Figure:
    models = metrics["models"]
    tasks = metrics["tasks"]
    targets = metrics["targets"]
    iterations = int(metrics["iterations"])
    styles = _target_styles(targets)
    figure = Figure(
        figsize=(max(8.0, 2.75 * len(tasks)), max(4.4, 2.35 * len(models))),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(models), len(tasks), squeeze=False, sharex=True)
    rows = {
        (row["model_id"], row["task_ref"], row["target"], row["iteration"]): row
        for row in metrics["curve_rows"]
    }

    for model_index, model_id in enumerate(models):
        for task_index, task in enumerate(tasks):
            axis = axes[model_index][task_index]
            for target_index, target in enumerate(targets):
                color, marker, line_style = styles[target]
                values: list[float] = []
                missing_iterations: list[int] = []
                for iteration in range(1, iterations + 1):
                    row = rows.get((model_id, task["ref"], target, iteration))
                    value = row.get("best_speedup") if row is not None else None
                    if value is None:
                        values.append(math.nan)
                        missing_iterations.append(iteration)
                    else:
                        values.append(float(value))
                axis.plot(
                    range(1, iterations + 1),
                    values,
                    color=color,
                    marker=marker,
                    linestyle=line_style,
                    label=_display_target(target),
                )
                if missing_iterations:
                    axis.scatter(
                        missing_iterations,
                        [0.025 + 0.027 * target_index] * len(missing_iterations),
                        transform=axis.get_xaxis_transform(),
                        color=color,
                        marker="x",
                        linewidths=1.1,
                        clip_on=False,
                        zorder=4,
                    )

            axis.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
            axis.grid(axis="y")
            axis.set_xticks(range(1, iterations + 1))
            if model_index == 0:
                axis.set_title(_task_title(task))
            if task_index == 0:
                axis.set_ylabel(f"{model_id}\nBest correct speedup (x)")
            if model_index == len(models) - 1:
                axis.set_xlabel("Agent iteration")

    handles = [
        Line2D(
            [0],
            [0],
            color=styles[target][0],
            marker=styles[target][1],
            linestyle=styles[target][2],
            label=_display_target(target),
        )
        for target in targets
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#777777",
                marker="x",
                linestyle="none",
                label="No correct result",
            ),
            Line2D(
                [0],
                [0],
                color="#777777",
                linestyle="--",
                label="1.0x baseline",
            ),
        ]
    )
    figure.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
    return figure


def _winner_and_gain_figure(metrics: Mapping[str, Any]) -> Figure:
    models = metrics["models"]
    tasks = metrics["tasks"]
    targets = metrics["targets"]
    iterations = int(metrics["iterations"])
    styles = _target_styles(targets)
    figure = Figure(
        figsize=(11.0, max(4.8, 2.75 * len(models))),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(len(models), 2, width_ratios=(1.45, 1.0))
    winner_rows = {
        (row["model_id"], row["task_ref"], row["iteration"]): row
        for row in metrics["winners"]
    }
    aggregate_rows = {
        (row["model_id"], row["iteration"]): row for row in metrics["aggregates"]
    }
    gain_rows = {
        (row["model_id"], row["task_ref"], row["iteration"]): row
        for row in metrics["task_oracle_gains"]
    }
    color_map = ListedColormap([_MISSING_COLOR, *(styles[target][0] for target in targets)])

    for model_index, model_id in enumerate(models):
        heatmap_axis = figure.add_subplot(grid[model_index, 0])
        gain_axis = figure.add_subplot(grid[model_index, 1])
        cells: list[list[int]] = []
        labels: list[list[str]] = []
        for task in tasks:
            task_cells: list[int] = []
            task_labels: list[str] = []
            for iteration in range(1, iterations + 1):
                row = winner_rows.get((model_id, task["ref"], iteration))
                winner = row.get("winner_target") if row is not None else None
                if winner is None:
                    task_cells.append(0)
                    task_labels.append("NA")
                else:
                    task_cells.append(targets.index(winner) + 1)
                    task_labels.append(_display_target(winner))
            cells.append(task_cells)
            labels.append(task_labels)
        heatmap_axis.imshow(
            cells,
            cmap=color_map,
            vmin=-0.5,
            vmax=len(targets) + 0.5,
            aspect="auto",
            interpolation="nearest",
        )
        for task_index, task_labels in enumerate(labels):
            for iteration_index, label in enumerate(task_labels):
                heatmap_axis.text(
                    iteration_index,
                    task_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#202020",
                )
        heatmap_axis.set_xticks(range(iterations), range(1, iterations + 1))
        heatmap_axis.set_yticks(
            range(len(tasks)),
            [task["name"].replace("_", " ") for task in tasks],
        )
        heatmap_axis.set_xlabel("Agent iteration")
        heatmap_axis.set_title(f"{model_id}: winning target")

        final_iteration = iterations
        final_aggregate = aggregate_rows[(model_id, final_iteration)]
        final_gains = [
            float(gain_rows[(model_id, task["ref"], final_iteration)]["oracle_gain_percent"])
            for task in tasks
        ]
        final_winners = [
            winner_rows[(model_id, task["ref"], final_iteration)]["winner_target"]
            for task in tasks
        ]
        bar_colors = [
            styles[winner][0] if winner is not None else _MISSING_COLOR
            for winner in final_winners
        ]
        positions = list(range(len(tasks)))
        gain_axis.barh(positions, final_gains, color=bar_colors, height=0.62)
        gain_axis.axvline(0.0, color="#777777", linewidth=0.8)
        gain_axis.set_yticks(
            positions,
            [task["name"].replace("_", " ") for task in tasks],
        )
        gain_axis.invert_yaxis()
        gain_axis.set_xlabel("Oracle gain over best fixed (%)")
        gain_axis.grid(axis="x")
        for position, value in zip(positions, final_gains, strict=True):
            gain_axis.text(
                value,
                position,
                f" {value:.1f}%",
                ha="left",
                va="center",
                fontsize=7,
            )
        gain_axis.set_title(
            f"Iteration {final_iteration}: best fixed = "
            f"{_display_target(final_aggregate['best_fixed_target'])}; "
            f"overall +{final_aggregate['oracle_gain_percent']:.1f}%"
        )

    legend_handles = [
        Patch(facecolor=styles[target][0], label=_display_target(target)) for target in targets
    ]
    legend_handles.append(Patch(facecolor=_MISSING_COLOR, label="No correct result"))
    figure.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=len(legend_handles),
        frameon=False,
    )
    return figure


def _save_figure(figure: Figure, base_path: Path) -> tuple[Path, Path]:
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    metadata = {
        "Creator": "AbstraK KernelBench agent pilot",
        "Title": base_path.name,
    }
    figure.savefig(png_path, dpi=180, metadata=metadata, facecolor="white", edgecolor="white")
    figure.savefig(pdf_path, metadata=metadata, facecolor="white", edgecolor="white")
    figure.clear()
    return png_path, pdf_path


def plot_agent_run(run_directory: str | Path) -> tuple[Path, Path, Path, Path]:
    """Render exactly the two planned figure bases from derived metrics."""

    import matplotlib

    run_path = Path(run_directory).expanduser().resolve()
    metrics = load_agent_metrics(run_path)
    figure_path = run_path / "figures"
    figure_path.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(_style()):
        profile_paths = _save_figure(
            _profile_figure(metrics), figure_path / "01_anytime_performance_profiles"
        )
        winner_paths = _save_figure(
            _winner_and_gain_figure(metrics), figure_path / "02_winner_map_and_oracle_gain"
        )
    return (*profile_paths, *winner_paths)
