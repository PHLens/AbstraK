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
from matplotlib.ticker import PercentFormatter

from abstrak.evaluation.agent_analysis import load_agent_metrics

_TARGET_COLORS = ("#0072B2", "#D55E00", "#009E73")
_MARKERS = ("o", "s", "^")
_LINE_STYLES = ("-", "--", ":")
_MISSING_COLOR = "#D9D9D9"
_TIE_COLOR = "#9E9E9E"
_STRATEGY_STYLES = {
    "best_fixed": ("#0072B2", "o", "-", "Best fixed (B to one target)"),
    "oracle": ("#009E73", "^", "--", "Free best-of (B per target)"),
    "equal_split": ("#D55E00", "s", ":", "Equal split (B total)"),
}


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


def _reference_lookup(metrics: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (row["task_ref"], row["target"]): row
        for row in metrics.get("reference_rows", [])
    }


def _profile_figure(metrics: Mapping[str, Any]) -> Figure:
    models = metrics["models"]
    tasks = metrics["tasks"]
    targets = metrics["targets"]
    iterations = int(metrics["iterations"])
    styles = _target_styles(targets)
    references = _reference_lookup(metrics)
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
                reference = references.get((task["ref"], target))
                if reference is not None:
                    axis.scatter(
                        [iterations],
                        [float(reference["reference_speedup"])],
                        marker="D",
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.1,
                        zorder=5,
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
    if references:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="D",
                markerfacecolor="none",
                linestyle="none",
                label="Measured reference",
            )
        )
    figure.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
    return figure


def _token_profile_figure(metrics: Mapping[str, Any]) -> Figure:
    models = metrics["models"]
    tasks = metrics["tasks"]
    targets = metrics["targets"]
    styles = _target_styles(targets)
    references = _reference_lookup(metrics)
    token_rows = metrics["token_curve_rows"]
    usage_rows = {
        (row["model_id"], row["task_ref"], row["target"]): row
        for row in metrics["trajectory_usage_rows"]
    }
    figure = Figure(
        figsize=(max(8.0, 2.75 * len(tasks)), max(4.4, 2.35 * len(models))),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(models), len(tasks), squeeze=False, sharex="row")

    for model_index, model_id in enumerate(models):
        model_budgets = sorted(
            {
                int(row["token_budget"])
                for row in token_rows
                if row["model_id"] == model_id
            }
        )
        reference_x = model_budgets[-1] if model_budgets else 0
        for task_index, task in enumerate(tasks):
            axis = axes[model_index][task_index]
            for target_index, target in enumerate(targets):
                color, marker, line_style = styles[target]
                rows = sorted(
                    (
                        row
                        for row in token_rows
                        if row["model_id"] == model_id
                        and row["task_ref"] == task["ref"]
                        and row["target"] == target
                        and row["budget_status"] == "exact"
                    ),
                    key=lambda row: int(row["token_budget"]),
                )
                budgets = [int(row["token_budget"]) for row in rows]
                values = [
                    (
                        float(row["best_speedup"])
                        if row.get("best_speedup") is not None
                        else math.nan
                    )
                    for row in rows
                ]
                axis.step(
                    budgets,
                    values,
                    where="post",
                    color=color,
                    marker=marker,
                    linestyle=line_style,
                    label=_display_target(target),
                )
                missing_budgets = [
                    int(row["token_budget"])
                    for row in rows
                    if row.get("best_speedup") is None
                ]
                if missing_budgets:
                    axis.scatter(
                        missing_budgets,
                        [0.025 + 0.027 * target_index] * len(missing_budgets),
                        transform=axis.get_xaxis_transform(),
                        color=color,
                        marker="x",
                        linewidths=1.0,
                        clip_on=False,
                        zorder=4,
                    )

                usage = usage_rows[(model_id, task["ref"], target)]
                if usage["censored"]:
                    prefix = int(usage["exact_prefix_tokens"])
                    prefix_row = next(
                        (row for row in reversed(rows) if int(row["token_budget"]) <= prefix),
                        None,
                    )
                    prefix_value = (
                        prefix_row.get("best_speedup") if prefix_row is not None else None
                    )
                    if prefix_value is None:
                        axis.scatter(
                            [prefix],
                            [0.11 + 0.027 * target_index],
                            transform=axis.get_xaxis_transform(),
                            color=color,
                            marker="|",
                            s=80,
                            linewidths=1.5,
                            clip_on=False,
                            zorder=5,
                        )
                    else:
                        axis.scatter(
                            [prefix],
                            [float(prefix_value)],
                            color=color,
                            marker="|",
                            s=80,
                            linewidths=1.5,
                            zorder=5,
                        )

                reference = references.get((task["ref"], target))
                if reference is not None:
                    axis.scatter(
                        [reference_x],
                        [float(reference["reference_speedup"])],
                        marker="D",
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.1,
                        zorder=6,
                    )

            axis.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
            axis.grid(axis="y")
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
            if model_budgets:
                axis.set_xlim(
                    (0, model_budgets[-1] * 1.05)
                    if model_budgets[-1] > 0
                    else (-0.5, 0.5)
                )
            if model_index == 0:
                axis.set_title(_task_title(task))
            if task_index == 0:
                axis.set_ylabel(f"{model_id}\nBest correct speedup (x)")
            if model_index == len(models) - 1:
                axis.set_xlabel("Cumulative model tokens")

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
                marker="|",
                markersize=9,
                linestyle="none",
                label="Exact prefix ends",
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
    if references:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="D",
                markerfacecolor="none",
                linestyle="none",
                label="Measured reference",
            )
        )
    figure.legend(
        handles=handles,
        loc="outside upper center",
        ncol=min(4, len(handles)),
        frameon=False,
    )
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
    color_map = ListedColormap(
        [_MISSING_COLOR, *(styles[target][0] for target in targets), _TIE_COLOR]
    )

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
                tied = row.get("tied_targets", []) if row is not None else []
                winner_status = row.get("winner_status") if row is not None else None
                if winner_status == "tie" or len(tied) > 1:
                    task_cells.append(len(targets) + 1)
                    task_labels.append("Tie")
                elif winner is None:
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
            vmax=len(targets) + 1.5,
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
        final_winner_rows = [
            winner_rows[(model_id, task["ref"], final_iteration)] for task in tasks
        ]
        bar_colors = []
        for row in final_winner_rows:
            tied = row.get("tied_targets", [])
            if row.get("winner_status") == "tie" or len(tied) > 1:
                bar_colors.append(_TIE_COLOR)
            elif row["winner_target"] is None:
                bar_colors.append(_MISSING_COLOR)
            else:
                bar_colors.append(styles[row["winner_target"]][0])
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
        gain_axis.margins(x=0.12)
        for position, value in zip(positions, final_gains, strict=True):
            gain_axis.text(
                value,
                position,
                f" {value:.1f}%",
                ha="left",
                va="center",
                fontsize=7,
            )
        tied_fixed = final_aggregate.get("tied_best_fixed_targets", [])
        fixed_label = (
            "Tie: " + " / ".join(_display_target(target) for target in tied_fixed)
            if len(tied_fixed) > 1
            else _display_target(final_aggregate["best_fixed_target"])
        )
        fixed_coverage = final_aggregate.get("best_fixed_correctness_coverage")
        oracle_coverage = final_aggregate.get("oracle_correctness_coverage")
        coverage_label = (
            f"; correct {fixed_coverage:.0%} -> {oracle_coverage:.0%}"
            if fixed_coverage is not None and oracle_coverage is not None
            else ""
        )
        gain_axis.set_title(
            f"Iteration {final_iteration}: best fixed = {fixed_label}\n"
            f"Oracle gain +{final_aggregate['oracle_gain_percent']:.1f}%{coverage_label}"
        )

    legend_handles = [
        Patch(facecolor=styles[target][0], label=_display_target(target)) for target in targets
    ]
    legend_handles.append(Patch(facecolor=_TIE_COLOR, label="Tie"))
    legend_handles.append(Patch(facecolor=_MISSING_COLOR, label="No correct result"))
    figure.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=len(legend_handles),
        frameon=False,
    )
    return figure


def _portfolio_figure(metrics: Mapping[str, Any]) -> Figure:
    models = metrics["models"]
    aggregate_rows = metrics["token_aggregates"]
    figure = Figure(
        figsize=(10.5, max(4.5, 2.8 * len(models))),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(models), 2, squeeze=False, sharex="row")
    series = (
        (
            "best_fixed",
            "budget_status",
            "best_fixed_geomean_utility",
            "best_fixed_correctness_coverage",
        ),
        (
            "oracle",
            "budget_status",
            "oracle_geomean_utility",
            "oracle_correctness_coverage",
        ),
        (
            "equal_split",
            "equal_split_budget_status",
            "equal_split_geomean_utility",
            "equal_split_correctness_coverage",
        ),
    )

    for model_index, model_id in enumerate(models):
        utility_axis = axes[model_index][0]
        coverage_axis = axes[model_index][1]
        rows = sorted(
            (row for row in aggregate_rows if row["model_id"] == model_id),
            key=lambda row: int(row["token_budget"]),
        )
        for strategy, status_key, utility_key, coverage_key in series:
            color, marker, line_style, label = _STRATEGY_STYLES[strategy]
            exact_rows = [
                row
                for row in rows
                if row.get(status_key) == "exact" and row.get(utility_key) is not None
            ]
            budgets = [int(row["token_budget"]) for row in exact_rows]
            utilities = [float(row[utility_key]) for row in exact_rows]
            coverages = [float(row[coverage_key]) for row in exact_rows]
            utility_axis.step(
                budgets,
                utilities,
                where="post",
                color=color,
                marker=marker,
                linestyle=line_style,
                label=label,
            )
            coverage_axis.step(
                budgets,
                coverages,
                where="post",
                color=color,
                marker=marker,
                linestyle=line_style,
                label=label,
            )
            if exact_rows and any(
                row.get(status_key) != "exact"
                and int(row["token_budget"]) > budgets[-1]
                for row in rows
            ):
                utility_axis.scatter(
                    [budgets[-1]],
                    [utilities[-1]],
                    color=color,
                    marker="|",
                    s=90,
                    linewidths=1.6,
                    zorder=5,
                )
                coverage_axis.scatter(
                    [budgets[-1]],
                    [coverages[-1]],
                    color=color,
                    marker="|",
                    s=90,
                    linewidths=1.6,
                    zorder=5,
                )

        utility_axis.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
        utility_axis.set_ylim(bottom=0.95)
        utility_axis.set_ylabel(f"{model_id}\nDeployment utility (geomean x)")
        utility_axis.set_title("Performance with 1.0x fallback")
        utility_axis.grid(axis="y")
        coverage_axis.set_ylim(0.0, 1.02)
        coverage_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        coverage_axis.set_ylabel("Correct workload coverage")
        coverage_axis.set_title("Correctness coverage")
        coverage_axis.grid(axis="y")
        for axis in (utility_axis, coverage_axis):
            axis.set_xlabel("Token budget B per workload")
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))

    handles = [
        Line2D(
            [0],
            [0],
            color=style[0],
            marker=style[1],
            linestyle=style[2],
            label=style[3],
        )
        for style in _STRATEGY_STYLES.values()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#777777",
            marker="|",
            markersize=9,
            linestyle="none",
            label="Exact range ends",
        )
    )
    figure.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
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


def plot_agent_run(run_directory: str | Path) -> tuple[Path, ...]:
    """Render iteration views and token views when exact token metrics exist."""

    import matplotlib

    run_path = Path(run_directory).expanduser().resolve()
    metrics = load_agent_metrics(run_path)
    figure_path = run_path / "figures"
    figure_path.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(_style()):
        paths: list[Path] = []
        profile_paths = _save_figure(
            _profile_figure(metrics), figure_path / "01_anytime_performance_profiles"
        )
        paths.extend(profile_paths)
        has_token_metrics = bool(metrics.get("token_curve_rows")) and bool(
            metrics.get("token_aggregates")
        )
        if has_token_metrics:
            paths.extend(
                _save_figure(
                    _token_profile_figure(metrics),
                    figure_path / "01b_token_performance_profiles",
                )
            )
        winner_paths = _save_figure(
            _winner_and_gain_figure(metrics), figure_path / "02_winner_map_and_oracle_gain"
        )
        paths.extend(winner_paths)
        if has_token_metrics:
            paths.extend(
                _save_figure(
                    _portfolio_figure(metrics),
                    figure_path / "03_token_budget_portfolio",
                )
            )
    return tuple(paths)
