"""Deterministic tables and publication figures for anytime analysis reports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from abstrak.anytime.analysis import (
    AnytimeAnalysisDataset,
    AnytimeAnalysisReport,
    AnytimeAnalysisSpec,
    AnytimeAnalysisTable,
    AnytimeMatchedObservation,
    AnytimeWinnerRow,
    anytime_analysis_tables,
    build_anytime_analysis,
)
from abstrak.anytime.contracts import IDENTIFIER_PATTERN, SHA256_PATTERN, AnytimeModel
from abstrak.providers.contracts import sha256_json

_TARGET_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
_LINESTYLES = ("-", "--", "-.", ":", "-", "--", "-.", ":")
_STAGE_LABELS = {
    "synthetic_fixture": "SYNTHETIC FIXTURE",
    "shakeout": "SHAKEOUT",
    "formal": "FORMAL",
}


class AnytimeFigureError(ValueError):
    """Raised when a deterministic analysis bundle cannot be rendered safely."""


class AnytimeDerivedFile(AnytimeModel):
    schema_version: Literal["abstrak-anytime-derived-file.v1"] = "abstrak-anytime-derived-file.v1"
    relative_path: str = Field(min_length=1)
    role: Literal[
        "analysis_report",
        "analysis_table_json",
        "analysis_table_csv",
        "figure_png",
        "figure_svg",
        "figure_pdf",
        "figure_manifest",
    ]
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=1)

    @field_validator("relative_path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("derived file path must be a safe relative POSIX path")
        return value


class AnytimeSeriesStyle(AnytimeModel):
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    color: str = Field(pattern=r"^#[0-9A-F]{6}$")
    marker: str = Field(min_length=1, max_length=2)
    line_style: str = Field(min_length=1)


class AnytimeRendererEnvironment(AnytimeModel):
    schema_version: Literal["abstrak-anytime-renderer-environment.v1"] = (
        "abstrak-anytime-renderer-environment.v1"
    )
    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    matplotlib_version: str = Field(min_length=1)
    numpy_version: str = Field(min_length=1)
    pillow_version: str = Field(min_length=1)
    freetype_version: str = Field(min_length=1)
    dejavu_sans_sha256: str = Field(pattern=SHA256_PATTERN)
    uv_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    backend_strategy: Literal["object-oriented-format-canvases"] = "object-oriented-format-canvases"

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeFigureArtifact(AnytimeModel):
    schema_version: Literal["abstrak-anytime-figure-artifact.v1"] = (
        "abstrak-anytime-figure-artifact.v1"
    )
    figure_id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: Literal[
        "formal_anytime_small_multiples",
        "exploratory_anytime_small_multiples",
        "stage_rates",
        "final_winner_map",
        "formal_hindsight_gain",
    ]
    study_stage: Literal["synthetic_fixture", "shakeout", "formal"]
    evidence_scope: Literal["formal_only", "formal_and_exploratory", "all_turns"]
    agent_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    title: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    width_inches: float = Field(gt=0)
    height_inches: float = Field(gt=0)
    files: tuple[AnytimeDerivedFile, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def files_cover_raster_and_vector_formats(self) -> AnytimeFigureArtifact:
        roles = tuple(item.role for item in self.files)
        expected = ("figure_png", "figure_svg", "figure_pdf")
        if roles != expected:
            raise ValueError("figure files must be ordered PNG, SVG, PDF")
        stems = {PurePosixPath(item.relative_path).stem for item in self.files}
        if stems != {self.figure_id}:
            raise ValueError("figure files must share the declared figure ID")
        return self


class AnytimeFigureManifest(AnytimeModel):
    schema_version: Literal["abstrak-anytime-figure-manifest.v1"] = (
        "abstrak-anytime-figure-manifest.v1"
    )
    analysis_report_sha256: str = Field(pattern=SHA256_PATTERN)
    input_dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    analysis_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    analysis_code_sha256: str = Field(pattern=SHA256_PATTERN)
    renderer: str = Field(pattern=r"^matplotlib-[0-9]+\.[0-9]+\.[0-9]+$")
    renderer_environment: AnytimeRendererEnvironment
    study_stage: Literal["synthetic_fixture", "shakeout", "formal"]
    target_styles: tuple[AnytimeSeriesStyle, ...]
    figures: tuple[AnytimeFigureArtifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def figure_ids_and_styles_are_unique(self) -> AnytimeFigureManifest:
        ids = tuple(item.figure_id for item in self.figures)
        targets = tuple(item.target_id for item in self.target_styles)
        if len(ids) != len(set(ids)):
            raise ValueError("figure IDs must be unique")
        if len(targets) != len(set(targets)):
            raise ValueError("target figure styles must be unique")
        files = tuple(item.relative_path for figure in self.figures for item in figure.files)
        if len(files) != len(set(files)):
            raise ValueError("figure file paths must be unique across the manifest")
        if any(figure.study_stage != self.study_stage for figure in self.figures):
            raise ValueError("figure stage must match the figure manifest")
        if self.renderer != f"matplotlib-{self.renderer_environment.matplotlib_version}":
            raise ValueError("renderer version must match its environment record")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeAnalysisBundleManifest(AnytimeModel):
    schema_version: Literal["abstrak-anytime-analysis-bundle.v1"] = (
        "abstrak-anytime-analysis-bundle.v1"
    )
    input_dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    analysis_report_sha256: str = Field(pattern=SHA256_PATTERN)
    figure_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    files: tuple[AnytimeDerivedFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def inventory_is_sorted_and_unique(self) -> AnytimeAnalysisBundleManifest:
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("analysis bundle inventory must be sorted and unique")
        if "bundle-manifest.json" in paths:
            raise ValueError("bundle manifest cannot inventory itself")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _canonical_json_bytes(value: Any) -> bytes:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_bytes(
    root: Path,
    relative_path: str,
    payload: bytes,
    *,
    role: str,
) -> AnytimeDerivedFile:
    relative = PurePosixPath(relative_path)
    destination = root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise AnytimeFigureError(f"derived file already exists: {relative_path}") from error
    return AnytimeDerivedFile(
        relative_path=relative_path,
        role=role,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _analysis_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).with_name("analysis.py"), Path(__file__)):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _renderer_environment() -> AnytimeRendererEnvironment:
    import matplotlib
    import matplotlib.font_manager
    import matplotlib.ft2font
    import numpy
    import PIL

    lock_path = Path(__file__).resolve().parents[3] / "uv.lock"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise AnytimeFigureError("renderer environment requires the repository uv.lock")
    font_path = Path(matplotlib.font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if not font_path.is_file() or font_path.is_symlink():
        raise AnytimeFigureError("renderer environment requires a regular DejaVu Sans font")
    return AnytimeRendererEnvironment(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        matplotlib_version=matplotlib.__version__,
        numpy_version=numpy.__version__,
        pillow_version=PIL.__version__,
        freetype_version=matplotlib.ft2font.__freetype_version__,
        dejavu_sans_sha256=hashlib.sha256(font_path.read_bytes()).hexdigest(),
        uv_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    )


def _target_styles(spec: AnytimeAnalysisSpec) -> tuple[AnytimeSeriesStyle, ...]:
    if len(spec.targets) > len(_TARGET_COLORS):
        raise AnytimeFigureError("figure renderer supports at most eight target series")
    return tuple(
        AnytimeSeriesStyle(
            target_id=target,
            color=_TARGET_COLORS[index],
            marker=_MARKERS[index],
            line_style=_LINESTYLES[index % len(_LINESTYLES)],
        )
        for index, target in enumerate(spec.targets)
    )


def _matplotlib_style() -> dict[str, object]:
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
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "abstrak-anytime-figures-v1",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }


def _stage_title(spec: AnytimeAnalysisSpec, title: str) -> str:
    return f"[{_STAGE_LABELS[spec.study_stage]}] {title}"


def _stage_caption(spec: AnytimeAnalysisSpec, caption: str) -> str:
    prefix = {
        "synthetic_fixture": "Synthetic fixture only; no model or GPU result is claimed.",
        "shakeout": "Shakeout evidence only; not a formal result.",
        "formal": (
            "Formal study artifact population; timing claims use independently sealed "
            "checkpoint measurements."
        ),
    }[spec.study_stage]
    return f"{caption} {prefix}"


def _rows_for_point(
    report: AnytimeAnalysisReport,
    *,
    agent_id: str,
    workload_id: str,
    target_id: str,
    call: int,
    formal_only: bool,
) -> tuple[AnytimeMatchedObservation, ...]:
    scope = "formal_only" if formal_only else "formal_and_exploratory"
    return tuple(
        row
        for row in report.matches
        if row.agent_id == agent_id
        and row.workload_id == workload_id
        and row.target_id == target_id
        and row.match_mode == "iteration"
        and row.iteration_budget == call
        and row.evidence_scope == scope
    )


def _median_band(
    rows: Sequence[AnytimeMatchedObservation],
    expected_replicates: int,
) -> tuple[float, float, float]:
    values = sorted(
        float(row.eager_speedup)
        for row in rows
        if row.missing_reason == "none" and row.eager_speedup is not None
    )
    if len(values) != expected_replicates:
        return (math.nan, math.nan, math.nan)
    import statistics

    return (float(statistics.median(values)), values[0], values[-1])


def _anytime_figure(
    dataset: AnytimeAnalysisDataset,
    report: AnytimeAnalysisReport,
    *,
    agent_id: str,
    exploratory: bool,
) -> tuple[Any, str, str, str, str]:
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    spec = dataset.spec
    styles = {style.target_id: style for style in _target_styles(spec)}
    agent_axis = next(axis for axis in spec.agents if axis.agent_id == agent_id)
    calls = (
        tuple(range(1, spec.max_scientific_calls + 1)) if exploratory else spec.formal_checkpoints
    )
    columns = min(3, len(spec.workloads))
    rows_count = math.ceil(len(spec.workloads) / columns)
    width = 3.6 * columns
    height = 2.75 * rows_count + 0.75
    figure = Figure(figsize=(width, height))
    axes = figure.subplots(rows_count, columns, squeeze=False)
    all_axes = list(axes.flat)
    floor_map = {(item.workload_id, item.target_id): item for item in dataset.floors}
    for axis, workload in zip(all_axes, spec.workloads, strict=False):
        plotted_values: list[float] = [1.0]
        for target in spec.targets:
            style = styles[target]
            medians: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for call in calls:
                formal_only = call in spec.formal_checkpoints
                rows = _rows_for_point(
                    report,
                    agent_id=agent_id,
                    workload_id=workload.workload_id,
                    target_id=target,
                    call=call,
                    formal_only=formal_only,
                )
                median, low, high = _median_band(rows, len(agent_axis.replicates))
                medians.append(median)
                lows.append(low)
                highs.append(high)
                plotted_values.extend(
                    value for value in (median, low, high) if math.isfinite(value)
                )
            axis.plot(
                calls,
                medians,
                color=style.color,
                marker=style.marker,
                linestyle=style.line_style,
                label=target,
                markerfacecolor=(style.color if not exploratory else "white"),
                markeredgecolor=style.color,
                markeredgewidth=1.0,
            )
            if len(agent_axis.replicates) > 1:
                axis.fill_between(
                    calls,
                    lows,
                    highs,
                    color=style.color,
                    alpha=0.10,
                    linewidth=0,
                )
        floor = floor_map[(workload.workload_id, spec.targets[0])]
        if floor.status == "valid":
            assert floor.eager_latency_ms is not None and floor.bstar_latency_ms is not None
            bstar_speedup = floor.eager_latency_ms / floor.bstar_latency_ms
            plotted_values.append(bstar_speedup)
            axis.axhline(bstar_speedup, color="#333333", linestyle=(0, (2, 2)), linewidth=1.0)
        axis.axhline(1.0, color="#777777", linestyle=":", linewidth=0.9)
        axis.set_title(textwrap.fill(workload.workload_id, width=30))
        axis.set_xticks(calls)
        axis.set_xlim(min(calls) - 0.25, max(calls) + 0.25)
        axis.set_ylim(0.0, max(1.15, max(plotted_values) * 1.12))
        axis.grid(axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    for axis in all_axes[len(spec.workloads) :]:
        axis.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("Speedup vs eager")
    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel("Agent iteration")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=styles[target].color,
            marker=styles[target].marker,
            linestyle=styles[target].line_style,
            markerfacecolor=(styles[target].color if not exploratory else "white"),
            label=target,
        )
        for target in spec.targets
    ]
    legend_handles.extend(
        (
            Line2D([0], [0], color="#777777", linestyle=":", label="Eager parity"),
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle=(0, (2, 2)),
                label="B* baseline",
            ),
        )
    )
    kind = (
        "exploratory_anytime_small_multiples" if exploratory else "formal_anytime_small_multiples"
    )
    figure_id = f"{'exploratory' if exploratory else 'formal'}-anytime-{agent_id}"
    title = _stage_title(
        spec,
        f"{'Exploratory' if exploratory else 'Formal'} anytime performance: {agent_id}",
    )
    figure.suptitle(title, y=0.985)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 5),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.21,
        top=0.91,
        wspace=0.28,
        hspace=0.42,
    )
    final_rows = tuple(
        row
        for row in report.winners
        if row.agent_id == agent_id
        and row.match_mode == "iteration"
        and row.iteration_budget == spec.max_scientific_calls
        and row.evidence_scope == "formal_only"
    )
    resolved = sum(row.status in {"selected", "tie"} for row in final_rows)
    caption = _stage_caption(
        spec,
        (
            f"At the final formal checkpoint, {resolved}/{len(spec.workloads)} workloads have "
            "complete winner evidence; curves report median trajectory-replicate speedup with "
            "min-max bands."
            if not exploratory
            else "Intermediate dev measurements expose trajectory shape and crossovers; hollow "
            "markers distinguish this exploratory view from the confirmatory checkpoint figure."
        ),
    )
    return figure, figure_id, kind, title, caption


def _rates_figure(
    dataset: AnytimeAnalysisDataset,
    report: AnytimeAnalysisReport,
) -> tuple[Any, str, str, str, str]:
    from matplotlib.figure import Figure

    spec = dataset.spec
    columns = len(spec.agents)
    figure = Figure(figsize=(4.6 * columns, 3.6))
    axes = figure.subplots(1, columns, squeeze=False)
    stages = (
        ("compiled_rate", "Compiled", "#0072B2", "o", "-"),
        ("correct_rate", "Correct", "#D55E00", "s", "--"),
        ("qualification_rate", "Qualified", "#009E73", "^", "-."),
        ("eligible_rate", "Eligible", "#CC79A7", "D", ":"),
    )
    for axis, agent in zip(axes.flat, spec.agents, strict=True):
        rows = [row for row in report.rates if row.agent_id == agent.agent_id]
        calls = [row.scientific_call_index for row in rows]
        for field, label, color, marker, linestyle in stages:
            axis.plot(
                calls,
                [getattr(row, field) for row in rows],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
            )
        for checkpoint in spec.formal_checkpoints:
            axis.axvline(checkpoint, color="#BBBBBB", linewidth=0.7, zorder=0)
        axis.set_title(agent.agent_id)
        axis.set_xlabel("Agent iteration")
        axis.set_ylabel("Planned-population rate")
        axis.set_xticks(tuple(range(1, spec.max_scientific_calls + 1)))
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    title = _stage_title(spec, "Candidate-stage rates by agent iteration")
    figure.suptitle(title, y=0.98)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.83, wspace=0.28)
    final_rates = [
        row.eligible_rate
        for row in report.rates
        if row.scientific_call_index == spec.max_scientific_calls
    ]
    caption = _stage_caption(
        spec,
        "Final eligible rates span "
        f"{min(final_rates):.1%}-{max(final_rates):.1%}; denominators retain every planned "
        "trajectory, including censored and exhausted cells.",
    )
    return figure, "stage-rates", "stage_rates", title, caption


def _final_winner_rows(
    report: AnytimeAnalysisReport,
    spec: AnytimeAnalysisSpec,
) -> tuple[AnytimeWinnerRow, ...]:
    return tuple(
        row
        for row in report.winners
        if row.match_mode == "iteration"
        and row.iteration_budget == spec.max_scientific_calls
        and row.evidence_scope == "formal_only"
    )


def _winner_figure(
    dataset: AnytimeAnalysisDataset,
    report: AnytimeAnalysisReport,
) -> tuple[Any, str, str, str, str]:
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle

    spec = dataset.spec
    styles = {style.target_id: style for style in _target_styles(spec)}
    figure = Figure(
        figsize=(
            5.1 * len(spec.agents),
            max(3.6, 0.42 * len(spec.workloads) + 1.7),
        )
    )
    axes = figure.subplots(1, len(spec.agents), squeeze=False)
    rows = _final_winner_rows(report, spec)
    for axis, agent in zip(axes.flat, spec.agents, strict=True):
        by_workload = {row.workload_id: row for row in rows if row.agent_id == agent.agent_id}
        for y, workload in enumerate(spec.workloads):
            row = by_workload[workload.workload_id]
            for x, target in enumerate(spec.targets):
                winner = target in row.winner_target_ids
                face = styles[target].color if winner else "#F2F2F2"
                rectangle = Rectangle(
                    (x - 0.46, y - 0.42),
                    0.92,
                    0.84,
                    facecolor=face,
                    edgecolor="#FFFFFF",
                    linewidth=1.0,
                    hatch=("///" if row.status not in {"selected", "tie"} else None),
                )
                axis.add_patch(rectangle)
                if winner:
                    axis.text(
                        x,
                        y,
                        "T" if row.status == "tie" else "W",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                        fontweight="bold",
                    )
        axis.set_xlim(-0.5, len(spec.targets) - 0.5)
        axis.set_ylim(len(spec.workloads) - 0.5, -0.5)
        axis.set_xticks(range(len(spec.targets)), spec.targets, rotation=25, ha="right")
        axis.set_yticks(
            range(len(spec.workloads)),
            [textwrap.fill(item.workload_id, width=28) for item in spec.workloads],
        )
        axis.set_title(agent.agent_id)
        axis.tick_params(length=0)
        axis.spines[:].set_visible(False)
    legend = [Patch(facecolor=styles[target].color, label=target) for target in spec.targets] + [
        Patch(facecolor="#F2F2F2", edgecolor="#777777", hatch="///", label="Unavailable")
    ]
    title = _stage_title(spec, "Final formal target winner sets")
    figure.suptitle(title, y=0.985)
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=min(len(legend), 4),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.subplots_adjust(left=0.24, right=0.98, bottom=0.17, top=0.86, wspace=0.55)
    resolved = [row for row in rows if row.status in {"selected", "tie"}]
    winner_sets = {row.winner_target_ids for row in resolved}
    ties = sum(row.status == "tie" for row in resolved)
    caption = _stage_caption(
        spec,
        f"The final checkpoint contains {len(winner_sets)} distinct winner sets across "
        f"{len(resolved)} resolved workload-agent cells, including {ties} ties; W marks a unique "
        "winner and T marks membership in a tie.",
    )
    return figure, "final-winner-map", "final_winner_map", title, caption


def _hindsight_figure(
    dataset: AnytimeAnalysisDataset,
    report: AnytimeAnalysisReport,
) -> tuple[Any, str, str, str, str]:
    from matplotlib.figure import Figure

    spec = dataset.spec
    figure = Figure(figsize=(6.4, 3.8))
    axis = figure.subplots()
    observed_values: list[float] = []
    for index, agent in enumerate(spec.agents):
        points = {
            row.iteration_budget: row.oracle_over_fixed_gain
            for row in report.hindsight
            if row.agent_id == agent.agent_id
            and row.match_mode == "iteration"
            and row.evidence_scope == "formal_only"
            and row.iteration_budget in spec.formal_checkpoints
            and row.status == "complete"
        }
        y = [float(points.get(call, math.nan)) for call in spec.formal_checkpoints]
        observed_values.extend(value for value in y if math.isfinite(value))
        axis.plot(
            spec.formal_checkpoints,
            y,
            color=_TARGET_COLORS[index % len(_TARGET_COLORS)],
            marker=_MARKERS[index % len(_MARKERS)],
            linestyle=_LINESTYLES[index % len(_LINESTYLES)],
            label=agent.agent_id,
        )
    axis.axhline(1.0, color="#777777", linestyle=":", linewidth=0.9)
    axis.set_xlabel("Agent iteration")
    axis.set_ylabel("Hindsight oracle / best fixed target")
    axis.set_xticks(spec.formal_checkpoints)
    axis.set_ylim(0.0, max((1.0, *observed_values)) * 1.12)
    axis.grid(axis="y")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    title = _stage_title(spec, "Per-workload target choice versus one fixed target")
    figure.suptitle(title, y=0.97)
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.84)
    finding = (
        f"Across complete formal checkpoints, the largest observed oracle-over-fixed gain is "
        f"{max(observed_values):.3f}x; 1.0x denotes no benefit from workload-specific target "
        "selection."
        if observed_values
        else "No formal checkpoint has complete hindsight evidence; 1.0x is shown only as the "
        "no-benefit reference."
    )
    caption = _stage_caption(spec, finding)
    return figure, "formal-hindsight-gain", "formal_hindsight_gain", title, caption


def _figure_bytes(figure: Any, figure_id: str, title: str) -> dict[str, bytes]:
    formats: dict[str, bytes] = {}
    metadata = {
        "png": {"Software": "AbstraK anytime figure renderer v1"},
        "svg": {"Creator": "AbstraK anytime figure renderer v1", "Date": None, "Title": title},
        "pdf": {
            "Creator": "AbstraK anytime figure renderer v1",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
            "Title": title,
            "Subject": figure_id,
        },
    }
    for extension in ("png", "svg", "pdf"):
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format=extension,
            dpi=180,
            metadata=metadata[extension],
            facecolor="white",
            edgecolor="white",
        )
        formats[extension] = buffer.getvalue()
    return formats


def _persist_figure(
    root: Path,
    built: tuple[Any, str, str, str, str],
    *,
    spec: AnytimeAnalysisSpec,
    agent_id: str | None,
    evidence_scope: str,
) -> AnytimeFigureArtifact:
    figure, figure_id, kind, title, caption = built
    width, height = figure.get_size_inches()
    payloads = _figure_bytes(figure, figure_id, title)
    figure.clear()
    files = tuple(
        _write_bytes(
            root,
            f"figures/{figure_id}.{extension}",
            payloads[extension],
            role=f"figure_{extension}",
        )
        for extension in ("png", "svg", "pdf")
    )
    return AnytimeFigureArtifact(
        figure_id=figure_id,
        kind=kind,
        study_stage=spec.study_stage,
        evidence_scope=evidence_scope,
        agent_id=agent_id,
        title=title,
        caption=caption,
        width_inches=float(width),
        height_inches=float(height),
        files=files,
    )


def _table_csv_bytes(table: AnytimeAnalysisTable) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(table.columns)
    writer.writerows(table.rows)
    return stream.getvalue().encode()


def write_anytime_analysis_bundle(
    dataset: AnytimeAnalysisDataset,
    output_directory: str | Path,
    *,
    include_exploratory: bool = True,
) -> AnytimeAnalysisBundleManifest:
    """Write one immutable derived bundle without discovering or executing study artifacts."""

    import matplotlib

    checked = AnytimeAnalysisDataset.model_validate_json(dataset.model_dump_json())
    report = build_anytime_analysis(checked)
    final = Path(output_directory).expanduser()
    staging = final.with_name(f"{final.name}.incomplete")
    if final.exists() or final.is_symlink() or staging.exists() or staging.is_symlink():
        raise AnytimeFigureError("analysis output or staging directory already exists")
    final.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging.mkdir(mode=0o700)

    files: list[AnytimeDerivedFile] = []
    files.append(
        _write_bytes(
            staging,
            "analysis-report.json",
            _canonical_json_bytes(report),
            role="analysis_report",
        )
    )
    for table in anytime_analysis_tables(report):
        files.append(
            _write_bytes(
                staging,
                f"tables/{table.name}.json",
                _canonical_json_bytes(table),
                role="analysis_table_json",
            )
        )
        files.append(
            _write_bytes(
                staging,
                f"tables/{table.name}.csv",
                _table_csv_bytes(table),
                role="analysis_table_csv",
            )
        )

    figures: list[AnytimeFigureArtifact] = []
    with matplotlib.rc_context(_matplotlib_style()):
        for agent in checked.spec.agents:
            figures.append(
                _persist_figure(
                    staging,
                    _anytime_figure(
                        checked,
                        report,
                        agent_id=agent.agent_id,
                        exploratory=False,
                    ),
                    spec=checked.spec,
                    agent_id=agent.agent_id,
                    evidence_scope="formal_only",
                )
            )
            if include_exploratory:
                figures.append(
                    _persist_figure(
                        staging,
                        _anytime_figure(
                            checked,
                            report,
                            agent_id=agent.agent_id,
                            exploratory=True,
                        ),
                        spec=checked.spec,
                        agent_id=agent.agent_id,
                        evidence_scope="formal_and_exploratory",
                    )
                )
        figures.extend(
            (
                _persist_figure(
                    staging,
                    _rates_figure(checked, report),
                    spec=checked.spec,
                    agent_id=None,
                    evidence_scope="all_turns",
                ),
                _persist_figure(
                    staging,
                    _winner_figure(checked, report),
                    spec=checked.spec,
                    agent_id=None,
                    evidence_scope="formal_only",
                ),
                _persist_figure(
                    staging,
                    _hindsight_figure(checked, report),
                    spec=checked.spec,
                    agent_id=None,
                    evidence_scope="formal_only",
                ),
            )
        )
    figure_manifest = AnytimeFigureManifest(
        analysis_report_sha256=report.sha256,
        input_dataset_sha256=checked.sha256,
        analysis_spec_sha256=checked.spec.sha256,
        analysis_code_sha256=_analysis_code_sha256(),
        renderer=f"matplotlib-{matplotlib.__version__}",
        renderer_environment=_renderer_environment(),
        study_stage=checked.spec.study_stage,
        target_styles=_target_styles(checked.spec),
        figures=tuple(figures),
    )
    figure_manifest_file = _write_bytes(
        staging,
        "figure-manifest.json",
        _canonical_json_bytes(figure_manifest),
        role="figure_manifest",
    )
    files.append(figure_manifest_file)
    files.extend(file for figure in figures for file in figure.files)
    bundle = AnytimeAnalysisBundleManifest(
        input_dataset_sha256=checked.sha256,
        analysis_report_sha256=report.sha256,
        figure_manifest_sha256=figure_manifest.sha256,
        files=tuple(sorted(files, key=lambda item: item.relative_path)),
    )
    _write_bytes(
        staging,
        "bundle-manifest.json",
        _canonical_json_bytes(bundle),
        role="figure_manifest",
    )
    os.rename(staging, final)
    descriptor = os.open(final.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return bundle


def verify_anytime_analysis_bundle(
    directory: str | Path,
) -> AnytimeAnalysisBundleManifest:
    """Freshly checksum every listed derived file and reject extras or broken bindings."""

    root = Path(directory).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise AnytimeFigureError("analysis bundle directory is missing or unsafe")
    manifest_path = root / "bundle-manifest.json"
    try:
        bundle = AnytimeAnalysisBundleManifest.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValueError) as error:
        raise AnytimeFigureError(f"invalid analysis bundle manifest: {error}") from error
    expected_paths = {item.relative_path for item in bundle.files} | {"bundle-manifest.json"}
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AnytimeFigureError("symbolic links are forbidden in analysis bundles")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != expected_paths:
        raise AnytimeFigureError("analysis bundle inventory differs from its manifest")
    for item in bundle.files:
        path = root.joinpath(*PurePosixPath(item.relative_path).parts)
        payload = path.read_bytes()
        if len(payload) != item.size_bytes or hashlib.sha256(payload).hexdigest() != item.sha256:
            raise AnytimeFigureError(f"analysis bundle checksum mismatch: {item.relative_path}")
    inventory = {item.relative_path: item for item in bundle.files}
    if (
        inventory.get("analysis-report.json", None) is None
        or inventory["analysis-report.json"].role != "analysis_report"
    ):
        raise AnytimeFigureError("analysis report inventory role is missing")
    if (
        inventory.get("figure-manifest.json", None) is None
        or inventory["figure-manifest.json"].role != "figure_manifest"
    ):
        raise AnytimeFigureError("figure manifest inventory role is missing")
    try:
        figure_manifest = AnytimeFigureManifest.model_validate_json(
            (root / "figure-manifest.json").read_bytes()
        )
        report = AnytimeAnalysisReport.model_validate_json(
            (root / "analysis-report.json").read_bytes()
        )
    except ValueError as error:
        raise AnytimeFigureError(f"invalid bound analysis artifact: {error}") from error
    if figure_manifest.sha256 != bundle.figure_manifest_sha256:
        raise AnytimeFigureError("figure manifest differs from the bundle binding")
    if report.sha256 != bundle.analysis_report_sha256:
        raise AnytimeFigureError("analysis report differs from the bundle binding")
    if (
        report.input_dataset_sha256 != bundle.input_dataset_sha256
        or figure_manifest.input_dataset_sha256 != bundle.input_dataset_sha256
    ):
        raise AnytimeFigureError("input dataset bindings disagree across analysis manifests")
    if figure_manifest.analysis_report_sha256 != report.sha256:
        raise AnytimeFigureError("figure manifest is bound to another analysis report")
    if figure_manifest.analysis_spec_sha256 != report.analysis_spec_sha256:
        raise AnytimeFigureError("analysis spec bindings disagree across analysis manifests")

    nested_figures = {
        item.relative_path: item for figure in figure_manifest.figures for item in figure.files
    }
    outer_figures = {
        path: item
        for path, item in inventory.items()
        if item.role in {"figure_png", "figure_svg", "figure_pdf"}
    }
    if nested_figures != outer_figures:
        raise AnytimeFigureError("nested figure inventory differs from the bundle inventory")

    expected_tables: dict[str, bytes] = {}
    for table in anytime_analysis_tables(report):
        expected_tables[f"tables/{table.name}.json"] = _canonical_json_bytes(table)
        expected_tables[f"tables/{table.name}.csv"] = _table_csv_bytes(table)
    table_paths = {
        path
        for path, item in inventory.items()
        if item.role in {"analysis_table_json", "analysis_table_csv"}
    }
    if table_paths != set(expected_tables):
        raise AnytimeFigureError("analysis table inventory differs from the report schema")
    for path, expected in expected_tables.items():
        if root.joinpath(*PurePosixPath(path).parts).read_bytes() != expected:
            raise AnytimeFigureError(f"analysis table differs from its report: {path}")
    return bundle
