"""The simulation study: the variant grid, per-seed evaluation and aggregation.

The study runs every pipeline variant on many evaluation simulations and reports
the distribution of its skill and net performance, so a difference between two
variants can be compared with the variation between seeds. Per-seed results are
cached on disk and filled incrementally, so the study can run one seed at a time
and resume.

Design choices in the variants below were fixed on development simulations before
any evaluation seed was run. See :data:`DEVELOPMENT_SEEDS` and
:data:`EVALUATION_SEEDS`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spendsignal.evaluation import bootstrap_mean_ci
from spendsignal.pipeline import ORIGINAL, PipelineConfig, PipelineResult, oracle_ic, replace, run
from spendsignal.simulate import DEFAULT_SECTORS, SimulationConfig, simulate

#: Seeds used to choose and freeze the design. No evaluation number comes from these.
DEVELOPMENT_SEEDS: tuple[int, ...] = (7, 8, 9, 10, 11)
#: Seeds the frozen design is reported on. Run once, after the design was fixed.
EVALUATION_SEEDS: tuple[int, ...] = tuple(range(3001, 3051))

#: Partial-adjustment fraction for the cost-aware book, chosen on the development seeds.
REBALANCE_SPEED: float = 0.4
#: Assumed general-collateral borrow fee on the short side, in basis points a year.
BORROW_BPS_ANNUAL: float = 50.0

FINAL = PipelineConfig()

#: Every variant reported in the study. Each changes one choice relative to the final pipeline.
VARIANTS: dict[str, PipelineConfig] = {
    "Final pipeline": FINAL,
    "Pooled winsorisation instead of per-series filter": replace(FINAL, cleaning="pooled_winsor"),
    "Cohort offsets removed before averaging": replace(FINAL, normalisation="adjusted"),
    "Year-on-year growth instead of calendar buckets": replace(FINAL, seasonality="yoy"),
    "Lagged values instead of trailing means": replace(FINAL, features="lags"),
    "50-day moving average instead of ridge": replace(FINAL, model="smooth"),
    "All sectors traded (no sector filter)": replace(FINAL, sector_selection="none"),
    "Sectors chosen on the test period": replace(FINAL, sector_selection="test"),
    "Data available 3 days late": replace(FINAL, publication_lag=3),
    "Costs of 10 bp instead of 5 bp": replace(FINAL, cost_bps=10.0),
    "Gradient-boosted trees instead of ridge": replace(FINAL, model="tree"),
    "Beta- and sector-neutral spread": replace(FINAL, portfolio="neutral"),
    "Cost-aware rebalancing": replace(FINAL, rebalance="partial", rebalance_speed=REBALANCE_SPEED),
    "Borrow cost on the short side": replace(FINAL, borrow_bps_annual=BORROW_BPS_ANNUAL),
    "Original design (see text)": ORIGINAL,
}

#: The variant each paired comparison is made against.
FINAL_NAME = "Final pipeline"
#: Metrics stored per variant per seed.
METRICS: tuple[str, ...] = (
    "mean_ic",
    "t_stat_nw",
    "gross_sharpe",
    "net_sharpe",
    "net_return",
    "net_vol",
    "max_drawdown",
    "turnover",
)


def _atomic_write(path: Path, text: str) -> None:
    """Write a file by renaming a temporary, so a concurrent reader never sees a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def variant_metrics(result: PipelineResult) -> dict[str, float]:
    """Pull the reported metrics out of one pipeline result.

    Args:
        result: The output of :func:`spendsignal.pipeline.run`.

    Returns:
        A mapping from each name in :data:`METRICS` to its value.
    """
    s, g, n = result.ic_stats, result.perf_gross, result.perf_net
    return {
        "mean_ic": s["mean_ic"],
        "t_stat_nw": s["t_stat_nw"],
        "gross_sharpe": g["sharpe"],
        "net_sharpe": n["sharpe"],
        "net_return": n["ann_return"],
        "net_vol": n["ann_vol"],
        "max_drawdown": n["max_drawdown"],
        "turnover": float(result.backtest.turnover.mean()),
    }


def evaluate_seed(
    seed: int, variants: dict[str, PipelineConfig] | None = None, previous: dict | None = None
) -> dict:
    """Evaluate every variant on one simulation, reusing any already-computed variants.

    Args:
        seed: Simulation seed to evaluate.
        variants: (Optional) variant grid; defaults to :data:`VARIANTS`.
        previous: (Optional) a cached result for this seed; variants already present
            in it are kept and only missing ones are computed.

    Returns:
        A record with the seed, the two oracle ICs and a metric mapping per variant.
        The same object as ``previous`` is returned when nothing was missing.
    """
    variants = variants or VARIANTS
    done = dict(previous["variants"]) if previous else {}
    missing = {name: cfg for name, cfg in variants.items() if name not in done}
    if previous and not missing:
        return previous
    data = simulate(SimulationConfig(seed=seed))
    for name, cfg in missing.items():
        done[name] = variant_metrics(run(data, cfg))
    return {
        "seed": int(seed),
        "oracle_ic": oracle_ic(data)["mean_ic"],
        "oracle_full_ic": oracle_ic(data, card_only=False)["mean_ic"],
        "variants": done,
    }


def load_or_evaluate_seed(
    seed: int, cache_dir: Path, variants: dict[str, PipelineConfig] | None = None
) -> dict:
    """Return a seed's evaluation from cache, computing and caching any missing variants.

    Args:
        seed: Simulation seed to evaluate.
        cache_dir: Directory holding one JSON file per seed.
        variants: (Optional) variant grid; defaults to :data:`VARIANTS`.

    Returns:
        The evaluation record for the seed.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"seed_{seed}.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    result = evaluate_seed(seed, variants, previous)
    if result is not previous:
        _atomic_write(path, json.dumps(result, indent=2))
    return result


def _metric_array(seed_results: list[dict], name: str, metric: str) -> np.ndarray:
    return np.array([record["variants"][name][metric] for record in seed_results], dtype=float)


def aggregate(seed_results: list[dict], variants: dict[str, PipelineConfig] | None = None) -> dict:
    """Summarise evaluation records into distributions and paired differences.

    For every variant the distribution of mean IC and net Sharpe across seeds is
    summarised by its mean, a 95% bootstrap confidence interval and its standard
    deviation. For every variant except the final pipeline, the per-seed paired
    difference against the final pipeline is summarised the same way, with the
    share of seeds in which the variant wins.

    Args:
        seed_results: Evaluation records from :func:`load_or_evaluate_seed`.
        variants: (Optional) variant grid; defaults to :data:`VARIANTS`.

    Returns:
        A mapping with a ``distribution`` frame, a ``paired`` frame and a
        ``summary`` dict of headline figures.
    """
    variants = variants or VARIANTS
    names = list(variants)

    distribution = {}
    for name in names:
        ic = _metric_array(seed_results, name, "mean_ic")
        sharpe = _metric_array(seed_results, name, "net_sharpe")
        ic_ci, sharpe_ci = bootstrap_mean_ci(ic), bootstrap_mean_ci(sharpe)
        distribution[name] = {
            "ic_mean": ic_ci["mean"],
            "ic_lo": ic_ci["lo"],
            "ic_hi": ic_ci["hi"],
            "ic_std": float(ic.std(ddof=1)),
            "sharpe_mean": sharpe_ci["mean"],
            "sharpe_lo": sharpe_ci["lo"],
            "sharpe_hi": sharpe_ci["hi"],
            "sharpe_std": float(sharpe.std(ddof=1)),
            "t_mean": float(_metric_array(seed_results, name, "t_stat_nw").mean()),
            "net_return_mean": float(_metric_array(seed_results, name, "net_return").mean()),
            "turnover_mean": float(_metric_array(seed_results, name, "turnover").mean()),
            "sharpe_positive_frac": float((sharpe > 0).mean()),
        }
    distribution = pd.DataFrame(distribution).T.reindex(names)
    distribution.index.name = "Variant"

    final_ic = _metric_array(seed_results, FINAL_NAME, "mean_ic")
    final_sharpe = _metric_array(seed_results, FINAL_NAME, "net_sharpe")
    paired = {}
    for name in names:
        if name == FINAL_NAME:
            continue
        d_ic = _metric_array(seed_results, name, "mean_ic") - final_ic
        d_sharpe = _metric_array(seed_results, name, "net_sharpe") - final_sharpe
        ic_ci, sharpe_ci = bootstrap_mean_ci(d_ic), bootstrap_mean_ci(d_sharpe)
        paired[name] = {
            "d_ic": ic_ci["mean"],
            "d_ic_lo": ic_ci["lo"],
            "d_ic_hi": ic_ci["hi"],
            "d_sharpe": sharpe_ci["mean"],
            "d_sharpe_lo": sharpe_ci["lo"],
            "d_sharpe_hi": sharpe_ci["hi"],
            "ic_better_frac": float((d_ic > 0).mean()),
            "sharpe_better_frac": float((d_sharpe > 0).mean()),
        }
    paired = pd.DataFrame(paired).T.reindex([name for name in names if name != FINAL_NAME])
    paired.index.name = "Variant"

    oracle = _array(seed_results, "oracle_ic")
    recovery = final_ic / oracle
    summary = {
        "n_seeds": len(seed_results),
        "seeds": [int(record["seed"]) for record in seed_results],
        "oracle_ic": bootstrap_mean_ci(oracle),
        "oracle_full_ic": bootstrap_mean_ci(_array(seed_results, "oracle_full_ic")),
        "final_ic": bootstrap_mean_ci(final_ic),
        "final_net_sharpe": bootstrap_mean_ci(final_sharpe),
        "final_recovery": {
            "mean": float(recovery.mean()),
            "min": float(recovery.min()),
            "max": float(recovery.max()),
        },
        "final_t_min": float(_metric_array(seed_results, FINAL_NAME, "t_stat_nw").min()),
        "final_t_max": float(_metric_array(seed_results, FINAL_NAME, "t_stat_nw").max()),
        "final_sharpe_min": float(final_sharpe.min()),
        "final_sharpe_max": float(final_sharpe.max()),
    }
    return {"distribution": distribution, "paired": paired, "summary": summary}


def _array(seed_results: list[dict], key: str) -> np.ndarray:
    return np.array([record[key] for record in seed_results], dtype=float)


# --- Dependence on the simulator -----------------------------------------------------------

#: Evaluation seeds used for the simulator sweeps (a subset of the evaluation seeds).
SENSITIVITY_SEEDS: tuple[int, ...] = tuple(range(3001, 3011))
#: Variants carried through the sweeps: enough to test the headline conclusions.
SWEEP_VARIANTS: tuple[str, ...] = (
    "Final pipeline",
    "50-day moving average instead of ridge",
    "Lagged values instead of trailing means",
    "All sectors traded (no sector filter)",
)


def _scaled_sectors(scale: float) -> dict[int, tuple[str, int, float]]:
    return {
        code: (name, count, min(1.0, share * scale))
        for code, (name, count, share) in DEFAULT_SECTORS.items()
    }


def _baseline_factory(seed: int) -> SimulationConfig:
    return SimulationConfig(seed=seed)


def _sim_factory(field: str, value: float):
    return lambda seed: SimulationConfig(seed=seed, **{field: value})


def _cardshare_factory(scale: float):
    return lambda seed: SimulationConfig(seed=seed, sectors=_scaled_sectors(scale))


def sweep_cells() -> list[tuple[str, str, object, dict]]:
    """Describe the simulator sweep as a baseline cell plus one cell per level.

    Returns:
        A list of ``(parameter, level, data_factory, pipeline_override)`` tuples.
        ``data_factory(seed)`` builds a :class:`SimulationConfig`; ``pipeline_override``
        is applied to each swept variant's config. The baseline cell is the default
        data-generating process and is shared by every parameter.
    """
    cells: list[tuple[str, str, object, dict]] = [("Baseline", "base", _baseline_factory, {})]
    sim_grids = {
        "Signal strength": ("alpha_scale", {"weak (0.5x)": 0.0003, "strong (2x)": 0.0012}),
        "Learning half-life": ("learning_halflife_days", {"fast (10d)": 10.0, "slow (40d)": 40.0}),
        "Panel noise": ("panel_noise", {"low (0.05)": 0.05, "high (0.20)": 0.20}),
    }
    for param, (field, levels) in sim_grids.items():
        for label, value in levels.items():
            cells.append((param, label, _sim_factory(field, value), {}))
    cells.append(("Card share", "low (0.6x)", _cardshare_factory(0.6), {}))
    cells.append(("Card share", "high (1.4x)", _cardshare_factory(1.4), {}))
    cells.append(("Publication lag", "1 day", _baseline_factory, {"publication_lag": 1}))
    cells.append(("Publication lag", "3 days", _baseline_factory, {"publication_lag": 3}))
    return cells


def _slug(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text).strip("_").lower()


def sweep_cell_path(param: str, level: str, seed: int, cache_dir: Path) -> Path:
    """Cache-file path for one sweep cell and seed."""
    return Path(cache_dir) / f"{_slug(param)}__{_slug(level)}__seed_{seed}.json"


def evaluate_sweep_cell(param: str, level: str, data_factory, override: dict, seed: int) -> dict:
    """Evaluate the swept variants for one parameter level on one simulation.

    Args:
        param: Name of the swept parameter.
        level: Name of the level.
        data_factory: Callable mapping a seed to a :class:`SimulationConfig`.
        override: Pipeline-config overrides applied to each swept variant.
        seed: Simulation seed.

    Returns:
        A record with the parameter, level, seed, oracle IC and a metric mapping per
        swept variant.
    """
    data = simulate(data_factory(seed))
    metrics = {
        name: variant_metrics(run(data, replace(VARIANTS[name], **override))) for name in SWEEP_VARIANTS
    }
    return {
        "param": param,
        "level": level,
        "seed": int(seed),
        "oracle_ic": oracle_ic(data)["mean_ic"],
        "variants": metrics,
    }


def load_or_evaluate_sweep_cell(
    param: str, level: str, data_factory, override: dict, seed: int, cache_dir: Path
) -> dict:
    """Return a sweep cell from cache, computing and caching it if missing."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = sweep_cell_path(param, level, seed, cache_dir)
    if path.exists():
        cached = json.loads(path.read_text())
        if set(cached.get("variants", {})) >= set(SWEEP_VARIANTS):
            return cached
    result = evaluate_sweep_cell(param, level, data_factory, override, seed)
    _atomic_write(path, json.dumps(result, indent=2))
    return result


def aggregate_sensitivity(records: list[dict]) -> pd.DataFrame:
    """Average the sweep records over seeds, one row per parameter and level.

    Args:
        records: Sweep records from :func:`load_or_evaluate_sweep_cell`.

    Returns:
        A frame indexed by (parameter, level) with the mean final IC, oracle IC,
        recovery fraction, final net Sharpe, the moving-average IC, the single-lag
        net Sharpe and the no-filter net Sharpe across seeds.
    """
    def mean_over(cells: list[dict], variant: str, metric: str) -> float:
        return float(np.mean([cell["variants"][variant][metric] for cell in cells]))

    rows = {}
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["param"], record["level"]), []).append(record)
    for (param, level), cells in grouped.items():
        final_ic = np.array([cell["variants"]["Final pipeline"]["mean_ic"] for cell in cells])
        oracle = np.array([cell["oracle_ic"] for cell in cells])
        rows[(param, level)] = {
            "final_ic": float(final_ic.mean()),
            "oracle_ic": float(oracle.mean()),
            "recovery": float((final_ic / oracle).mean()),
            "final_net_sharpe": mean_over(cells, "Final pipeline", "net_sharpe"),
            "ma_ic": mean_over(cells, "50-day moving average instead of ridge", "mean_ic"),
            "lags_net_sharpe": mean_over(cells, "Lagged values instead of trailing means", "net_sharpe"),
            "no_filter_net_sharpe": mean_over(cells, "All sectors traded (no sector filter)", "net_sharpe"),
        }
    table = pd.DataFrame(rows).T
    table.index.names = ["parameter", "level"]
    return table
