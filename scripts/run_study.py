"""Run the full study and write every file the README and the report depend on.

The study evaluates each pipeline variant on all evaluation seeds, sweeps the main
simulator parameters, and draws the figures. Per-seed results are cached, so the
run is resumable and can be done one seed at a time:

    python scripts/run_study.py                           # the whole study, resuming from cache
    python scripts/run_study.py --seeds 3001 --fill-only  # fill one evaluation seed into the cache

Results and figures are written only once the cache is complete, so a partial run never
leaves inconsistent files behind. All numbers in README.md and docs/report.pdf come from the
files this script writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spendsignal import forecast, panel, plotting, seasonality, study
from spendsignal.evaluation import ic_by_group
from spendsignal.pipeline import ORIGINAL, PipelineConfig, replace, run
from spendsignal.simulate import SimulationConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
FINAL = PipelineConfig()


def _complete_records(seeds: tuple[int, ...], cache_dir: Path) -> list[dict]:
    """Load cached evaluation records that hold every variant."""
    records = []
    for seed in seeds:
        path = cache_dir / f"seed_{seed}.json"
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # a concurrent writer is mid-file; it will parse on the next run
        if set(record.get("variants", {})) >= set(study.VARIANTS):
            records.append(record)
    return records


def _sweep_records(cache_dir: Path) -> list[dict]:
    records = []
    for param, level, _, _ in study.sweep_cells():
        for seed in study.SENSITIVITY_SEEDS:
            path = study.sweep_cell_path(param, level, seed, cache_dir)
            if path.exists():
                records.append(json.loads(path.read_text()))
    return records


def markdown_table(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a frame as a GitHub-flavoured Markdown table."""
    header = "| " + " | ".join([frame.index.name or ""] + [str(c) for c in frame.columns]) + " |"
    sep = "|" + "---|" * (len(frame.columns) + 1)
    lines = [header, sep]
    for name, row in frame.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (int, float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(name)] + cells) + " |")
    return "\n".join(lines)


def illustrative_diagnostics(seed: int, fig_dir: Path) -> dict:
    """Recompute one simulation in full for the diagnostic figures and detail numbers.

    The performance claims come from the seed distribution; this single seed only
    supplies the illustrative figures (cleaning, seasonality, lag profile, sector
    skill, cumulative P&L) and the matching detail fields.

    Args:
        seed: The illustrative evaluation seed.
        fig_dir: Directory the figures are written to.

    Returns:
        The detail fields that go into ``summary.json`` for this seed.
    """
    data = simulate(SimulationConfig(seed=seed))
    meta = data.metadata.set_index("stock")
    final = run(data, FINAL)
    original = run(data, ORIGINAL)
    neutral = run(data, replace(FINAL, portfolio="neutral"))
    cost_aware = run(data, replace(FINAL, rebalance="partial", rebalance_speed=study.REBALANCE_SPEED))
    all_sectors = run(data, replace(FINAL, sector_selection="none"))

    cube, _ = panel.to_cube(data.spend)
    logs, _, _ = panel.log_and_filter(cube)
    active = panel.active_cohorts(logs)
    total = panel.total_log_spend(logs)
    normalised = panel.cohort_mean(logs)
    level = normalised.asfreq("D")
    adjusted = seasonality.bucket_adjust(level, meta["sector"])
    clipped = panel.clipped_share_pooled(cube)

    windows = list(FINAL.windows)
    profiles = pd.DataFrame(
        {
            year: forecast.implied_lag_weights(coefs, max_lag=max(windows))
            for year, coefs in final.coefficients.iterrows()
        }
    ).T
    sector_ic = ic_by_group(all_sectors.prediction, all_sectors.fwd_returns, meta["sector_name"])
    exposure = meta.groupby("sector_name")["card_share"].mean()

    plotting.cohorts_vs_spend(total, normalised, active, fig_dir / "cohorts_vs_spend.png")
    disc = meta.index[meta["sector"] == 25]
    plotting.seasonality(normalised[disc], adjusted[disc], fig_dir / "seasonality.png")
    size = meta["revenue"] + np.log(meta["card_share"])
    plotting.clipping_vs_size(clipped, size, fig_dir / "pooled_clipping.png")
    plotting.lag_profile(profiles, None, fig_dir / "lag_profile.png")
    plotting.ic_by_sector(sector_ic, exposure, fig_dir / "ic_by_sector.png")
    plotting.cumulative_pnl(
        {"net of 5 bp costs": final.backtest.net, "gross": final.backtest.gross},
        fig_dir / "pnl_final.png",
        "Final pipeline: dollar-neutral quintile spread",
    )
    market = final.fwd_returns.mean(axis=1)
    plotting.cumulative_pnl(
        {
            "original long-only portfolio": original.backtest.net,
            "equal-weight market": market.loc[original.backtest.net.index],
        },
        fig_dir / "pnl_original_vs_market.png",
        "Original design: long-only top half vs the market",
    )
    plotting.cumulative_pnl(
        {
            "final long-short": final.backtest.net,
            "beta- and sector-neutral": neutral.backtest.net,
            "cost-aware rebalancing": cost_aware.backtest.net,
        },
        fig_dir / "portfolio_variants.png",
        "Portfolio variants on the illustrative seed, net of costs",
    )

    sector_ic.to_csv(ROOT / "results" / "ic_by_sector.csv")
    stacked_active = active.stack()
    demeaned_total = (total - total.mean()).stack().reindex(stacked_active.index).fillna(0.0)
    corr_active = float(np.corrcoef(stacked_active, demeaned_total)[0, 1])
    return {
        "cleaning_report": final.cleaning,
        "final_ridge_alphas": {int(k): float(v) for k, v in final.alphas.items()},
        "final_window_coefficients": {
            int(y): {int(w): float(c) for w, c in r.items()} for y, r in final.coefficients.iterrows()
        },
        "max_clipped_share": float(clipped.max()),
        "stocks_over_half_clipped": int((clipped > 0.5).sum()),
        "corr_total_vs_active": corr_active,
        "net_results": {
            "final": final.perf_net,
            "original": original.perf_net,
            "beta_sector_neutral": neutral.perf_net,
            "cost_aware": cost_aware.perf_net,
            "market_equal_weight": {
                "ann_return": float(market.mean() * 252),
                "ann_vol": float(market.std() * np.sqrt(252)),
            },
        },
        "turnover": {
            "final": float(final.backtest.turnover.mean()),
            "beta_sector_neutral": float(neutral.backtest.turnover.mean()),
            "cost_aware": float(cost_aware.backtest.turnover.mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(study.EVALUATION_SEEDS))
    parser.add_argument("--illustrative-seed", type=int, default=study.EVALUATION_SEEDS[0])
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / ".cache")
    parser.add_argument("--fill-only", action="store_true", help="only fill the cache, write no results")
    args = parser.parse_args()

    results_dir, fig_dir = ROOT / "results", ROOT / "docs" / "figures"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    eval_cache, sweep_cache = args.cache_dir / "eval", args.cache_dir / "sweep"

    for seed in args.seeds:
        study.load_or_evaluate_seed(seed, eval_cache)
    if args.fill_only:
        done = len(_complete_records(study.EVALUATION_SEEDS, eval_cache))
        print(f"filled cache: {done}/{len(study.EVALUATION_SEEDS)} evaluation seeds complete")
        return

    for param, level, factory, override in study.sweep_cells():
        for seed in study.SENSITIVITY_SEEDS:
            study.load_or_evaluate_sweep_cell(param, level, factory, override, seed, sweep_cache)

    seed_results = _complete_records(study.EVALUATION_SEEDS, eval_cache)
    sweeps = _sweep_records(sweep_cache)
    expected_sweeps = len(study.sweep_cells()) * len(study.SENSITIVITY_SEEDS)
    if len(seed_results) < len(study.EVALUATION_SEEDS) or len(sweeps) < expected_sweeps:
        print(
            f"cache incomplete: {len(seed_results)}/{len(study.EVALUATION_SEEDS)} evaluation seeds and "
            f"{len(sweeps)}/{expected_sweeps} sweep cells. Run again to continue; "
            "results are written once the cache is complete."
        )
        return

    agg = study.aggregate(seed_results)
    distribution, paired = agg["distribution"], agg["paired"]
    sensitivity = study.aggregate_sensitivity(sweeps)

    detail = illustrative_diagnostics(args.illustrative_seed, fig_dir)
    strip_variants = [
        "Final pipeline",
        "Beta- and sector-neutral spread",
        "All sectors traded (no sector filter)",
        "Original design (see text)",
    ]
    samples = {
        name: np.array([record["variants"][name]["net_sharpe"] for record in seed_results])
        for name in strip_variants
    }
    plotting.sharpe_distribution(samples, fig_dir / "sharpe_distribution.png")
    plotting.paired_differences(paired, fig_dir / "ablation_differences.png")
    plotting.sensitivity(sensitivity, fig_dir / "sensitivity.png")

    summary = {
        "evaluation_seeds": {
            "n": len(seed_results),
            "first": int(min(r["seed"] for r in seed_results)),
            "last": int(max(r["seed"] for r in seed_results)),
        },
        "illustrative_seed": args.illustrative_seed,
        "oracle_ic": agg["summary"]["oracle_ic"],
        "oracle_full_ic": agg["summary"]["oracle_full_ic"],
        "final_ic": agg["summary"]["final_ic"],
        "final_net_sharpe": agg["summary"]["final_net_sharpe"],
        "final_recovery": agg["summary"]["final_recovery"],
        "final_t_min": agg["summary"]["final_t_min"],
        "final_t_max": agg["summary"]["final_t_max"],
        "final_sharpe_min": agg["summary"]["final_sharpe_min"],
        "final_sharpe_max": agg["summary"]["final_sharpe_max"],
        "distribution": distribution.to_dict(orient="index"),
        "paired": paired.to_dict(orient="index"),
        "sensitivity": {f"{p} / {lv}": row for (p, lv), row in sensitivity.to_dict(orient="index").items()},
        **detail,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    distribution.to_csv(results_dir / "ablation_distribution.csv")
    paired.to_csv(results_dir / "paired_diffs.csv")
    sensitivity.to_csv(results_dir / "sensitivity.csv")

    span = summary["evaluation_seeds"]
    tables = [
        f"# Results across {len(seed_results)} evaluation seeds ({span['first']} to {span['last']})",
        "",
        f"Oracle mean IC: {agg['summary']['oracle_ic']['mean']:.4f}",
        "",
        "## Distribution of skill and net performance",
        "",
        markdown_table(distribution.round(4)),
        "",
        "## Paired difference from the final pipeline",
        "",
        markdown_table(paired.round(4)),
        "",
        "## Dependence on the simulator",
        "",
        markdown_table(sensitivity.round(4)),
        "",
    ]
    (results_dir / "tables.md").write_text("\n".join(tables))
    print(f"wrote results for {len(seed_results)} evaluation seeds and {len(sweeps)} sweep records")


if __name__ == "__main__":
    main()
