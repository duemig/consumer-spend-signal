"""Run the full study: ablations on the main evaluation seed, robustness across seeds, figures.

Usage:
    python scripts/run_study.py            # writes results/ and docs/figures/

All numbers in README.md and docs/report.pdf come from the files this script writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spendsignal import forecast, panel, plotting, seasonality
from spendsignal.evaluation import ic_by_group
from spendsignal.pipeline import ORIGINAL, PipelineConfig, oracle_ic, replace, run
from spendsignal.simulate import SimulationConfig, simulate

ROOT = Path(__file__).resolve().parents[1]

FINAL = PipelineConfig()
ABLATIONS: dict[str, PipelineConfig] = {
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
    "Original design (see text)": ORIGINAL,
}


def row(result) -> dict[str, float]:
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


def markdown_table(df: pd.DataFrame) -> str:
    fmt = {
        "mean_ic": "{:.4f}",
        "t_stat_nw": "{:.1f}",
        "gross_sharpe": "{:.2f}",
        "net_sharpe": "{:.2f}",
        "net_return": "{:.1%}",
        "net_vol": "{:.1%}",
        "max_drawdown": "{:.1%}",
        "turnover": "{:.2f}",
    }
    header = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    lines = [header, sep]
    for name, r in df.iterrows():
        cells = [fmt.get(c, "{:.4f}").format(v) if pd.notna(v) else "" for c, v in r.items()]
        lines.append("| " + " | ".join([str(name)] + cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--extra-seeds", type=int, nargs="*", default=[2025, 2026, 2027, 2028])
    args = parser.parse_args()

    results_dir, fig_dir = ROOT / "results", ROOT / "docs" / "figures"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = simulate(SimulationConfig(seed=args.seed))
    meta = data.metadata.set_index("stock")

    # --- Ablations on the main evaluation seed ---------------------------------------------
    results = {name: run(data, cfg) for name, cfg in ABLATIONS.items()}
    table = pd.DataFrame({name: row(r) for name, r in results.items()}).T
    table.index.name = "Variant"
    oracle = oracle_ic(data)
    oracle_full = oracle_ic(data, card_only=False)
    final = results["Final pipeline"]

    # --- Robustness across further evaluation seeds ---------------------------------------
    seeds = [args.seed, *args.extra_seeds]
    robust_rows = []
    for seed in seeds:
        d = data if seed == args.seed else simulate(SimulationConfig(seed=seed))
        f = results["Final pipeline"] if seed == args.seed else run(d, FINAL)
        o = results["Original design (see text)"] if seed == args.seed else run(d, ORIGINAL)
        ts_cfg = ABLATIONS["All sectors traded (no sector filter)"]
        ts = results["All sectors traded (no sector filter)"] if seed == args.seed else run(d, ts_cfg)
        robust_rows.append(
            {
                "seed": seed,
                "oracle_ic": oracle_ic(d)["mean_ic"],
                "oracle_full_ic": oracle_ic(d, card_only=False)["mean_ic"],
                "final_ic": f.ic_stats["mean_ic"],
                "final_t": f.ic_stats["t_stat_nw"],
                "final_net_sharpe": f.perf_net["sharpe"],
                "no_filter_ic": ts.ic_stats["mean_ic"],
                "no_filter_net_sharpe": ts.perf_net["sharpe"],
                "original_ic": o.ic_stats["mean_ic"],
                "original_net_sharpe": o.perf_net["sharpe"],
            }
        )
    robust = pd.DataFrame(robust_rows).set_index("seed")

    # --- Diagnostics for figures ---------------------------------------------------------------
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
    sector_ic = ic_by_group(final.prediction, final.fwd_returns, meta["sector_name"])
    exposure = meta.groupby("sector_name")["card_share"].mean()

    plotting.cohorts_vs_spend(total, normalised, active, fig_dir / "cohorts_vs_spend.png")
    disc = meta.index[meta["sector"] == 25]
    plotting.seasonality(normalised[disc], adjusted[disc], fig_dir / "seasonality.png")
    plotting.clipping_vs_size(
        clipped, meta["revenue"] + np.log(meta["card_share"]), fig_dir / "pooled_clipping.png"
    )
    plotting.lag_profile(profiles, None, fig_dir / "lag_profile.png")
    plotting.ic_by_sector(sector_ic, exposure, fig_dir / "ic_by_sector.png")
    plotting.cumulative_pnl(
        {"net of 5 bp costs": final.backtest.net, "gross": final.backtest.gross},
        fig_dir / "pnl_final.png",
        "Final pipeline: dollar-neutral quintile spread",
    )
    orig = results["Original design (see text)"]
    market = final.fwd_returns.mean(axis=1)
    plotting.cumulative_pnl(
        {
            "original long-only portfolio": orig.backtest.net,
            "equal-weight market": market.loc[orig.backtest.net.index],
        },
        fig_dir / "pnl_original_vs_market.png",
        "Original design: long-only top half vs the market",
    )

    # --- Write results ------------------------------------------------------------------
    summary = {
        "evaluation_seed": args.seed,
        "test_period": [str(final.prediction.index.min().date()), str(final.prediction.index.max().date())],
        "cleaning_report": final.cleaning,
        "oracle": oracle,
        "oracle_full_information": oracle_full,
        "final_ic": final.ic_stats,
        "final_gross": final.perf_gross,
        "final_net": final.perf_net,
        "final_ridge_alphas": {int(k): float(v) for k, v in final.alphas.items()},
        "final_window_coefficients": {
            int(y): {int(w): float(c) for w, c in r.items()} for y, r in final.coefficients.iterrows()
        },
        "original_ic": orig.ic_stats,
        "original_net": orig.perf_net,
        "original_sectors": {int(k): v for k, v in orig.sectors.items()},
        "market_equal_weight": {
            "ann_return": float(market.mean() * 252),
            "ann_vol": float(market.std() * np.sqrt(252)),
        },
        "max_clipped_share": float(clipped.max()),
        "stocks_over_half_clipped": int((clipped > 0.5).sum()),
        "corr_total_vs_active": float(
            np.corrcoef(
                active.stack(), (total - total.mean()).stack().reindex(active.stack().index).fillna(0)
            )[0, 1]
        ),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    table.to_csv(results_dir / "ablation.csv")
    robust.to_csv(results_dir / "robustness.csv")
    sector_ic.to_csv(results_dir / "ic_by_sector.csv")
    md = [
        f"# Results (evaluation seed {args.seed})",
        "",
        f"Oracle mean IC: {oracle['mean_ic']:.4f}",
        "",
        markdown_table(table),
        "",
        "## Robustness across evaluation seeds",
        "",
        markdown_table(robust),
        "",
    ]
    (results_dir / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
