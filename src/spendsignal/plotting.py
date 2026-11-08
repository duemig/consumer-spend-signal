"""Figures for the README and the report (static PNGs, light theme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cohorts_vs_spend(total: pd.DataFrame, normalised: pd.DataFrame, active: pd.DataFrame, path: Path) -> None:
    """Total spend rises with the number of active cohorts; the cohort average does not."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharex=True)
    rng = np.random.default_rng(0)
    for ax, frame, title in (
        (axes[0], total, "Log total spend"),
        (axes[1], normalised, "Average log spend per active cohort"),
    ):
        demeaned = frame - frame.mean()
        x = active.reindex_like(frame).stack()
        y = demeaned.stack().reindex(x.index)
        keep = y.notna() & x.notna()
        idx = rng.choice(np.flatnonzero(keep.to_numpy()), size=min(20_000, int(keep.sum())), replace=False)
        xs = x.to_numpy()[idx] + rng.uniform(-0.25, 0.25, size=len(idx))
        ax.scatter(xs, y.to_numpy()[idx], s=2, alpha=0.15, color=BLUE, linewidths=0)
        means = y[keep].groupby(x[keep]).mean()
        ax.plot(means.index, means.to_numpy(), color=ORANGE, marker="o", markersize=4, label="mean")
        corr = np.corrcoef(x[keep], y[keep])[0, 1]
        ax.set_title(f"{title} (corr {corr:.2f})")
        ax.set_xlabel("Active cohorts")
    axes[0].set_ylabel("Deviation from stock mean (log)")
    axes[1].legend(loc="lower right")
    _save(fig, path)


def seasonality(raw: pd.DataFrame, adjusted: pd.DataFrame, path: Path) -> None:
    """Cross-stock average before and after seasonal adjustment."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    raw_avg = (raw - raw.mean()).mean(axis=1)
    adj_avg = adjusted.mean(axis=1)
    dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = raw_avg.groupby(raw_avg.index.dayofweek).mean()
    axes[0].bar(dow, by_dow.to_numpy(), color=BLUE, width=0.6)
    axes[0].set_title("Day-of-week pattern, consumer discretionary")
    axes[0].set_ylabel("Average deviation (log)")
    window = slice("2018-09-01", "2019-02-28")
    axes[1].plot(raw_avg.loc[window], color=BLUE, linewidth=1.0, label="raw")
    axes[1].plot(adj_avg.loc[window], color=ORANGE, linewidth=1.0, label="adjusted")
    axes[1].set_title("Year-end season 2018-19, consumer discretionary")
    axes[1].legend(loc="upper left")
    axes[1].tick_params(axis="x", rotation=30)
    _save(fig, path)


def clipping_vs_size(share: pd.Series, size: pd.Series, path: Path) -> None:
    """Share of each stock's observations capped by pooled winsorisation, against stock size."""
    _style()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    size = size.reindex(share.index)
    ax.scatter(size, share * 100, s=10, color=BLUE, alpha=0.7, linewidths=0)
    ax.set_xlabel("Panel spend level (log daily consumer card revenue)")
    ax.set_ylabel("Observations capped (%)")
    ax.set_title("Pooled 95th-percentile cap hits large stocks")
    _save(fig, path)


def lag_profile(profiles: pd.DataFrame, truth: pd.Series | None, path: Path) -> None:
    """Implied weight on each daily lag, one line per refit year, with the average highlighted."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for _, row in profiles.iterrows():
        ax.plot(row.index, row.to_numpy(), color=BLUE, alpha=0.25, linewidth=1.0)
    ax.plot(
        profiles.columns,
        profiles.mean().to_numpy(),
        color=BLUE,
        linewidth=2.0,
        label="ridge, average of yearly fits",
    )
    if truth is not None:
        scale = profiles.mean().abs().max() / truth.abs().max()
        ax.plot(
            truth.index, truth.to_numpy() * scale, color=ORANGE, linestyle="--", label="true shape (rescaled)"
        )
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Weight (shape only)")
    ax.set_title("Weight on lagged spend implied by the ridge model")
    ax.legend(loc="upper right")
    _save(fig, path)


def ic_by_sector(table: pd.DataFrame, exposure: pd.Series, path: Path) -> None:
    """Mean out-of-sample rank IC per sector, ordered by card-revenue exposure."""
    _style()
    order = exposure.sort_values(ascending=False).index
    table = table.reindex(order)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(table))
    ax.barh(y, table["mean_ic"].to_numpy(), color=BLUE, height=0.6)
    ax.set_yticks(y, [f"{name} ({exposure[name]:.2f})" for name in table.index])
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Mean daily rank IC, out of sample")
    ax.set_title("Forecast skill by sector (average card share of revenue in brackets)")
    _save(fig, path)


def cumulative_pnl(series: dict[str, pd.Series], path: Path, title: str) -> None:
    """Cumulative simple returns of up to three strategies on one axis."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for (label, pnl), color in zip(series.items(), (BLUE, ORANGE, AQUA), strict=False):
        ax.plot((1 + pnl).cumprod() - 1, color=color, label=label)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)
