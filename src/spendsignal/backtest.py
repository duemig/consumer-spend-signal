"""Daily portfolio backtests driven by a cross-sectional forecast.

Weights formed at the close of day ``t`` earn the forward return r(t, t+1).
Costs are charged on traded notional: ``cost_bps`` per unit of one-way turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    gross: pd.Series
    net: pd.Series
    turnover: pd.Series


def long_short_weights(
    prediction: pd.DataFrame, fraction: float = 0.2, universe: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Equal-weight long the top ``fraction`` and short the bottom ``fraction`` each day.

    Each side has a gross weight of 1, so the book is dollar neutral.
    """
    pred = prediction if universe is None else prediction.where(universe)
    n = pred.notna().sum(axis=1)
    k = np.floor(n * fraction).replace(0, np.nan)
    rank = pred.rank(axis=1, method="first")
    longs = rank.gt(n - k, axis=0).astype(float)
    shorts = rank.le(k, axis=0).astype(float)
    return (longs.div(k, axis=0) - shorts.div(k, axis=0)).fillna(0.0)


def long_only_weights(
    prediction: pd.DataFrame, fraction: float = 0.5, universe: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Equal-weight long the top ``fraction`` of names each day (gross weight 1)."""
    pred = prediction if universe is None else prediction.where(universe)
    n = pred.notna().sum(axis=1)
    k = np.floor(n * fraction)
    rank = pred.rank(axis=1, method="first")
    longs = rank.gt(n - k, axis=0)
    return longs.astype(float).div(k.replace(0, np.nan), axis=0).fillna(0.0)


def run_backtest(weights: pd.DataFrame, fwd_returns: pd.DataFrame, cost_bps: float = 5.0) -> BacktestResult:
    """Daily P&L of a weight matrix, before and after linear trading costs."""
    weights, fwd = weights.align(fwd_returns, join="inner")
    gross = (weights * fwd.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    net = gross - turnover * cost_bps / 1e4
    keep = fwd.notna().any(axis=1)
    return BacktestResult(weights[keep], gross[keep], net[keep], turnover[keep])


def performance(pnl: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Annualised return and volatility, Sharpe ratio, maximum drawdown and hit rate.

    Returns are treated as simple daily returns on capital and compounded for the drawdown.
    """
    pnl = pnl.dropna()
    ann_ret = pnl.mean() * periods_per_year
    ann_vol = pnl.std() * np.sqrt(periods_per_year)
    wealth = (1.0 + pnl).cumprod()
    drawdown = 1.0 - wealth / wealth.cummax()
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else float("nan"),
        "max_drawdown": float(drawdown.max()),
        "hit_rate": float((pnl > 0).mean()),
    }
