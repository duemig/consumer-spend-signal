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


def estimate_beta(returns: pd.DataFrame, window: int = 252, min_periods: int = 120) -> pd.DataFrame:
    """Rolling market beta of each stock from trailing returns.

    The market return each day is the equal-weight cross-sectional mean. Beta on
    date ``t`` is the trailing ``window``-day covariance of a stock's return with
    the market divided by the market variance, using returns up to and including
    ``t``, so it is known at ``t``'s close.

    Args:
        returns: Backward returns as a (dates x stocks) frame.
        window: Length of the trailing estimation window in days.
        min_periods: Minimum observations before a beta is returned.

    Returns:
        A (dates x stocks) frame of betas, null until enough history exists.
    """
    market = returns.mean(axis=1)
    cov = returns.rolling(window, min_periods=min_periods).cov(market)
    var = market.rolling(window, min_periods=min_periods).var()
    return cov.div(var, axis=0)


def neutral_weights(
    prediction: pd.DataFrame,
    beta: pd.DataFrame,
    sector: pd.Series,
    universe: pd.DataFrame | None = None,
    gross: float = 2.0,
) -> pd.DataFrame:
    """Continuous long-short weights orthogonal to dollar, beta and sector.

    Each day the cross-sectionally demeaned prediction is projected onto the
    space orthogonal to a constant, the stock betas and the sector dummies. The
    residual therefore has zero net dollar, market-beta and per-sector exposure.
    It is scaled so that the gross notional (sum of absolute weights) equals
    ``gross``, matching the long-short quintile book.

    Args:
        prediction: Forecast as a (dates x stocks) frame.
        beta: Market betas aligned to ``prediction``; names with a null beta on a
            given day are left out of that day's book.
        sector: GICS sector code per stock.
        universe: (Optional) boolean mask of tradable names; ``None`` trades all.
        gross: Target gross notional (sum of absolute weights) each day.

    Returns:
        A (dates x stocks) weight frame, zero where a name is not held.
    """
    pred = prediction if universe is None else prediction.where(universe)
    dummies = pd.get_dummies(sector.reindex(pred.columns)).to_numpy(dtype=float)
    beta = beta.reindex(index=pred.index, columns=pred.columns)
    out = pd.DataFrame(0.0, index=pred.index, columns=pred.columns)
    pred_values, beta_values = pred.to_numpy(dtype=float), beta.to_numpy(dtype=float)
    for row, (signal, betas) in enumerate(zip(pred_values, beta_values, strict=True)):
        valid = ~np.isnan(signal) & ~np.isnan(betas)
        if valid.sum() < 10:
            continue
        centred = signal[valid] - signal[valid].mean()
        design = np.column_stack([np.ones(valid.sum()), betas[valid], dummies[valid]])
        coefficients, *_ = np.linalg.lstsq(design, centred, rcond=None)
        residual = centred - design @ coefficients
        scale = np.abs(residual).sum()
        if scale > 0:
            out.iloc[row, np.flatnonzero(valid)] = residual / scale * gross
    return out


def apply_rebalance_rule(
    targets: pd.DataFrame, rule: str = "full", band: float = 0.0, speed: float = 1.0
) -> pd.DataFrame:
    """Turn daily target weights into applied weights that trade less often.

    ``"full"`` trades to the target every day. ``"band"`` leaves a name's weight
    unchanged until its target moves more than ``band`` away from the held
    weight, a no-trade band. ``"partial"`` moves a fraction ``speed`` of the way
    from the held weight to the target each day, partial adjustment. Both rules
    carry the held weight forward across days.

    Args:
        targets: Target weights as a (dates x stocks) frame; nulls count as zero.
        rule: One of ``"full"``, ``"band"`` or ``"partial"``.
        band: No-trade threshold for the ``"band"`` rule, in weight units.
        speed: Adjustment fraction in ``(0, 1]`` for the ``"partial"`` rule.

    Returns:
        Applied weights as a (dates x stocks) frame.
    """
    if rule == "full":
        return targets
    target_values = np.nan_to_num(targets.to_numpy(dtype=float))
    applied = np.zeros_like(target_values)
    held = np.zeros(target_values.shape[1])
    for row, target in enumerate(target_values):
        if rule == "band":
            held = np.where(np.abs(target - held) > band, target, held)
        elif rule == "partial":
            held = held + speed * (target - held)
        else:
            raise ValueError(f"unknown rebalance rule: {rule}")
        applied[row] = held
    return pd.DataFrame(applied, index=targets.index, columns=targets.columns)


def run_backtest(
    weights: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_annual: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Daily P&L of a weight matrix, before and after costs.

    Linear trading costs of ``cost_bps`` per unit of one-way turnover are charged
    on traded notional. A short borrow fee of ``borrow_bps_annual`` a year is
    charged daily on the gross short notional, so a dollar-neutral book with a
    gross short of one pays ``borrow_bps_annual`` basis points a year.
    """
    weights, fwd = weights.align(fwd_returns, join="inner")
    gross = (weights * fwd.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    short_notional = weights.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * (borrow_bps_annual / 1e4) / periods_per_year
    net = gross - turnover * cost_bps / 1e4 - borrow
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
