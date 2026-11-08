"""Forecast evaluation: cross-sectional rank information coefficients."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Turn backward returns r(t-1, t), dated t, into forward returns r(t, t+1), dated t."""
    return returns.shift(-1)


def daily_rank_ic(prediction: pd.DataFrame, target: pd.DataFrame, min_names: int = 10) -> pd.Series:
    """Spearman correlation across stocks between prediction and target, for each date."""
    prediction, target = prediction.align(target, join="inner")
    valid = prediction.notna() & target.notna()
    p = prediction.where(valid).rank(axis=1)
    f = target.where(valid).rank(axis=1)
    p = p.sub(p.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)
    num = (p * f).sum(axis=1)
    den = np.sqrt((p**2).sum(axis=1) * (f**2).sum(axis=1))
    ic = num / den.replace(0.0, np.nan)
    ic[valid.sum(axis=1) < min_names] = np.nan
    return ic.dropna().rename("rank_ic")


def newey_west_tstat(x: pd.Series, lags: int = 10) -> float:
    """t-statistic of the mean of ``x`` with a Newey-West (Bartlett) variance estimate."""
    v = x.dropna().to_numpy()
    n = len(v)
    if n < 2:
        return float("nan")
    e = v - v.mean()
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        var += 2.0 * weight * (e[lag:] @ e[:-lag]) / n
    return float(v.mean() / np.sqrt(var / n))


def ic_summary(ic: pd.Series, nw_lags: int = 10) -> dict[str, float]:
    """Mean, volatility, Newey-West t-statistic and hit rate of a daily IC series."""
    return {
        "mean_ic": float(ic.mean()),
        "ic_vol": float(ic.std()),
        "icir_annual": float(ic.mean() / ic.std() * np.sqrt(252)),
        "t_stat_nw": newey_west_tstat(ic, nw_lags),
        "hit_rate": float((ic > 0).mean()),
        "days": int(ic.count()),
    }


def ic_by_group(
    prediction: pd.DataFrame, target: pd.DataFrame, groups: pd.Series, min_names: int = 5
) -> pd.DataFrame:
    """Daily rank IC computed within each group of stocks (for example GICS sectors)."""
    rows = {}
    for name, members in groups.groupby(groups).groups.items():
        cols = [c for c in members if c in prediction.columns]
        ic = daily_rank_ic(prediction[cols], target[cols], min_names=min_names)
        rows[name] = ic_summary(ic)
    return pd.DataFrame(rows).T
