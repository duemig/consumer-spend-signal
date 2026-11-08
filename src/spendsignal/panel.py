"""Turning a raw long-format card spending panel into a per-stock spend estimate.

Every function here is point-in-time: the value for date ``t`` only uses rows
dated ``t`` or earlier. ``tests/test_no_lookahead.py`` checks this.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _nanmean(values: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    """``np.nanmean`` without the warning for all-NaN slices (which return NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis, keepdims=keepdims)


@dataclass
class SpendCube:
    """Spend as a dense (days x stocks x cohorts) array. Missing rows are NaN."""

    values: np.ndarray
    dates: pd.DatetimeIndex
    stocks: pd.Index
    cohorts: pd.Index

    def copy(self) -> SpendCube:
        return SpendCube(self.values.copy(), self.dates, self.stocks, self.cohorts)


@dataclass
class CleaningReport:
    rows: int
    non_numeric: int
    non_positive: int
    outliers: int

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


def to_cube(spend: pd.DataFrame) -> tuple[SpendCube, int]:
    """Pivot a long panel (date, cohort, stock, spend) into a :class:`SpendCube`.

    Non-numeric spend values are set to NaN. Duplicate keys keep the last row.
    Returns the cube and the number of non-numeric values found.
    """
    numeric = pd.to_numeric(spend["spend"], errors="coerce")
    non_numeric = int(numeric.isna().sum() - spend["spend"].isna().sum())
    dates = pd.DatetimeIndex(np.sort(spend["date"].unique()))
    stocks = pd.Index(np.sort(spend["stock"].unique()), name="stock")
    cohorts = pd.Index(np.sort(spend["cohort"].unique()), name="cohort")
    t = dates.get_indexer(spend["date"])
    i = stocks.get_indexer(spend["stock"])
    k = cohorts.get_indexer(spend["cohort"])
    values = np.full((len(dates), len(stocks), len(cohorts)), np.nan)
    values[t, i, k] = numeric.to_numpy(dtype=float)
    return SpendCube(values, dates, stocks, cohorts), non_numeric


def winsorize_pooled(cube: SpendCube, lower: float = 0.05, upper: float = 0.95) -> SpendCube:
    """Clip each day's values at quantiles taken over all stocks and cohorts together.

    This is the treatment used in the original study. It is kept for the
    ablation only: because stocks differ in size by orders of magnitude, it
    caps the largest stocks rather than their outliers.
    """
    out = cube.copy()
    flat = out.values.reshape(len(out.dates), -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanquantile(flat, lower, axis=1)
        hi = np.nanquantile(flat, upper, axis=1)
    flat[:] = np.clip(flat, lo[:, None], hi[:, None])
    return out


def clipped_share_pooled(cube: SpendCube, upper: float = 0.95) -> pd.Series:
    """Share of each stock's observations that :func:`winsorize_pooled` caps at the top."""
    flat = cube.values.reshape(len(cube.dates), -1)
    hi = np.nanquantile(flat, upper, axis=1)
    above = cube.values > hi[:, None, None]
    observed = ~np.isnan(cube.values)
    share = above.sum(axis=(0, 2)) / observed.sum(axis=(0, 2))
    return pd.Series(share, index=cube.stocks, name="clipped_share")


def log_and_filter(
    cube: SpendCube, window: int = 90, min_periods: int = 30, threshold: float = 6.0
) -> tuple[SpendCube, int, int]:
    """Take logs and remove outliers series by series.

    Non-positive values (refunds and reversals) cannot be logged and are set to
    NaN. Each (stock, cohort) series is then compared with the median and the
    median absolute deviation of its own trailing ``window`` days, excluding the
    current day. Values more than ``threshold`` robust standard deviations away
    are set to NaN.

    Returns the filtered log cube, the number of non-positive values and the
    number of outliers removed.
    """
    out = cube.copy()
    vals = out.values
    non_positive = int(np.sum(vals <= 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        logs = np.where(vals > 0, np.log(vals), np.nan)

    n_days = logs.shape[0]
    frame = pd.DataFrame(logs.reshape(n_days, -1))
    med = frame.rolling(window, min_periods=min_periods).median().shift(1)
    dev = (frame - med).abs()
    mad = dev.rolling(window, min_periods=min_periods).median().shift(1)
    z = (frame - med) / (1.4826 * mad)
    outlier = (z.abs() > threshold).to_numpy()
    flat = frame.to_numpy(copy=True)
    n_out = int(outlier.sum())
    flat[outlier] = np.nan
    out.values = flat.reshape(logs.shape)
    return out, non_positive, n_out


def log_only(cube: SpendCube) -> SpendCube:
    """Take logs without any filtering (non-positive values become NaN)."""
    out = cube.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out.values = np.where(out.values > 0, np.log(out.values), np.nan)
    return out


def active_cohorts(cube: SpendCube) -> pd.DataFrame:
    """Number of cohorts with a valid observation, per day and stock."""
    counts = (~np.isnan(cube.values)).sum(axis=2)
    return pd.DataFrame(counts, index=cube.dates, columns=cube.stocks)


def total_log_spend(log_cube: SpendCube) -> pd.DataFrame:
    """Log of total spend summed over active cohorts (depends on how many are active)."""
    with np.errstate(invalid="ignore"):
        total = np.nansum(np.exp(log_cube.values), axis=2)
    total[total == 0] = np.nan
    return pd.DataFrame(np.log(total), index=log_cube.dates, columns=log_cube.stocks)


def cohort_mean(log_cube: SpendCube) -> pd.DataFrame:
    """Average log spend over active cohorts (the original normalisation)."""
    avg = _nanmean(log_cube.values, axis=2)
    return pd.DataFrame(avg, index=log_cube.dates, columns=log_cube.stocks)


def cohort_mean_adjusted(log_cube: SpendCube, warmup: int = 60) -> pd.DataFrame:
    """Average log spend over active cohorts after removing each cohort's size offset.

    A cohort's offset is the expanding mean, over its own history, of its log
    spend minus the cross-cohort average, averaged across stocks. A cohort only
    enters the estimate after ``warmup`` days of history, so that composition
    changes do not shift the series.
    """
    vals = log_cube.values
    centre = _nanmean(vals, axis=2, keepdims=True)
    rel = _nanmean(vals - centre, axis=1)  # days x cohorts
    rel_df = pd.DataFrame(rel, index=log_cube.dates, columns=log_cube.cohorts)
    history = rel_df.notna().cumsum()
    offset = rel_df.expanding(min_periods=1).mean()
    eligible = (history >= warmup).to_numpy()
    adjusted = np.where(eligible[:, None, :], vals - offset.to_numpy()[:, None, :], np.nan)
    avg = _nanmean(adjusted, axis=2)
    return pd.DataFrame(avg, index=log_cube.dates, columns=log_cube.stocks)
