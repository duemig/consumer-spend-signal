"""Seasonal adjustment of daily spend series and alignment to trading days.

Inputs are DataFrames indexed by consecutive calendar days with one column per
stock. All transforms only use information up to and including each date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def yoy_growth(level: pd.DataFrame, lag_days: int = 364, smooth_days: int = 7) -> pd.DataFrame:
    """Year-on-year change in log spend, compared with the same weekday a year earlier.

    A 364-day lag keeps weekdays aligned, so the day-of-week pattern and most of
    the annual pattern cancel, together with each stock's size. A trailing
    ``smooth_days`` mean then removes what is left of the weekly cycle.
    """
    _check_calendar(level)
    growth = level - level.shift(lag_days)
    return growth.rolling(smooth_days, min_periods=max(1, smooth_days // 2 + 1)).mean()


def bucket_adjust(
    level: pd.DataFrame,
    groups: pd.Series,
    min_weekday_obs: int = 261,
    min_weekend_obs: int = 105,
    min_offseason_obs: int = 304,
    min_season_obs: int = 61,
) -> pd.DataFrame:
    """Calendar-bucket seasonal adjustment (the original study's method).

    Step 1 subtracts, separately for weekdays and weekends, the expanding mean of
    the group-average log spend. Step 2 does the same for November-December and
    for the rest of the year. With ``groups`` mapping stocks to sectors this is the
    original method; mapping each stock to itself also removes stock-level size.
    """
    _check_calendar(level)
    dates = level.index
    weekend = np.asarray(dates.dayofweek >= 5)
    season = np.asarray(dates.month >= 11)
    step1 = _subtract_expanding_group_mean(
        level, groups, weekend, min_obs_true=min_weekend_obs, min_obs_false=min_weekday_obs
    )
    return _subtract_expanding_group_mean(
        step1, groups, season, min_obs_true=min_season_obs, min_obs_false=min_offseason_obs
    )


def _subtract_expanding_group_mean(
    values: pd.DataFrame, groups: pd.Series, mask: np.ndarray, min_obs_true: int, min_obs_false: int
) -> pd.DataFrame:
    groups = groups.reindex(values.columns)
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for flag, min_obs in ((True, min_obs_true), (False, min_obs_false)):
        rows = mask == flag
        part = values.loc[rows]
        group_avg = part.T.groupby(groups).mean().T  # dates x groups
        expanding = group_avg.expanding(min_periods=min_obs).mean()
        mapped = expanding.reindex(columns=groups.to_numpy())
        mapped.columns = values.columns
        out.loc[rows] = part - mapped
    return out


def to_business_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Align calendar-day values to trading days.

    Saturday and Sunday are averaged together with the following Monday, so the
    Monday value is known at Monday's close and no weekend data is lost.
    """
    _check_calendar(daily)
    dates = daily.index
    shift = np.select([dates.dayofweek == 5, dates.dayofweek == 6], [2, 1], default=0)
    target = dates + pd.to_timedelta(shift, unit="D")
    out = daily.groupby(target).mean()
    out = out.loc[out.index <= dates[-1]]  # a trailing weekend has no Monday yet
    out.index.name = daily.index.name
    return out


def _check_calendar(frame: pd.DataFrame) -> None:
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("expected a DatetimeIndex")
    if len(idx) > 1 and not (pd.Series(idx).diff().iloc[1:] == pd.Timedelta(days=1)).all():
        raise ValueError("expected consecutive calendar days")
