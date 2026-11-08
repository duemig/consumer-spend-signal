import numpy as np
import pandas as pd
import pytest

from spendsignal import seasonality


def test_weekends_fold_into_monday():
    dates = pd.date_range("2021-01-01", "2021-01-11", freq="D")  # Fri .. Mon
    daily = pd.DataFrame({"A": np.arange(len(dates), dtype=float)}, index=dates)
    bd = seasonality.to_business_days(daily)
    assert (bd.index.dayofweek < 5).all()
    # Monday 4 Jan = mean of Sat 2, Sun 3, Mon 4 -> values 1, 2, 3
    assert bd.loc["2021-01-04", "A"] == pytest.approx(2.0)


def test_trailing_weekend_is_dropped():
    dates = pd.date_range("2021-01-04", "2021-01-09", freq="D")  # Mon .. Sat
    daily = pd.DataFrame({"A": 1.0}, index=dates)
    bd = seasonality.to_business_days(daily)
    assert bd.index.max() == pd.Timestamp("2021-01-08")


def test_yoy_removes_weekly_and_annual_pattern():
    dates = pd.date_range("2018-01-01", "2020-12-31", freq="D")
    weekly = np.array([0, 0, 0, 0, 0.1, 0.3, 0.2])[dates.dayofweek]
    annual = 0.3 * (dates.month == 12)
    level = pd.DataFrame({"A": 5.0 + weekly + annual}, index=dates)
    growth = seasonality.yoy_growth(level).dropna()
    # December shifts by one weekday per year, so allow a small residual at the season edges
    assert growth["A"].abs().quantile(0.95) < 0.05


def test_bucket_adjust_removes_weekend_level():
    dates = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    rng = np.random.default_rng(1)
    weekend = (dates.dayofweek >= 5).astype(float) * 0.5
    level = pd.DataFrame({"A": weekend + rng.normal(0, 0.05, len(dates))}, index=dates)
    groups = pd.Series({"A": "g"})
    adj = seasonality.bucket_adjust(level, groups).dropna()
    by_weekend = adj["A"].groupby(adj.index.dayofweek >= 5).mean()
    assert abs(by_weekend[True] - by_weekend[False]) < 0.05


def test_calendar_check():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-03"])
    with pytest.raises(ValueError):
        seasonality.yoy_growth(pd.DataFrame({"A": [1.0, 2.0]}, index=idx))
