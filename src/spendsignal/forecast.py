"""From a daily spend predictor to a return forecast.

Two forecasters are provided:

* :func:`smoothed_signal`, a trailing moving average of the predictor (the
  original study's first model), and
* :func:`walk_forward_ridge`, a ridge regression of next-day returns on lagged
  values of the predictor, refitted every year on an expanding window with the
  penalty chosen on the last training year.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from spendsignal.evaluation import daily_rank_ic, newey_west_tstat


def cross_sectional_zscore(frame: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Standardise each row across stocks and clip at ``clip`` standard deviations."""
    mu = frame.mean(axis=1)
    sd = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


def smoothed_signal(predictor: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Trailing moving average over ``window`` trading days, standardised across stocks."""
    smooth = predictor.rolling(window, min_periods=window // 2).mean()
    return cross_sectional_zscore(smooth)


def lagged_features(predictor: pd.DataFrame, lags: Sequence[int]) -> dict[int, pd.DataFrame]:
    """Cross-sectionally standardised copies of the predictor lagged by ``lags`` trading days."""
    return {lag: cross_sectional_zscore(predictor.shift(lag)) for lag in lags}


def window_features(predictor: pd.DataFrame, windows: Sequence[int]) -> dict[int, pd.DataFrame]:
    """Cross-sectionally standardised trailing means of the predictor over ``windows`` trading days."""
    return {
        w: cross_sectional_zscore(predictor.rolling(w, min_periods=max(1, w // 2)).mean()) for w in windows
    }


def implied_lag_weights(coefficients: pd.Series, max_lag: int | None = None) -> pd.Series:
    """Weight on each daily lag implied by ridge coefficients on trailing-mean features.

    A trailing mean over ``w`` days puts weight ``1 / w`` on lags ``0 .. w-1``, so a
    combination of such means is itself a weighted sum of lagged values.
    Standardisation is ignored, so the profile shows shape, not scale.
    """
    windows = [int(w) for w in coefficients.index]
    max_lag = max_lag or max(windows)
    weights = np.zeros(max_lag)
    for w, c in zip(windows, coefficients.to_numpy(), strict=True):
        weights[: min(w, max_lag)] += c / w
    return pd.Series(weights, index=pd.RangeIndex(max_lag, name="lag"))


def _stack(features: dict[int, pd.DataFrame], target: pd.DataFrame | None, dates: pd.DatetimeIndex):
    cols = {f"lag_{lag}": f.loc[f.index.intersection(dates)].stack() for lag, f in features.items()}
    X = pd.DataFrame(cols)
    if target is not None:
        y = target.loc[target.index.intersection(dates)].stack().rename("y")
        data = X.join(y, how="inner").dropna()
        return data.drop(columns="y"), data["y"]
    return X.dropna(), None


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    alphas: pd.Series
    in_sample: dict[int, pd.DataFrame] = field(default_factory=dict)


def walk_forward_ridge(
    predictor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    lags: Sequence[int] = (0, 5, 10, 15, 20, 30, 40, 50, 60),
    features: str = "lags",
    first_test_year: int = 2019,
    first_train_date: str | pd.Timestamp | None = None,
    alphas: Sequence[float] = (1.0, 10.0, 100.0, 1_000.0, 10_000.0),
) -> WalkForwardResult:
    """Yearly expanding-window ridge forecasts of next-day returns.

    For test year ``Y`` the model is trained on dates before 1 January ``Y``,
    excluding the last training date, whose forward return is only known in
    year ``Y``. The penalty is chosen by fitting on dates before ``Y - 1`` and
    maximising the mean daily rank IC in year ``Y - 1``. The target is the
    forward return minus its cross-sectional mean.

    ``features="lags"`` regresses on lagged values (the original design);
    ``features="windows"`` regresses on trailing means over the given windows.
    """
    if features == "lags":
        feats = lagged_features(predictor, lags)
    elif features == "windows":
        feats = window_features(predictor, lags)
    else:
        raise ValueError(f"unknown feature set: {features}")
    features_ = feats
    target = fwd_returns.sub(fwd_returns.mean(axis=1), axis=0)
    dates = predictor.index
    if first_train_date is not None:
        dates = dates[dates >= pd.Timestamp(first_train_date)]
    last_year = int(dates.max().year)

    preds, coefs, chosen, in_sample = [], {}, {}, {}
    for year in range(first_test_year, last_year + 1):
        train_dates = dates[dates < pd.Timestamp(year=year, month=1, day=1)][:-1]
        test_dates = dates[dates.year == year]
        if len(train_dates) == 0 or len(test_dates) == 0:
            continue
        alpha = _choose_alpha(features_, target, train_dates, alphas)
        X, y = _stack(features_, target, train_dates)
        if len(X) == 0:
            continue
        model = Ridge(alpha=alpha).fit(X.to_numpy(), y.to_numpy())
        X_test, _ = _stack(features_, None, test_dates)
        p = pd.Series(model.predict(X_test.to_numpy()), index=X_test.index).unstack()
        preds.append(p)
        coefs[year] = pd.Series(model.coef_, index=list(lags))
        chosen[year] = alpha
        fitted = pd.Series(model.predict(X.to_numpy()), index=X.index).unstack()
        in_sample[year] = fitted

    predictions = pd.concat(preds).reindex(columns=predictor.columns)
    return WalkForwardResult(
        predictions=predictions,
        coefficients=pd.DataFrame(coefs).T.rename_axis(index="test_year", columns="lag"),
        alphas=pd.Series(chosen, name="alpha"),
        in_sample=in_sample,
    )


def _choose_alpha(
    features: dict[int, pd.DataFrame],
    target: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    alphas: Sequence[float],
) -> float:
    last_year = train_dates.max().year
    inner_train = train_dates[train_dates.year < last_year][:-1]
    valid = train_dates[train_dates.year == last_year]
    if len(inner_train) < 250 or len(valid) < 100:
        return float(np.median(alphas))
    X, y = _stack(features, target, inner_train)
    X_val, _ = _stack(features, None, valid)
    if len(X) < 5_000 or len(X_val) < 1_000:  # not enough history to compare penalties
        return float(np.median(alphas))
    scores = {}
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(X.to_numpy(), y.to_numpy())
        p = pd.Series(model.predict(X_val.to_numpy()), index=X_val.index).unstack()
        scores[alpha] = daily_rank_ic(p, target).mean()
    return float(max(scores, key=scores.get))


def select_sectors(
    prediction: pd.DataFrame, target: pd.DataFrame, sector: pd.Series, min_t: float = 2.0
) -> list[int]:
    """Sectors whose within-sector daily rank IC has a Newey-West t-statistic above ``min_t``."""
    keep = []
    for code, members in sector.groupby(sector).groups.items():
        cols = [c for c in members if c in prediction.columns]
        ic = daily_rank_ic(prediction[cols], target[cols], min_names=5)
        if len(ic) > 50 and newey_west_tstat(ic) > min_t:
            keep.append(int(code))
    return sorted(keep)
