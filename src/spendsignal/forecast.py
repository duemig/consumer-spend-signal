"""From a daily spend predictor to a return forecast.

Three forecasters are provided:

* :func:`smoothed_signal`, a trailing moving average of the predictor (the
  original study's first model),
* :func:`walk_forward_ridge`, a ridge regression of next-day returns on lagged
  values of the predictor, refitted every year on an expanding window with the
  penalty chosen on the last training year, and
* :func:`walk_forward_tree`, a gradient-boosted-tree forecaster with the same
  features, walk-forward schedule and held-out hyperparameter choice, so it can
  be compared with the ridge on equal terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
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


def _build_features(
    predictor: pd.DataFrame, spans: Sequence[int], features: Literal["lags", "windows"]
) -> dict[int, pd.DataFrame]:
    """Single lags or trailing means of the predictor over ``spans`` trading days."""
    if features == "lags":
        return lagged_features(predictor, spans)
    if features == "windows":
        return window_features(predictor, spans)
    raise ValueError(f"unknown feature set: {features}")


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


def _yearly_splits(dates: pd.DatetimeIndex, first_test_year: int):
    """Yield ``(year, train_dates, test_dates)`` for each out-of-sample year.

    Training uses every date before 1 January of the test year except the last
    one, whose forward return is only known inside the test year.
    """
    for year in range(first_test_year, int(dates.max().year) + 1):
        train_dates = dates[dates < pd.Timestamp(year=year, month=1, day=1)][:-1]
        test_dates = dates[dates.year == year]
        if len(train_dates) and len(test_dates):
            yield year, train_dates, test_dates


def _validation_data(features: dict[int, pd.DataFrame], target: pd.DataFrame, train_dates: pd.DatetimeIndex):
    """Inner-train design and target plus a last-year validation design, for tuning.

    Returns ``(X, y, X_val)`` stacked for a held-out comparison of hyperparameters,
    or ``None`` when there is too little history for a reliable comparison.
    """
    last_year = train_dates.max().year
    inner_train = train_dates[train_dates.year < last_year][:-1]
    valid = train_dates[train_dates.year == last_year]
    if len(inner_train) < 250 or len(valid) < 100:
        return None
    X, y = _stack(features, target, inner_train)
    X_val, _ = _stack(features, None, valid)
    if len(X) < 5_000 or len(X_val) < 1_000:
        return None
    return X, y, X_val


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    alphas: pd.Series
    in_sample: dict[int, pd.DataFrame] = field(default_factory=dict)


def walk_forward_ridge(
    predictor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    spans: Sequence[int] = (0, 5, 10, 15, 20, 30, 40, 50, 60),
    features: Literal["lags", "windows"] = "lags",
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
    feats = _build_features(predictor, spans, features)
    target = fwd_returns.sub(fwd_returns.mean(axis=1), axis=0)
    dates = predictor.index
    if first_train_date is not None:
        dates = dates[dates >= pd.Timestamp(first_train_date)]

    preds, coefs, chosen, in_sample = [], {}, {}, {}
    for year, train_dates, test_dates in _yearly_splits(dates, first_test_year):
        alpha = _choose_alpha(feats, target, train_dates, alphas)
        X, y = _stack(feats, target, train_dates)
        if len(X) == 0:
            continue
        model = Ridge(alpha=alpha).fit(X.to_numpy(), y.to_numpy())
        X_test, _ = _stack(feats, None, test_dates)
        p = pd.Series(model.predict(X_test.to_numpy()), index=X_test.index).unstack()
        preds.append(p)
        coefs[year] = pd.Series(model.coef_, index=list(spans))
        chosen[year] = alpha
        in_sample[year] = pd.Series(model.predict(X.to_numpy()), index=X.index).unstack()

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
    split = _validation_data(features, target, train_dates)
    if split is None:
        return float(np.median(alphas))
    X, y, X_val = split
    scores = {}
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(X.to_numpy(), y.to_numpy())
        p = pd.Series(model.predict(X_val.to_numpy()), index=X_val.index).unstack()
        scores[alpha] = daily_rank_ic(p, target).mean()
    return float(max(scores, key=scores.get))


def walk_forward_tree(
    predictor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    spans: Sequence[int] = (5, 10, 20, 40, 60, 120),
    features: Literal["lags", "windows"] = "windows",
    first_test_year: int = 2019,
    first_train_date: str | pd.Timestamp | None = None,
    learning_rates: Sequence[float] = (0.02, 0.05, 0.1),
    max_iter: int = 100,
    max_leaf_nodes: int = 15,
    l2_regularization: float = 10.0,
) -> WalkForwardResult:
    """Yearly expanding-window gradient-boosted-tree forecasts of next-day returns.

    This mirrors :func:`walk_forward_ridge`: the same features, the same yearly
    refit on an expanding window, and the same held-out rule for the one tuned
    setting. A ``HistGradientBoostingRegressor`` replaces the ridge, and the
    learning rate takes the place of the ridge penalty, chosen by maximising the
    mean daily rank IC in the last training year. The remaining tree settings are
    fixed. The returned ``coefficients`` frame is empty, because a tree has no
    linear coefficients; ``alphas`` holds the chosen learning rate per year.

    Args:
        predictor: Daily spend predictor as a (dates x stocks) frame.
        fwd_returns: Forward returns r(t, t+1) dated t, aligned to the predictor.
        spans: Trailing-mean windows (``features="windows"``) or lags in days.
        features: ``"windows"`` for trailing means, ``"lags"`` for single lags.
        first_test_year: First calendar year forecast out of sample.
        first_train_date: (Optional) earliest date used for training; ``None``
            uses all available history.
        learning_rates: Grid of learning rates compared on the validation year.
        max_iter: Number of boosting iterations.
        max_leaf_nodes: Maximum leaves per tree.
        l2_regularization: L2 penalty on leaf values.

    Returns:
        A :class:`WalkForwardResult` with out-of-sample predictions, an empty
        coefficient frame, the chosen learning rate per year and the in-sample
        fitted values per year.
    """
    feats = _build_features(predictor, spans, features)
    target = fwd_returns.sub(fwd_returns.mean(axis=1), axis=0)
    dates = predictor.index
    if first_train_date is not None:
        dates = dates[dates >= pd.Timestamp(first_train_date)]
    settings = {
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "l2_regularization": l2_regularization,
    }

    preds, chosen, in_sample = [], {}, {}
    for year, train_dates, test_dates in _yearly_splits(dates, first_test_year):
        rate = _choose_learning_rate(feats, target, train_dates, learning_rates, settings)
        X, y = _stack(feats, target, train_dates)
        if len(X) == 0:
            continue
        model = HistGradientBoostingRegressor(learning_rate=rate, random_state=0, **settings)
        model.fit(X.to_numpy(), y.to_numpy())
        X_test, _ = _stack(feats, None, test_dates)
        p = pd.Series(model.predict(X_test.to_numpy()), index=X_test.index).unstack()
        preds.append(p)
        chosen[year] = rate
        in_sample[year] = pd.Series(model.predict(X.to_numpy()), index=X.index).unstack()

    predictions = pd.concat(preds).reindex(columns=predictor.columns)
    return WalkForwardResult(
        predictions=predictions,
        coefficients=pd.DataFrame(),
        alphas=pd.Series(chosen, name="learning_rate"),
        in_sample=in_sample,
    )


def _choose_learning_rate(
    features: dict[int, pd.DataFrame],
    target: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    rates: Sequence[float],
    settings: dict[str, float],
) -> float:
    split = _validation_data(features, target, train_dates)
    if split is None:
        return float(np.median(rates))
    X, y, X_val = split
    scores = {}
    for rate in rates:
        model = HistGradientBoostingRegressor(learning_rate=rate, random_state=0, **settings)
        model.fit(X.to_numpy(), y.to_numpy())
        p = pd.Series(model.predict(X_val.to_numpy()), index=X_val.index).unstack()
        scores[rate] = daily_rank_ic(p, target).mean()
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
