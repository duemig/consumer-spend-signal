"""Point-in-time checks: removing future data must not change past values."""

import numpy as np
import pandas as pd
import pytest

from spendsignal import backtest, forecast
from spendsignal.evaluation import forward_returns
from spendsignal.pipeline import PipelineConfig, build_predictor, returns_wide
from spendsignal.simulate import SimulatedData

CUTOFF = pd.Timestamp("2019-06-14")  # a Friday


def _truncate(data: SimulatedData, cutoff: pd.Timestamp) -> SimulatedData:
    return SimulatedData(
        spend=data.spend[data.spend["date"] <= cutoff].reset_index(drop=True),
        returns=data.returns[data.returns["date"] <= cutoff].reset_index(drop=True),
        metadata=data.metadata,
        truth=data.truth,
    )


@pytest.mark.parametrize(
    "cfg",
    [
        PipelineConfig(),
        PipelineConfig(cleaning="pooled_winsor", seasonality="yoy", normalisation="adjusted"),
        PipelineConfig(seasonality="bucket_stock"),
    ],
    ids=["final", "alternatives", "per-stock-buckets"],
)
def test_predictor_is_point_in_time(small_data, cfg):
    full, _ = build_predictor(small_data, cfg)
    part, _ = build_predictor(_truncate(small_data, CUTOFF), cfg)
    common = part.index
    pd.testing.assert_frame_equal(full.loc[common], part, check_freq=False, atol=1e-12)


def test_forecast_is_point_in_time(small_data):
    cfg = PipelineConfig()
    predictor, _ = build_predictor(small_data, cfg)
    fwd = forward_returns(returns_wide(small_data)).reindex(predictor.index)
    full = forecast.walk_forward_ridge(
        predictor, fwd, spans=cfg.windows, features="windows", first_test_year=2019
    )

    truncated = _truncate(small_data, CUTOFF)
    p2, _ = build_predictor(truncated, cfg)
    fwd2 = forward_returns(returns_wide(truncated)).reindex(p2.index)
    part = forecast.walk_forward_ridge(p2, fwd2, spans=cfg.windows, features="windows", first_test_year=2019)

    common = part.predictions.index
    assert len(common) > 50
    np.testing.assert_allclose(
        full.predictions.loc[common].to_numpy(), part.predictions.to_numpy(), atol=1e-12
    )


def test_tree_forecast_is_point_in_time(small_data):
    cfg = PipelineConfig()
    predictor, _ = build_predictor(small_data, cfg)
    fwd = forward_returns(returns_wide(small_data)).reindex(predictor.index)
    full = forecast.walk_forward_tree(predictor, fwd, spans=cfg.windows, first_test_year=2019)

    truncated = _truncate(small_data, CUTOFF)
    p2, _ = build_predictor(truncated, cfg)
    fwd2 = forward_returns(returns_wide(truncated)).reindex(p2.index)
    part = forecast.walk_forward_tree(p2, fwd2, spans=cfg.windows, first_test_year=2019)

    common = part.predictions.index
    assert len(common) > 20
    np.testing.assert_allclose(
        full.predictions.loc[common].to_numpy(), part.predictions.to_numpy(), atol=1e-12
    )


def test_estimate_beta_is_point_in_time():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2019-01-01", periods=400)
    columns = [f"S{i}" for i in range(10)]
    returns = pd.DataFrame(rng.normal(0.0, 0.01, size=(400, 10)), index=index, columns=columns)
    cutoff = index[300]
    full = backtest.estimate_beta(returns, window=120, min_periods=60)
    part = backtest.estimate_beta(returns.loc[:cutoff], window=120, min_periods=60)
    pd.testing.assert_frame_equal(full.loc[part.index], part, atol=1e-12)


def test_neutral_weights_are_point_in_time():
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2019-01-01", periods=60)
    columns = [f"S{i}" for i in range(12)]
    prediction = pd.DataFrame(rng.normal(size=(60, 12)), index=index, columns=columns)
    beta = pd.DataFrame(1.0 + 0.3 * rng.normal(size=(60, 12)), index=index, columns=columns)
    sector = pd.Series({col: [10, 20, 30][i % 3] for i, col in enumerate(columns)})
    cutoff = index[40]
    full = backtest.neutral_weights(prediction, beta, sector)
    part = backtest.neutral_weights(prediction.loc[:cutoff], beta.loc[:cutoff], sector)
    pd.testing.assert_frame_equal(full.loc[part.index], part, atol=1e-12)
