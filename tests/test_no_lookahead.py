"""Point-in-time checks: removing future data must not change past values."""

import numpy as np
import pandas as pd
import pytest

from spendsignal import forecast
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
        predictor, fwd, lags=cfg.windows, features="windows", first_test_year=2019
    )

    truncated = _truncate(small_data, CUTOFF)
    p2, _ = build_predictor(truncated, cfg)
    fwd2 = forward_returns(returns_wide(truncated)).reindex(p2.index)
    part = forecast.walk_forward_ridge(p2, fwd2, lags=cfg.windows, features="windows", first_test_year=2019)

    common = part.predictions.index
    assert len(common) > 50
    np.testing.assert_allclose(
        full.predictions.loc[common].to_numpy(), part.predictions.to_numpy(), atol=1e-12
    )
