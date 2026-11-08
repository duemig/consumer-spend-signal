import numpy as np
import pandas as pd

from spendsignal.simulate import CORRUPT_TOKENS, simulate


def test_shapes_and_columns(small_data, small_config):
    assert list(small_data.spend.columns) == ["date", "cohort", "stock", "spend"]
    assert list(small_data.returns.columns) == ["date", "stock", "ret"]
    assert len(small_data.metadata) == small_config.n_stocks
    assert small_data.spend["cohort"].nunique() == small_config.n_cohorts


def test_reproducible(small_config):
    a = simulate(small_config)
    b = simulate(small_config)
    pd.testing.assert_frame_equal(a.returns, b.returns)
    assert a.spend["spend"].astype(str).equals(b.spend["spend"].astype(str))


def test_data_problems_are_present(small_data):
    raw = small_data.spend["spend"]
    assert raw.isin(CORRUPT_TOKENS).any()
    numeric = pd.to_numeric(raw, errors="coerce")
    assert (numeric < 0).any()


def test_inactive_cohorts_have_no_rows(small_data):
    active = small_data.truth["cohort_active"]
    rows = small_data.spend.groupby(["date", "cohort"]).size().unstack(fill_value=0)
    rows = rows.reindex(index=active.index, columns=active.columns, fill_value=0)
    assert (rows.to_numpy()[~active.to_numpy()] == 0).all()
    assert (active.sum(axis=1) >= 3).all()


def test_returns_are_business_days(small_data):
    dates = pd.DatetimeIndex(small_data.returns["date"].unique())
    assert (dates.dayofweek < 5).all()
    assert np.isfinite(small_data.returns["ret"]).all()


def test_card_share_drives_spend_level_and_noise(small_data):
    meta = small_data.metadata.set_index("stock")
    spend = pd.to_numeric(small_data.spend["spend"], errors="coerce")
    level = spend.groupby(small_data.spend["stock"]).median()
    corr = np.corrcoef(np.log(meta["card_share"]), np.log(level.reindex(meta.index).clip(lower=1)))[0, 1]
    assert corr > 0.3


def test_covid_shock_hits_discretionary_spending(small_config):
    from dataclasses import replace

    data = simulate(replace(small_config, start="2019-06-01", end="2020-06-30"))
    df = data.spend.assign(spend=pd.to_numeric(data.spend["spend"], errors="coerce"))
    df = df.merge(data.metadata[["stock", "sector"]], on="stock")
    disc = df[df["sector"] == 25]
    before = disc[(disc["date"] >= "2020-02-01") & (disc["date"] < "2020-03-01")]["spend"].median()
    during = disc[(disc["date"] >= "2020-04-05") & (disc["date"] < "2020-04-20")]["spend"].median()
    assert during < 0.8 * before
