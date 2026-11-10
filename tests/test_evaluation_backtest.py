import numpy as np
import pandas as pd
import pytest

from spendsignal import backtest, evaluation, forecast


def _panel(n_days=300, n_stocks=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n_stocks)]
    return pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=idx, columns=cols)


def test_rank_ic_perfect_and_inverse():
    x = _panel()
    assert evaluation.daily_rank_ic(x, x).mean() == pytest.approx(1.0)
    assert evaluation.daily_rank_ic(x, -x).mean() == pytest.approx(-1.0)


def test_rank_ic_of_noise_is_small():
    ic = evaluation.daily_rank_ic(_panel(seed=1), _panel(seed=2))
    assert abs(ic.mean()) < 0.02
    assert abs(evaluation.newey_west_tstat(ic)) < 3


def test_forward_returns_alignment():
    r = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=pd.bdate_range("2020-01-01", periods=3))
    fwd = evaluation.forward_returns(r)
    assert fwd["A"].iloc[0] == 0.02 and np.isnan(fwd["A"].iloc[-1])


def test_long_short_weights_are_dollar_neutral():
    w = backtest.long_short_weights(_panel(), fraction=0.2)
    assert np.allclose(w.sum(axis=1), 0.0)
    assert np.allclose(w.clip(lower=0).sum(axis=1), 1.0)
    assert np.allclose((w > 0).sum(axis=1), 8)


def test_backtest_costs_and_alignment():
    idx = pd.bdate_range("2020-01-01", periods=3)
    w = pd.DataFrame({"A": [1.0, 1.0, -1.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=idx)
    res = backtest.run_backtest(w, fwd, cost_bps=10)
    assert res.turnover.tolist() == [1.0, 0.0, 2.0]
    assert res.gross.tolist() == pytest.approx([0.01, 0.02, -0.03])
    assert res.net.tolist() == pytest.approx([0.01 - 0.001, 0.02, -0.03 - 0.002])


def test_performance_metrics():
    pnl = pd.Series([0.01, -0.01] * 126)
    perf = backtest.performance(pnl)
    assert perf["ann_return"] == pytest.approx(0.0)
    assert perf["hit_rate"] == pytest.approx(0.5)


def test_implied_lag_weights():
    coefs = pd.Series({5: 1.0, 10: -1.0})
    w = forecast.implied_lag_weights(coefs, max_lag=10)
    assert w.iloc[:5].tolist() == pytest.approx([0.1] * 5)
    assert w.iloc[5:].tolist() == pytest.approx([-0.1] * 5)


def test_walk_forward_recovers_planted_signal():
    signal = _panel(n_days=900, n_stocks=60, seed=3)
    noise = _panel(n_days=900, n_stocks=60, seed=4)
    fwd = 0.2 * signal + noise
    wf = forecast.walk_forward_ridge(signal, fwd, spans=(0, 5), first_test_year=2022, alphas=(1.0, 10.0))
    ic = evaluation.daily_rank_ic(wf.predictions, fwd)
    assert ic.mean() > 0.1
    assert wf.coefficients.loc[2022, 0] > abs(wf.coefficients.loc[2022, 5])


def test_walk_forward_tree_recovers_planted_signal():
    signal = _panel(n_days=900, n_stocks=60, seed=3)
    noise = _panel(n_days=900, n_stocks=60, seed=4)
    fwd = 0.2 * signal + noise
    wf = forecast.walk_forward_tree(
        signal, fwd, spans=(0, 5), features="lags", first_test_year=2022, learning_rates=(0.05, 0.1)
    )
    ic = evaluation.daily_rank_ic(wf.predictions, fwd)
    assert ic.mean() > 0.05
    assert wf.coefficients.empty  # a tree has no linear coefficients
    assert wf.alphas.loc[2022] in (0.05, 0.1)  # a learning rate was chosen on held-out data
