import numpy as np
import pandas as pd

from spendsignal import backtest


def _frame(n_days=20, n_stocks=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n_stocks)]
    return pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=idx, columns=cols)


def test_estimate_beta_tracks_true_beta():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=400)
    true_beta = np.linspace(0.5, 1.5, 15)  # mean 1, so the equal-weight market has beta 1
    factor = rng.normal(0.0, 0.01, size=400)
    noise = rng.normal(0.0, 0.002, size=(400, 15))
    returns = pd.DataFrame(
        true_beta * factor[:, None] + noise, index=idx, columns=[f"S{i}" for i in range(15)]
    )
    estimated = backtest.estimate_beta(returns, window=250, min_periods=120).iloc[-1].to_numpy()
    assert np.corrcoef(estimated, true_beta)[0, 1] > 0.95
    assert estimated[-1] > estimated[0] + 0.5


def test_neutral_weights_remove_dollar_beta_and_sector():
    pred = _frame(seed=0)
    beta = 1.0 + 0.3 * _frame(seed=1)
    sector = pd.Series({col: [10, 20, 30][i % 3] for i, col in enumerate(pred.columns)})
    weights = backtest.neutral_weights(pred, beta, sector)
    assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-8)  # dollar neutral
    assert np.allclose((weights * beta).sum(axis=1), 0.0, atol=1e-8)  # beta neutral
    assert np.allclose(weights.T.groupby(sector).sum().to_numpy(), 0.0, atol=1e-8)  # sector neutral
    assert np.allclose(weights.abs().sum(axis=1), 2.0, atol=1e-8)  # gross of 2
    # the book leans long the high-prediction names within the neutral constraints
    assert (np.sign(weights) == np.sign(pred.sub(pred.mean(axis=1), axis=0))).mean().mean() > 0.5


def test_neutral_weights_skip_names_without_beta():
    pred = _frame(seed=0)
    beta = 1.0 + 0.3 * _frame(seed=1)
    beta.iloc[:, 0] = np.nan  # no beta estimate for the first name
    sector = pd.Series({col: [10, 20, 30][i % 3] for i, col in enumerate(pred.columns)})
    weights = backtest.neutral_weights(pred, beta, sector)
    assert (weights.iloc[:, 0] == 0.0).all()
    assert np.allclose((weights * beta.fillna(0.0)).sum(axis=1), 0.0, atol=1e-8)


def test_rebalance_rules_reduce_turnover():
    targets = _frame(seed=2)
    full = backtest.apply_rebalance_rule(targets, "full")
    assert full is targets
    partial_full = backtest.apply_rebalance_rule(targets, "partial", speed=1.0)
    pd.testing.assert_frame_equal(partial_full, targets.fillna(0.0), check_dtype=False)

    def total_turnover(weights):
        return float(weights.diff().abs().sum(axis=1).sum() + weights.iloc[0].abs().sum())

    base = total_turnover(targets)
    assert total_turnover(backtest.apply_rebalance_rule(targets, "band", band=0.3)) < base
    assert total_turnover(backtest.apply_rebalance_rule(targets, "partial", speed=0.5)) < base


def test_borrow_cost_charged_on_short_notional():
    idx = pd.bdate_range("2020-01-01", periods=2)
    weights = pd.DataFrame({"A": [-1.0, -1.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.0, 0.0]}, index=idx)
    res = backtest.run_backtest(weights, fwd, cost_bps=0.0, borrow_bps_annual=2520.0)
    daily_borrow = 2520.0 / 1e4 / 252  # 0.001 on a gross short of 1
    assert res.gross.tolist() == [0.0, 0.0]
    assert res.net.tolist() == [-daily_borrow, -daily_borrow]
