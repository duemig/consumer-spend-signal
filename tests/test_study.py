import numpy as np

from spendsignal import study
from spendsignal.evaluation import bootstrap_mean_ci
from spendsignal.pipeline import PipelineConfig


def test_bootstrap_mean_ci_brackets_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    sample = rng.normal(1.0, 0.5, size=200)
    ci = bootstrap_mean_ci(sample, seed=0)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert abs(ci["mean"] - sample.mean()) < 1e-12
    assert bootstrap_mean_ci(sample, seed=0) == bootstrap_mean_ci(sample, seed=0)
    assert np.isnan(bootstrap_mean_ci([])["mean"])


def _record(seed, final, other):
    def metrics(mean_ic, net_sharpe):
        return {name: 0.0 for name in study.METRICS} | {"mean_ic": mean_ic, "net_sharpe": net_sharpe}

    return {
        "seed": seed,
        "oracle_ic": 0.02,
        "oracle_full_ic": 0.04,
        "variants": {"Final pipeline": metrics(*final), "Other": metrics(*other)},
    }


def test_aggregate_distribution_and_paired_difference():
    variants = {"Final pipeline": PipelineConfig(), "Other": PipelineConfig()}
    records = [_record(1, (0.01, 1.0), (0.02, 0.5)), _record(2, (0.03, 2.0), (0.02, 1.5))]
    agg = study.aggregate(records, variants)
    distribution, paired = agg["distribution"], agg["paired"]
    assert abs(distribution.loc["Final pipeline", "ic_mean"] - 0.02) < 1e-12
    assert abs(distribution.loc["Other", "sharpe_mean"] - 1.0) < 1e-12
    # Other minus Final: d_sharpe = [-0.5, -0.5], d_ic = [0.01, -0.01]
    assert abs(paired.loc["Other", "d_sharpe"] - (-0.5)) < 1e-12
    assert abs(paired.loc["Other", "d_ic"]) < 1e-12
    assert paired.loc["Other", "sharpe_better_frac"] == 0.0
    assert agg["summary"]["n_seeds"] == 2


def test_evaluate_seed_skips_simulation_when_cached(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("evaluate_seed should not simulate when every variant is cached")

    monkeypatch.setattr(study, "simulate", fail)
    variants = {"Final pipeline": PipelineConfig()}
    previous = {
        "seed": 5,
        "oracle_ic": 0.02,
        "oracle_full_ic": 0.04,
        "variants": {"Final pipeline": {name: 0.0 for name in study.METRICS}},
    }
    assert study.evaluate_seed(5, variants, previous) is previous


def _sweep_record(param, level, seed, final_ic, oracle):
    def metrics(mean_ic):
        return {name: 0.0 for name in study.METRICS} | {"mean_ic": mean_ic}

    return {
        "param": param,
        "level": level,
        "seed": seed,
        "oracle_ic": oracle,
        "variants": {
            name: metrics(final_ic if name == "Final pipeline" else 0.0) for name in study.SWEEP_VARIANTS
        },
    }


def test_aggregate_sensitivity_groups_and_recovers():
    records = [
        _sweep_record("Signal strength", "weak", 3001, 0.005, 0.02),
        _sweep_record("Signal strength", "weak", 3002, 0.015, 0.04),
    ]
    table = study.aggregate_sensitivity(records)
    assert ("Signal strength", "weak") in table.index
    assert abs(table.loc[("Signal strength", "weak"), "final_ic"] - 0.010) < 1e-12
    # recovery is the mean of per-seed final/oracle = mean(0.25, 0.375) = 0.3125
    assert abs(table.loc[("Signal strength", "weak"), "recovery"] - 0.3125) < 1e-12
