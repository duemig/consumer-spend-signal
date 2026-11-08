import numpy as np
import pandas as pd

from spendsignal import panel


def _toy_long() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    rows = []
    for d in dates:
        for stock, size in (("A", 1_000.0), ("B", 1_000_000.0)):
            for cohort in ("C0", "C1", "C2"):
                rows.append((d, cohort, stock, size * np.exp(rng.normal(0, 0.1))))
    df = pd.DataFrame(rows, columns=["date", "cohort", "stock", "spend"]).astype({"spend": object})
    return df


def test_to_cube_coerces_strings():
    df = _toy_long()
    df.loc[5, "spend"] = "n/a"
    cube, non_numeric = panel.to_cube(df)
    assert non_numeric == 1
    assert cube.values.shape == (200, 2, 3)
    assert np.isnan(cube.values).sum() == 1


def test_filter_removes_spikes_and_refunds():
    df = _toy_long()
    spike_row = df.index[(df["stock"] == "A") & (df["cohort"] == "C1")][150]
    refund_row = df.index[(df["stock"] == "B") & (df["cohort"] == "C2")][120]
    df.loc[spike_row, "spend"] = float(df.loc[spike_row, "spend"]) * 50
    df.loc[refund_row, "spend"] = -100.0
    cube, _ = panel.to_cube(df)
    logs, non_positive, outliers = panel.log_and_filter(cube)
    assert non_positive == 1
    assert outliers >= 1
    t = cube.dates.get_loc(df.loc[spike_row, "date"])
    assert np.isnan(logs.values[t, 0, 1])


def test_pooled_winsorisation_caps_the_large_stock():
    cube, _ = panel.to_cube(_toy_long())
    share = panel.clipped_share_pooled(cube, upper=0.95)
    assert share["B"] > share["A"]
    capped = panel.winsorize_pooled(cube)
    assert np.nanmax(capped.values[:, 1, :]) < np.nanmax(cube.values[:, 1, :])


def test_cohort_mean_is_independent_of_active_count():
    df = _toy_long()
    df = df[~((df["cohort"] == "C2") & (df["date"] < "2020-03-01"))]  # C2 joins late
    cube, _ = panel.to_cube(df)
    logs = panel.log_only(cube)
    total = panel.total_log_spend(logs)
    mean = panel.cohort_mean(logs)
    jump_total = total.loc["2020-03-01":].mean() - total.loc[:"2020-02-29"].mean()
    jump_mean = mean.loc["2020-03-01":].mean() - mean.loc[:"2020-02-29"].mean()
    assert (jump_total > 0.3).all()  # about log(3/2)
    assert (jump_mean.abs() < 0.05).all()
