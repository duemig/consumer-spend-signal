"""End-to-end research pipeline with switchable components for ablations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import pandas as pd

from spendsignal import backtest, evaluation, forecast, panel, seasonality
from spendsignal.simulate import SimulatedData


@dataclass(frozen=True)
class PipelineConfig:
    """Choices at each stage.

    The defaults are the final pipeline. They were fixed on a development
    simulation (seed 7) before the evaluation simulations were run.
    """

    cleaning: Literal["robust", "pooled_winsor"] = "robust"
    normalisation: Literal["mean", "adjusted"] = "mean"
    seasonality: Literal["bucket", "yoy", "bucket_stock"] = "bucket"
    model: Literal["ridge", "smooth", "tree"] = "ridge"
    features: Literal["windows", "lags"] = "windows"
    sector_selection: Literal["train", "none", "test"] = "train"
    portfolio: Literal["long_short", "long_only", "neutral"] = "long_short"
    publication_lag: int = 0
    forecast_halflife: float = 0.0
    smooth_window: int = 50
    lags: tuple[int, ...] = (0, 5, 10, 15, 20, 30, 40, 50, 60)
    windows: tuple[int, ...] = (5, 10, 20, 40, 60, 120)
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
    first_test_year: int = 2019
    train_start: str = "2017-03-01"
    cost_bps: float = 5.0
    borrow_bps_annual: float = 0.0
    rebalance: Literal["full", "band", "partial"] = "full"
    no_trade_band: float = 0.0
    rebalance_speed: float = 1.0
    beta_window: int = 252


#: Closest equivalent of the original study within this framework: pooled
#: winsorisation, an untuned ridge on lagged values, sectors chosen on the test
#: period and a long-only top-half portfolio.
ORIGINAL = PipelineConfig(
    cleaning="pooled_winsor",
    normalisation="mean",
    seasonality="bucket",
    model="ridge",
    features="lags",
    ridge_alphas=(1.0,),
    sector_selection="test",
    portfolio="long_only",
)


@dataclass
class PipelineResult:
    config: PipelineConfig
    predictor: pd.DataFrame
    prediction: pd.DataFrame
    fwd_returns: pd.DataFrame
    ic: pd.Series
    ic_stats: dict[str, float]
    sectors: dict[int, list[int]]
    backtest: backtest.BacktestResult
    perf_gross: dict[str, float]
    perf_net: dict[str, float]
    coefficients: pd.DataFrame | None = None
    alphas: pd.Series | None = None
    cleaning: dict[str, int] = field(default_factory=dict)


def returns_wide(data: SimulatedData) -> pd.DataFrame:
    """Backward returns as a (business days x stocks) frame."""
    return data.returns.pivot(index="date", columns="stock", values="ret")


def build_predictor(data: SimulatedData, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    """Daily spend predictor on trading days, plus a summary of the cleaning step."""
    cube, non_numeric = panel.to_cube(data.spend)
    report = {"rows": int(len(data.spend)), "non_numeric": non_numeric}
    if cfg.cleaning == "robust":
        logs, non_positive, outliers = panel.log_and_filter(cube)
        report.update(non_positive=non_positive, outliers=outliers)
    else:
        logs = panel.log_only(panel.winsorize_pooled(cube))
        report.update(non_positive=0, outliers=0)

    level = panel.cohort_mean_adjusted(logs) if cfg.normalisation == "adjusted" else panel.cohort_mean(logs)
    level = level.asfreq("D")

    if cfg.seasonality == "yoy":
        adjusted = seasonality.yoy_growth(level)
    elif cfg.seasonality == "bucket":
        sector = data.metadata.set_index("stock")["sector"]
        adjusted = seasonality.bucket_adjust(level, sector)
    else:
        own = pd.Series(level.columns, index=level.columns)
        adjusted = seasonality.bucket_adjust(level, own)

    predictor = seasonality.to_business_days(adjusted)
    if cfg.publication_lag:
        predictor = predictor.shift(cfg.publication_lag)
    return predictor, report


def run(data: SimulatedData, cfg: PipelineConfig | None = None) -> PipelineResult:
    """Run one pipeline variant and evaluate it out of sample."""
    cfg = cfg or PipelineConfig()
    predictor, report = build_predictor(data, cfg)
    fwd = evaluation.forward_returns(returns_wide(data)).reindex(predictor.index)
    sector = data.metadata.set_index("stock")["sector"]
    test_start = pd.Timestamp(year=cfg.first_test_year, month=1, day=1)

    coefficients = alphas = None
    in_sample: dict[int, pd.DataFrame] = {}
    if cfg.model in ("ridge", "tree"):
        spans = cfg.windows if cfg.features == "windows" else cfg.lags
        if cfg.model == "ridge":
            wf = forecast.walk_forward_ridge(
                predictor,
                fwd,
                spans=spans,
                features=cfg.features,
                first_test_year=cfg.first_test_year,
                first_train_date=cfg.train_start,
                alphas=cfg.ridge_alphas,
            )
        else:
            wf = forecast.walk_forward_tree(
                predictor,
                fwd,
                spans=spans,
                features=cfg.features,
                first_test_year=cfg.first_test_year,
                first_train_date=cfg.train_start,
            )
        prediction = wf.predictions
        coefficients, alphas, in_sample = wf.coefficients, wf.alphas, wf.in_sample
    else:
        smooth = forecast.smoothed_signal(predictor, cfg.smooth_window)
        prediction = smooth.loc[smooth.index >= test_start]
        for year in sorted(set(prediction.index.year)):
            train = smooth.loc[(smooth.index >= cfg.train_start) & (smooth.index.year < year)]
            in_sample[year] = train

    prediction = prediction.loc[prediction.index >= test_start].dropna(how="all")
    if cfg.forecast_halflife > 0:
        prediction = prediction.ewm(halflife=cfg.forecast_halflife, min_periods=1).mean()
    fwd_test = fwd.loc[prediction.index]

    sectors: dict[int, list[int]] = {}
    universe = pd.DataFrame(True, index=prediction.index, columns=prediction.columns)
    all_codes = sorted(sector.unique().tolist())
    for year in sorted(set(prediction.index.year)):
        rows = prediction.index.year == year
        if cfg.sector_selection == "train":
            fitted = in_sample[year]
            chosen = forecast.select_sectors(fitted, fwd.loc[fitted.index], sector)
        elif cfg.sector_selection == "test":
            chosen = forecast.select_sectors(prediction, fwd_test, sector)
        else:
            chosen = all_codes
        chosen = chosen or all_codes  # nothing passes the test: trade the full universe
        sectors[year] = chosen
        keep = sector.reindex(prediction.columns).isin(chosen).to_numpy()
        universe.loc[rows] = keep

    pred_universe = prediction.where(universe)
    ic = evaluation.daily_rank_ic(pred_universe, fwd_test)
    if cfg.portfolio == "neutral":
        beta = backtest.estimate_beta(returns_wide(data), window=cfg.beta_window)
        targets = backtest.neutral_weights(prediction, beta, sector, universe=universe)
    elif cfg.portfolio == "long_only":
        targets = backtest.long_only_weights(prediction, universe=universe)
    else:
        targets = backtest.long_short_weights(prediction, universe=universe)
    weights = backtest.apply_rebalance_rule(targets, cfg.rebalance, cfg.no_trade_band, cfg.rebalance_speed)
    bt = backtest.run_backtest(
        weights, fwd_test, cost_bps=cfg.cost_bps, borrow_bps_annual=cfg.borrow_bps_annual
    )

    return PipelineResult(
        config=cfg,
        predictor=predictor,
        prediction=prediction,
        fwd_returns=fwd_test,
        ic=ic,
        ic_stats=evaluation.ic_summary(ic),
        sectors=sectors,
        backtest=bt,
        perf_gross=backtest.performance(bt.gross),
        perf_net=backtest.performance(bt.net),
        coefficients=coefficients,
        alphas=alphas,
        cleaning=report,
    )


def oracle_ic(data: SimulatedData, first_test_year: int = 2019, card_only: bool = True) -> dict[str, float]:
    """IC of the true expected-return driver.

    With ``card_only`` the oracle knows only the consumer part of revenue news,
    the most a card panel can reveal; otherwise it knows all revenue news.
    """
    info = data.truth["info_card" if card_only else "info_total"]
    fwd = evaluation.forward_returns(returns_wide(data)).reindex(info.index)
    rows = info.index.year >= first_test_year
    return evaluation.ic_summary(evaluation.daily_rank_ic(info.loc[rows], fwd.loc[rows]))


__all__ = ["ORIGINAL", "PipelineConfig", "PipelineResult", "build_predictor", "oracle_ic", "replace", "run"]
