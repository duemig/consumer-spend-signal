"""Simulator for a daily credit card spending panel and matching stock returns.

The goal is data that behaves like a real card panel, with a data-generating
process that is fully known, so the research pipeline can be judged by how
well it recovers the signal.

What the panel measures
-----------------------
A card panel sees what its card holders spend at a company's stores and
websites. That is the company's *consumer card revenue*, which is most of the
revenue of a retailer or a restaurant chain and almost none of the revenue of a
bank or an industrial supplier. Each company therefore has a card share
``e[i]`` (0.02 to 1). Its total revenue surprise mixes a consumer part, which
the panel can see, and an other part, which it cannot:

    x_total[i, t] = e[i] * x_consumer[i, t] + (1 - e[i]) * x_other[i, t]

Both parts are persistent AR(1) processes; the consumer part has a component
common to all consumer-facing companies.

Panel spend
-----------
Daily panel spend of cohort ``k`` at company ``i`` is consumer revenue times
the cohort's share of all card holders:

    log spend[i, k, t] = rev[i] + log e[i] + g[i] * years + season_s(t) + covid_s(t)
                         + x_consumer[i, t] + cohort[k] + noise

* ``season_s`` depends on the sector: retail and restaurants peak on Fridays
  and Saturdays and between Black Friday and Christmas, groceries peak on
  Saturdays, fuel is flat.
* ``covid_s`` is the 2020 shock: discretionary and fuel spending collapse from
  mid-March and recover over the following year, groceries spike with
  stockpiling.
* The noise of a daily total falls with the number of transactions behind it,
  so small merchants and small cohorts are noisier. Net spend can be negative
  when refunds exceed purchases, which mostly happens when few transactions
  are involved.
* Cohorts join and leave the panel at different dates. Rows for inactive
  cohorts are absent, and a few rows are missing at random.
* Files contain recording errors: decimal shifts (x10), duplicated batches
  (x2) and corrupt entries.

Returns
-------
Investors learn about revenue gradually: their expectation ``m`` is an
exponentially weighted average of ``x_total``. The unpriced part drives the
next-day expected return:

    r[i, t -> t+1] = beta[i] * f_mkt + h_sector + alpha_scale * z(x_total - m)[i, t] + u

The market factor includes the February-March 2020 crash and the recovery
that followed. Because the panel only sees the consumer part of revenue news,
it can only forecast returns of companies with a high card share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: GICS sector code -> (name, number of stocks, typical card share of revenue)
DEFAULT_SECTORS: dict[int, tuple[str, int, float]] = {
    25: ("Consumer Discretionary", 45, 0.8),
    30: ("Consumer Staples", 30, 0.7),
    50: ("Communication Services", 15, 0.4),
    10: ("Energy", 10, 0.3),
    45: ("Information Technology", 15, 0.25),
    35: ("Health Care", 10, 0.1),
    20: ("Industrials", 15, 0.05),
    40: ("Financials", 10, 0.03),
}

#: Day-of-week profile of log spend (Mon..Sun) and holiday-season amplitude, by sector
SECTOR_SEASONALITY: dict[int, tuple[tuple[float, ...], float]] = {
    25: ((0.0, 0.0, 0.0, 0.02, 0.12, 0.30, 0.15), 1.0),  # retail, restaurants, travel
    30: ((0.0, -0.02, 0.0, 0.02, 0.08, 0.20, 0.05), 0.3),  # groceries, household
    50: ((0.0, 0.0, 0.0, 0.0, 0.03, 0.08, 0.06), 0.4),  # subscriptions, entertainment
    10: ((0.0, 0.0, 0.0, 0.0, 0.04, 0.02, -0.05), 0.05),  # fuel
    45: ((0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05), 0.6),  # devices, software, online
    35: ((0.05, 0.03, 0.02, 0.0, -0.02, -0.30, -0.40), 0.0),  # pharmacies, clinics
    20: ((0.05, 0.05, 0.05, 0.05, 0.0, -0.50, -0.60), 0.0),  # mostly business purchases
    40: ((0.0, 0.0, 0.0, 0.0, 0.0, -0.20, -0.30), 0.0),  # fees, premiums
}

#: COVID-19 shock to log consumer spending: (date, level) knots per sector, linear in between
COVID_PATHS: dict[int, list[tuple[str, float]]] = {
    25: [("2020-03-08", 0.0), ("2020-04-12", -0.45), ("2020-07-31", -0.15), ("2021-06-30", 0.0)],
    30: [("2020-03-08", 0.0), ("2020-03-18", 0.30), ("2020-04-15", 0.08), ("2021-06-30", 0.0)],
    10: [("2020-03-08", 0.0), ("2020-04-12", -0.40), ("2020-09-30", -0.10), ("2021-06-30", 0.0)],
    50: [("2020-03-08", 0.0), ("2020-04-12", 0.08), ("2021-06-30", 0.0)],
    45: [("2020-03-08", 0.0), ("2020-04-30", 0.10), ("2021-06-30", 0.0)],
}

CORRUPT_TOKENS = ("n/a", "#VALUE!", "ERR")
AVERAGE_TICKET = 45.0  # dollars per card transaction


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of the simulated panel. Defaults are used for the published results."""

    n_cohorts: int = 12
    start: str = "2016-01-01"
    end: str = "2023-12-31"
    seed: int = 7
    sectors: dict[int, tuple[str, int, float]] = field(default_factory=lambda: dict(DEFAULT_SECTORS))
    # Revenue surprises (calendar-day AR(1))
    surprise_phi: float = 0.995
    surprise_vol: float = 0.02
    consumer_macro_share: float = 0.3
    # Investor learning
    learning_halflife_days: float = 20.0
    # Panel
    cohort_share: float = 5e-4  # average share of all card holders in one cohort
    panel_noise: float = 0.10  # noise unrelated to transaction counts (card holder churn etc.)
    missing_prob: float = 0.01
    decimal_shift_prob: float = 0.0003
    duplicate_batch_prob: float = 0.0005
    corrupt_prob: float = 0.001
    covid: bool = True
    # Returns (daily)
    market_vol: float = 0.010
    sector_vol: float = 0.006
    idio_vol: float = 0.015
    alpha_scale: float = 0.0006

    @property
    def n_stocks(self) -> int:
        return sum(n for _, n, _ in self.sectors.values())


@dataclass
class SimulatedData:
    """Output of :func:`simulate`. ``truth`` holds latent quantities for diagnostics only."""

    spend: pd.DataFrame  # long: date, cohort, stock, spend (object dtype: may contain strings)
    returns: pd.DataFrame  # long: date, stock, ret  (ret is the backward return r(t-1, t))
    metadata: pd.DataFrame  # stock, sector, sector_name, card_share, revenue
    truth: dict[str, pd.DataFrame]


def _holiday_season(dates: pd.DatetimeIndex) -> np.ndarray:
    """Year-end pattern in log spend for a retailer, by calendar date."""
    out = np.zeros(len(dates))
    month = dates.month.to_numpy()
    day = dates.day.to_numpy()
    for year in np.unique(dates.year):
        nov1 = pd.Timestamp(year=year, month=11, day=1)
        thanksgiving = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7 + 21)
        black_friday = thanksgiving + pd.Timedelta(days=1)
        peak = (dates >= black_friday) & (dates <= pd.Timestamp(year=year, month=12, day=24))
        out[np.asarray(peak)] += 0.35
        out[np.asarray(dates == thanksgiving)] -= 0.30
    out[month == 11] += 0.10
    out[(month == 12) & (day == 25)] -= 0.60
    out[month == 1] -= 0.10
    return out


def _covid_path(dates: pd.DatetimeIndex, knots: list[tuple[str, float]]) -> np.ndarray:
    x = pd.DatetimeIndex([d for d, _ in knots]).asi8
    y = np.array([v for _, v in knots])
    out = np.interp(dates.asi8, x, y)
    out[dates.asi8 < x[0]] = 0.0
    out[dates.asi8 > x[-1]] = 0.0
    return out


def _ar1(rng: np.random.Generator, n_steps: int, n_series: int, phi: float, vol: float) -> np.ndarray:
    shocks = rng.normal(0.0, vol, size=(n_steps, n_series))
    out = np.empty_like(shocks)
    out[0] = shocks[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n_steps):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def _ewma(x: np.ndarray, halflife: float) -> np.ndarray:
    lam = 1.0 - 0.5 ** (1.0 / halflife)
    m = np.empty_like(x)
    m[0] = x[0]
    for t in range(1, len(x)):
        m[t] = m[t - 1] + lam * (x[t] - m[t - 1])
    return m


def _cohort_activity(rng: np.random.Generator, n_days: int, n_cohorts: int) -> np.ndarray:
    """Boolean (days x cohorts) activity mask with staggered joins and drop-outs."""
    active = np.zeros((n_days, n_cohorts), dtype=bool)
    n_initial = max(3, n_cohorts // 2)
    for k in range(n_cohorts):
        start = 0 if k < n_initial else int(rng.integers(n_days // 10, int(n_days * 0.7)))
        end = n_days
        if k < n_initial and rng.random() < 0.4:
            end = int(rng.integers(int(n_days * 0.5), n_days))
        active[start:end, k] = True
    active[active.sum(axis=1) < 3, :3] = True
    return active


def _market_factor(rng: np.random.Generator, bdays: pd.DatetimeIndex, vol: float, covid: bool) -> np.ndarray:
    f = rng.normal(0.0003, vol, size=len(bdays))
    if covid:
        crash = (bdays >= "2020-02-20") & (bdays <= "2020-03-23")
        rebound = (bdays > "2020-03-23") & (bdays <= "2020-08-31")
        stress = (bdays >= "2020-02-20") & (bdays <= "2020-05-31")
        f[stress] = rng.normal(0.0, 3 * vol, size=stress.sum())
        f[crash] += np.log(0.66) / crash.sum()  # about -34% peak to trough
        f[rebound] += np.log(1.50) / rebound.sum()
    return f


def simulate(config: SimulationConfig | None = None) -> SimulatedData:
    """Simulate a credit card spending panel and daily stock returns."""
    cfg = config or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)

    days = pd.date_range(cfg.start, cfg.end, freq="D")
    n_days, n_cohorts = len(days), cfg.n_cohorts

    # --- Companies ------------------------------------------------------------------
    codes, names, shares = [], [], []
    for code, (name, count, share) in cfg.sectors.items():
        codes += [code] * count
        names += [name] * count
        shares.append(share * rng.uniform(0.6, 1.25, size=count))
    n_stocks = len(codes)
    codes_arr = np.asarray(codes)
    card_share = np.clip(np.concatenate(shares), 0.02, 1.0)
    stocks = pd.Index([f"S{i:03d}" for i in range(n_stocks)], name="stock")
    revenue = rng.normal(16.5, 1.3, size=n_stocks)  # log total daily revenue in dollars
    growth = rng.normal(0.03, 0.05, size=n_stocks)

    # --- Revenue surprises and investor learning --------------------------------------------
    consumer_facing = card_share > 0.2
    idio = _ar1(rng, n_days, n_stocks, cfg.surprise_phi, cfg.surprise_vol)
    macro = _ar1(rng, n_days, 1, cfg.surprise_phi, cfg.surprise_vol)
    w = cfg.consumer_macro_share
    x_consumer = np.sqrt(1 - w) * idio + np.sqrt(w) * macro * consumer_facing
    x_other = _ar1(rng, n_days, n_stocks, cfg.surprise_phi, cfg.surprise_vol)
    x_total = card_share * x_consumer + (1 - card_share) * x_other
    info_total = x_total - _ewma(x_total, cfg.learning_halflife_days)
    info_card = card_share * (x_consumer - _ewma(x_consumer, cfg.learning_halflife_days))

    # --- Expected panel spend -----------------------------------------------------------
    years = ((days - days[0]).days / 365.25).to_numpy()
    holiday = _holiday_season(days)
    dow = days.dayofweek.to_numpy()
    season = np.zeros((n_days, n_stocks))
    covid = np.zeros((n_days, n_stocks))
    for code in np.unique(codes_arr):
        cols = codes_arr == code
        profile, amp = SECTOR_SEASONALITY.get(int(code), ((0.0,) * 7, 0.0))
        season[:, cols] = (np.asarray(profile)[dow] + amp * holiday)[:, None]
        if cfg.covid and int(code) in COVID_PATHS:
            covid[:, cols] = _covid_path(days, COVID_PATHS[int(code)])[:, None]
    log_consumer_rev = revenue + np.log(card_share) + np.outer(years, growth) + season + covid + x_consumer

    active = _cohort_activity(rng, n_days, n_cohorts)
    cohort_offset = np.log(cfg.cohort_share) + rng.normal(0.0, 0.4, size=n_cohorts)
    expected = log_consumer_rev[:, :, None] + cohort_offset[None, None, :]

    # Noise of a daily total: falls with the number of transactions behind it
    n_tx = np.exp(expected) / AVERAGE_TICKET
    sd = np.sqrt(cfg.panel_noise**2 + np.log1p(1.5**2 / np.maximum(n_tx, 0.05)))
    spend = np.exp(expected + rng.normal(0.0, 1.0, size=expected.shape) * sd)

    # Net spend turns negative when refunds exceed purchases (mostly with few transactions)
    p_negative = 0.0003 + 0.15 * np.exp(-n_tx / 3.0)
    negative = rng.random(spend.shape) < p_negative
    spend[negative] = -rng.uniform(0.1, 1.0, size=negative.sum()) * np.exp(expected[negative])

    # Recording errors
    shift = rng.random(spend.shape) < cfg.decimal_shift_prob
    spend[shift] *= 10.0
    duplicate = rng.random(spend.shape) < cfg.duplicate_batch_prob
    spend[duplicate] *= 2.0
    present = active[:, None, :] & (rng.random(spend.shape) >= cfg.missing_prob)

    t_idx, i_idx, k_idx = np.nonzero(present)
    cohorts = [f"C{k:02d}" for k in range(n_cohorts)]
    values = np.round(spend[t_idx, i_idx, k_idx], 2).astype(object)
    corrupt = np.flatnonzero(rng.random(len(values)) < cfg.corrupt_prob)
    values[corrupt] = rng.choice(np.asarray(CORRUPT_TOKENS, dtype=object), size=len(corrupt))
    spend_long = pd.DataFrame(
        {
            "date": days[t_idx],
            "cohort": np.asarray(cohorts)[k_idx],
            "stock": stocks.to_numpy()[i_idx],
            "spend": values,
        }
    )

    # --- Returns (business days) ---------------------------------------------------------
    bdays = pd.bdate_range(cfg.start, cfg.end)
    pos = days.get_indexer(bdays)
    info_b = info_total[pos]
    z = (info_b - info_b.mean()) / info_b.std()
    alpha = cfg.alpha_scale * z  # expected r(t, t+1), known at t

    n_b = len(bdays)
    sector_list = list(cfg.sectors)
    sector_pos = np.array([sector_list.index(c) for c in codes])
    beta = rng.normal(1.0, 0.25, size=n_stocks)
    f_mkt = _market_factor(rng, bdays, cfg.market_vol, cfg.covid)
    f_sec = rng.normal(0.0, cfg.sector_vol, size=(n_b, len(sector_list)))
    eps = rng.standard_t(df=5, size=(n_b, n_stocks)) * cfg.idio_vol / np.sqrt(5 / 3)
    ret = np.full((n_b, n_stocks), np.nan)
    ret[1:] = beta * f_mkt[1:, None] + f_sec[1:, sector_pos] + alpha[:-1] + eps[1:]
    ret[1:] = np.expm1(ret[1:])  # simple returns
    returns_long = (
        pd.DataFrame(ret, index=bdays, columns=stocks)
        .iloc[1:]
        .rename_axis(index="date")
        .stack()
        .rename("ret")
        .reset_index()
    )

    metadata = pd.DataFrame(
        {
            "stock": stocks,
            "sector": codes,
            "sector_name": names,
            "card_share": card_share,
            "revenue": revenue,
        }
    )
    truth = {
        "info_total": pd.DataFrame(info_b, index=bdays, columns=stocks),
        "info_card": pd.DataFrame(info_card[pos], index=bdays, columns=stocks),
        "cohort_active": pd.DataFrame(active, index=days, columns=cohorts),
    }
    return SimulatedData(spend=spend_long, returns=returns_long, metadata=metadata, truth=truth)
