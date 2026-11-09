"""Build docs/report.pdf from the package and the files in results/.

Run scripts/run_study.py first. The report is HTML printed to an A3 PDF with Playwright and Chromium,
in the layout of the original 2024 write-up. All numbers come from results/ or are computed here on the
illustrative evaluation seed named in summary.json.
"""

import base64
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

from spendsignal import backtest, evaluation, forecast, panel, study  # noqa: E402
from spendsignal.pipeline import PipelineConfig, build_predictor, returns_wide, run  # noqa: E402
from spendsignal.simulate import SimulationConfig, simulate  # noqa: E402

plt.rcdefaults()
RESULTS, FIGURES = REPO / "results", REPO / "docs" / "figures"
summary = json.loads((RESULTS / "summary.json").read_text())
dist = pd.read_csv(RESULTS / "ablation_distribution.csv", index_col=0)
paired = pd.read_csv(RESULTS / "paired_diffs.csv", index_col=0)
sens = pd.read_csv(RESULTS / "sensitivity.csv", index_col=[0, 1])

ill_seed = summary["illustrative_seed"]
data = simulate(SimulationConfig(seed=ill_seed))
meta = data.metadata.set_index("stock")
cfg = PipelineConfig()
res = run(data, cfg)


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


figs = {}

# Summary statistics table (time-averaged cross-sectional statistics of raw spend by cohort)
raw = data.spend.assign(spend=pd.to_numeric(data.spend["spend"], errors="coerce"))
wide = raw.pivot_table(index=["date", "stock"], columns="cohort", values="spend", aggfunc="last")
qs = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]


def cs_stats(frame):
    d = frame.quantile(qs)
    d.index = [f"{int(q * 100)} %" for q in qs]
    extra = pd.DataFrame(
        {"count": frame.count(), "ex_kurt": frame.kurt(), "max": frame.max(), "mean": frame.mean(),
         "min": frame.min(), "skew": frame.skew(), "std": frame.std()}
    ).T
    return pd.concat([d, extra])


stats = wide.groupby(level="date").apply(cs_stats)
stats.index.names = ["date", "stat"]
order = [f"{int(q * 100)} %" for q in qs] + ["count", "ex_kurt", "max", "mean", "min", "skew", "std"]
avg_stats = stats.groupby(level="stat").mean().reindex(order)

# Fig 1: boxplot of cohorts (raw values)
fig, ax = plt.subplots(figsize=(10, 6))
wide.plot.box(vert=False, ax=ax)
figs["box"] = b64(fig)

# Fig 2: log total spend vs active cohorts
cube, _ = panel.to_cube(data.spend)
logs, _, _ = panel.log_and_filter(cube)
active = panel.active_cohorts(logs).stack().rename("Active Cohorts")
tot_w = panel.total_log_spend(logs)
total = (tot_w - tot_w.mean()).stack().rename("Log Total Spent (deviation from stock mean)")
sc = pd.concat([active, total], axis=1).dropna().sample(6000, random_state=0)
fig, ax = plt.subplots(figsize=(10, 6))
ycol = "Log Total Spent (deviation from stock mean)"
sns.scatterplot(data=sc, x="Active Cohorts", y=ycol, ax=ax, s=12)
sns.regplot(data=sc, x="Active Cohorts", y=ycol, scatter=False, color="red", ax=ax)
r, p = pearsonr(sc["Active Cohorts"], sc[ycol])
ax.annotate(f"Correlation: {r:.2f}\nP-value: {p:.4f}", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
figs["scatter"] = b64(fig)
corr_total = r
norm = panel.cohort_mean(logs)
an = pd.concat([active, (norm - norm.mean()).stack().rename("n")], axis=1).dropna()
corr_norm = an.corr().iloc[0, 1]

# Fig 3 and 4: seasonality with highlights
ts = norm.mean(axis=1)


def highlight_plot(series, crit, weekday):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(series.index, series, color="blue")
    labels = set()
    for day in series.index:
        if weekday and day.dayofweek in crit:
            ax.axvspan(day, day + pd.DateOffset(1), color="lightgrey", alpha=0.5)
            labels.add(day.strftime("%A"))
        if not weekday and day.month in crit and day.day == 1:
            ax.axvspan(day, day + pd.offsets.MonthEnd(0), color="lightgrey", alpha=0.5)
            labels.add(day.strftime("%B"))
    order_l = sorted(labels, key=lambda s: ["November", "December", "Saturday", "Sunday"].index(s))
    ax.legend(handles=[Patch(color="lightgrey", label=lab) for lab in order_l])
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


figs["holiday"] = b64(highlight_plot(ts, [11, 12], False))
figs["weekend"] = b64(highlight_plot(ts.loc["2019-08"], [5, 6], True))

# Fig 5: per-stock correlation of the adjusted predictor with next-day returns
predictor, _ = build_predictor(data, cfg)
fwd = evaluation.forward_returns(returns_wide(data)).reindex(predictor.index)
test = predictor.index.year >= 2019
corr_stock = predictor.loc[test].corrwith(fwd.loc[test])
raw_ic = evaluation.daily_rank_ic(forecast.cross_sectional_zscore(predictor).loc[test], fwd.loc[test]).mean()
smooth_ic = evaluation.daily_rank_ic(forecast.smoothed_signal(predictor, 50).loc[test], fwd.loc[test]).mean()
fig, ax = plt.subplots(figsize=(10, 6))
corr_stock.plot(kind="bar", ax=ax, color="blue")
ax.set_ylabel("Correlation")
ax.set_xlabel("Stock")
ax.tick_params(axis="x", labelrotation=90, labelsize=5)
med = corr_stock.median()
ax.axhline(med, color="red", linestyle="--", label=f"Median Correlation: {med:.2f}")
ax.text(0.5, 0.95, f"Predictive Power (rank IC): {raw_ic:.4f}", ha="center", va="center", transform=ax.transAxes, fontsize=12)
figs["corr"] = b64(fig)

# Fig 6: ridge coefficients by trailing-mean window (average of yearly fits)
coefs = res.coefficients.mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(coefs)), coefs.to_numpy())
ax.set_ylabel("Coefficient Value")
ax.set_title("Ridge Regression Coefficients for Trailing-Mean Features")
ax.set_xticks(range(len(coefs)))
ax.set_xticklabels([f"mean_{w}d" for w in coefs.index], rotation=45)
ax.grid(True)
figs["coef"] = b64(fig)

# Fig 7: predictive power by sector (all-stock forecasts)
wf_all = run(data, PipelineConfig(sector_selection="none"))
sector_ic = evaluation.ic_by_group(wf_all.prediction, wf_all.fwd_returns, meta["sector_name"])["mean_ic"]
order_s = ["Energy", "Industrials", "Consumer Discretionary", "Consumer Staples", "Health Care", "Financials", "Information Technology", "Communication Services"]
fig, ax = plt.subplots(figsize=(10, 6))
sector_ic.reindex(order_s).plot(kind="bar", legend=None, ax=ax)
ax.set_ylabel("Predictive Power (rank IC)")
ax.set_xticklabels(order_s, rotation=45)
figs["sector"] = b64(fig)

# Fig 8: High, Low and High - Low portfolios on the illustrative seed
sector = meta["sector"]
universe = pd.DataFrame(False, index=res.prediction.index, columns=res.prediction.columns)
for year, chosen in res.sectors.items():
    universe.loc[res.prediction.index.year == year] = sector.reindex(res.prediction.columns).isin(chosen).to_numpy()
high = backtest.run_backtest(backtest.long_only_weights(res.prediction, 0.2, universe), res.fwd_returns, 5.0)
low = backtest.run_backtest(backtest.long_only_weights(-res.prediction, 0.2, universe), res.fwd_returns, 5.0)
spread = res.backtest
cum = pd.DataFrame({k: ((1 + v.net).cumprod() - 1) * 100 for k, v in {"High": high, "Low": low, "High - Low": spread}.items()})
fig, ax = plt.subplots(figsize=(10, 6))
cum.plot(ax=ax)
ax.set_ylabel("Cumulative Return (%)")
ax.set_title("Cumulative Returns by Portfolio")
ax.grid(True)
ax.set_xlabel("date")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.legend(title="Portfolio")
figs["pnl"] = b64(fig)
perf = {k: {**backtest.performance(v.net), "n": float((v.weights != 0).sum(axis=1).mean())} for k, v in {"High": high, "Low": low, "High - Low": spread}.items()}

# --- Figures the study already wrote (shared with the README) --------------------------------
clean = summary["cleaning_report"]
oracle_ic = summary["oracle_ic"]["mean"]
ridge_all_ic = evaluation.daily_rank_ic(wf_all.prediction, wf_all.fwd_returns).mean()

n_seeds = summary["evaluation_seeds"]["n"]
first_seed, last_seed = summary["evaluation_seeds"]["first"], summary["evaluation_seeds"]["last"]
final_sharpe, recovery = summary["final_net_sharpe"], summary["final_recovery"]
s_min, s_max = summary["final_sharpe_min"], summary["final_sharpe_max"]
t_min, t_max = summary["final_t_min"], summary["final_t_max"]

ridge_ic = dist.loc["Final pipeline", "ic_mean"]
tree_ic = dist.loc["Gradient-boosted trees instead of ridge", "ic_mean"]
lags_turnover = dist.loc["Lagged values instead of trailing means", "turnover_mean"]
cost10_sharpe = dist.loc["Costs of 10 bp instead of 5 bp", "sharpe_mean"]
final_turnover = dist.loc["Final pipeline", "turnover_mean"]
neutral_sharpe = dist.loc["Beta- and sector-neutral spread", "sharpe_mean"]
neutral_turnover = dist.loc["Beta- and sector-neutral spread", "turnover_mean"]
costaware_sharpe = dist.loc["Cost-aware rebalancing", "sharpe_mean"]
costaware_turnover = dist.loc["Cost-aware rebalancing", "turnover_mean"]
borrow_sharpe = dist.loc["Borrow cost on the short side", "sharpe_mean"]


def phrase(names) -> str:
    names = [name[0].lower() + name[1:] for name in names]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


within_noise = [name for name, row in paired.iterrows() if row["d_sharpe_lo"] <= 0 <= row["d_sharpe_hi"]]


def sens_value(param, level, column):
    return sens.loc[(param, level), column]


base_recovery = sens_value("Baseline", "base", "recovery")
cardshare_recovery = sens_value("Card share", "high (1.4x)", "recovery")
signal_recovery = sens_value("Signal strength", "strong (2x)", "recovery")
lag_recovery = sens_value("Publication lag", "3 days", "recovery")
noise_recovery = sens_value("Panel noise", "high (0.20)", "recovery")
ma_lo, ma_hi = sens["ma_ic"].min(), sens["ma_ic"].max()


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def img(key, cap_num, cap, width=78):
    return (
        f"<figure><img src='data:image/png;base64,{figs[key]}' style='width:{width}%'/>"
        f"<figcaption><b>Fig. {cap_num}</b> {cap}</figcaption></figure>"
    )


def img_file(name, cap_num, cap, width=78):
    return (
        f"<figure><img src='data:image/png;base64,{b64_file(FIGURES / name)}' style='width:{width}%'/>"
        f"<figcaption><b>Fig. {cap_num}</b> {cap}</figcaption></figure>"
    )


def table_html(df, fmt):
    head = "".join(f"<th>{c}</th>" for c in [""] + list(df.columns))
    rows = "".join(
        "<tr><td class='rl'>" + str(i) + "</td>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>" for i, r in df.iterrows()
    )
    return f"<table class='nt'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


stats_html = table_html(avg_stats.rename_axis(None), lambda v: f"{v:.1e}")
perf_rows = "".join(
    f"<tr><td class='rl'>{lab}</td>" + "".join(f"<td>{f(perf[c][k])}</td>" for c in ["High", "Low", "High - Low"]) + "</tr>"
    for lab, k, f in [
        ("Ann. Returns:", "ann_return", lambda v: f"{v:.2f}"),
        ("Ann. Vola:", "ann_vol", lambda v: f"{v:.2f}"),
        ("Sharpe Ratio:", "sharpe", lambda v: f"{v:.2f}"),
        ("Max. Drawdown:", "max_drawdown", lambda v: f"{v:.2f}"),
        ("Ø num. Stocks:", "n", lambda v: f"{v:.0f}"),
    ]
)
perf_html = (
    "<table class='nt small'><thead><tr><th>Metric</th><th>Portfolio (High)</th><th>Portfolio (Low)</th>"
    f"<th>High - Low</th></tr></thead><tbody>{perf_rows}</tbody></table>"
)


def distribution_html():
    rows = "".join(
        f"<tr><td class='rl'>{name}</td><td>{r.ic_mean:.4f}</td><td>{r.sharpe_mean:.2f}</td>"
        f"<td>{r.sharpe_lo:.2f} to {r.sharpe_hi:.2f}</td><td>{r.turnover_mean:.2f}</td></tr>"
        for name, r in dist.iterrows()
    )
    return (
        "<table class='nt small'><thead><tr><th>Variant</th><th>Mean IC</th><th>Net Sharpe</th>"
        f"<th>Net Sharpe 95% CI</th><th>Turnover</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def paired_html():
    rows = "".join(
        f"<tr><td class='rl'>{name}</td><td>{r.d_ic:+.4f}</td><td>{r.d_sharpe:+.2f}</td>"
        f"<td>{r.d_sharpe_lo:+.2f} to {r.d_sharpe_hi:+.2f}</td><td>{r.sharpe_better_frac:.0%}</td></tr>"
        for name, r in paired.iterrows()
    )
    return (
        "<table class='nt small'><thead><tr><th>Variant minus final pipeline</th><th>IC difference</th>"
        f"<th>Net Sharpe difference</th><th>95% CI</th><th>Seeds better</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def sensitivity_html():
    rows = "".join(
        f"<tr><td class='rl'>{param}</td><td>{level}</td><td>{r.oracle_ic:.4f}</td><td>{r.final_ic:.4f}</td>"
        f"<td>{r.recovery:.0%}</td><td>{r.final_net_sharpe:.2f}</td><td>{r.ma_ic:.4f}</td></tr>"
        for (param, level), r in sens.iterrows()
    )
    return (
        "<table class='nt small'><thead><tr><th>Parameter</th><th>Level</th><th>Oracle IC</th><th>Final IC</th>"
        f"<th>Recovery</th><th>Final net Sharpe</th><th>50-day MA IC</th></tr></thead><tbody>{rows}</tbody></table>"
    )


dev_seeds = ", ".join(str(s) for s in study.DEVELOPMENT_SEEDS)

html = f"""<!doctype html><html><head><meta charset='utf-8'><style>
@page {{ size: A3; margin: 28mm 30mm 24mm 30mm; }}
body {{ font-family: 'Inter', sans-serif; color: #37352f; font-size: 15px; line-height: 1.55; }}
h1 {{ font-size: 40px; font-weight: 700; line-height: 1.2; margin: 0 0 6px 0; color: #000; }}
.byline {{ color: #787774; font-size: 14px; margin-bottom: 40px; }}
ol.top {{ padding-left: 26px; margin: 0; }}
ol.top > li {{ margin: 0 0 28px 0; }}
.q {{ text-decoration: underline; text-underline-offset: 3px; }}
ol.sub {{ list-style-type: lower-alpha; padding-left: 26px; margin-top: 12px; }}
ol.sub > li {{ margin-bottom: 18px; }}
p {{ margin: 8px 0 10px 0; }}
figure {{ margin: 22px 0 18px 0; text-align: center; page-break-inside: avoid; }}
figcaption {{ color: #9b9a97; font-size: 12px; margin-top: 6px; }}
figcaption b {{ color: #787774; }}
table.nt {{ border-collapse: collapse; margin: 14px 0; font-size: 12px; }}
table.nt th, table.nt td {{ border: 1px solid #e9e9e7; padding: 5px 8px; text-align: left; }}
table.nt th {{ color: #787774; font-weight: 500; }}
table.nt td.rl {{ text-decoration: underline; text-underline-offset: 2px; }}
table.small {{ font-size: 13px; }}
.tt {{ font-weight: 700; font-size: 13px; margin-top: 16px; }}
.eq {{ text-align: center; font-style: italic; margin: 14px 0; font-size: 16px; }}
h2 {{ font-size: 22px; font-weight: 700; margin: 36px 0 10px 0; color: #000; }}
.pb {{ page-break-before: always; }}
</style></head><body>
<h1>Credit card spending and<br/>stock returns</h1>
<div class='byline'>Data analysis and modelling research · David Dümig · Based on research from February 2024</div>
<ol class='top'>
<li><span class='q'>Why can credit card transactions be predictive of future stock returns?</span>
<p>Credit card transactions can serve as predictive indicators of future stock returns due to their ability to provide insights into consumer behavior and economic activity. For example, heightened economic activity, as reflected in increased credit card spending, may suggest potential stock price upticks for companies reliant on consumer expenditures. This works if investors absorb revenue news gradually: spending that is not yet reflected in prices then predicts the following day's return.</p></li>
<li><span class='q'>Is this data equally predictive of all stocks?</span>
<p>Probably not. Stocks in sectors heavily reliant on consumer spending, such as Consumer Discretionary or Consumer Staples, may show a stronger correlation with credit card transaction data compared to stocks in other sectors. A card panel only sees consumer card revenue, which is most of a retailer's revenue and almost none of a bank's or an industrial supplier's.</p></li>
<li><span class='q'>Exploratory analysis: effects, anomalies and corrections</span>
<p>The data is a simulated panel of daily credit card spending by 12 cohorts of card holders at 150 listed companies from 2016 to 2023 (see appendix). The table and graph provided below illustrate both time-averaged cross-sectional statistics (quantile, count, ex_kurt, max, min, mean, std, skew) and panel statistics (median, Q1, Q3, 1.5 * IQR), with outliers displayed beyond the 1.5 * IQR range.</p>
<div class='tt'>Time-series averages of the cross-sectional summary stats</div>
{stats_html}
{img('box', 1, 'Median, Q1, Q3, and Outliers Represented Beyond 1.5*IQR', 70)}
<p>Both analyses reveal the presence of extreme outliers in the dataset. Some are real-world effects: spending at the largest merchants is orders of magnitude higher than at small ones, and net spend is negative on days when refunds exceed purchases ({clean['non_positive']:,} values), mostly for merchants with few card transactions. Others are artefacts of the data collection process: decimal shifts, duplicated batches and corrupt entries such as 'n/a' or '#VALUE!' ({clean['non_numeric']:,} values). Corrupt entries are replaced with NaN values, and non-positive values are dropped because they cannot be logged. A log transformation is then applied, which fosters normality, stabilizes variance, and alleviates skewness. Remaining recording errors are removed by comparing each merchant and cohort series with the median and median absolute deviation of its own previous 90 days ({clean['outliers']:,} values). Clipping each day at the 5th and 95th percentiles of all merchants together would instead cap the largest merchants rather than their outliers. Furthermore, the returns are shifted, ensuring that a feature on day t is aligned with the return from t to t+1, as the returns are backward-looking. On a final note, in practice, one should confine the exploratory data analysis solely to a well-specified training set to prevent any forward-looking bias from leaking into the investment process.</p></li>
<li>Cohort Normalization
<ol class='sub'>
<li><span class='q'>Total spend per stock and the number of active cohorts</span>
<p>The following graph shows a clear and statistically significant correlation between the logarithm of total spent and the number of active cohorts, validating the hypothesis of a strong relationship.</p>
{img('scatter', 2, 'Log Total Spent vs. Active Cohorts: Demonstrating a Strong Relationship', 70)}</li>
<li><span class='q'>Normalization that corrects for the number of active cohorts</span>
<p>Given that each cohort broadly represents the entire demographic spectrum of the population, and to maintain simplicity, the average across cohorts is computed to mitigate variations in the number of active cohorts. Upon normalization, the correlation between log spend (as deviation from each stock's mean) and active cohorts drops from {corr_total:.2f} to {corr_norm:.2f}. Removing estimated cohort size offsets before averaging changes the forecast very little, because a cohort joining or leaving shifts all merchants by the same amount.</p></li>
</ol></li>
<li>We now explore the empirical relationship between stock returns and total sales.
<ol class='sub'>
<li><span class='q'>Seasonality of the total spend per stock averaged across all stocks</span>
<p>When plotting the normalised time series of log total spending averaged across multiple stocks, a clear seasonality pattern emerges, particularly in November and December. One major factor driving increased credit card spending during these months can probably be related to holiday shopping. November and December encompass major holidays such as Thanksgiving, Black Friday, Cyber Monday, Christmas, and New Year's Eve. During this time, people typically spend more on gifts, decorations, travel, and hosting gatherings. The series also shows the collapse in discretionary spending in spring 2020 and the stockpiling spike in groceries.</p>
{img('holiday', 3, 'Seasonal Spikes in Credit Card Spending During Holiday Months', 70)}
<p>However, upon closer examination of a specific month, such as August 2019, a second distinct seasonality pattern emerges, particularly noticeable on Saturdays and Sundays. This uptick in credit card spending during weekends likely stems from increased leisure activities, such as dining out, entertainment, and retail purchases, as well as heightened social gatherings. The pattern differs by sector: groceries peak on Saturdays, fuel is flat, and business-related spending falls at weekends.</p>
{img('weekend', 4, 'Seasonal Spikes in Credit Card Spending During Weekends', 70)}</li>
<li><span class='q'>Seasonality correction and predictive power of the predictor</span>
<p>To maintain simplicity and avoid any forward-looking bias, the following approach is employed to address seasonality: Initially, the expanding sector average of weekend spending (Saturday and Sunday) is subtracted from the respective weekend spending group. Similarly, the expanding sector average of spending on non-weekend days is deducted from the remaining weekdays. Next, the adjustment is refined by subtracting the expanding sector average from the year-end spending (November and December). For the rest of the year, the expanding sector average is subtracted from the spending outside of November and December. Mathematically, this adjustment is represented by the equation:</p>
<div class='eq'>z<sub>i,t</sub> = x<sub>i,t</sub> − <span style='font-style:normal'>(1 / (t − t<sub>0</sub> + 1))</span> · Σ<sub>s=t<sub>0</sub></sub><sup>t</sup> m<sub>c(i),s</sub></div>
<p>where m<sub>c,t</sub> is the equal weighted sector average of x at time t for sector code c, and c(i) is the sector code of stock i. Calculating the sector averages robustly requires a burn-in period of about 22 months.</p>
<p>To explore the empirical relationship between stock returns ('target') and seasonality-adjusted total sales (the 'predictor'), the frequencies are aligned by resampling total sales to a business daily frequency. Considering the potential significance of weekend sales data, averaging during the resampling procedure is employed; i.e., the first weekday represents the average of Saturday, Sunday, and Monday to ensure that no valuable information is lost and to avoid any forward-looking bias. Predictive power is measured as the mean daily rank correlation between predictor and next-day return across stocks (rank IC). However, even after normalization and seasonality correction, the predictor still appears to be too noisy and lacks any discernible predictive power, as illustrated by the following figure.</p>
{img('corr', 5, 'Time Series Correlation of the Predictor with Next-Day Returns for Individual Stocks, the Median Correlation, and Predictive Power', 78)}</li>
<li><span class='q'>Simple smoothing of the predictor</span>
<p>Smoothing filters out noise, but in this data a 50-day moving average has no predictive power either (rank IC {smooth_ic:.4f}). The reason is the way investors learn: a long average of spending mostly measures what is already priced. What predicts returns is recent spending relative to its longer history, which the model in section 6 captures.</p></li>
</ol></li>
<li><span class='q'>Constrained linear model with lagged values of the predictor</span>
<p>The Ridge Regression uses trailing means of the predictor over 5, 10, 20, 40, 60 and 120 days instead of single lagged values. Single lags give a similar IC but a forecast so noisy that the portfolio turns over {lags_turnover:.1f} times its value per day. Following the recommendation of the first version, the features are standardized before applying the penalty, and the penalty is determined on held-out data: the model is refitted every January on all earlier data, and the penalty is the one that maximises the rank IC in the last training year. The following graph illustrates how the resulting regression weights vary across the features: positive on the short windows and negative on the longer ones, the shape implied by gradual learning.</p>
{img('coef', 6, 'Regression Weights on the Trailing-Mean Features (average of the yearly fits)', 70)}
<p>Out of sample (2019 to 2023), the forecast has a rank IC of {ridge_all_ic:.4f} across all stocks on the illustrative seed, against {oracle_ic:.4f} for a forecast that knew the consumer part of revenue news exactly, which is the most a card panel can reveal. A more flexible model does not help here: a gradient-boosted tree on the same six features, chosen and refitted on the same yearly schedule, reaches a mean rank IC of only {tree_ic:.4f} across the {n_seeds} evaluation seeds, against {ridge_ic:.4f} for the ridge. The extra flexibility finds nothing the linear filter misses, because the simulated signal is linear in the predictor.</p></li>
<li><span class='q'>Predictive power across GICS sectors</span>
<p>In exploring the predictive power of the model across GICS sectors, a clear pattern emerges, aligning with intuitive expectations. Sectors such as Consumer Discretionary and Consumer Staples exhibit higher predictability, reflecting their stronger sensitivity to consumer spending compared to the other sectors.</p>
{img('sector', 7, 'Predictive Power of the Model Across GICS Sectors', 70)}
<p>The decision of whether to remove less predictive sectors depends on the benchmark of the strategy. If the benchmark is represented by an equity index, excluding the less predictive sectors introduces active systematic risk, essentially implying a bet against these sectors. The potential consequences of this active systemic risk were starkly highlighted by recent events such as the COVID-19 pandemic or the conflict in Ukraine and subsequent sector and style rotations. Therefore, in this context, it would probably be more sensible to simply track the less predictable sectors instead of completely underweighting them. However, if the strategy is measured against a cash benchmark, more active bets can be made, provided a diversified portfolio is maintained and absolute risk levels are closely monitored.</p>
<p>If sectors are removed, the choice has to be made with training data only. Here a sector is kept if the in-sample IC in the training data has a t-statistic above 2, which selected Consumer Discretionary and Consumer Staples in every year. Across the evaluation seeds this filter raised the mean IC and the mean net Sharpe ratio relative to trading every sector.</p></li>
<li><span class='q'>Additional data sources that might improve the model</span>
<p>In the context of credit card data, segmenting cohorts could yield significant benefits in understanding the sources of sales. By categorizing customers based on demographic, behavioral, and psychographic attributes, segmentation empowers investors to assess the effectiveness of marketing initiatives, validate business strategies, and optimize investment decisions according to the dynamics of segmented markets. Augmenting this segmentation data with supplementary economic indicators, such as inflation rates, credit card interest rates, consumer sentiment surveys, and payment trends, provides deeper insights into the factors influencing credit card spending patterns. For example, fluctuations in inflation rates may significantly sway consumers' spending inclinations, while shifts in credit card interest rates can profoundly influence borrowing and spending behaviors.</p></li>
<li><span class='q'>Trading strategy, cumulative PnL and performance metrics</span>
<p>Assuming the strategy is measured against cash, a more active stance can be taken, focusing on the sectors selected in the training data. An actionable strategy, albeit simplified and overlooking certain crucial factors like total risk, operational constraints, and regulatory considerations, would involve daily sorting of stocks into two portfolios: High and Low. The High portfolio comprises the fifth of stocks with the highest predictions, while the Low portfolio includes the fifth with the lowest predictions. Subsequently, one can opt to either go long on the High portfolio or adopt a more sophisticated approach by simultaneously going short on the Low portfolio (High - Low), which removes market exposure. Trading costs of 5 bp per unit of turnover are deducted. The following graph shows the cumulative generated PnL over time on the illustrative seed, and the table below provides some performance metrics.</p>
{img('pnl', 8, 'Cumulative Profit and Loss (PnL) Over Time for High, Low and High - Low Portfolios', 70)}
{perf_html}
<p>The High portfolio carries the full market risk, including the 2020 crash, and the Low portfolio also earns a positive return for the same reason. Only the High - Low spread isolates the forecast. A single seed says little about how good the spread is. Across the {n_seeds} evaluation seeds the spread earns a mean net Sharpe ratio of {final_sharpe['mean']:.2f}, with a 95% confidence interval of {final_sharpe['lo']:.2f} to {final_sharpe['hi']:.2f}, and the pipeline recovers on average {recovery['mean']:.0%} of the card oracle's IC. The net Sharpe ratio of a single seed ranges from {s_min:.2f} to {s_max:.2f} and the t-statistic of the IC from {t_min:.1f} to {t_max:.1f}, so one backtest is a weak guide.</p>
{img_file('sharpe_distribution.png', 9, 'Net Sharpe of selected variants across the evaluation seeds, one point per seed, mean marked', 70)}
<p>Measured against the final pipeline on the same seeds, {phrase(within_noise)} cannot be told apart from it, because the 95% interval for their net-Sharpe difference includes zero. Single lags, the plain moving average and the gradient-boosted tree each lose to it by a wide margin, and trading every sector lowers the net Sharpe as well. Choosing sectors on the test period does a little better, but it uses the future and serves only as a benchmark.</p>
{img_file('ablation_differences.png', 10, 'Difference in net Sharpe from the final pipeline, with a 95% confidence interval over seeds', 70)}
<p>Three refinements to the portfolio were added and measured the same way. A beta- and sector-neutral book, which trades the forecast after projecting out estimated market beta and the sector dummies, raises the mean net Sharpe ratio to {neutral_sharpe:.2f} and lowers turnover to {neutral_turnover:.2f}. Partial adjustment towards the target weights each day, moving part of the way rather than trading fully, lowers turnover from {final_turnover:.2f} to {costaware_turnover:.2f} while holding the mean net Sharpe ratio at {costaware_sharpe:.2f}. A borrow fee of 50 basis points a year on the short side lowers it only to {borrow_sharpe:.2f}. The result stays sensitive to trading costs: at 10 basis points per unit of turnover the mean net Sharpe ratio falls to {cost10_sharpe:.2f}.</p>
{img_file('portfolio_variants.png', 11, 'Cumulative net return of the final, beta- and sector-neutral, and cost-aware books on the illustrative seed', 70)}</li>
<li><span class='q'>How far these conclusions depend on the simulator</span>
<p>The study measures a pipeline against a known data-generating process, so the natural question is which conclusions survive a change in that process. Varying the five main simulator parameters one at a time, across ten evaluation seeds each, separates the two. The pipeline recovers {base_recovery:.0%} of the oracle at the baseline. Recovery rises to {cardshare_recovery:.0%} when the card share is 40% higher and to {signal_recovery:.0%} when the signal is twice as strong, and falls to {lag_recovery:.0%} when the data arrives three days late and to {noise_recovery:.0%} at double the panel noise. Through all of this the 50-day moving average keeps a near-zero rank IC (between {ma_lo:.4f} and {ma_hi:.4f}), the single-lag portfolio keeps losing money after costs, and the sector filter keeps helping. What depends on the simulator is the level of skill, not the ranking of the methods.</p>
{img_file('sensitivity.png', 12, 'Oracle, final and moving-average IC as each simulator parameter is varied around its baseline', 92)}</li>
</ol>
<p>In conclusion, while these insights offer valuable perspectives, it's crucial to recognize their limitations within the context of real-world applications. The simulated nature of the data used in this project may not fully capture the complexities and nuances of actual card data and market dynamics, and which methods work depends on the assumed data-generating process. Therefore, while these findings provide valuable insights and avenues for exploration, they should be interpreted with caution and supplemented with empirical validation in real-world scenarios.</p>
<h2 class='pb'>Appendix</h2>
<p><b>Simulated credit card data.</b> Each company has a card share e<sub>i</sub>, the part of its revenue paid by consumers with cards (about 0.7 in Consumer Discretionary, 0.6 in Consumer Staples, 0.03 in Financials). Revenue news mixes a consumer part, which the panel sees, and another part, which it does not: x<sub>i,t</sub> = e<sub>i</sub> x<sup>c</sup><sub>i,t</sub> + (1 − e<sub>i</sub>) x<sup>o</sup><sub>i,t</sub>, both persistent AR(1) processes. Daily spend of cohort k at company i is consumer card revenue times the cohort's share of all card holders, with sector-specific weekday and holiday patterns, the 2020 COVID-19 shock, and noise that falls with the number of transactions behind the daily total. Net spend turns negative when refunds exceed purchases. Files contain decimal shifts, duplicated batches and corrupt entries, and cohorts join and leave the panel at different dates. Investors learn gradually: their expectation m<sub>i,t</sub> is an exponentially weighted average of x<sub>i,t</sub> with a 20-day half-life, and the next-day expected return is proportional to x<sub>i,t</sub> − m<sub>i,t</sub>. The market factor includes the February to March 2020 crash and the recovery that followed.</p>
<p><b>Changes from the first version.</b> This note rebuilds a take-home research exercise completed for Man Group in February 2024. The firm supplied a simulated credit card dataset, which cannot be shared; all data in this note comes from a new simulator written for the rebuild. The original structure and reasoning are kept. Changes: a per-series outlier filter instead of pooled winsorization; trailing-mean features instead of single lags; standardized features and a penalty chosen on held-out data with yearly walk-forward refits; sector selection on training data instead of the test period; a dollar-neutral portfolio with trading costs; and tests that check every step uses only past data. Design choices were made on development simulations (seeds {dev_seeds}); the configuration was then frozen and run once on {n_seeds} evaluation simulations (seeds {first_seed} to {last_seed}).</p>
<div class='tt'>Distribution of skill and net performance across the {n_seeds} evaluation simulations (card oracle IC {oracle_ic:.4f})</div>
{distribution_html()}
<div class='tt'>Paired difference of each variant from the final pipeline, with a 95% confidence interval over seeds</div>
{paired_html()}
<div class='tt'>Dependence on the simulator: each parameter varied one at a time around its baseline</div>
{sensitivity_html()}
</body></html>"""

from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.set_content(html, wait_until="load")
    page.pdf(
        path=str(REPO / "docs" / "report.pdf"),
        format="A3",
        display_header_footer=True,
        header_template="<span></span>",
        footer_template=(
            "<div style='font-family:Inter,sans-serif;font-size:8px;color:#9b9a97;width:100%;"
            "padding:0 30mm;display:flex;justify-content:space-between'>"
            "<span>Credit card spending and stock returns</span><span class='pageNumber'></span></div>"
        ),
        margin={"top": "28mm", "bottom": "24mm", "left": "30mm", "right": "30mm"},
        print_background=True,
    )
    browser.close()
print("ok", round(corr_total, 2), round(corr_norm, 2), round(raw_ic, 4), round(smooth_ic, 4), round(ridge_all_ic, 4))
