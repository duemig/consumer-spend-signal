# Results (evaluation seed 2024)

Oracle mean IC: 0.0203

| Variant | mean_ic | t_stat_nw | gross_sharpe | net_sharpe | net_return | net_vol | max_drawdown | turnover |
|---|---|---|---|---|---|---|---|---|
| Final pipeline | 0.0132 | 3.2 | 1.35 | 0.90 | 9.4% | 10.4% | 16.4% | 0.37 |
| Pooled winsorisation instead of per-series filter | 0.0122 | 3.2 | 1.10 | 0.63 | 6.4% | 10.3% | 15.6% | 0.38 |
| Cohort offsets removed before averaging | 0.0153 | 3.9 | 1.61 | 1.13 | 12.0% | 10.6% | 15.2% | 0.40 |
| Year-on-year growth instead of calendar buckets | 0.0150 | 4.0 | 1.44 | 1.11 | 11.7% | 10.6% | 13.3% | 0.28 |
| Lagged values instead of trailing means | 0.0125 | 3.3 | 1.07 | -0.65 | -7.2% | 11.2% | 46.7% | 1.52 |
| 50-day moving average instead of ridge | 0.0031 | 0.7 | 1.09 | 1.08 | 12.2% | 11.3% | 22.1% | 0.01 |
| All sectors traded (no sector filter) | 0.0112 | 3.8 | 1.72 | 1.05 | 8.3% | 7.9% | 9.5% | 0.41 |
| Sectors chosen on the test period | 0.0132 | 3.2 | 1.35 | 0.90 | 9.4% | 10.4% | 16.4% | 0.37 |
| Data available 3 days late | 0.0115 | 3.3 | 1.56 | 1.13 | 10.5% | 9.3% | 14.7% | 0.32 |
| Costs of 10 bp instead of 5 bp | 0.0132 | 3.2 | 1.35 | 0.45 | 4.7% | 10.4% | 18.8% | 0.37 |
| Original design (see text) | 0.0119 | 3.7 | 0.93 | 0.61 | 13.3% | 21.9% | 57.0% | 0.57 |

## Robustness across evaluation seeds

| seed | oracle_ic | oracle_full_ic | final_ic | final_t | final_net_sharpe | no_filter_ic | no_filter_net_sharpe | original_ic | original_net_sharpe |
|---|---|---|---|---|---|---|---|---|---|
| 2024 | 0.0203 | 0.0398 | 0.0132 | 3.2058 | 0.9013 | 0.0112 | 1.0537 | 0.0119 | 0.6087 |
| 2025 | 0.0176 | 0.0329 | 0.0110 | 3.1873 | 0.3102 | 0.0090 | 0.2109 | 0.0108 | 0.4587 |
| 2026 | 0.0217 | 0.0378 | 0.0099 | 2.2360 | -0.3097 | 0.0074 | -0.0751 | 0.0098 | 0.1959 |
| 2027 | 0.0264 | 0.0396 | 0.0182 | 4.5743 | 1.0986 | 0.0105 | 0.3455 | 0.0131 | 0.5627 |
| 2028 | 0.0208 | 0.0374 | 0.0205 | 5.9552 | 1.7586 | 0.0130 | 1.4441 | 0.0216 | 1.6954 |
