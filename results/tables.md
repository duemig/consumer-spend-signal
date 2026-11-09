# Results across 50 evaluation seeds (3001 to 3050)

Oracle mean IC: 0.0218

## Distribution of skill and net performance

| Variant | ic_mean | ic_lo | ic_hi | ic_std | sharpe_mean | sharpe_lo | sharpe_hi | sharpe_std | t_mean | net_return_mean | turnover_mean | sharpe_positive_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Final pipeline | 0.0156 | 0.0145 | 0.0167 | 0.0041 | 1.0493 | 0.9211 | 1.1746 | 0.4618 | 4.1278 | 0.1099 | 0.4320 | 0.9800 |
| Pooled winsorisation instead of per-series filter | 0.0156 | 0.0145 | 0.0167 | 0.0041 | 1.0329 | 0.9173 | 1.1450 | 0.4171 | 4.1108 | 0.1099 | 0.4519 | 0.9600 |
| Cohort offsets removed before averaging | 0.0149 | 0.0136 | 0.0162 | 0.0047 | 0.9611 | 0.8365 | 1.0817 | 0.4481 | 3.8688 | 0.1029 | 0.4240 | 1.0000 |
| Year-on-year growth instead of calendar buckets | 0.0121 | 0.0111 | 0.0131 | 0.0036 | 0.8290 | 0.6996 | 0.9590 | 0.4690 | 3.2093 | 0.0861 | 0.3200 | 0.9600 |
| Lagged values instead of trailing means | 0.0116 | 0.0102 | 0.0129 | 0.0049 | -0.8345 | -0.9830 | -0.6840 | 0.5470 | 2.9629 | -0.0951 | 1.7355 | 0.0600 |
| 50-day moving average instead of ridge | -0.0004 | -0.0017 | 0.0009 | 0.0049 | -0.1029 | -0.2493 | 0.0443 | 0.5358 | -0.2233 | -0.0073 | 0.0123 | 0.4800 |
| All sectors traded (no sector filter) | 0.0110 | 0.0101 | 0.0118 | 0.0032 | 0.7482 | 0.6172 | 0.8801 | 0.4834 | 4.0564 | 0.0569 | 0.4877 | 0.9600 |
| Sectors chosen on the test period | 0.0175 | 0.0165 | 0.0186 | 0.0037 | 1.2850 | 1.1781 | 1.3933 | 0.3951 | 4.7966 | 0.1306 | 0.4181 | 1.0000 |
| Data available 3 days late | 0.0118 | 0.0104 | 0.0130 | 0.0047 | 0.6567 | 0.5295 | 0.7814 | 0.4583 | 3.0032 | 0.0717 | 0.4175 | 0.9000 |
| Costs of 10 bp instead of 5 bp | 0.0156 | 0.0145 | 0.0167 | 0.0041 | 0.5331 | 0.4054 | 0.6623 | 0.4676 | 4.1278 | 0.0554 | 0.4320 | 0.9000 |
| Gradient-boosted trees instead of ridge | 0.0013 | 0.0005 | 0.0021 | 0.0029 | -0.2151 | -0.3559 | -0.0710 | 0.5191 | 0.5255 | -0.0175 | 0.2957 | 0.3000 |
| Beta- and sector-neutral spread | 0.0156 | 0.0145 | 0.0167 | 0.0041 | 1.4534 | 1.2826 | 1.6210 | 0.6068 | 4.1278 | 0.1041 | 0.3701 | 0.9800 |
| Cost-aware rebalancing | 0.0156 | 0.0145 | 0.0167 | 0.0041 | 1.1651 | 1.0384 | 1.2909 | 0.4617 | 4.1278 | 0.1145 | 0.2870 | 0.9800 |
| Borrow cost on the short side | 0.0156 | 0.0145 | 0.0167 | 0.0041 | 1.0020 | 0.8744 | 1.1272 | 0.4597 | 4.1278 | 0.1049 | 0.4320 | 0.9800 |
| Original design (see text) | 0.0153 | 0.0142 | 0.0164 | 0.0040 | 0.4305 | 0.3140 | 0.5486 | 0.4288 | 4.0372 | 0.0910 | 0.5677 | 0.8400 |

## Paired difference from the final pipeline

| Variant | d_ic | d_ic_lo | d_ic_hi | d_sharpe | d_sharpe_lo | d_sharpe_hi | ic_better_frac | sharpe_better_frac |
|---|---|---|---|---|---|---|---|---|
| Pooled winsorisation instead of per-series filter | -0.0000 | -0.0008 | 0.0008 | -0.0163 | -0.1060 | 0.0748 | 0.5200 | 0.4200 |
| Cohort offsets removed before averaging | -0.0007 | -0.0016 | 0.0001 | -0.0882 | -0.2080 | 0.0335 | 0.4600 | 0.4200 |
| Year-on-year growth instead of calendar buckets | -0.0035 | -0.0047 | -0.0023 | -0.2202 | -0.3871 | -0.0510 | 0.2400 | 0.4000 |
| Lagged values instead of trailing means | -0.0040 | -0.0053 | -0.0028 | -1.8838 | -2.0481 | -1.7142 | 0.2200 | 0.0000 |
| 50-day moving average instead of ridge | -0.0160 | -0.0178 | -0.0143 | -1.1522 | -1.3519 | -0.9479 | 0.0200 | 0.0400 |
| All sectors traded (no sector filter) | -0.0047 | -0.0055 | -0.0038 | -0.3010 | -0.4263 | -0.1720 | 0.0600 | 0.2600 |
| Sectors chosen on the test period | 0.0019 | 0.0013 | 0.0026 | 0.2357 | 0.1447 | 0.3319 | 0.8000 | 0.7800 |
| Data available 3 days late | -0.0039 | -0.0051 | -0.0027 | -0.3925 | -0.5153 | -0.2673 | 0.1600 | 0.1800 |
| Costs of 10 bp instead of 5 bp | 0.0000 | 0.0000 | 0.0000 | -0.5161 | -0.5552 | -0.4802 | 0.0000 | 0.0000 |
| Gradient-boosted trees instead of ridge | -0.0143 | -0.0157 | -0.0130 | -1.2644 | -1.4390 | -1.0849 | 0.0000 | 0.0200 |
| Beta- and sector-neutral spread | 0.0000 | 0.0000 | 0.0000 | 0.4041 | 0.2892 | 0.5194 | 0.0000 | 0.8800 |
| Cost-aware rebalancing | 0.0000 | 0.0000 | 0.0000 | 0.1158 | 0.0807 | 0.1514 | 0.0000 | 0.8000 |
| Borrow cost on the short side | 0.0000 | 0.0000 | 0.0000 | -0.0473 | -0.0486 | -0.0459 | 0.0000 | 0.0000 |
| Original design (see text) | -0.0003 | -0.0015 | 0.0009 | -0.6188 | -0.7443 | -0.4871 | 0.3800 | 0.0800 |

## Dependence on the simulator

|  | final_ic | oracle_ic | recovery | final_net_sharpe | ma_ic | lags_net_sharpe | no_filter_net_sharpe |
|---|---|---|---|---|---|---|---|
| ('Baseline', 'base') | 0.0146 | 0.0220 | 0.6552 | 0.9410 | 0.0001 | -0.6679 | 0.6741 |
| ('Signal strength', 'weak (0.5x)') | 0.0023 | 0.0115 | 0.1622 | -0.1066 | -0.0006 | -1.3171 | -0.4240 |
| ('Signal strength', 'strong (2x)') | 0.0350 | 0.0429 | 0.8155 | 3.2434 | 0.0004 | 1.2423 | 2.7511 |
| ('Learning half-life', 'fast (10d)') | 0.0102 | 0.0223 | 0.4520 | 0.3786 | -0.0003 | -0.9996 | 0.1881 |
| ('Learning half-life', 'slow (40d)') | 0.0167 | 0.0217 | 0.7658 | 1.2524 | 0.0009 | -0.4510 | 1.0654 |
| ('Panel noise', 'low (0.05)') | 0.0152 | 0.0220 | 0.6845 | 0.9468 | 0.0001 | -0.4938 | 0.7530 |
| ('Panel noise', 'high (0.20)') | 0.0140 | 0.0220 | 0.6300 | 0.8587 | -0.0000 | -1.0448 | 0.5515 |
| ('Card share', 'low (0.6x)') | 0.0066 | 0.0126 | 0.5642 | 0.2055 | 0.0031 | -1.3087 | -0.1171 |
| ('Card share', 'high (1.4x)') | 0.0229 | 0.0253 | 0.9035 | 1.7436 | -0.0023 | -0.2405 | 1.4259 |
| ('Publication lag', '1 day') | 0.0150 | 0.0220 | 0.6729 | 0.8736 | 0.0002 | -0.9741 | 0.6206 |
| ('Publication lag', '3 days') | 0.0114 | 0.0220 | 0.5044 | 0.5131 | -0.0004 | -1.0170 | 0.4658 |
