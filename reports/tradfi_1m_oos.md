# TradFi 1m OOS backtest

161 USD-M TradFi USDT perps, canonical **1m** bars from DuckDB. Formula search is trained on all names with the last 30 calendar days held out; the last 14 days are a nested OOS window. Robust-norm medians/MADs are fit on train only, and the last two train labels are zeroed so they cannot peek at test opens.

## Setup

| Item | Value |
|---|---|
| Symbols | 161 |
| Interval | 1m |
| train | 2026-01-28T14:30 → 2026-07-23T07:47 (253,038 bars) |
| test_30d | 2026-07-23T07:48 → 2026-08-22T07:48 (43,201 bars) |
| test_14d | 2026-08-08T07:48 → 2026-08-22T07:48 (20,161 bars) |
| Fee | 5 bps + impact, $1000 notional |
| Min quote volume | 100 |
| Position | long-only if sigmoid(factor) > 0.7 and bar is liquid |
| Search | AlphaGPT 80 steps × batch 32, formula length 8 |

## Mined formula

`LOG_VOL PRESSURE LOG_VOL VOL_CHG LIQ ADD GATE SUB`  tokens=[5, 2, 5, 3, 1, 6, 13, 7]

Train fitness (median *period* Sharpe across names) is **0**. Many names never clear the 1m liquidity/threshold filter, so the median symbol is flat. Equal-weight PnL of the names that *do* trade is dominated by 1m turnover × 5 bps.

Annualized EW return is `mean(per-bar EW net pnl) × 525,600`. It is not a compounded NAV.

## Results

### train (253,038 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|
| best | 0.000 | -578.169 | -97.182 | 0.012 | 221440 |
| baseline_RET | 0.000 | -918.127 | -292.102 | 0.006 | 610872 |
| feature_LIQ | 0.000 | -620.284 | -53.700 | 0.008 | 101616 |
| feature_PRESSURE | -145.511 | -753.527 | -740.159 | 0.004 | 1501922 |
| feature_VOL_CHG | -149.620 | -829.819 | -687.655 | 0.010 | 1333449 |
| feature_DEV | -129.220 | -634.112 | -392.064 | 0.035 | 771622 |
| feature_LOG_VOL | -149.620 | -697.141 | -564.698 | 0.024 | 1076391 |

- Mined formula, traded names only — top 5 Sharpe: EWYUSDT (-114.10, trades=3547), AAPLUSDT (-118.48, trades=3297), NVDAUSDT (-125.44, trades=4072), TSMUSDT (-171.78, trades=6737), INTCUSDT (-174.69, trades=8827)
- Bottom 5 Sharpe: BABAUSDT (-330.01, trades=21864), AVGOUSDT (-327.48, trades=21697), AMZNUSDT (-239.54, trades=12508), QQQUSDT (-235.53, trades=12592), GOOGLUSDT (-233.33, trades=11998)

### test_30d (43,201 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|
| best | 0.000 | -1181.693 | -206.529 | 0.008 | 87015 |
| baseline_RET | 0.000 | -1076.672 | -328.053 | 0.002 | 132484 |
| feature_LIQ | 0.000 | -426.741 | -22.077 | 0.004 | 7133 |
| feature_PRESSURE | -344.566 | -2057.873 | -2202.586 | 0.000 | 788946 |
| feature_VOL_CHG | -372.279 | -3231.649 | -2065.916 | 0.001 | 679722 |
| feature_DEV | -281.778 | -2003.249 | -1376.122 | 0.000 | 467082 |
| feature_LOG_VOL | -352.680 | -3046.282 | -1931.432 | 0.001 | 625339 |

- Mined formula, traded names only — top 5 Sharpe: HOODUSDT (-161.89, trades=1080), EWYUSDT (-182.86, trades=1640), AAPLUSDT (-200.36, trades=1537), NVDAUSDT (-200.88, trades=1786), TSMUSDT (-219.85, trades=1823)
- Bottom 5 Sharpe: BABAUSDT (-595.56, trades=8797), AVGOUSDT (-541.05, trades=7770), GOOGLUSDT (-455.96, trades=6335), QQQUSDT (-386.85, trades=5091), AMZNUSDT (-380.95, trades=4747)

### test_14d (20,161 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|
| best | 0.000 | -1159.739 | -206.577 | 0.008 | 39881 |
| baseline_RET | 0.000 | -1014.046 | -287.367 | 0.002 | 53479 |
| feature_LIQ | 0.000 | -381.055 | -17.276 | 0.002 | 2604 |
| feature_PRESSURE | -347.587 | -2124.621 | -2250.337 | 0.000 | 373050 |
| feature_VOL_CHG | -375.435 | -3536.590 | -2189.723 | 0.001 | 335858 |
| feature_DEV | -287.577 | -2168.065 | -1440.620 | 0.000 | 226908 |
| feature_LOG_VOL | -364.519 | -3348.712 | -2052.540 | 0.001 | 310176 |

- Mined formula, traded names only — top 5 Sharpe: EWYUSDT (-165.11, trades=565), HOODUSDT (-167.45, trades=534), NVDAUSDT (-192.00, trades=754), AAPLUSDT (-207.72, trades=767), TSMUSDT (-219.45, trades=848)
- Bottom 5 Sharpe: BABAUSDT (-594.03, trades=4101), AVGOUSDT (-547.02, trades=3683), GOOGLUSDT (-460.37, trades=2981), QQQUSDT (-401.02, trades=2476), SPYUSDT (-392.26, trades=2304)

## Takeaway

This fill model cannot support 1m trading on this universe. High-frequency features (`PRESSURE`, `VOL_CHG`, `LOG_VOL`) fire on almost every liquid bar and pay the taker fee on huge turnover. `LIQ` (Amihud) trades least and loses least out of sample, but equal-weight annualized return is still negative (~-22% on the 30-day holdout, ~-17% on the last 14 days). The mined formula is closer to `LIQ` than to the hyperactive features, yet still bleeds ~-207% annualized EW on both OOS windows.

Default research interval remains **1h**. Use 1m as the storage/resample source, not as a 5 bp taker execution clock.

