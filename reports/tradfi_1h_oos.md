# TradFi 1h OOS backtest

161 USD-M TradFi USDT perps, **1h** bars from DuckDB. Formula search is trained on all names with the last 30 calendar days held out; the last 14 days are a nested OOS window. Robust-norm medians/MADs are fit on train only, and the last two train labels are zeroed so they cannot peek at test opens.

## Setup

| Item | Value |
|---|---|
| Symbols | 161 |
| Interval | 1h |
| train | 2026-01-28T14:00 → 2026-07-23T06:00 (4,217 bars) |
| test_30d | 2026-07-23T07:00 → 2026-08-22T07:00 (721 bars) |
| test_14d | 2026-08-08T07:00 → 2026-08-22T07:00 (337 bars) |
| Fee | 5 bps + impact, $1000 notional |
| Min quote volume | 1000 |
| Position | long-only if sigmoid(factor) > 0.7 and bar is liquid |
| Search | AlphaGPT 200 steps × batch 256, formula length 8 |

## Mined formula

`PRESSURE DEV ABS DECAY SUB ABS MAX3 NEG`  tokens=[2, 4, 11, 15, 7, 11, 17, 10]

Train fitness (median *period* Sharpe across names) is **0.0**.

Annualized EW return is `mean(per-bar EW net pnl) × 8,760`. It is not a compounded NAV.

## Results

### train (4,217 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|
| best | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| baseline_RET | 0.000 | -69.745 | -1.835 | 0.082 | 11726 |
| feature_LIQ | 0.000 | -17.469 | -0.043 | 0.002 | 106 |
| feature_PRESSURE | -20.005 | -71.448 | -9.921 | 0.050 | 34073 |
| feature_VOL_CHG | -2.275 | -25.664 | -3.988 | 0.203 | 11927 |
| feature_DEV | -8.001 | -36.316 | -3.666 | 0.205 | 11178 |
| feature_LOG_VOL | -2.034 | -18.813 | -2.894 | 0.221 | 9330 |

### test_30d (721 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|
| best | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| baseline_RET | 0.000 | -47.256 | -1.528 | 0.117 | 2505 |
| feature_LIQ | 0.000 | -28.356 | -0.096 | 0.004 | 32 |
| feature_PRESSURE | -50.220 | -162.827 | -38.926 | 0.014 | 20051 |
| feature_VOL_CHG | -10.130 | -33.089 | -11.901 | 0.266 | 5121 |
| feature_DEV | -20.019 | -64.753 | -15.433 | 0.126 | 7338 |
| feature_LOG_VOL | -9.276 | -31.061 | -11.224 | 0.269 | 4740 |

### test_14d (337 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|
| best | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| baseline_RET | 0.000 | -62.539 | -1.166 | 0.083 | 965 |
| feature_LIQ | 0.000 | -25.067 | -0.077 | 0.003 | 12 |
| feature_PRESSURE | -54.106 | -235.060 | -40.994 | 0.000 | 9667 |
| feature_VOL_CHG | -13.604 | -51.082 | -13.530 | 0.214 | 2621 |
| feature_DEV | -25.236 | -89.606 | -17.146 | 0.110 | 3660 |
| feature_LOG_VOL | -11.884 | -48.229 | -12.673 | 0.220 | 2438 |

## Takeaway

The miner collapsed to a **zero-trade** formula (`PRESSURE DEV ABS DECAY SUB ABS MAX3 NEG`). That is the same local optimum as 1m: doing nothing scores 0, while almost every active 5 bp long-only rule scores below 0. On the 30-day holdout, `LIQ` is the least-bad active feature (EW ann return **-0.096**, 32 trades). `RET` is **-1.53**, `PRESSURE` **-38.9**. Costs still dominate, but they are an order of magnitude smaller than at 5m/1m.

