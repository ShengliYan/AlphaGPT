# TradFi 1h OOS backtest

161 USD-M TradFi USDT perps, **1h** bars from DuckDB. Formula search uses a train / valid / test split: robust-norm stats are fit on train only; the last two labels at each window boundary are zeroed so they cannot peek at the next open. The miner is scored on walk-forward train folds; the king is the formula that maximizes `min(train_fold_mean, valid) - GAP_PENALTY * |train-valid|`. Test windows stay out of selection.

## Setup

| Item | Value |
|---|---|
| Symbols | 161 |
| Interval | 1h |
| train | 2026-01-28T14:00 → 2026-06-23T06:00 (3,497 bars) |
| valid | 2026-06-23T07:00 → 2026-07-23T06:00 (720 bars) |
| test_30d | 2026-07-23T07:00 → 2026-08-22T07:00 (721 bars) |
| test_14d | 2026-08-08T07:00 → 2026-08-22T07:00 (337 bars) |
| Fee | 0 bps maker (no impact), $1000 notional |
| Min quote volume | 1000 |
| Position | cross-sectional long/short vs mean among liquid names |
| Search | AlphaGPT 200 steps × batch 256, formula length 8 |
| Valid days | 30 |
| Train folds | 4 |
| Gap / turnover penalty | 1.0 / 1.0 |

## Mined formula

`LOG_VOL DECAY LOG_VOL DELAY1 JUMP ADD MAX3 NEG`  tokens=[5, 15, 5, 16, 14, 6, 17, 10]

Train selection score is **0.97** (fold mean 0.98, valid period score 0.98, folds=[0.61, 1.40, 1.52, 0.39]).

Annualized EW return is `mean(per-bar EW net pnl) × 8,760`. It is not a compounded NAV.

## Results

### train (3,497 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit | Long/Short bars | Trades |
|---|---:|---:|---:|---:|---:|---:|
| best | 0.000 | 1.482 | 0.086 | 0.491 | 39533/53636 | 3918 |
| baseline_RET | 0.000 | 1.374 | 0.096 | 0.431 | 43100/50027 | 39996 |
| feature_LIQ | 0.000 | -0.539 | -0.043 | 0.502 | 64030/29139 | 21810 |
| feature_PRESSURE | 0.000 | -4.089 | -0.233 | 0.411 | 47153/46016 | 50062 |
| feature_VOL_CHG | 0.000 | 0.795 | 0.056 | 0.496 | 60058/31569 | 13812 |
| feature_DEV | 0.000 | -2.022 | -0.111 | 0.422 | 47225/45915 | 13588 |
| feature_LOG_VOL | 0.000 | -1.093 | -0.064 | 0.490 | 53957/39212 | 7444 |

- Mined formula, traded names only — top 5 Sharpe: MUUSDT (4.24, trades=34), CRMUSDT (3.79, trades=48), SNDKUSDT (3.75, trades=48), QQQUSDT (3.44, trades=34), KORUUSDT (3.26, trades=2)
- Bottom 5 Sharpe: AMZNUSDT (-2.61, trades=232), AMATUSDT (-2.41, trades=6), STXXUSDT (-2.32, trades=4), MSTRUSDT (-2.25, trades=202), DRAMUSDT (-1.98, trades=2)

### valid (720 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit | Long/Short bars | Trades |
|---|---:|---:|---:|---:|---:|---:|
| best | 0.277 | 3.565 | 0.739 | 0.494 | 14353/63220 | 2634 |
| baseline_RET | 0.000 | -0.591 | -0.138 | 0.485 | 28749/48824 | 29390 |
| feature_LIQ | -0.601 | -4.428 | -1.069 | 0.488 | 60194/17379 | 17526 |
| feature_PRESSURE | 0.000 | -0.469 | -0.088 | 0.428 | 38523/39050 | 40982 |
| feature_VOL_CHG | 0.000 | -1.702 | -0.411 | 0.492 | 65814/10712 | 5861 |
| feature_DEV | -1.238 | -4.336 | -0.942 | 0.449 | 37888/39685 | 11478 |
| feature_LOG_VOL | -0.643 | -3.509 | -0.731 | 0.501 | 63297/14276 | 2590 |

- Mined formula, traded names only — top 5 Sharpe: ORCLUSDT (6.02, trades=0), APPUSDT (5.82, trades=10), USARUSDT (5.21, trades=4), FLNCUSDT (4.91, trades=6), OPENAIUSDT (4.83, trades=0)
- Bottom 5 Sharpe: VUSDT (-3.94, trades=72), SHAZUSDT (-3.94, trades=2), MUUUSDT (-3.91, trades=2), XLEUSDT (-3.28, trades=92), ADBEUSDT (-3.08, trades=20)

### test_30d (721 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit | Long/Short bars | Trades |
|---|---:|---:|---:|---:|---:|---:|
| best | -0.006 | -0.040 | -0.011 | 0.460 | 14432/85303 | 3465 |
| baseline_RET | -1.813 | -5.027 | -1.469 | 0.447 | 31872/67863 | 33811 |
| feature_LIQ | 1.709 | 3.804 | 1.149 | 0.540 | 80224/19511 | 18973 |
| feature_PRESSURE | -0.925 | -2.322 | -0.622 | 0.445 | 50777/48958 | 53418 |
| feature_VOL_CHG | -0.103 | -0.337 | -0.104 | 0.526 | 87518/10780 | 6808 |
| feature_DEV | -0.665 | -1.517 | -0.410 | 0.472 | 51659/48076 | 14619 |
| feature_LOG_VOL | 0.006 | 0.031 | 0.009 | 0.534 | 85397/14338 | 3403 |

- Mined formula, traded names only — top 5 Sharpe: LYTEUSDT (5.33, trades=2), PLTRUSDT (4.89, trades=2), KUAISHOUUSDT (4.82, trades=22), MEITUANUSDT (4.42, trades=19), APPUSDT (4.15, trades=38)
- Bottom 5 Sharpe: BITOUSDT (-7.61, trades=96), ZMUSDT (-6.16, trades=34), STRCUSDT (-6.06, trades=14), ANTHROPICUSDT (-5.87, trades=2), MSFTUSDT (-5.77, trades=0)

### test_14d (337 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit | Long/Short bars | Trades |
|---|---:|---:|---:|---:|---:|---:|
| best | 0.572 | 1.579 | 0.323 | 0.445 | 6749/41924 | 1840 |
| baseline_RET | -2.449 | -7.515 | -1.601 | 0.430 | 13641/35032 | 14623 |
| feature_LIQ | 0.688 | 2.875 | 0.620 | 0.537 | 40562/8111 | 7404 |
| feature_PRESSURE | -2.208 | -5.688 | -0.994 | 0.430 | 24773/23900 | 26253 |
| feature_VOL_CHG | -1.422 | -1.653 | -0.370 | 0.528 | 43031/5033 | 3302 |
| feature_DEV | -1.447 | -3.337 | -0.630 | 0.451 | 24701/23972 | 7344 |
| feature_LOG_VOL | -0.572 | -1.473 | -0.301 | 0.543 | 41982/6691 | 1802 |

- Mined formula, traded names only — top 5 Sharpe: APPUSDT (9.59, trades=8), AVGOUSDT (9.34, trades=0), CRCLUSDT (8.81, trades=0), COINUSDT (7.92, trades=0), UVXYUSDT (7.86, trades=4)
- Bottom 5 Sharpe: ANTHROPICUSDT (-14.02, trades=2), XLEUSDT (-13.59, trades=16), BITOUSDT (-12.67, trades=52), GDXUSDT (-8.10, trades=3), FWDIUSDT (-6.80, trades=16)

## Sign consistency

| Variant | Train | Valid | Test 30d | Train=Valid | Valid=Test |
|---|---:|---:|---:|---|---|
| best | 1.482 | 3.565 | -0.040 | yes | no |
| baseline_RET | 1.374 | -0.591 | -5.027 | no | yes |
| feature_LIQ | -0.539 | -4.428 | 3.804 | yes | no |
| feature_PRESSURE | -4.089 | -0.469 | -2.322 | yes | yes |
| feature_VOL_CHG | 0.795 | -1.702 | -0.337 | no | yes |
| feature_DEV | -2.022 | -4.336 | -1.517 | yes | yes |
| feature_LOG_VOL | -1.093 | -3.509 | 0.031 | yes | no |

## Takeaway

Compared with the previous train-max king (`RET` stack: train EW Sharpe +3.8, 30-day test **−4.4**):

| Book | Train | Valid | Test 30d | Test 14d | Train trades |
|---|---:|---:|---:|---:|---:|
| Old train-max | +3.79 | (none) | −4.41 | −6.40 | high |
| **This run** | **+1.48** | **+3.57** | **−0.04** | **+1.58** | 3.9k |
| RET | +1.37 | −0.59 | −5.03 | −7.52 | 40k |

All four train folds of the new formula are positive. Turnover is ~10× lower than `RET`. The 30-day test is flat rather than a crash; the 14-day slice is positive. That is the consistency gain.

It is **not** a guarantee of a profitable next month: `valid=test` is still "no" because −0.04 is slightly negative. `LIQ` remains the best 30-day test book but loses on both train and valid, so a leak-free selector will keep missing it until that regime shows up before the test window.

What actually moved the needle: walk-forward fold mean, `min(train, valid)` selection, and a turnover penalty. Test data still never enters selection.


