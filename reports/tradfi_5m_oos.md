# TradFi 5m OOS backtest

161 USD-M TradFi USDT perps, **5m** bars from DuckDB. Formula search is trained on all names with the last 30 calendar days held out; the last 14 days are a nested OOS window. Robust-norm medians/MADs are fit on train only, and the last two train labels are zeroed so they cannot peek at test opens.

## Setup

| Item | Value |
|---|---|
| Symbols | 161 |
| Interval | 5m |
| train | 2026-01-28T14:30 → 2026-07-23T07:40 (50,607 bars) |
| test_30d | 2026-07-23T07:45 → 2026-08-22T07:45 (8,641 bars) |
| test_14d | 2026-08-08T07:45 → 2026-08-22T07:45 (4,033 bars) |
| Fee | 5 bps + impact, $1000 notional |
| Min quote volume | 300 |
| Position | long-only if sigmoid(factor) > 0.7 and bar is liquid |
| Search | AlphaGPT 100 steps × batch 32, formula length 8 |

## Mined formula

`DEV JUMP DECAY VOL_CHG DECAY MUL SIGN JUMP`  tokens=[4, 14, 15, 3, 15, 8, 12, 14]

Train fitness (median *period* Sharpe across names) is **0.0**.

Annualized EW return is `mean(per-bar EW net pnl) × 105,120`. It is not a compounded NAV.

## Results

### train (50,607 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|
| best | 0.000 | -76.340 | -2.765 | 0.004 | 5424 |
| baseline_RET | 0.000 | -413.886 | -52.286 | 0.009 | 134126 |
| feature_LIQ | 0.000 | -146.191 | -3.117 | 0.002 | 5923 |
| feature_PRESSURE | -79.191 | -328.046 | -161.327 | 0.005 | 362787 |
| feature_VOL_CHG | -59.830 | -317.865 | -124.837 | 0.029 | 259382 |
| feature_DEV | -61.443 | -255.503 | -76.963 | 0.057 | 162734 |
| feature_LOG_VOL | -59.646 | -250.362 | -96.120 | 0.049 | 195676 |

- Mined formula, traded names only — top 5 Sharpe: POPMARTUSDT (-3.44, trades=3), SKHYUSDT (-8.00, trades=144), HK1810USDT (-9.15, trades=24), SHAZUSDT (-9.56, trades=39), TENCENTUSDT (-11.71, trades=32)
- Bottom 5 Sharpe: XBIUSDT (-43.71, trades=455), APPUSDT (-43.37, trades=444), SNOWUSDT (-42.18, trades=430), GEVUSDT (-41.86, trades=426), WENUSDT (-40.63, trades=394)

### test_30d (8,641 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|
| best | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| baseline_RET | 0.000 | -482.278 | -49.774 | 0.007 | 28623 |
| feature_LIQ | 0.000 | -90.471 | -1.020 | 0.000 | 330 |
| feature_PRESSURE | -190.728 | -1033.325 | -511.144 | 0.000 | 200299 |
| feature_VOL_CHG | -164.508 | -869.085 | -408.536 | 0.018 | 140206 |
| feature_DEV | -140.157 | -844.780 | -303.919 | 0.007 | 108791 |
| feature_LOG_VOL | -148.264 | -842.163 | -383.924 | 0.020 | 127318 |

### test_14d (4,033 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|
| best | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| baseline_RET | 0.000 | -465.440 | -42.867 | 0.007 | 11530 |
| feature_LIQ | 0.000 | -75.969 | -0.743 | 0.000 | 112 |
| feature_PRESSURE | -196.475 | -1152.348 | -538.407 | 0.000 | 97113 |
| feature_VOL_CHG | -177.595 | -1033.104 | -446.918 | 0.010 | 71008 |
| feature_DEV | -143.572 | -991.551 | -326.214 | 0.002 | 53942 |
| feature_LOG_VOL | -161.173 | -1020.544 | -420.265 | 0.011 | 64954 |

## Takeaway

The miner found `DEV JUMP DECAY VOL_CHG DECAY MUL SIGN JUMP`, which trades a little in-sample then **goes silent on both OOS windows**. Single-feature 5m books still overtrade: 30-day `PRESSURE` EW ann return **-511**, `RET` **-50**, `LIQ` **-1.02** with only 330 trades. 5m sits between 1m and 1h: same sign, smaller cost drag than 1m, still not a 5 bp taker clock.

