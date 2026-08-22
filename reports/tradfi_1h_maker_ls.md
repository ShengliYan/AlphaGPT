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
| Fee | 0 bps maker (no impact), $1000 notional |
| Min quote volume | 1000 |
| Position | cross-sectional long/short vs mean among liquid names |
| Search | AlphaGPT 200 steps × batch 256, formula length 8 |

## Mined formula

`RET LOG_VOL VOL_CHG ABS DIV SIGN DIV DECAY`  tokens=[0, 5, 3, 11, 9, 12, 9, 15]

Train fitness (equal-weight book period Sharpe) is **2.63**.

Annualized EW return is `mean(per-bar EW net pnl) × 8,760`. It is not a compounded NAV.

## Results

### train (4,217 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit | Long/Short bars | Trades |
|---|---:|---:|---:|---:|---:|---:|
| best | 0.521 | 3.785 | 0.433 | 0.494 | 64397/106345 | 39993 |
| baseline_RET | 0.000 | 0.324 | 0.036 | 0.425 | 72547/98139 | 72461 |
| feature_LIQ | -0.283 | -1.725 | -0.220 | 0.491 | 120650/50092 | 45446 |
| feature_PRESSURE | -0.211 | -2.240 | -0.208 | 0.414 | 85676/85066 | 91044 |
| feature_VOL_CHG | 0.000 | -0.512 | -0.059 | 0.493 | 120778/48233 | 21181 |
| feature_DEV | -0.577 | -2.495 | -0.253 | 0.426 | 84337/86376 | 25269 |
| feature_LOG_VOL | -0.266 | -1.703 | -0.166 | 0.494 | 114193/56549 | 11269 |

- Mined formula, traded names only — top 5 Sharpe: LITEUSDT (4.70, trades=330), MRVLUSDT (4.43, trades=328), CRWVUSDT (3.88, trades=318), AAOIUSDT (3.85, trades=210), ALABUSDT (3.79, trades=162)
- Bottom 5 Sharpe: COINUSDT (-2.77, trades=1106), SQQQUSDT (-2.21, trades=112), SPYUSDT (-2.21, trades=410), TSLAUSDT (-2.04, trades=1134), TZAUSDT (-1.87, trades=36)

### test_30d (721 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit | Long/Short bars | Trades |
|---|---:|---:|---:|---:|---:|---:|
| best | -1.322 | -4.414 | -1.287 | 0.460 | 26410/73325 | 19562 |
| baseline_RET | -1.838 | -5.488 | -1.580 | 0.447 | 32934/66801 | 35209 |
| feature_LIQ | 1.149 | 3.053 | 0.927 | 0.538 | 78688/21047 | 20593 |
| feature_PRESSURE | -0.925 | -2.322 | -0.622 | 0.445 | 50777/48958 | 53418 |
| feature_VOL_CHG | -0.518 | -1.192 | -0.356 | 0.526 | 85550/13161 | 6883 |
| feature_DEV | -0.726 | -1.500 | -0.403 | 0.469 | 51341/48394 | 14720 |
| feature_LOG_VOL | -0.004 | -0.050 | -0.013 | 0.528 | 83233/16502 | 3401 |

- Mined formula, traded names only — top 5 Sharpe: LLYUSDT (6.41, trades=124), TZAUSDT (6.09, trades=164), IBMUSDT (5.74, trades=114), COSTUSDT (5.32, trades=170), PYPLUSDT (5.15, trades=136)
- Bottom 5 Sharpe: STRCUSDT (-9.78, trades=120), URNMUSDT (-8.75, trades=174), MINIMAXUSDT (-7.28, trades=114), USARUSDT (-7.10, trades=114), SOFIUSDT (-6.93, trades=180)

### test_14d (337 bars)

| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit | Long/Short bars | Trades |
|---|---:|---:|---:|---:|---:|---:|
| best | -2.165 | -6.403 | -1.378 | 0.445 | 11342/37331 | 8356 |
| baseline_RET | -1.697 | -6.044 | -1.290 | 0.430 | 14529/34144 | 15697 |
| feature_LIQ | 1.189 | 3.703 | 0.805 | 0.549 | 39888/8785 | 8246 |
| feature_PRESSURE | -2.208 | -5.688 | -0.994 | 0.430 | 24773/23900 | 26253 |
| feature_VOL_CHG | -1.340 | -1.895 | -0.414 | 0.531 | 41934/6130 | 3415 |
| feature_DEV | -1.447 | -3.272 | -0.615 | 0.445 | 24488/24185 | 7392 |
| feature_LOG_VOL | -0.276 | -1.047 | -0.212 | 0.540 | 40970/7703 | 1800 |

- Mined formula, traded names only — top 5 Sharpe: PANWUSDT (6.83, trades=68), KUAISHOUUSDT (5.93, trades=52), KOUSDT (5.92, trades=64), HIMSUSDT (5.90, trades=52), HDUSDT (5.85, trades=80)
- Bottom 5 Sharpe: AMZNUSDT (-14.93, trades=96), SAMSUNGUSDT (-12.54, trades=42), SKHYUSDT (-11.77, trades=42), KORUUSDT (-11.53, trades=42), SKHYNIXUSDT (-11.47, trades=42)

## Takeaway

The first mined formulas looked dead because the **backtest was wrong for these perps**, not because the features have no signal.

1. **Long-only.** `sigmoid(factor) > 0.7` never took a short. `PRESSURE` / `DEV` are balanced ~50/50 long/short here; that side was previously thrown away.
2. **5 bp taker.** Any 1h flip paid 10 bps round-trip. Cash (Sharpe 0) beat every active rule, so the miner learned to sit out.
3. **Median-name Sharpe.** Most names never cleared 0.7, so the median was locked at 0 even when the book had edge.

This run is maker **0 bps**, cross-sectional **long/short vs the mean**, fitness = equal-weight book Sharpe.

The miner now finds a real in-sample book (`RET LOG_VOL VOL_CHG ABS DIV SIGN DIV DECAY`, train EW ann Sharpe **+3.79**, mix of longs and shorts) but it **does not hold** on the 30-day / 14-day holdouts (−4.4 / −6.4). That formula is RET-like; `RET` itself also flips from +0.32 in-sample to −5.5 OOS.

The feature that *does* work out of sample is **`LIQ` (Amihud)**: 30-day EW ann Sharpe **+3.05**, 14-day **+3.70**, hit ~0.54. It was **negative in-sample** (−1.73), so a train-only miner will not pick it. That is a regime/overfit issue, not a “shorts or fees” issue.

Replay the old book with `FEE_BPS=5 POSITION_MODE=long_only IMPACT_COEFF=1`.


