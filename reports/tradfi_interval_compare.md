# TradFi interval comparison (1m / 5m / 1h)

Same 161 USD-M TradFi USDT perps, same 5 bps + impact / $1000 notional / `sigmoid > 0.7` long-only rule, same **30-day holdout** and nested **14-day** window. EW ann return is `mean(per-bar equal-weight net pnl) × bars_per_year` (linear, not compounded NAV). `1.0` means +100% annualized under that linear sum.

## Panels

| Interval | Train bars | test_30d | test_14d | Min quote vol | Search |
|---|---:|---:|---:|---:|---|
| 1m | 253,038 | 43,201 | 20,161 | 100 | 80 × 32 |
| 5m | 50,607 | 8,641 | 4,033 | 300 | 100 × 32 |
| 1h | 4,217 | 721 | 337 | 1000 | 200 × 256 |

## Mined formulas

| Interval | Formula | Train fitness | OOS trades (30d / 14d) |
|---|---|---:|---|
| 1m | `LOG_VOL PRESSURE LOG_VOL VOL_CHG LIQ ADD GATE SUB` | 0.0 | 87015 / 39881 |
| 5m | `DEV JUMP DECAY VOL_CHG DECAY MUL SIGN JUMP` | 0.0 | 0 / 0 |
| 1h | `PRESSURE DEV ABS DECAY SUB ABS MAX3 NEG` | 0.0 | 0 / 0 |

At 5m and 1h the miner prefers **doing nothing** (fitness 0 beats any positive-turnover 5 bp rule). The 1m formula still trades OOS and bleeds.

## 30-day holdout — EW annualized return

| Interval | Mined | RET | LIQ | PRESSURE | VOL_CHG | DEV | LOG_VOL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1m | -206.529 | -328.053 | -22.077 | -2202.586 | -2065.916 | -1376.122 | -1931.432 |
| 5m | 0.000 | -49.774 | -1.020 | -511.144 | -408.536 | -303.919 | -383.924 |
| 1h | 0.000 | -1.528 | -0.096 | -38.926 | -11.901 | -15.433 | -11.224 |

## 14-day window — EW annualized return

| Interval | Mined | RET | LIQ | PRESSURE | VOL_CHG | DEV | LOG_VOL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1m | -206.577 | -287.367 | -17.276 | -2250.337 | -2189.723 | -1440.620 | -2052.540 |
| 5m | 0.000 | -42.867 | -0.743 | -538.407 | -446.918 | -326.214 | -420.265 |
| 1h | 0.000 | -1.166 | -0.077 | -40.994 | -13.530 | -17.146 | -12.673 |

## 30-day holdout — trades

| Interval | Mined | RET | LIQ | PRESSURE |
|---|---:|---:|---:|---:|
| 1m | 87015 | 132484 | 7133 | 788946 |
| 5m | 0 | 28623 | 330 | 200299 |
| 1h | 0 | 2505 | 32 | 20051 |

## Takeaway

Cost drag **scales with bar frequency**. Under 5 bp taker fills, every interval's active long-only books are negative out of sample; the ranking is `1h << 5m << 1m` in how badly they lose.

- **1h** is the only horizon where a sparse feature (`LIQ`) is close to flat (−0.096 EW ann, 32 trades in 30 days). `RET` still linear-annualizes to −1.53. The miner correctly learns to sit out.
- **5m** `LIQ` is −1.02 EW ann with 330 trades; `RET` −50. Mined formula goes silent OOS.
- **1m** remains unusable: even `LIQ` is −22 EW ann, and the mined formula −207.

Default research clock should stay **1h**. If you want a signal that actually trades, the next lever is the fill model (lower fee, cooldown, or rank long/short) — not a faster bar.

Per-interval detail: [1h](tradfi_1h_oos.md), [5m](tradfi_5m_oos.md), [1m](tradfi_1m_oos.md).

