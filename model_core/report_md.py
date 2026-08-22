from pathlib import Path

from .config import ModelConfig
from .vocab import FORMULA_VOCAB


def _fmt(value, nd=3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{nd}f}"


def _variant_keys():
    return ["best", "baseline_RET"] + [
        f"feature_{name}" for name in FORMULA_VOCAB.feature_names if name != "RET"
    ]


def _short_ts(value) -> str:
    if not value:
        return "n/a"
    return str(value)[:16]


def write_oos_markdown(report: dict, path: str | Path, takeaway: str | None = None) -> Path:
    interval = report.get("bar_interval", ModelConfig.BAR_INTERVAL)
    n_symbols = report.get("n_symbols", 0)
    bars_per_year = ModelConfig.BARS_PER_YEAR.get(interval, 365 * 24)
    windows = report.get("windows") or {}
    lines = [
        f"# TradFi {interval} OOS backtest",
        "",
        f"{n_symbols} USD-M TradFi USDT perps, **{interval}** bars from DuckDB. "
        "Formula search uses a train / valid / test split: robust-norm stats are fit on train only; "
        "the last two labels at each window boundary are zeroed so they cannot peek at the next open. "
        "The miner is scored on walk-forward train folds; the king is the formula that maximizes "
        "`min(train_fold_mean, valid) - GAP_PENALTY * |train-valid|`. Test windows stay out of selection.",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Symbols | {n_symbols} |",
        f"| Interval | {interval} |",
    ]
    for name, window in windows.items():
        n_bars = int(window.get("n_bars") or 0)
        lines.append(
            f"| {name} | {_short_ts(window.get('start'))} → {_short_ts(window.get('end'))} ({n_bars:,} bars) |"
        )
    pos_mode = report.get("position_mode") or "long_short"
    if pos_mode == "long_only":
        pos_desc = "long-only if sigmoid(factor) > 0.7 and bar is liquid"
    else:
        thresh = report.get("ls_z_thresh")
        pos_desc = (
            f"cross-sectional long/short vs mean among liquid names"
            + (f" (|z| > {thresh:g})" if thresh else "")
        )
    fee_bps = float(report.get("fee_bps") or 0)
    impact = report.get("impact_coeff")
    fee_desc = f"{fee_bps:.0f} bps"
    if impact:
        fee_desc += f" + impact×{impact:g}"
    else:
        fee_desc += " maker (no impact)"
    lines.extend(
        [
            f"| Fee | {fee_desc}, ${float(report.get('trade_size_usd') or 0):.0f} notional |",
            f"| Min quote volume | {float(report.get('min_quote_volume') or 0):.0f} |",
            f"| Position | {pos_desc} |",
            f"| Search | AlphaGPT {report.get('train_steps')} steps × batch {report.get('batch_size')}, formula length 8 |",
            f"| Valid days | {report.get('valid_days')} |",
            f"| Train folds | {report.get('n_folds')} |",
            f"| Gap / turnover penalty | {report.get('gap_penalty')} / {report.get('turnover_penalty')} |",
            "",
            "## Mined formula",
            "",
            "`" + " ".join(report.get("formula_decoded") or []) + f"`  tokens={report.get('formula')}",
            "",
            f"Train selection score is **{report.get('train_fitness')}** "
            f"(fold mean {report.get('train_fold_mean')}, valid {report.get('valid_score')}, "
            f"folds={report.get('train_fold_scores')}).",
            "",
            f"Annualized EW return is `mean(per-bar EW net pnl) × {bars_per_year:,}`. It is not a compounded NAV.",
            "",
            "## Results",
            "",
        ]
    )
    keys = _variant_keys()
    for window_name, window in windows.items():
        n_bars = int(window.get("n_bars") or 0)
        lines.append(f"### {window_name} ({n_bars:,} bars)")
        lines.append("")
        lines.append("| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit | Long/Short bars | Trades |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for key in keys:
            stats = window.get(key)
            if not stats:
                continue
            lines.append(
                f"| {key} | {_fmt(stats.get('median_ann_sharpe'))} | "
                f"{_fmt(stats.get('ew_ann_sharpe'))} | {_fmt(stats.get('ew_ann_return'))} | "
                f"{_fmt(stats.get('ew_hit_rate'))} | "
                f"{float(stats.get('long_bars') or 0):.0f}/{float(stats.get('short_bars') or 0):.0f} | "
                f"{float(stats.get('total_trades') or 0):.0f} |"
            )
        lines.append("")
        best = window.get("best") or {}
        if best.get("top5") and any(float(row.get("trades") or 0) > 0 for row in best["top5"]):
            tops = ", ".join(
                f"{row['symbol']} ({row['ann_sharpe']:.2f}, trades={row['trades']:.0f})"
                for row in best["top5"]
            )
            bots = ", ".join(
                f"{row['symbol']} ({row['ann_sharpe']:.2f}, trades={row['trades']:.0f})"
                for row in best["bottom5"]
            )
            lines.append(f"- Mined formula, traded names only — top 5 Sharpe: {tops}")
            lines.append(f"- Bottom 5 Sharpe: {bots}")
            lines.append("")

    consistency = report.get("consistency") or []
    if consistency:
        lines.extend(
            [
                "## Sign consistency",
                "",
                "| Variant | Train | Valid | Test 30d | Train=Valid | Valid=Test |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        test_key = f"test_{report.get('test_days') or 30}d"
        for row in consistency:
            sharpes = row.get("sharpes") or {}
            test_s = sharpes.get(test_key)
            if test_s is None:
                test_s = sharpes.get("test_30d")
            lines.append(
                f"| {row.get('variant')} | {_fmt(sharpes.get('train'))} | {_fmt(sharpes.get('valid'))} | "
                f"{_fmt(test_s)} | "
                f"{'yes' if row.get('train_valid_same_sign') else 'no'} | "
                f"{'yes' if row.get('valid_test_same_sign') else 'no'} |"
            )
        lines.append("")

    if takeaway:
        lines.extend(["## Takeaway", "", takeaway, ""])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
