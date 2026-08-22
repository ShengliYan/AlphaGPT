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
        "Formula search is trained on all names with the last 30 calendar days held out; "
        "the last 14 days are a nested OOS window. Robust-norm medians/MADs are fit on train only, "
        "and the last two train labels are zeroed so they cannot peek at test opens.",
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
    lines.extend(
        [
            f"| Fee | {float(report.get('fee_bps') or 0):.0f} bps + impact, ${float(report.get('trade_size_usd') or 0):.0f} notional |",
            f"| Min quote volume | {float(report.get('min_quote_volume') or 0):.0f} |",
            "| Position | long-only if sigmoid(factor) > 0.7 and bar is liquid |",
            f"| Search | AlphaGPT {report.get('train_steps')} steps × batch {report.get('batch_size')}, formula length 8 |",
            "",
            "## Mined formula",
            "",
            "`" + " ".join(report.get("formula_decoded") or []) + f"`  tokens={report.get('formula')}",
            "",
            f"Train fitness (median *period* Sharpe across names) is **{report.get('train_fitness')}**.",
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
        lines.append("| Variant | Med ann Sharpe | EW ann Sharpe | EW ann return | Hit rate | Trades |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for key in keys:
            stats = window.get(key)
            if not stats:
                continue
            lines.append(
                f"| {key} | {_fmt(stats.get('median_ann_sharpe'))} | "
                f"{_fmt(stats.get('ew_ann_sharpe'))} | {_fmt(stats.get('ew_ann_return'))} | "
                f"{_fmt(stats.get('ew_hit_rate'))} | {float(stats.get('total_trades') or 0):.0f} |"
            )
        lines.append("")
        best = window.get("best") or {}
        if best.get("top5"):
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

    if takeaway:
        lines.extend(["## Takeaway", "", takeaway, ""])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
