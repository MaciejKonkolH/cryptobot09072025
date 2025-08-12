import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class Trade:
    timestamp: str
    signal: str  # "LONG" or "SHORT"
    probability: float
    true_label: str  # "LONG" | "SHORT" | "NEUTRAL"
    result: str  # "WIN" | "LOSS"


def parse_filename_metadata(file_path: Path) -> Tuple[str, float, float, float]:
    """Parse symbol, tp, sl, thr from filename like:
    trades_{SYMBOL}_tp{TP}_sl{SL}_thr{THR}_{YYYYMMDD_HHMMSS}.json
    - TP/SL are floats with dot decimal
    - THR is percent as integer (e.g., 40)
    """
    name = file_path.name
    pattern = r"^trades_(?P<symbol>[A-Z0-9]+)_tp(?P<tp>\d+(?:\.\d+)?)_sl(?P<sl>\d+(?:\.\d+)?)_thr(?P<thr>\d+)_\d{8}_\d{6}\.json$"
    m = re.match(pattern, name)
    if not m:
        raise SystemExit(
            f"Niepoprawna nazwa pliku: {name}. Oczekiwany wzorzec: trades_{SYMBOL}_tp{TP}_sl{SL}_thr{THR}_{YYYYMMDD_HHMMSS}.json"
        )
    symbol = m.group("symbol")
    tp = float(m.group("tp"))
    sl = float(m.group("sl"))
    thr_pct = float(m.group("thr"))
    return symbol, tp, sl, thr_pct


def load_trades_json(file_path: Path) -> Tuple[str, float, float, float, List[Trade]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    symbol = data.get("symbol")
    tp = float(data.get("tp"))
    sl = float(data.get("sl"))
    thr = float(data.get("confidence_threshold"))  # 0..1
    trades_raw = data.get("trades", [])
    trades: List[Trade] = []
    for t in trades_raw:
        trades.append(
            Trade(
                timestamp=str(t.get("timestamp")),
                signal=str(t.get("signal")),
                probability=float(t.get("probability")),
                true_label=str(t.get("true_label")),
                result=str(t.get("result")),
            )
        )
    return symbol, tp, sl, thr, trades


def compute_metrics(tp: float, sl: float, trades: List[Trade]) -> Dict[str, float]:
    """Compute summary metrics.
    dochod_netto_per_trade = p_win * tp - (1 - p_win) * sl
    Precision per direction measured as wins / (wins + losses) for predictions of that direction.
    """
    total = len(trades)
    wins = sum(1 for t in trades if t.result == "WIN")
    losses = sum(1 for t in trades if t.result == "LOSS")
    p_win_overall = wins / total if total > 0 else 0.0

    # Directional effectiveness (only on predicted LONG/SHORT, trades should already be filtered as such)
    long_trades = [t for t in trades if t.signal == "LONG"]
    short_trades = [t for t in trades if t.signal == "SHORT"]
    long_total = len(long_trades)
    short_total = len(short_trades)
    long_wins = sum(1 for t in long_trades if t.result == "WIN")
    short_wins = sum(1 for t in short_trades if t.result == "WIN")
    long_precision = long_wins / long_total if long_total > 0 else 0.0
    short_precision = short_wins / short_total if short_total > 0 else 0.0

    # Profitability per trade (%)
    net_income_perc = p_win_overall * tp - (1.0 - p_win_overall) * sl

    # Break-even and margin
    breakeven = sl / (tp + sl) if (tp + sl) > 0 else 0.0
    safety_margin = p_win_overall - breakeven

    return {
        "n_trades": float(total),
        "wins": float(wins),
        "losses": float(losses),
        "p_win_overall": p_win_overall,
        "long_precision": long_precision,
        "short_precision": short_precision,
        "breakeven": breakeven,
        "safety_margin": safety_margin,
        "net_income_perc": net_income_perc,
    }


def simulate_equity(
    trades: List[Trade],
    tp: float,
    sl: float,
    start_balance: float,
    position_frac: float,
    fee_entry_pct: float,
    fee_exit_win_pct: float,
    fee_exit_loss_pct: float,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Simulate sequential equity curve using fraction of balance per trade and fees.

    - Each trade uses position_size = balance_before * position_frac
    - ROI gross: +tp% for WIN, -sl% for LOSS (regardless of LONG/SHORT)
    - Fees: asymetryczne
      * WIN: total_fee_pct = fee_entry_pct + fee_exit_win_pct
      * LOSS: total_fee_pct = fee_entry_pct + fee_exit_loss_pct
    - Net ROI = ROI_gross - total_fee_pct
    - balance_after = balance_before + position_size * (net_roi/100)
    Returns list of points and summary stats (min/max/final).
    """
    balance = float(start_balance)
    min_balance = balance
    max_balance = balance
    min_idx = 0
    max_idx = 0
    curve: List[Dict[str, float]] = []

    # Ensure chronological order by timestamp string (ISO-like)
    trades_sorted = sorted(trades, key=lambda t: str(t.timestamp))

    for i, t in enumerate(trades_sorted):
        balance_before = balance
        position_size = balance_before * position_frac
        roi_gross = (tp if t.result == "WIN" else -sl)
        fee_total_pct = (fee_entry_pct + fee_exit_win_pct) if t.result == "WIN" else (fee_entry_pct + fee_exit_loss_pct)
        roi_net = roi_gross - fee_total_pct
        pnl = position_size * (roi_net / 100.0)
        balance = balance_before + pnl

        if balance < min_balance:
            min_balance = balance
            min_idx = i
        if balance > max_balance:
            max_balance = balance
            max_idx = i

        curve.append(
            {
                "i": i,
                "timestamp": t.timestamp,
                "signal": 1.0 if t.signal == "LONG" else -1.0,
                "result": 1.0 if t.result == "WIN" else 0.0,
                "roi_gross_pct": roi_gross,
                "fee_total_pct": fee_total_pct,
                "roi_net_pct": roi_net,
                "balance_before": balance_before,
                "position_frac": position_frac,
                "position_size": position_size,
                "pnl_abs": pnl,
                "balance_after": balance,
            }
        )

    total_return_pct = ((balance / start_balance) - 1.0) * 100.0 if start_balance > 0 else 0.0
    summary = {
        "start_balance": start_balance,
        "final_balance": balance,
        "min_balance": min_balance,
        "min_balance_index": float(min_idx),
        "max_balance": max_balance,
        "max_balance_index": float(max_idx),
        "total_return_pct": total_return_pct,
    }
    return curve, summary


def save_outputs(
    symbol: str,
    tp: float,
    sl: float,
    thr_pct: float,
    metrics: Dict[str, float],
    out_dir: Path,
    equity_summary: Dict[str, float],
    equity_curve: List[Dict[str, float]],
    params: Optional[Dict[str, float]] = None,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON summary
    json_path = out_dir / f"income_{symbol}_tp{tp}_sl{sl}_thr{int(thr_pct)}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": symbol,
                "tp": tp,
                "sl": sl,
                "thr_percent": thr_pct,
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
                "equity": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in equity_summary.items()},
                "params": params or {},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Markdown summary
    md_path = out_dir / f"income_{symbol}_tp{tp}_sl{sl}_thr{int(thr_pct)}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"Symbol: {symbol}\n")
        f.write(f"TP: {tp:.1f}%, SL: {sl:.1f}%, Próg pewności: {thr_pct:.0f}%\n\n")
        f.write(f"Liczba transakcji: {int(metrics['n_trades'])}\n")
        f.write(f"Wygrane: {int(metrics['wins'])}, Przegrane: {int(metrics['losses'])}\n")
        f.write(f"Skuteczność ogółem: {metrics['p_win_overall']*100:.2f}%\n")
        f.write(f"Skuteczność LONG: {metrics['long_precision']*100:.2f}%\n")
        f.write(f"Skuteczność SHORT: {metrics['short_precision']*100:.2f}%\n")
        f.write(f"Próg opłacalności: {metrics['breakeven']*100:.2f}%\n")
        f.write(f"Marża bezpieczeństwa: {metrics['safety_margin']*100:.2f}%\n")
        f.write(f"Dochód netto (per transakcję): ~{metrics['net_income_perc']:.3f}%\n")
        f.write("\n")
        f.write("Symulacja salda (po kolei):\n")
        f.write(f"- Saldo początkowe: {equity_summary['start_balance']:.2f}\n")
        f.write(f"- Saldo końcowe: {equity_summary['final_balance']:.2f}\n")
        f.write(f"- Minimum salda: {equity_summary['min_balance']:.2f} (idx {int(equity_summary['min_balance_index'])})\n")
        f.write(f"- Maksimum salda: {equity_summary['max_balance']:.2f} (idx {int(equity_summary['max_balance_index'])})\n")
        f.write(f"- Zwrot łączny: {equity_summary['total_return_pct']:.2f}%\n")
        if params:
            f.write("\nUżyte parametry:\n")
            f.write(f"- Ułamek salda na transakcję: {params.get('position_frac')}\n")
            f.write(f"- Prowizja wejście: {params.get('fee_entry')}%\n")
            f.write(f"- Prowizja wyjście (TP): {params.get('fee_exit_win')}%\n")
            f.write(f"- Prowizja wyjście (SL): {params.get('fee_exit_loss')}%\n")

    # Equity curve CSV
    csv_path = out_dir / f"equity_curve_{symbol}_tp{tp}_sl{sl}_thr{int(thr_pct)}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = [
            "i",
            "timestamp",
            "signal",
            "result",
            "roi_gross_pct",
            "fee_total_pct",
            "roi_net_pct",
            "balance_before",
            "position_frac",
            "position_size",
            "pnl_abs",
            "balance_after",
        ]
        f.write(",".join(headers) + "\n")
        for row in equity_curve:
            values = [row.get(h) for h in headers]
            f.write(",".join(str(v) for v in values) + "\n")
    return md_path, json_path, csv_path


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Income calculator: wczytuje plik trades_*.json, oblicza metryki i zapisuje podsumowanie."
        )
    )
    parser.add_argument(
        "--config",
        default=str(Path("income_calculator") / "config.json"),
        help="Ścieżka do pliku konfiguracyjnego JSON (domyślnie income_calculator/config.json)",
    )
    parser.add_argument(
        "--file",
        required=True,
        help=(
            "Ścieżka do pliku trades_*.json (np. crypto/training5/output/reports/BTCUSDT/trades_BTCUSDT_tp0.8_sl0.2_thr40_YYYYMMDD_HHMMSS.json)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path("income_calculator") / "output"),
        help="Katalog wyjściowy na podsumowania (JSON/MD)",
    )
    parser.add_argument(
        "--start_balance",
        type=float,
        default=None,
        help="Saldo początkowe. Jeśli nie podasz, zostanie pobrane z configu lub domyślne 1000.0",
    )
    parser.add_argument(
        "--position_frac",
        type=float,
        default=None,
        help="Jaka część salda wchodzi w pojedynczą transakcję (0..1). Jeśli nie podasz, z configu lub 0.2",
    )
    # Asymetryczne prowizje (w %)
    parser.add_argument(
        "--fee_entry",
        type=float,
        default=None,
        help="Prowizja na wejście w %. Jeśli nie podasz, z configu lub 0.018",
    )
    parser.add_argument(
        "--fee_exit_win",
        type=float,
        default=None,
        help="Prowizja na wyjście w % dla transakcji zakończonej TP. Jeśli nie podasz, z configu lub 0.018",
    )
    parser.add_argument(
        "--fee_exit_loss",
        type=float,
        default=None,
        help="Prowizja na wyjście w % dla transakcji zakończonej SL. Jeśli nie podasz, z configu lub 0.045",
    )
    # Dla kompatybilności: jednolita stawka per strona (opcjonalnie). Jeśli podana, nadpisze powyższe fee_exit_win=fee_exit_loss=fee_entry=fee_per_side.
    parser.add_argument(
        "--fee_per_side",
        type=float,
        default=None,
        help="(Opcjonalne) Jednolita prowizja per strona w %, jeśli podana zastępuje fee_entry/fee_exit_win/fee_exit_loss",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"Plik nie istnieje: {file_path}")

    symbol_f, tp_f, sl_f, thr_pct = parse_filename_metadata(file_path)
    symbol_j, tp_j, sl_j, thr_j, trades = load_trades_json(file_path)

    # Walidacja spójności metadanych
    if symbol_j and symbol_j != symbol_f:
        print(
            f"Ostrzeżenie: symbol w nazwie pliku ({symbol_f}) != w JSON ({symbol_j}). Używam tego z JSON.",
            file=sys.stderr,
        )
        symbol = symbol_j
    else:
        symbol = symbol_f

    if abs(tp_f - tp_j) > 1e-6 or abs(sl_f - sl_j) > 1e-6:
        print(
            f"Ostrzeżenie: TP/SL w nazwie pliku ({tp_f}/{sl_f}) != w JSON ({tp_j}/{sl_j}). Używam wartości z JSON.",
            file=sys.stderr,
        )
        tp, sl = tp_j, sl_j
    else:
        tp, sl = tp_f, sl_f

    # Wczytaj config
    cfg_path = Path(args.config)
    cfg = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"Ostrzeżenie: nie udało się wczytać configu {cfg_path}: {e}", file=sys.stderr)

    # Ustal parametry z kolejnością priorytetu: CLI -> config -> domyślne
    start_balance = (
        float(args.start_balance)
        if args.start_balance is not None
        else float(cfg.get("start_balance", 1000.0))
    )
    position_frac = (
        float(args.position_frac)
        if args.position_frac is not None
        else float(cfg.get("position_frac", 0.2))
    )

    # Ustal opłaty: fee_per_side nadpisuje wszystkie
    if args.fee_per_side is not None:
        fee_entry = float(args.fee_per_side)
        fee_exit_win = float(args.fee_per_side)
        fee_exit_loss = float(args.fee_per_side)
    else:
        fee_entry = (
            float(args.fee_entry) if args.fee_entry is not None else float(cfg.get("fee_entry", 0.018))
        )
        fee_exit_win = (
            float(args.fee_exit_win)
            if args.fee_exit_win is not None
            else float(cfg.get("fee_exit_win", 0.018))
        )
        fee_exit_loss = (
            float(args.fee_exit_loss)
            if args.fee_exit_loss is not None
            else float(cfg.get("fee_exit_loss", 0.045))
        )

    metrics = compute_metrics(tp=tp, sl=sl, trades=trades)

    equity_curve, equity_summary = simulate_equity(
        trades=trades,
        tp=tp,
        sl=sl,
        start_balance=start_balance,
        position_frac=position_frac,
        fee_entry_pct=fee_entry,
        fee_exit_win_pct=fee_exit_win,
        fee_exit_loss_pct=fee_exit_loss,
    )

    out_dir = Path(args.output_dir) / symbol
    params = {
        "start_balance": start_balance,
        "position_frac": position_frac,
        "fee_entry": fee_entry,
        "fee_exit_win": fee_exit_win,
        "fee_exit_loss": fee_exit_loss,
    }

    md_path, json_path, csv_path = save_outputs(
        symbol=symbol,
        tp=tp,
        sl=sl,
        thr_pct=thr_pct,
        metrics=metrics,
        out_dir=out_dir,
        equity_summary=equity_summary,
        equity_curve=equity_curve,
        params=params,
    )

    print(f"Zapisano podsumowania: {md_path} | {json_path} | {csv_path}")


if __name__ == "__main__":
    run_cli()

