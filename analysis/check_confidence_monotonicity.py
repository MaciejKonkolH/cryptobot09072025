import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def find_latest_report_json(base_dir: Path, symbol: str) -> Optional[Path]:
    reports_dir = base_dir / "output" / "reports" / symbol
    if not reports_dir.exists():
        return None
    # Najczęstszy format: results_{SYMBOL}_YYYY-MM-DD_HH-MM-SS.json
    candidates = sorted(reports_dir.glob(f"results_{symbol}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    # Fallback: any json in symbol dir
    candidates = sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_report(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _profit_threshold(tp: float, sl: float) -> float:
    # returns fraction (0..1)
    denom = (tp or 0.0) + (sl or 0.0)
    return (sl / denom) if denom > 0 else 0.0


def _predicted_trades_count_from_cm(cm: List[List[int]]) -> int:
    # sum of predicted LONG and SHORT columns
    if not cm:
        return 0
    col_long = sum(row[0] for row in cm)
    col_short = sum(row[1] for row in cm)
    return int(col_long + col_short)


def _precision_from_cm(cm: List[List[int]]) -> float:
    if not cm:
        return 0.0
    tp_long = cm[0][0]
    tp_short = cm[1][1]
    pred_long = sum(row[0] for row in cm)
    pred_short = sum(row[1] for row in cm)
    denom = pred_long + pred_short
    return ((tp_long + tp_short) / denom) if denom > 0 else 0.0


def extract_precision_series(report: Dict) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Return per-level series of (threshold, long_short_precision) for monotonicity check."""
    series: List[Tuple[str, List[Tuple[float, float]]]] = []
    for level in report.get("levels", []):
        tp = level.get("tp_pct")
        sl = level.get("sl_pct")
        label = f"TP: {tp:.1f}%, SL: {sl:.1f}%" if isinstance(tp, (int, float)) and isinstance(sl, (int, float)) else level.get("level_index", "?")
        conf_list = level.get("confidence_levels", level.get("confidence", []))
        points: List[Tuple[float, float]] = []
        for c in conf_list:
            thr = float(c.get("threshold", 0.0))
            # Keep only thresholds: 0.3, 0.4, 0.45, 0.5
            if abs(thr - 0.3) > 1e-9 and abs(thr - 0.4) > 1e-9 and abs(thr - 0.45) > 1e-9 and abs(thr - 0.5) > 1e-9:
                continue
            ls = c.get("classification_report", {}).get("LONG_SHORT", {})
            prec = ls.get("precision")
            if prec is None:
                cm = c.get("confusion_matrix") or []
                prec = _precision_from_cm(cm)
            if prec is None:
                continue
            points.append((thr, float(prec)))
        points.sort(key=lambda x: x[0])
        series.append((label, points))
    return series


def is_non_decreasing(values: List[float]) -> bool:
    for i in range(1, len(values)):
        if values[i] < values[i - 1] - 1e-12:
            return False
    return True


def analyze_report(path: Path, allow_equal: bool = True, csv_rows: Optional[list] = None, json_acc: Optional[list] = None) -> int:
    report = load_report(path)
    print(f"Analiza raportu: {path}")
    series = extract_precision_series(report)
    total = len(series)
    passed = 0
    json_report_obj = {
        'report': str(path),
        'summary': {
            'num_levels_total': total,
            'num_levels_pass': 0,
        },
        'levels': []
    }
    for idx, (label, points) in enumerate(series):
        # per-level additional details
        level = report.get("levels", [])[idx]
        tp = float(level.get("tp_pct", 0.0))
        sl = float(level.get("sl_pct", 0.0))
        thr_profit = _profit_threshold(tp, sl)  # fraction
        conf_list = level.get("confidence_levels", level.get("confidence", []))

        if not points:
            print(f"- {label}: BRAK DANYCH")
            json_level = {
                'label': label,
                'tp_pct': tp,
                'sl_pct': sl,
                'monotonic_pass': False,
                'thresholds': []
            }
            json_report_obj['levels'].append(json_level)
            continue
        thresholds = [p[0] for p in points]
        precisions = [p[1] for p in points]
        ok = is_non_decreasing(precisions) if allow_equal else all(precisions[i] > precisions[i - 1] for i in range(1, len(precisions)))
        status = "PASS" if ok else "FAIL"
        # Build detailed lines per threshold
        details: List[str] = []
        # Map from threshold to cm to get trades and compute precision if needed
        cm_by_thr: Dict[float, List[List[int]]] = {}
        for c in conf_list:
            thr = float(c.get("threshold", 0.0))
            if abs(thr - 0.3) > 1e-9 and abs(thr - 0.4) > 1e-9 and abs(thr - 0.45) > 1e-9 and abs(thr - 0.5) > 1e-9:
                continue
            cm_by_thr[thr] = c.get("confusion_matrix") or []
        json_level = {
            'label': label,
            'tp_pct': tp,
            'sl_pct': sl,
            'monotonic_pass': bool(ok),
            'thresholds': []
        }
        for thr, prec in points:
            cm = cm_by_thr.get(thr, [])
            trades = _predicted_trades_count_from_cm(cm)
            margin = prec - thr_profit
            details.append(f"{thr*100:.0f}%:prec={prec:.3f},trades={trades},thr={thr_profit:.3f},margin={margin:.3f}")
            if csv_rows is not None:
                csv_rows.append({
                    'report': str(path),
                    'level_label': label,
                    'tp_pct': tp,
                    'sl_pct': sl,
                    'threshold': thr,
                    'precision': prec,
                    'trades': trades,
                    'thr_profit': thr_profit,
                    'margin': margin,
                    'monotonic_pass': int(ok),
                })
            json_level['thresholds'].append({
                'threshold': thr,
                'precision': prec,
                'trades': trades,
                'thr_profit': thr_profit,
                'margin': margin,
            })
        print(f"- {label}: {status}  |  " + "; ".join(details))
        if ok:
            passed += 1
        json_report_obj['levels'].append(json_level)
    print(f"Wynik: {passed}/{total} poziomów spełnia warunek niemalejącego LONG_SHORT precision wraz z rosnącym confidence.")
    json_report_obj['summary']['num_levels_pass'] = passed
    if json_acc is not None:
        json_acc.append(json_report_obj)
    return 0 if passed == total else 1


def main():
    parser = argparse.ArgumentParser(description="Sprawdza monotoniczność LONG_SHORT precision względem confidence thresholds w raportach JSON.")
    parser.add_argument("--json", dest="json_paths", nargs="*", help="Ścieżki do plików JSON raportów")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol do automatycznego wyszukania najnowszych raportów")
    parser.add_argument("--modules", choices=["training3", "training5", "both"], default="training3", help="Których modułów raporty analizować, jeśli nie podano --json")
    parser.add_argument("--strict", action="store_true", help="Wymagaj ściśle rosnącej precyzji (zamiast niemalejącej)")
    parser.add_argument("--csv", dest="csv_out", default=None, help="Ścieżka do pliku CSV na wynik analizy (opcjonalnie)")
    parser.add_argument("--json-out", dest="json_out", default=None, help="Ścieżka do pliku JSON na wynik analizy (opcjonalnie)")
    args = parser.parse_args()

    json_files: List[Path] = []
    if args.json_paths:
        json_files = [Path(p) for p in args.json_paths]
    else:
        if args.modules in ("training3", "both"):
            p = find_latest_report_json(Path("crypto/training3"), args.symbol)
            if p:
                json_files.append(p)
        if args.modules in ("training5", "both"):
            p = find_latest_report_json(Path("crypto/training5"), args.symbol)
            if p:
                json_files.append(p)

    if not json_files:
        print("Brak plików JSON do analizy. Podaj --json lub użyj --modules ze wskazanym symbolem.")
        raise SystemExit(2)

    exit_code = 0
    all_rows = [] if args.csv_out else None
    all_reports = [] if args.json_out else None
    for jf in json_files:
        if not jf.exists():
            print(f"Plik nie istnieje: {jf}")
            exit_code = 2
            continue
        code = analyze_report(jf, allow_equal=not args.strict, csv_rows=all_rows, json_acc=all_reports)
        if code != 0:
            exit_code = code
    if args.csv_out and all_rows is not None:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['report', 'level_label', 'tp_pct', 'sl_pct', 'threshold', 'precision', 'trades', 'thr_profit', 'margin', 'monotonic_pass']
        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Zapisano CSV: {out_path}")
    if args.json_out and all_reports is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(all_reports, f, ensure_ascii=False, indent=2)
        print(f"Zapisano JSON: {out_path}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

