from datetime import datetime
from pathlib import Path
import json


def _get_tp_sl_for_level(cfg, level_index: int):
    tp, sl = cfg.TP_SL_LEVELS[level_index]
    return float(tp), float(sl)


def _profit_threshold(tp: float, sl: float) -> float:
    return sl / (sl + tp) * 100.0 if (sl + tp) > 0 else 0.0


def _weighted_trade_accuracy(cm, class_report) -> float:
    """Weighted precision over predicted LONG/SHORT trades only.

    We weight by counts of predicted trades per side (column sums), and we
    exclude any side with zero predictions from both numerator and denominator.
    This avoids penalizing the metric when a side has no predictions (precision
    reported as 0.0 by some reporters).
    """
    # Column sums = number of predictions for a given class
    col_long = cm[0][0] + cm[1][0] + cm[2][0]
    col_short = cm[0][1] + cm[1][1] + cm[2][1]

    # Compute precision directly from CM to avoid reporter quirks
    p_long = (cm[0][0] / col_long) if col_long > 0 else None
    p_short = (cm[1][1] / col_short) if col_short > 0 else None

    numerator = 0.0
    denominator = 0
    if col_long and p_long is not None:
        numerator += col_long * p_long
        denominator += col_long
    if col_short and p_short is not None:
        numerator += col_short * p_short
        denominator += col_short

    if denominator == 0:
        return 0.0
    return numerator / float(denominator)


def _calc_profit_from_cm(cm, tp: float, sl: float, signal_type: str) -> float:
    """Per-trade expected net profit (%), using precision as win rate.

    profit = p_win * TP - (1 - p_win) * SL
    """
    if tp <= 0 or sl <= 0:
        return 0.0
    if signal_type == 'LONG':
        profitable = cm[0][0]
        total = cm[0][0] + cm[1][0] + cm[2][0]
    elif signal_type == 'SHORT':
        profitable = cm[1][1]
        total = cm[0][1] + cm[1][1] + cm[2][1]
    else:
        return 0.0
    if total == 0:
        return 0.0
    p_win = profitable / total
    return p_win * tp - (1.0 - p_win) * sl


def _calc_combined_profit_from_cm(cm, tp: float, sl: float) -> float:
    """Weighted average of per-trade profits across LONG and SHORT, weighted by number of trades per side."""
    col_long = cm[0][0] + cm[1][0] + cm[2][0]
    col_short = cm[0][1] + cm[1][1] + cm[2][1]
    if (col_long + col_short) == 0:
        return 0.0
    p_long = _calc_profit_from_cm(cm, tp, sl, 'LONG')
    p_short = _calc_profit_from_cm(cm, tp, sl, 'SHORT')
    return (p_long * col_long + p_short * col_short) / (col_long + col_short)


def save_markdown_report(evaluation_results: dict, model_params: dict, data_info: dict, cfg, symbol: str, out_dir: Path | None = None) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rep_dir = out_dir if out_dir is not None else cfg.get_report_dir(symbol)
    out = rep_dir / f"results_{symbol}_{ts}.md"

    with open(out, 'w', encoding='utf-8') as f:
        f.write(f"# WYNIKI TRENINGU - {symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Wykorzystane cechy
        feature_names = data_info.get('feature_names') or []
        if feature_names:
            f.write("## 🧩 WYKORZYSTANE CECHY\n")
            f.write(f"Liczba cech: {len(feature_names)}\n\n")
            for name in feature_names:
                f.write(f"- {name}\n")
            f.write("\n")

        # Parametry modelu
        f.write("## 📊 PARAMETRY MODELU\n")
        for k, v in model_params.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        # Informacje o danych
        f.write("## 📋 INFORMACJE O DANYCH\n")
        for k, v in data_info.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        # Best validation mlogloss per level (if provided)
        best_map = data_info.get('best_validation_mlogloss')
        if isinstance(best_map, dict) and best_map:
            f.write("## 🔎 NAJLEPSZE WARTOŚCI VALIDATION MLOGLOSS (per poziom)\n")
            # Expect keys are label columns in order of levels
            for col, d in best_map.items():
                try:
                    lvl_idx = int(evaluation_results.get(col, {}).get('level_index', 0))
                except Exception:
                    lvl_idx = 0
                tp, sl = _get_tp_sl_for_level(cfg, lvl_idx)
                best_ll = d.get('best_mlogloss')
                best_it = d.get('best_iteration')
                f.write(f"- TP={tp:.1f}%, SL={sl:.1f}% [{col}]: best_mlogloss={best_ll} @ iter {best_it}\n")
            f.write("\n")

        # Wyniki per poziom
        f.write("## 📈 WYNIKI DLA KAŻDEGO POZIOMU TP/SL\n\n")
        for col, res in evaluation_results.items():
            lvl_idx = int(res.get('level_index', 0))
            tp, sl = _get_tp_sl_for_level(cfg, lvl_idx)
            f.write(f"### TP: {tp:.1f}%, SL: {sl:.1f}%:\n")
            acc = res.get('accuracy', 0)
            cls = res.get('classification_report', {})
            cm = res.get('confusion_matrix')

            if cm is not None:
                total_samples = sum(sum(row) for row in cm)
                long_count = sum(cm[0])
                short_count = sum(cm[1])
                neutral_count = sum(cm[2])
                f.write("Stosunek wszystkich etykiet w zbiorze testowym\n")
                f.write(f"LONG {long_count/total_samples*100:.0f}%\n")
                f.write(f"SHORT {short_count/total_samples*100:.0f}%\n")
                f.write(f"NEUTRAL {neutral_count/total_samples*100:.0f}%\n\n")

            f.write("Standardowe metryki (bez progów):\n")

            if cm is not None:
                f.write("Predicted\n")
                f.write("Actual    LONG  SHORT  NEUTRAL\n")
                f.write(f"LONG      {cm[0][0]:<6} {cm[0][1]:<6} {cm[0][2]:<6}\n")
                f.write(f"SHORT     {cm[1][0]:<6} {cm[1][1]:<6} {cm[1][2]:<6}\n")
                f.write(f"NEUTRAL   {cm[2][0]:<6} {cm[2][1]:<6} {cm[2][2]:<6}\n\n")

            f.write(f"Accuracy: {acc:.4f}\n")

            for cname in ['LONG', 'SHORT', 'NEUTRAL']:
                if cname in cls:
                    prec = cls[cname].get('precision', 0)
                    rec = cls[cname].get('recall', 0)
                    f1 = cls[cname].get('f1-score', 0)
                    f.write(f"{cname}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}\n")

            if cm is not None:
                wacc = _weighted_trade_accuracy(cm, cls)
                thr = _profit_threshold(tp, sl)
                f.write(f"Accuracy SHORT+LONG (ważona): {wacc:.4f}\n")
                f.write(f"Próg opłacalności: {thr:.2f}%\n")
                if wacc > 0:
                    safety = (wacc * 100.0) - thr
                    f.write(f"Marża bezpieczeństwa: {safety:.2f}%\n\n")
                else:
                    f.write("Marża bezpieczeństwa: BRAK SYGNAŁÓW (model nie przewiduje transakcji)\n\n")

                long_prec = cls.get('LONG', {}).get('precision', 0)
                short_prec = cls.get('SHORT', {}).get('precision', 0)
                long_profit = _calc_profit_from_cm(cm, tp, sl, 'LONG')
                short_profit = _calc_profit_from_cm(cm, tp, sl, 'SHORT')
                combined_profit = _calc_combined_profit_from_cm(cm, tp, sl)
                f.write(f"TP: {tp:.1f}%, SL: {sl:.1f}% - LONG: {long_prec*100:.1f}% (dochód netto ~{long_profit:.3f}%)\n")
                f.write(f"TP: {tp:.1f}%, SL: {sl:.1f}% - SHORT: {short_prec*100:.1f}% (dochód netto ~{short_profit:.3f}%)\n")
                f.write(f"TP: {tp:.1f}%, SL: {sl:.1f}% - ŁĄCZNY DOCHÓD: {combined_profit:.3f}%\n\n")
            # Confidence thresholds blocks (30/40/45/50%)
            conf = res.get('confidence_results', {})
            for thr_percent in [30.0, 40.0, 45.0, 50.0]:
                thr = thr_percent / 100.0
                f.write(f"\nProgi pewności {thr_percent:.1f}%:\n")
                if conf.get(thr) is None:
                    f.write("Brak próbek z tak wysoką pewnością\n")
                    continue
                cres = conf[thr]
                cmh = cres.get('confusion_matrix')
                if cmh is not None:
                    f.write("Predicted\n")
                    f.write("Actual    LONG  SHORT  NEUTRAL\n")
                    f.write(f"LONG      {cmh[0][0]:<6} {cmh[0][1]:<6} {cmh[0][2]:<6}\n")
                    f.write(f"SHORT     {cmh[1][0]:<6} {cmh[1][1]:<6} {cmh[1][2]:<6}\n")
                    f.write(f"NEUTRAL   {cmh[2][0]:<6} {cmh[2][1]:<6} {cmh[2][2]:<6}\n\n")
                f.write(f"Próbki z wysoką pewnością: {cres['n_high_conf']:,}/{cres['n_total']:,} ({cres['percentage']:.1f}%)\n")
                f.write(f"Accuracy: {cres.get('accuracy', 0):.4f}\n")
                cr = cres.get('classification_report', {})
                for cname in ['LONG', 'SHORT', 'NEUTRAL']:
                    if cname in cr:
                        prec = cr[cname].get('precision', 0)
                        rec = cr[cname].get('recall', 0)
                        f1 = cr[cname].get('f1-score', 0)
                        f.write(f"{cname}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}\n")
                # Extended profitability/threshold metrics within the confidence block
                if cmh is not None:
                    wacc_h = _weighted_trade_accuracy(cmh, cr)
                    thr_profit = _profit_threshold(tp, sl)
                    f.write(f"\nAccuracy SHORT+LONG (ważona): {wacc_h:.4f}\n")
                    f.write(f"Próg opłacalności: {thr_profit:.2f}%\n")
                    if wacc_h > 0:
                        safety_h = (wacc_h * 100.0) - thr_profit
                        f.write(f"Marża bezpieczeństwa: {safety_h:.2f}%\n")
                    else:
                        f.write("Marża bezpieczeństwa: BRAK SYGNAŁÓW (model nie przewiduje transakcji)\n")

                    # Per-trade expected profits
                    long_prec_h = cr.get('LONG', {}).get('precision', 0)
                    short_prec_h = cr.get('SHORT', {}).get('precision', 0)
                    long_profit_h = _calc_profit_from_cm(cmh, tp, sl, 'LONG')
                    short_profit_h = _calc_profit_from_cm(cmh, tp, sl, 'SHORT')
                    combined_profit_h = _calc_combined_profit_from_cm(cmh, tp, sl)
                    f.write(f"\nTP: {tp:.1f}%, SL: {sl:.1f}% - LONG: {long_prec_h*100:.1f}% (dochód netto ~{long_profit_h:.3f}%)\n")
                    f.write(f"TP: {tp:.1f}%, SL: {sl:.1f}% - SHORT: {short_prec_h*100:.1f}% (dochód netto ~{short_profit_h:.3f}%)\n")
                    f.write(f"TP: {tp:.1f}%, SL: {sl:.1f}% - ŁĄCZNY DOCHÓD: {combined_profit_h:.3f}%\n")
            f.write("\n" + "-" * 68 + "\n\n")

    return out


def save_json_report(evaluation_results: dict, model_params: dict, data_info: dict, cfg, symbol: str, out_dir: Path | None = None) -> Path:
    """Save a unified JSON report mirroring the markdown content structure.

    Output path: reports/{symbol}/results_{symbol}_{timestamp}.json
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rep_dir = out_dir if out_dir is not None else cfg.get_report_dir(symbol)
    out = rep_dir / f"results_{symbol}_{ts}.json"

    payload = {
        "symbol": symbol,
        "generated_at": datetime.now().isoformat(),
        "model_params": model_params,
        "data_info": data_info,
        "evaluation_results": evaluation_results,
    }

    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out

