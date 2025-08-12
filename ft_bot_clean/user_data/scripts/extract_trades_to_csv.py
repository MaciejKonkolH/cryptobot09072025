import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


TRADE_KEYS_PRIMARY = {
    "open_date", "close_date", "open_rate", "close_rate"
}


def _find_trades_list(obj: Any) -> List[Dict[str, Any]]:
    """Recursively search for a list of trade-like dicts in freqtrade backtest JSON."""
    candidates: List[List[Dict[str, Any]]] = []

    def is_trade_list(node: Any) -> bool:
        if not isinstance(node, list) or not node:
            return False
        first = node[0]
        if not isinstance(first, dict):
            return False
        keys = set(first.keys())
        # Heuristic: at least 3 of primary keys present, or typical fields
        if len(keys & TRADE_KEYS_PRIMARY) >= 3:
            return True
        if {"open_time", "close_time"} <= keys and {"open_price", "close_price"} & keys:
            return True
        if {"open_date", "close_date"} <= keys and {"open_price", "close_price"} & keys:
            return True
        if {"open_timestamp", "close_timestamp"} <= keys and {"open_rate", "close_rate"} & keys:
            return True
        return False

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            if is_trade_list(node):
                candidates.append(node)  # type: ignore[arg-type]
            else:
                for v in node:
                    visit(v)

    visit(obj)
    if not candidates:
        raise ValueError("Could not locate trades list in JSON.")
    # Pick the longest candidate
    return max(candidates, key=len)


def _get_first(trade: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in trade and trade[k] is not None:
            return trade[k]
    return default


def _trade_to_row(t: Dict[str, Any]) -> Dict[str, Any]:
    open_dt = _get_first(t, ["open_date", "date_open", "open_time"], None)
    close_dt = _get_first(t, ["close_date", "date_close", "close_time"], None)

    open_rate = _get_first(t, ["open_rate", "open_price", "entry_price"], None)
    close_rate = _get_first(t, ["close_rate", "close_price", "exit_price"], None)

    is_short: Optional[bool] = None
    if "is_short" in t:
        is_short = bool(t["is_short"])
    else:
        direction = str(_get_first(t, ["direction", "trade_type", "type", "side"], "")).lower()
        is_short = direction == "short"

    profit_ratio = _get_first(t, ["profit_ratio", "profit_percent"], None)
    profit_abs = _get_first(t, ["profit_abs", "profit", "profit_usdt"], None)
    exit_reason = _get_first(t, ["exit_reason", "sell_reason"], None)
    pair = _get_first(t, ["pair", "symbol"], None)

    # Determine result label
    result: Optional[str] = None
    if profit_ratio is not None:
        try:
            result = "WIN" if float(profit_ratio) > 0 else "LOSS" if float(profit_ratio) < 0 else "DRAW"
        except Exception:
            result = None
    if result is None and profit_abs is not None:
        try:
            result = "WIN" if float(profit_abs) > 0 else "LOSS" if float(profit_abs) < 0 else "DRAW"
        except Exception:
            result = None
    if result is None and open_rate is not None and close_rate is not None:
        try:
            open_f = float(open_rate)
            close_f = float(close_rate)
            move = (open_f - close_f) if is_short else (close_f - open_f)
            result = "WIN" if move > 0 else "LOSS" if move < 0 else "DRAW"
        except Exception:
            result = "UNKNOWN"

    return {
        "pair": pair,
        "trade_type": "SHORT" if is_short else "LONG",
        "result": result or "UNKNOWN",
        "entry_time": open_dt,
        "exit_time": close_dt,
        "entry_price": open_rate,
        "exit_price": close_rate,
        "profit_abs": profit_abs,
        "profit_ratio": profit_ratio,
        "exit_reason": exit_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract trades from Freqtrade backtest JSON to CSV")
    parser.add_argument("json_path", type=str,
                        help="Path to backtest-result-*.json (inside extracted folder)")
    parser.add_argument("--out", dest="out_csv", type=str, default=None,
                        help="Output CSV path (default: alongside JSON as trades.csv)")
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    if not json_path.exists():
        raise SystemExit(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trades = _find_trades_list(data)
    rows = [_trade_to_row(t) for t in trades]
    df = pd.DataFrame(rows)

    # Normalize datetime columns to ISO strings
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S%z")
            except Exception:
                pass

    out_csv = Path(args.out_csv).resolve() if args.out_csv else json_path.with_name("trades.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved trades to {out_csv} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

