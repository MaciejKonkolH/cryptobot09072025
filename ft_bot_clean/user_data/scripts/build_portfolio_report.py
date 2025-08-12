import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
# Charts will be created using XlsxWriter engine to avoid Excel repair issues


@dataclass
class Config:
    starting_balance: float
    stake_pct: float
    fee: float
    leverage: float


def load_trades(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    # Parse datetimes
    for c in ["entry_time", "exit_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    # Ensure required columns
    required = [
        "pair",
        "trade_type",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in trades CSV: {missing}")
    # Sort by entry_time then original order
    df = df.sort_values(by=["entry_time"]).reset_index(drop=True)
    return df


def simulate(df_trades: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Build a Sim dataframe with only raw inputs filled.
    All computed columns will be Excel formulas.
    """
    base_cols = [
        "pair",
        "trade_type",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
    ]
    sim = df_trades[base_cols].copy()
    computed_cols = [
        "stake_pct_used",
        "stake_usd",
        "qty",
        "entry_fee",
        "exit_fee",
        "pnl_gross",
        "pnl_net",
        "equity_at_entry",
        "equity_after_close",
    ]
    for c in computed_cols:
        sim[c] = None
    # Normalize datetimes
    for c in ["entry_time", "exit_time"]:
        if c in sim.columns:
            sim[c] = pd.to_datetime(sim[c], utc=True)
    return sim


def _make_excel_safe_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[c]):
            # Convert to UTC-naive
            out[c] = out[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def write_excel(trades: pd.DataFrame, sim: pd.DataFrame, cfg: Config, out_xlsx: Path) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    # Use XlsxWriter to build charts reliably
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as writer:
        # Ensure Excel recalculates all formulas on open
        writer.book.set_calc_mode('auto')
        try:
            writer.book.set_calc_on_load()
        except Exception:
            pass
        # Config sheet
        cfg_df = pd.DataFrame({
            "parameter": ["starting_balance", "stake_pct", "fee", "leverage"],
            "value": [cfg.starting_balance, cfg.stake_pct, cfg.fee, cfg.leverage],
        })
        cfg_df.to_excel(writer, index=False, sheet_name="Config")
        # Define named ranges for easier formulas
        wb = writer.book
        ws_cfg = writer.sheets["Config"]
        # Values end up in column B rows 2..5
        wb.define_name("starting_balance", "=Config!$B$2")
        wb.define_name("stake_pct", "=Config!$B$3")
        wb.define_name("fee", "=Config!$B$4")
        wb.define_name("leverage", "=Config!$B$5")

        # Trades sheet
        trades_xlsx = _make_excel_safe_datetime(trades)
        trades_xlsx.to_excel(writer, index=False, sheet_name="Trades")

        # Sim sheet
        sim_xlsx = _make_excel_safe_datetime(sim)
        sim_xlsx.to_excel(writer, index=False, sheet_name="Sim")
        # Overwrite computation columns with formulas bound to Config values, using header-based indexing
        ws_sim = writer.sheets["Sim"]
        nrows = len(sim_xlsx)
        headers = list(sim_xlsx.columns)
        col_idx = {name: headers.index(name) for name in headers}
        def col_letter(idx: int) -> str:
            # 0-based to Excel letters
            s = ""
            n = idx + 1
            while n:
                n, r = divmod(n - 1, 26)
                s = chr(65 + r) + s
            return s
        # Shortcuts for key columns
        c_trade_type = col_letter(col_idx["trade_type"])
        c_entry_time = col_letter(col_idx["entry_time"])
        c_exit_time = col_letter(col_idx["exit_time"])
        c_entry_price = col_letter(col_idx["entry_price"])
        c_exit_price = col_letter(col_idx["exit_price"])
        c_stake_pct_used = col_letter(col_idx["stake_pct_used"])
        c_stake_usd = col_letter(col_idx["stake_usd"])
        c_qty = col_letter(col_idx["qty"])
        c_entry_fee = col_letter(col_idx["entry_fee"])
        c_exit_fee = col_letter(col_idx["exit_fee"])
        c_pnl_gross = col_letter(col_idx["pnl_gross"])
        c_pnl_net = col_letter(col_idx["pnl_net"])
        c_equity_at_entry = col_letter(col_idx["equity_at_entry"])
        c_equity_after_close = col_letter(col_idx["equity_after_close"])

        # Set number formats for datetime columns
        date_fmt = writer.book.add_format({'num_format': 'yyyy-mm-dd hh:mm'})
        ws_sim.set_column(f"{c_entry_time}:{c_exit_time}", None, date_fmt)

        for i in range(nrows):
            r = i + 2
            ws_sim.write_formula(f"{c_stake_pct_used}{r}", "=stake_pct")
            # equity_at_entry
            ws_sim.write_formula(
                f"{c_equity_at_entry}{r}",
                f"=starting_balance + SUMIFS(${c_pnl_net}:${c_pnl_net},${c_exit_time}:${c_exit_time},\"<\"&{c_entry_time}{r}) - (SUMIFS(${c_entry_fee}:${c_entry_fee},${c_entry_time}:${c_entry_time},\"<\"&{c_entry_time}{r}) - SUMIFS(${c_entry_fee}:${c_entry_fee},${c_exit_time}:${c_exit_time},\"<\"&{c_entry_time}{r}))"
            )
            ws_sim.write_formula(f"{c_stake_usd}{r}", f"={c_stake_pct_used}{r}*{c_equity_at_entry}{r}")
            ws_sim.write_formula(f"{c_qty}{r}", f"={c_stake_usd}{r}*leverage/{c_entry_price}{r}")
            ws_sim.write_formula(f"{c_entry_fee}{r}", f"={c_qty}{r}*{c_entry_price}{r}*fee")
            ws_sim.write_formula(f"{c_exit_fee}{r}", f"={c_qty}{r}*{c_exit_price}{r}*fee")
            ws_sim.write_formula(
                f"{c_pnl_gross}{r}",
                f"=IF({c_trade_type}{r}=\"SHORT\",({c_entry_price}{r}-{c_exit_price}{r})*{c_qty}{r},({c_exit_price}{r}-{c_entry_price}{r})*{c_qty}{r})"
            )
            ws_sim.write_formula(f"{c_pnl_net}{r}", f"={c_pnl_gross}{r}-{c_entry_fee}{r}-{c_exit_fee}{r}")
            ws_sim.write_formula(
                f"{c_equity_after_close}{r}",
                f"=starting_balance + SUMIFS(${c_pnl_net}:${c_pnl_net},${c_exit_time}:${c_exit_time},\"<=\"&{c_exit_time}{r}) - (SUMIFS(${c_entry_fee}:${c_entry_fee},${c_entry_time}:${c_entry_time},\"<=\"&{c_exit_time}{r}) - SUMIFS(${c_entry_fee}:${c_entry_fee},${c_exit_time}:${c_exit_time},\"<=\"&{c_exit_time}{r}))"
            )

        # Equity sheet (formulas referencing Sim so Excel calculates)
        ws_equity = writer.book.add_worksheet("Equity")
        ws_equity.write(0, 0, "exit_time")
        ws_equity.write(0, 1, "equity_after_close")
        for i in range(nrows):
            r = i + 2
            ws_equity.write_formula(i + 1, 0, f"=Sim!{c_exit_time}{r}")
            ws_equity.write_formula(i + 1, 1, f"=Sim!{c_equity_after_close}{r}")

        # Add line chart on Equity sheet using full range
        workbook = writer.book
        worksheet = writer.sheets.get("Equity")
        if worksheet is not None and nrows > 0:
            last_row = nrows + 1  # header at row 1
            chart = workbook.add_chart({'type': 'line'})
            chart.set_title({'name': 'Equity over closures'})
            chart.set_y_axis({'name': 'Equity (USDT)'})
            chart.set_x_axis({'name': 'Close Time', 'num_format': 'yyyy-mm-dd hh:mm'})
            chart.add_series({
                'name': 'Equity',
                'categories': ['Equity', 1, 0, last_row, 0],
                'values':     ['Equity', 1, 1, last_row, 1],
            })
            chart.set_legend({'none': True})
            worksheet.insert_chart('E2', chart, {'x_scale': 2.0, 'y_scale': 1.2})


def main() -> int:
    parser = argparse.ArgumentParser(description="Build portfolio Excel report from trades CSV")
    parser.add_argument("trades_csv", type=str, help="Path to trades.csv")
    parser.add_argument("--out", dest="out_xlsx", type=str, default=None, help="Output .xlsx path")
    parser.add_argument("--starting-balance", type=float, default=1_000_000.0)
    parser.add_argument("--stake-pct", type=float, default=0.05)
    parser.add_argument("--fee", type=float, default=0.00045)
    parser.add_argument("--leverage", type=float, default=1.0)
    args = parser.parse_args()

    trades_csv = Path(args.trades_csv).resolve()
    if not trades_csv.exists():
        raise SystemExit(f"Trades CSV not found: {trades_csv}")
    out_xlsx = (
        Path(args.out_xlsx).resolve()
        if args.out_xlsx
        else trades_csv.parent.parent / ".." / "reports" / "portfolio_report.xlsx"
    )
    out_xlsx = out_xlsx.resolve()

    cfg = Config(
        starting_balance=float(args.starting_balance),
        stake_pct=float(args.stake_pct),
        fee=float(args.fee),
        leverage=float(args.leverage),
    )

    trades = load_trades(trades_csv)
    sim = simulate(trades, cfg)
    write_excel(trades, sim, cfg, out_xlsx)
    print(f"Saved Excel report to {out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

