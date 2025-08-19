import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Użycie backendu bez ekranu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_latest_predictions_file(reports_root: Path, symbol: str) -> Optional[Path]:
	"""Wyszukaj najnowszy plik predictions_trades_* dla podanego symbolu.

	Struktura szukania: crypto/training5/output/reports/{SYMBOL}/**/predictions_trades_{SYMBOL}_*.csv
	"""
	symbol_dir = reports_root / symbol
	if not symbol_dir.exists():
		return None
	candidates = sorted(
		symbol_dir.rglob(f"predictions_trades_{symbol}_*.csv"),
		key=lambda p: p.stat().st_mtime,
		reverse=True,
	)
	return candidates[0] if candidates else None


def find_latest_run_dir(reports_root: Path, symbol: str) -> Optional[Path]:
	"""Zwróć najnowszy katalog run_* dla symbolu, sortując po znaczniku z nazwy.

	Oczekiwana nazwa: run_YYYYMMDD_HHMMSS. Jeśli brak dopasowania, fallback na mtime.
	"""
	symbol_dir = reports_root / symbol
	if not symbol_dir.exists():
		return None
	run_dirs = [p for p in symbol_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
	if not run_dirs:
		return None

	def run_key(p: Path) -> int:
		m = re.match(r"run_(\d{8})_(\d{6})$", p.name)
		if m:
			# int YYYYMMDDHHMMSS dla poprawnego sortowania
			return int(m.group(1) + m.group(2))
		# Fallback: mtime (mniej wiarygodny)
		return int(p.stat().st_mtime)

	run_dirs.sort(key=run_key, reverse=True)
	return run_dirs[0]


def parse_tp_sl_from_filename(path: Path) -> Tuple[Optional[float], Optional[float]]:
	"""Parsuj TP/SL w % z nazwy pliku.

	Oczekiwany fragment: ..._tp1p4_sl0p6_... -> tp=1.4%, sl=0.6%
	Zwraca wartości w formie ułamków (0.014, 0.006).
	"""
	m = re.search(r"tp(\d+)p(\d+)_sl(\d+)p(\d+)", path.name)
	if not m:
		return None, None
	tp = float(f"{m.group(1)}.{m.group(2)}") / 100.0
	sl = float(f"{m.group(3)}.{m.group(4)}") / 100.0
	return tp, sl


def load_trades_csv(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	# Czas i sortowanie
	df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
	df = df.sort_values("timestamp").reset_index(drop=True)
	return df


def compute_pnl(df: pd.DataFrame, tp_pct: float, sl_pct: float, fee_roundtrip_pct: float) -> pd.DataFrame:
	"""Wylicz wynik każdej transakcji jako procent (ułamek), po odjęciu kosztów round-trip.

	- WIN -> +tp_pct
	- LOSS -> -sl_pct
	- Następnie odejmujemy fee_roundtrip_pct (np. 0.002 dla 0.2%).
	"""
	df = df.copy()
	df["pnl_pct_raw"] = np.where(df["result"] == "WIN", float(tp_pct), -float(sl_pct))
	df["pnl_pct"] = df["pnl_pct_raw"] - float(fee_roundtrip_pct)
	return df


def weekly_analysis(df: pd.DataFrame) -> pd.DataFrame:
	"""Agregacja tygodniowa: zwraca liczność i zwrot tygodniowy oraz skumulowany czynnik."""
	df = df.copy()
	df["week_start"] = df["timestamp"].dt.to_period("W-MON").apply(lambda p: p.start_time)
	g = df.groupby("week_start", as_index=False)
	out = g.apply(
		lambda gdf: pd.Series({
			"n_trades": int(len(gdf)),
			"weekly_return_pct": float(np.prod(1.0 + gdf["pnl_pct"].values) - 1.0),
		})
	).reset_index(drop=True)
	out = out.sort_values("week_start").reset_index(drop=True)
	out["cum_factor"] = (1.0 + out["weekly_return_pct"]).cumprod()
	return out


def equity_curve_over_trades(df: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
	"""Krzywa kapitału po każdej transakcji."""
	factors = (1.0 + df["pnl_pct"].values)
	cum_factor = np.cumprod(factors) if len(factors) else np.array([])
	curve = pd.DataFrame({
		"timestamp": df["timestamp"].values,
		"trade_idx": np.arange(1, len(df) + 1),
		"cum_factor": cum_factor,
		"equity": initial_balance * cum_factor if len(cum_factor) else np.array([]),
	})
	return curve


def extract_label_dir_name(path: Path) -> Optional[str]:
	m = re.search(r"(tp\d+p\d+_sl\d+p\d+)", path.name)
	return m.group(1) if m else None


def ensure_outdir_for(csv_path: Path, custom_out: Optional[Path]) -> Path:
	if custom_out:
		custom_out.mkdir(parents=True, exist_ok=True)
		return custom_out
	# Domyślnie obok pliku, w podkatalogu analysis/weekly/(tp..._sl...)
	base = csv_path.parent / "analysis" / "weekly"
	label_dir = extract_label_dir_name(csv_path) or csv_path.stem
	out = base / label_dir
	out.mkdir(parents=True, exist_ok=True)
	return out

def analyze_one(csv_path: Path, initial_balance: float, fee_rt: float, outdir: Optional[Path], symbol_for_title: Optional[str]) -> None:
	df = load_trades_csv(csv_path)
	if df.empty:
		print(f"Pusty plik: {csv_path}")
		return

	tp_pct, sl_pct = parse_tp_sl_from_filename(csv_path)
	if tp_pct is None or sl_pct is None:
		print(f"Pominięto (nie wykryto TP/SL): {csv_path}")
		return

	df = compute_pnl(df, tp_pct=tp_pct, sl_pct=sl_pct, fee_roundtrip_pct=fee_rt)
	weekly = weekly_analysis(df)
	curve = equity_curve_over_trades(df, initial_balance=float(initial_balance))

	out_dir = ensure_outdir_for(csv_path, outdir)
	# Zapisy
	weekly_out = out_dir / "weekly_summary.csv"
	curve_out = out_dir / "equity_curve_trades.csv"
	weekly.to_csv(weekly_out, index=False)
	curve.to_csv(curve_out, index=False)

	label_dir = extract_label_dir_name(csv_path) or "label"
	sym = symbol_for_title or "PAIR"
	title = f"{sym} | {label_dir} | fee RT={fee_rt*100:.2f}%"
	plot_path = plot_equity(curve, weekly, out_dir, title)
	plot_time_path = plot_equity_time(curve, out_dir, title)

	final_factor = float(curve["cum_factor"].iloc[-1]) if not curve.empty else 1.0
	final_equity = float(curve["equity"].iloc[-1]) if not curve.empty else float(initial_balance)
	total_return_pct = (final_factor - 1.0) * 100.0
	print(f"Plik: {csv_path}")
	print(f"Transakcji: {len(df)} | Łączny zwrot: {total_return_pct:.2f}% | Końcowy kapitał: {final_equity:.2f}")
	print(f"Zapisano: {weekly_out}")
	print(f"Zapisano: {curve_out}")
	print(f"Zapisano wykres: {plot_path}")
	print(f"Zapisano wykres: {plot_time_path}")



def plot_equity(curve_trades: pd.DataFrame, weekly: pd.DataFrame, out_dir: Path, title: str) -> Path:
	plt.style.use("seaborn-v0_8-darkgrid")
	fig, ax = plt.subplots(figsize=(12, 6))
	if not curve_trades.empty:
		ax.plot(curve_trades["trade_idx"], curve_trades["equity"], label="Kapitał po transakcjach", color="#1f77b4")
	if not weekly.empty:
		# Druga oś: skumulowany czynnik tygodniowy (przeskalowany do początkowego kapitału)
		ax2 = ax.twinx()
		ax2.plot(np.arange(1, len(weekly) + 1), weekly["cum_factor"], label="Skumulowany czynnik (tyg.)", color="#ff7f0e")
		ax2.set_ylabel("Cum factor (tygodnie)")
	ax.set_title(title)
	ax.set_xlabel("Indeks transakcji")
	ax.set_ylabel("Kapitał")
	ax.legend(loc="upper left")
	# Zapisz
	out_path = out_dir / "equity.png"
	fig.tight_layout()
	fig.savefig(out_path, dpi=130)
	plt.close(fig)
	return out_path


def plot_equity_time(curve_trades: pd.DataFrame, out_dir: Path, title: str) -> Path:
	plt.style.use("seaborn-v0_8-darkgrid")
	fig, ax = plt.subplots(figsize=(12, 6))
	if not curve_trades.empty:
		ax.plot(pd.to_datetime(curve_trades["timestamp"]), curve_trades["equity"], label="Kapitał po transakcjach", color="#1f77b4")
	ax.set_title(title + " (czas)")
	ax.set_xlabel("Data")
	ax.set_ylabel("Kapitał")
	ax.legend(loc="upper left")
	for label in ax.get_xticklabels():
		label.set_rotation(15)
	label.set_horizontalalignment('right')
	out_path = out_dir / "equity_time.png"
	fig.tight_layout()
	fig.savefig(out_path, dpi=130)
	plt.close(fig)
	return out_path


def main():
	parser = argparse.ArgumentParser(description="Analiza tygodniowa i krzywa kapitału z plików predictions_trades_*.csv")
	parser.add_argument("--symbol", "-s", help="Symbol pary, np. ETHUSDT (gdy nie podano --file)")
	parser.add_argument("--file", "-f", help="Ścieżka do konkretnego pliku predictions_trades_*.csv")
	parser.add_argument("--initial-balance", type=float, default=10000.0, help="Początkowy kapitał do krzywej kapitału (domyślnie 10000)")
	parser.add_argument("--fee-roundtrip-pct", type=float, default=0.0, help="Łączny koszt wejście+wyjście jako ułamek (np. 0.002 = 0.2%)")
	parser.add_argument("--outdir", type=str, default=None, help="Katalog wyjściowy; domyślnie obok pliku w analysis/weekly")
	args = parser.parse_args()

	# Tryb pojedynczego pliku
	if args.file:
		csv_file = Path(args.file)
		if not csv_file.exists():
			raise SystemExit(f"Plik nie istnieje: {csv_file}")
		analyze_one(
			csv_file,
			initial_balance=float(args.initial_balance),
			fee_rt=float(args.fee_roundtrip_pct),
			outdir=Path(args.outdir) if args.outdir else None,
			symbol_for_title=args.symbol,
		)
		return

	# Tryb symbolu – analizuj wszystkie pliki predictions_trades_* z najnowszego runu
	if not args.symbol:
		raise SystemExit("Podaj --symbol lub --file")
	# Spróbuj kilku możliwych lokalizacji katalogu raportów
	script_path = Path(__file__).resolve()
	candidates = [
		Path("training5/output/reports"),
		Path("crypto/training5/output/reports"),
		script_path.parent.parent / "output" / "reports",
		script_path.parents[2] / "training5" / "output" / "reports",
	]
	reports_root = next((p for p in candidates if p.exists()), None)
	if reports_root is None:
		raise SystemExit("Nie znaleziono katalogu z raportami (próbowano: training5/output/reports i crypto/training5/output/reports)")
	run_dir = find_latest_run_dir(reports_root, args.symbol)
	if run_dir is None:
		one = find_latest_predictions_file(reports_root, args.symbol)
		if not one:
			raise SystemExit(f"Nie znaleziono plików predictions_trades_ dla symbolu {args.symbol} w {reports_root}")
		analyze_one(
			one,
			initial_balance=float(args.initial_balance),
			fee_rt=float(args.fee_roundtrip_pct),
			outdir=Path(args.outdir) if args.outdir else None,
			symbol_for_title=args.symbol,
		)
		return

	files = sorted(run_dir.glob(f"predictions_trades_{args.symbol}_*.csv"))
	if not files:
		raise SystemExit(f"Brak plików predictions_trades_ w {run_dir}")
	for fp in files:
		analyze_one(
			fp,
			initial_balance=float(args.initial_balance),
			fee_rt=float(args.fee_roundtrip_pct),
			outdir=Path(args.outdir) if args.outdir else None,
			symbol_for_title=args.symbol,
		)


if __name__ == "__main__":
	main()

