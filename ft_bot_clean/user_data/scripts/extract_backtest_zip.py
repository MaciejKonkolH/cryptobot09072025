import argparse
import sys
from pathlib import Path
import zipfile


def find_latest_zip(results_dir: Path) -> Path | None:
    if not results_dir.exists():
        return None
    zips = sorted(results_dir.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def safe_extract(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            # Prevent Zip Slip
            member_path = dest_dir / member.filename
            try:
                member_path.resolve().relative_to(dest_dir.resolve())
            except Exception:
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dest_dir)


def main(argv: list[str]) -> int:
    script_dir = Path(__file__).resolve().parent
    user_data_dir = script_dir.parent
    default_results_dir = user_data_dir / 'backtest_results'

    parser = argparse.ArgumentParser(description="Extract Freqtrade backtest result zip")
    parser.add_argument('--zip', dest='zip_path', type=str, default=None,
                        help='Path to specific backtest-result-*.zip to extract')
    parser.add_argument('--latest', action='store_true', help='Extract the most recent zip from backtest_results')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory. Defaults to backtest_results/extracted/<zip_basename>')
    parser.add_argument('--results-dir', type=str, default=str(default_results_dir),
                        help='Directory containing backtest zip files (default: user_data/backtest_results)')
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir).resolve()

    zip_path: Path | None = None
    if args.zip_path:
        zip_path = Path(args.zip_path)
        if not zip_path.is_absolute():
            zip_path = results_dir / zip_path
        zip_path = zip_path.resolve()
    elif args.latest:
        zip_path = find_latest_zip(results_dir)
        if zip_path is None:
            print(f"No zip files found in {results_dir}")
            return 1
    else:
        parser.error("Provide --zip <path> or --latest")

    if not zip_path.exists():
        print(f"Zip not found: {zip_path}")
        return 1

    base_name = zip_path.stem  # without .zip
    if args.outdir:
        outdir = Path(args.outdir)
        if not outdir.is_absolute():
            outdir = results_dir / outdir
    else:
        outdir = results_dir / 'extracted' / base_name

    outdir = outdir.resolve()
    try:
        safe_extract(zip_path, outdir)
        print(f"Extracted {zip_path} -> {outdir}")
        return 0
    except Exception as e:
        print(f"Extraction failed: {e}")
        return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

