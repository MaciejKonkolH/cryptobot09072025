import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot as plot_offline
import plotly.io as pio


def load_window_ending_at(feather_path: Path, window: int, end_dt: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_feather(feather_path)
    # Normalize datetime column
    if 'date' in df.columns:
        dt = pd.to_datetime(df['date'], utc=True, errors='coerce')
    elif 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    else:
        raise ValueError("Input file must contain 'date' or 'timestamp' column")
    df = df.assign(date=dt).dropna(subset=['date']).sort_values('date')

    # Ensure required OHLC columns
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns in file: {missing}")

    if end_dt is None:
        end_dt = df['date'].iloc[-1]
    else:
        end_dt = pd.to_datetime(end_dt, utc=True)

    # Select all rows up to end_dt (inclusive) and take the last 'window'
    df_cut = df[df['date'] <= end_dt]
    if len(df_cut) < window:
        raise ValueError(f"Not enough rows up to {end_dt}. Needed {window}, found {len(df_cut)}")
    out = df_cut.iloc[-window:].copy().reset_index(drop=True)
    return out


def fit_parallel_channel(close: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit a parallel channel to series 'close' using linear regression as center
    and min/max residuals for support/resistance offsets.

    Returns (slope, intercept, offset_low, offset_high) so that:
    support(t) = slope*t + intercept + offset_low
    resist (t) = slope*t + intercept + offset_high
    """
    n = close.shape[0]
    x = np.arange(n, dtype=float)
    # Fit center line by least squares
    slope, intercept = np.polyfit(x, close, 1)
    center = slope * x + intercept
    resid = close - center
    offset_low = float(resid.min())
    offset_high = float(resid.max())
    return float(slope), float(intercept), offset_low, offset_high


def fit_wedge_channel(high: np.ndarray, low: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit independent lines to highs (resistance) and lows (support).

    Returns (slope_low, intercept_low, slope_high, intercept_high)
    such that:
      support(t)  = slope_low  * t + intercept_low
      resistance(t)= slope_high * t + intercept_high
    """
    n = min(len(high), len(low))
    x = np.arange(n, dtype=float)
    slope_high, intercept_high = np.polyfit(x, high[:n], 1)
    slope_low, intercept_low = np.polyfit(x, low[:n], 1)
    return float(slope_low), float(intercept_low), float(slope_high), float(intercept_high)


def fit_envelope_lp(high: np.ndarray, low: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit tight envelope (non-parallel) using linear programming.
    Upper line y = au + bu * t covers all highs and minimizes max residual.
    Lower line y = al + bl * t lies below all lows and minimizes max residual.
    Returns (al, bl, au, bu) as (intercept_low, slope_low, intercept_high, slope_high).
    """
    import numpy as np
    from scipy.optimize import linprog

    n = min(len(high), len(low))
    x = np.arange(n, dtype=float)

    # Upper line: vars [a, b, m]
    # Constraints for each i:
    # 1) a + b*x_i - m <= high_i
    # 2) -(a + b*x_i) <= -high_i  (equiv a + b*x_i >= high_i)
    A_u = []
    b_u = []
    for xi, hi in zip(x, high[:n]):
        A_u.append([1.0, xi, -1.0])
        b_u.append(hi)
        A_u.append([-1.0, -xi, 0.0])
        b_u.append(-hi)
    bounds_u = [(None, None), (None, None), (0.0, None)]  # a, b unbounded; m >= 0
    res_u = linprog(c=[0.0, 0.0, 1.0], A_ub=np.array(A_u), b_ub=np.array(b_u), bounds=bounds_u, method="highs")
    if not res_u.success:
        raise RuntimeError(f"Upper envelope LP failed: {res_u.message}")
    au, bu, _mu = res_u.x

    # Lower line: vars [a, b, m]
    # residual r_i = low_i - (a + b*x_i) >= 0
    # Constraints:
    # 1) -a - b*x_i - m <= -low_i   (r_i <= m)
    # 2) a + b*x_i <= low_i         (r_i >= 0)
    A_l = []
    b_l = []
    for xi, lo in zip(x, low[:n]):
        A_l.append([-1.0, -xi, -1.0])
        b_l.append(-lo)
        A_l.append([1.0, xi, 0.0])
        b_l.append(lo)
    bounds_l = [(None, None), (None, None), (0.0, None)]
    res_l = linprog(c=[0.0, 0.0, 1.0], A_ub=np.array(A_l), b_ub=np.array(b_l), bounds=bounds_l, method="highs")
    if not res_l.success:
        raise RuntimeError(f"Lower envelope LP failed: {res_l.message}")
    al, bl, _ml = res_l.x

    return float(al), float(bl), float(au), float(bu)


def make_plotly_figure(
    df_base: pd.DataFrame,
    channels: list,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_base['date'], open=df_base['open'], high=df_base['high'], low=df_base['low'], close=df_base['close'],
        name='OHLC'))

    palette = [
        ('#1f77b4', '#aec7e8'),  # blue
        ('#ff7f0e', '#ffbb78'),  # orange
        ('#2ca02c', '#98df8a'),  # green
        ('#d62728', '#ff9896'),  # red
        ('#9467bd', '#c5b0d5'),  # purple
    ]

    for idx, ch in enumerate(channels):
        color_sup, color_res = palette[idx % len(palette)]
        df_w = ch['df']
        fig.add_trace(go.Scatter(
            x=df_w['date'], y=ch['support'], mode='lines',
            name=f"Support {ch['window']}m", line=dict(color=color_sup, width=2)))
        fig.add_trace(go.Scatter(
            x=df_w['date'], y=ch['resistance'], mode='lines',
            name=f"Resistance {ch['window']}m", line=dict(color=color_res, width=2, dash='solid')))

    fig.update_layout(
        title=f"Parallel channels (windows={','.join(str(c['window']) for c in channels)} min)",
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=50, b=20, l=50, r=20),
        yaxis_title='Price',
    )
    return fig


def write_html_with_metrics(fig: go.Figure, metrics_rows: list, out_path: Path) -> None:
    fig_div = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
    # Simple selectable metrics table (requested metrics only)
    header = [
        "window_m",
        "pos_in_channel",
        "slope_%_window",
        "width_%",
        "fit_score",
    ]
    header_html = "".join(f"<th>{h}</th>" for h in header)
    rows_html = "\n".join(
        "<tr>" + "".join(f"<td>{row.get(col, '')}</td>" for col in header) + "</tr>"
        for row in metrics_rows
    )
    metrics_html = f"""
    <div class="metrics">
      <h3>Metrics</h3>
      <table>
        <thead><tr>{header_html}</tr></thead>
        <tbody>
        {rows_html}
        </tbody>
      </table>
    </div>
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Channels with metrics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 10px; }}
    .metrics {{ margin-top: 16px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
    th {{ background: #f0f0f0; }}
  </style>
  <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta http-equiv=\"X-UA-Compatible\" content=\"IE=edge\" />
  <meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self' 'unsafe-inline' https://cdn.plot.ly data:; img-src 'self' data: https:;\" />
  <head></head>
<body>
{fig_div}
{metrics_html}
</body>
</html>
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot support/resistance channels over last N minutes for multiple windows")
    parser.add_argument('--file', type=str, default=str(Path('labeler5') / 'output' / 'labeled_BTCUSDT.feather'),
                        help='Path to feather file with OHLC data')
    parser.add_argument('--windows', type=str, default="240,180,120", help='Comma-separated window sizes in minutes (e.g., 240,180,120)')
    parser.add_argument('--end', type=str, default=None, help='End timestamp (e.g., 2025-05-01 12:34:00Z). Defaults to last row')
    parser.add_argument('--out', type=str, default=str(Path('resistance_and_support_lines') / 'output' / 'channel_plot.html'))
    args = parser.parse_args()

    feather_path = Path(args.file).resolve()
    end_dt = pd.to_datetime(args.end, utc=True) if args.end else None
    # Parse windows
    windows = [int(w.strip()) for w in str(args.windows).split(',') if w.strip()]
    windows = sorted(windows, reverse=True)

    # Load base dataframe for the largest window
    df_base = load_window_ending_at(feather_path, windows[0], end_dt)

    channels = []
    metrics_rows = []
    for w in windows:
        df_w = df_base.tail(w).copy().reset_index(drop=True)
        close_np = df_w['close'].to_numpy(dtype=float)
        slope, intercept, off_low, off_high = fit_parallel_channel(close_np)

        n = len(df_w)
        x = np.arange(n, dtype=float)
        center = slope * x + intercept
        support_par = center + off_low
        resist_par = center + off_high

        last_close = float(df_w['close'].iloc[-1])
        sup_last = float(support_par[-1])
        res_last = float(resist_par[-1])
        width_now = res_last - sup_last

        pos = (last_close - sup_last) / width_now if width_now > 0 else np.nan
        pos = float(np.clip(pos, 0.0, 1.0)) if np.isfinite(pos) else np.nan

        # Requested metrics
        slope_pct_window = 100.0 * (center[-1] - center[0]) / last_close if last_close != 0 else np.nan
        width_pct = 100.0 * (width_now / last_close) if last_close != 0 else np.nan

        # Simple fit metric: 1 - IQR(residuals)/channel_width, clipped to [0,1]
        resid = close_np - center
        q75, q25 = np.percentile(resid, [75, 25])
        iqr = float(q75 - q25)
        channel_width = float(resid.max() - resid.min())
        if channel_width <= 0:
            fit_score = np.nan
        else:
            fit_score = float(np.clip(1.0 - (iqr / channel_width), 0.0, 1.0))

        channels.append({
            'window': w,
            'df': df_w,
            'support': support_par,
            'resistance': resist_par,
        })
        metrics_rows.append({
            'window_m': str(w),
            'pos_in_channel': f"{pos:.4f}" if np.isfinite(pos) else "",
            'slope_%_window': f"{slope_pct_window:.3f}" if np.isfinite(slope_pct_window) else "",
            'width_%': f"{width_pct:.4f}" if np.isfinite(width_pct) else "",
            'fit_score': f"{fit_score:.3f}" if np.isfinite(fit_score) else "",
        })

    # Build Plotly figure
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = make_plotly_figure(df_base, channels)
    write_html_with_metrics(fig, metrics_rows, out_path)
    print(f"Saved interactive plot (with selectable metrics) to {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

