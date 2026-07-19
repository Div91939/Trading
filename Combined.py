"""
combined_new.py  —  Daily Signal Scanner (New Strategy)
========================================================
Two strategies, both regime-gated:

  REGIME DETECTION (runs first on every bar):
    UP    → price above rising MA50, ADX > threshold, DI+ > DI-
    DOWN  → price below falling MA50, ADX > threshold, DI- > DI+
    RANGE → everything else

  REV signal  (recall-optimized, no regime/ADX gate — v2, tuned per-stock):
    - N-day return  < RevDD       (meaningful dislocation; N = RevWindow, per-stock)
    - Volume ratio  > RevVol      (real participation, not drift)
    NOTE: earlier versions also required RANGE regime + ADX ceiling. Both were
    found to actively *hurt* recall AND return quality — the sharpest, most
    tradeable part of a reversal often gets misclassified as DOWN regime /
    high ADX, so gating on them threw out the best trades along with noise.
    Dropped both gates after a ground-truth minima audit + per-stock grid
    search (window/threshold sweep, 25% SL-capped forward returns).
    → Hold 30 days, 25% stop loss

  MOM signal (v2 — momentum INCEPTION, replaces v1 entirely):
    - Crossed above MA50 within last 8 trading days (catches the trend at
      birth, not mid-flight — structurally anti-peak)
    - MA50 10-bar slope > 0.5%     (the base itself has turned)
    - 5d/20d volume ratio > 1.1x   (sustained participation)
    No regime/ADX gate. Same fixed thresholds for every stock — validated
    OOS across the 30-stock universe (+10.18% avg 40d fwd return, 58.8% win,
    N=68, no train->OOS decay unlike every magnitude-threshold variant
    tested). See mom_v2_signal.py for the full research writeup.
    → EXIT: 20% trailing stop from the highest close since entry, with a
      25% hard stop as a backstop (protects against a violent single-day
      drop before the trailing stop can react). Re-validated OOS across the
      31-stock universe with this exit vs. the original fixed-40-day hold:
      +14.91% avg (was +9.95%), median +4.18% (was +1.99%), same 56.9% win
      rate — even with best+worst trade stripped: N=70, +12.85% avg. A
      fixed hold was capping exactly the trades that mattered most (real,
      multi-month trend continuations) — the trailing stop lets those run
      while the hard stop still protects the downside.

  Each stock's "Signals" config field controls which signal(s) actually run
  for it (REV / MOM / BOTH) — set per-stock from a ground-truth recall audit
  + grid search (REV) and train/OOS backtest (MOM). 5 stocks in the original
  30-stock universe had no viable signal of either kind and are excluded
  entirely (see bottom of STOCKS dict).

Run daily. Fetches latest bar, appends to CSV, checks signals, emails alert.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ta
import smtplib
import os
import json
import io
import yfinance as yf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LOG_PATH       = "email_log.json"
MOM_CROSS_LOG_PATH = "mom_cross_log.json"  # tracks last-alerted MA50-cross bar index per stock,
                                            # so MOM doesn't refire every day for up to 8 days
                                            # off a single crossing event
EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# Regime detection — fixed across all stocks
REG_MA_SLOPE = 0.3    # MA50 slope threshold (% over 20 bars)
REG_ADX      = 20     # ADX threshold for trend confirmation

# Per-stock config:
#   ticker      — yfinance ticker string
#   csv_path    — local CSV path
#   RevWindow   — lookback (days) for REV's return check (per-stock tuned: 5-20)
#   RevDD       — max RevWindow-day return to trigger REV (negative %)
#   RevVol      — min volume ratio vs 20d avg to trigger REV
#   Signals     — "REV", "MOM", or "BOTH" — which signal(s) run for this stock
#                 (MOM v2's thresholds are fixed/global — see MOM2_* constants
#                 near check_mom — not per-stock, since they generalized well
#                 across the whole universe without per-stock tuning)
#
# REV params below come from a per-stock grid search (lookback window x
# return threshold x volume threshold) maximizing 25%-SL-capped forward
# return quality, cross-checked against a ground-truth local-minima audit
# for BSE/EDELWEISS/EICHER/INDOTHAI/ORIRAIL/TITAGARH. E2E/HINDZINC/PARAS had
# no combination clear a positive-return bar — left on a conservative
# fallback and flagged; don't expect REV to fire much on those three.

STOCKS = {
    "ADANIENT": {
        "ticker":   "ADANIENT.NS",
        "csv_path": "Data/adanient.csv",
        "Signals":  "BOTH",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "ADANIGREEN": {
        "ticker":   "ADANIGREEN.NS",
        "csv_path": "Data/adanigreen.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "ADANIPOWER": {
        "ticker":   "ADANIPOWER.NS",
        "csv_path": "Data/adanipower.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "AFCOM": {
        "ticker":   "AFCONS.NS",  # VERIFY TICKER before relying on auto-fetch
        "csv_path": "Data/afcom.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "BEML": {
        "ticker":   "BEML.NS",
        "csv_path": "Data/beml.csv",
        "Signals":  "BOTH",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.9,
    },
    "BSE": {
        "ticker":   "BSE.NS",
        "csv_path": "Data/bse.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "CDSL": {
        "ticker":   "CDSL.NS",
        "csv_path": "Data/cdsl.csv",
        "Signals":  "BOTH",   # which signal(s) to check for this stock
        "RevWindow": 10, "RevDD": -3, "RevVol": 0.7,
    },
    "DOLATALGO": {
        "ticker":   "DOLATALGO.BO",  # VERIFY TICKER before relying on auto-fetch
        "csv_path": "Data/dolatalgo.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 10, "RevDD": -3, "RevVol": 0.7,
    },
    "E2E": {
        "ticker":   "E2E.NS",
        "csv_path": "Data/e2e.csv",
        "Signals":  "MOM",   # which signal(s) to check for this stock
    },
    "EDELWEISS": {
        "ticker":   "EDELWEISS.BO",
        "csv_path": "Data/edelweiss.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "EICHER": {
        "ticker":   "EICHERMOT.BO",
        "csv_path": "Data/eicher.csv",
        "Signals":  "BOTH",   # which signal(s) to check for this stock
        "RevWindow": 10, "RevDD": -3, "RevVol": 0.7,
    },
    "GRSE": {
        "ticker":   "GRSE.NS",
        "csv_path": "Data/grse.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "HINDCOPPER": {
        "ticker":   "HINDCOPPER.NS",
        "csv_path": "Data/hindcopper.csv",
        "Signals":  "BOTH",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "HINDZINC": {
        "ticker":   "HINDZINC.BO",
        "csv_path": "Data/hindzinc.csv",
        "Signals":  "MOM",   # which signal(s) to check for this stock
    },
    "INDOTHAI": {
        "ticker":   "INDOTHAI.NS",
        "csv_path": "Data/indothai.csv",
        "Signals":  "MOM",   # which signal(s) to check for this stock
    },
    "MAZDOCK": {
        "ticker":   "MAZDOCK.NS",
        "csv_path": "Data/mazdock.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.9,
    },
    "ORIRAIL": {
        "ticker":   "ORIRAIL.BO",
        "csv_path": "Data/orirail.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 10, "RevDD": -3, "RevVol": 0.8,
    },
    "RAPICUT": {
        "ticker":   "RAPICUT.BO",  # VERIFY TICKER before relying on auto-fetch
        "csv_path": "Data/rapicut.csv",
        "Signals":  "BOTH",   # which signal(s) to check for this stock
        "RevWindow": 20, "RevDD": -3, "RevVol": 0.7,
    },
    "RECLTD": {
        "ticker":   "RECLTD.NS",
        "csv_path": "Data/recltd.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 1.0,
    },
    "REFEX": {
        "ticker":   "REFEX.NS",
        "csv_path": "Data/refex.csv",
        "Signals":  "BOTH",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "SBIN": {
        "ticker":   "SBIN.NS",
        "csv_path": "Data/sbin.csv",
        "Signals":  "BOTH",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "SWITCHTE": {
        "ticker":   "SWITCHTE.BO",  # VERIFY TICKER before relying on auto-fetch
        "csv_path": "Data/switchte.csv",
        "Signals":  "MOM",   # which signal(s) to check for this stock
    },
    "TITAGARH": {
        "ticker":   "TITAGARH.NS",
        "csv_path": "Data/titagarh.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 5, "RevDD": -3, "RevVol": 0.7,
    },
    "TITAN": {
        "ticker":   "TITAN.NS",
        "csv_path": "Data/titan.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 10, "RevDD": -3, "RevVol": 0.7,
    },
    "TRENT": {
        "ticker":   "TRENT.NS",
        "csv_path": "Data/trent.csv",
        "Signals":  "REV",   # which signal(s) to check for this stock
        "RevWindow": 15, "RevDD": -3, "RevVol": 0.7,
    },
    # Excluded — no viable signal found in tuning (REV grid search + MOM v2 both
    # failed to clear a positive-return bar): ADANIENSOL, PARAS, RUDRA, SRMENERGY, TATATECH
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH + UPDATE CSV
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_update_csv(ticker, csv_path):
    stock = yf.Ticker(ticker)
    info  = stock.info

    company_name = info.get("longName") or info.get("shortName") or ticker
    avg_volume   = info.get("averageDailyVolume10Day") or 0
    volume       = info.get("volume") or 0

    hist = stock.history(period="1d", interval="1d")
    if hist.empty:
        return None, None

    today_str = hist.index[-1].strftime("%d-%m-%Y")
    new_row = {
        "Date":         today_str,
        "Open":         round(float(hist["Open"].iloc[-1]),  2),
        "High":         round(float(hist["High"].iloc[-1]),  2),
        "Low":          round(float(hist["Low"].iloc[-1]),   2),
        "Close":        round(float(hist["Close"].iloc[-1]), 2),
        "Volume":       volume,
        "Avg_Volume":   avg_volume,
        "Dividends":    round(float(hist["Dividends"].iloc[-1]), 2),
        "Stock Splits": round(float(hist["Stock Splits"].iloc[-1]), 2),
    }

    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame([new_row]).to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
        df = df.dropna(how="all").drop_duplicates(subset="Date", keep="last")
        df.columns = df.columns.str.strip()
        if today_str in df["Date"].values:
            idx = df.index[df["Date"] == today_str][0]
            for col in ["Open", "High", "Low", "Close", "Volume", "Avg_Volume"]:
                df.loc[idx, col] = new_row[col]
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    meta = {
        "company":    company_name,
        "ticker":     ticker,
        "date":       today_str,
        "volume":     volume,
        "avg_volume": avg_volume,
    }
    return df, meta

# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df):
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    close = np.array(df["Close"], dtype=float)
    opn   = np.array(df["Open"],  dtype=float)
    high  = np.array(df["High"],  dtype=float)
    low   = np.array(df["Low"],   dtype=float)
    vol   = np.array(df["Volume"], dtype=float)
    n     = len(close)

    # MA50 and slope
    ma50 = pd.Series(close).rolling(50).mean().values
    ma50_slope = np.full(n, np.nan)
    for i in range(20, n):
        if not np.isnan(ma50[i]) and not np.isnan(ma50[i-20]) and ma50[i-20] != 0:
            ma50_slope[i] = (ma50[i] - ma50[i-20]) / ma50[i-20] * 100

    price_vs_ma50 = np.where(ma50 > 0, (close - ma50) / ma50 * 100, np.nan)

    # ── MOM v2 features (momentum inception — see check_mom_v2) ──────────
    # 10-bar MA50 slope (distinct from the 20-bar ma50_slope used by regime/v1)
    ma50_slope_10 = np.full(n, np.nan)
    for i in range(10, n):
        if not np.isnan(ma50[i]) and not np.isnan(ma50[i-10]) and ma50[i-10] != 0:
            ma50_slope_10[i] = (ma50[i] - ma50[i-10]) / ma50[i-10] * 100

    vol_ma5  = pd.Series(vol).rolling(5).mean().values
    vol_ma20_own = pd.Series(vol).rolling(20).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_expansion = np.where(vol_ma20_own > 0, vol_ma5 / vol_ma20_own, np.nan)

    # Days since price last crossed ABOVE MA50 (inf until the first cross)
    above = close > ma50
    days_since_ma50_cross = np.full(n, np.inf)
    last_cross = -np.inf
    for i in range(n):
        if i > 0 and above[i] and not above[i-1]:
            last_cross = i
        days_since_ma50_cross[i] = i - last_cross

    # ADX, DI+, DI-
    adxi    = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    adx     = np.array(adxi.adx())
    di_plus = np.array(adxi.adx_pos())
    di_minus= np.array(adxi.adx_neg())

    # Volume ratio vs 20d average
    vol_ma20  = pd.Series(vol).rolling(20).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(vol_ma20 > 0, vol / vol_ma20, np.nan)

    # 20d and 60d returns
    ret20d = np.full(n, np.nan)
    ret60d = np.full(n, np.nan)
    for i in range(60, n):
        if close[i-20] > 0: ret20d[i] = (close[i] - close[i-20]) / close[i-20] * 100
        if close[i-60] > 0: ret60d[i] = (close[i] - close[i-60]) / close[i-60] * 100

    # RSI for chart display only
    rsi14 = np.array(ta.momentum.RSIIndicator(df["Close"], window=14).rsi())

    # BB for chart display only
    bb     = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    bb_up  = np.array(bb.bollinger_hband())
    bb_low = np.array(bb.bollinger_lband())
    bb_mid = np.array(bb.bollinger_mavg())

    return dict(
        close=close, high=high, low=low, vol=vol, open=opn,
        ma50=ma50, ma50_slope=ma50_slope, price_vs_ma50=price_vs_ma50,
        ma50_slope_10=ma50_slope_10, vol_expansion=vol_expansion,
        days_since_ma50_cross=days_since_ma50_cross,
        adx=adx, di_plus=di_plus, di_minus=di_minus,
        vol_ratio=vol_ratio, vol_ma20=vol_ma20,
        ret20d=ret20d, ret60d=ret60d,
        rsi14=rsi14, bb_up=bb_up, bb_low=bb_low, bb_mid=bb_mid,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3. REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_regime(ind, i):
    """Returns 'UP', 'DOWN', or 'RANGE'."""
    vals = [ind["ma50_slope"][i], ind["adx"][i],
            ind["price_vs_ma50"][i], ind["di_plus"][i], ind["di_minus"][i]]
    if any(np.isnan(v) for v in vals):
        return "RANGE"  # default to RANGE if insufficient data

    above = ind["price_vs_ma50"][i] > 0
    slope = ind["ma50_slope"][i]
    adv   = ind["adx"][i]
    bull  = ind["di_plus"][i] > ind["di_minus"][i]

    if above and slope > REG_MA_SLOPE and adv > REG_ADX and bull:
        return "UP"
    if not above and slope < -REG_MA_SLOPE and adv > REG_ADX and not bull:
        return "DOWN"
    return "RANGE"

# ─────────────────────────────────────────────────────────────────────────────
# 4. SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def rev_window_return(ind, i, window):
    """N-day % return ending at bar i, using cfg['RevWindow'] (per-stock)."""
    close = ind["close"]
    if i < window:
        return np.nan
    prev = close[i - window]
    if prev <= 0 or np.isnan(prev) or np.isnan(close[i]):
        return np.nan
    return (close[i] - prev) / prev * 100


def check_rev(ind, i, cfg):
    """
    REV v2 — Mean reversion entry. NO regime or ADX gate (see module docstring
    for why — both were found to hurt recall and quality in the ground-truth
    minima audit).
    Conditions:
      1. RevWindow-day return below RevDD → confirmed dislocation
         (RevWindow is per-stock tuned, typically 5-20 days — most stocks in
         this universe reverse fast, so a short window catches the move while
         it's forming instead of waiting for it to accumulate)
      2. Volume ratio above RevVol        → real participation, not drift
    """
    ret_w = rev_window_return(ind, i, cfg["RevWindow"])
    vol_r = ind["vol_ratio"][i]
    if np.isnan(ret_w) or np.isnan(vol_r):
        return False
    return ret_w < cfg["RevDD"] and vol_r > cfg["RevVol"]


MOM2_CROSS_DAYS = 8     # must have crossed above MA50 within this many trading days
MOM2_SLOPE_MIN  = 0.5   # MA50 10-bar slope must exceed this (%) — base itself has turned
MOM2_VOL_EXP    = 1.1   # 5d avg volume vs 20d avg volume — sustained participation

def check_mom(ind, i, cfg):
    """
    MOM v2 — Momentum INCEPTION entry (replaces v1's regime/ADX-gated design).
    Validated OOS (last 12mo, 30-stock universe): +10.18% avg 40d fwd return,
    58.8% win rate, N=68 — the only momentum definition tested that did NOT
    decay OOS vs train (train: +8.56%, 55.6%). See mom_v2_signal.py for the
    full research writeup.

    Conditions (all fixed, NOT per-stock — generalized well across the universe):
      1. Crossed ABOVE MA50 within the last MOM2_CROSS_DAYS trading days
         → anti-peak by construction: entry only possible near the base,
           structurally cannot be deep into an extended move
      2. MA50 10-bar slope > MOM2_SLOPE_MIN  → the base itself has turned up
      3. 5d/20d volume ratio > MOM2_VOL_EXP  → sustained participation

    No regime/ADX gate — dropped in favor of the direct MA50-cross timing,
    which carries the "is this a real new trend" signal more directly.
    """
    if np.isnan(ind["ma50_slope_10"][i]) or np.isnan(ind["vol_expansion"][i]):
        return False
    if not np.isfinite(ind["days_since_ma50_cross"][i]):
        return False
    return (ind["days_since_ma50_cross"][i] <= MOM2_CROSS_DAYS and
            ind["ma50_slope_10"][i]         >  MOM2_SLOPE_MIN and
            ind["vol_expansion"][i]         >  MOM2_VOL_EXP)


def mom_cross_id(ind, i):
    """Identifies which MA50-cross 'cycle' bar i belongs to, for de-duplication —
    without this, MOM would refire every day for up to MOM2_CROSS_DAYS days
    straight off a single crossing event."""
    d = ind["days_since_ma50_cross"][i]
    if not np.isfinite(d):
        return None
    return int(i - d)  # bar index of the actual cross


SIGNAL_DESCRIPTIONS = {
    "REV": (
        "REVERSAL ENTRY (v2 — no regime/ADX gate, per-stock tuned window)\n"
        "  Stock is down meaningfully over its tuned lookback window on\n"
        "  elevated-for-it volume, confirming a real dislocation. No regime\n"
        "  or ADX filter — dropped after ground-truth testing showed both\n"
        "  were suppressing the sharpest, most tradeable reversals.\n"
        "  Conditions: {RevWindow}d return < {RevDD}%  |  Volume > {RevVol}x avg\n"
        "  Strategy: Hold 30 days."
    ),
    "MOM": (
        "MOMENTUM INCEPTION ENTRY (v2 — replaces v1 entirely)\n"
        "  Price crossed above MA50 recently, the base itself has turned up,\n"
        "  and volume is expanding — catches a new trend near its start,\n"
        "  not mid-flight. No regime/ADX gate; fixed thresholds validated\n"
        "  OOS across the full stock universe (not per-stock tuned).\n"
        "  Conditions: crossed MA50 <= {MOM2_CROSS_DAYS}d ago  |  MA50 10d slope > {MOM2_SLOPE_MIN}%  |  Vol(5d/20d) > {MOM2_VOL_EXP}x\n"
        "  Strategy: 20% trailing stop from the highest close since entry,\n"
        "  with a 25% hard stop as backstop. Re-validated OOS: +14.91% avg\n"
        "  (vs +9.95% on a fixed 40-day hold), median +4.18%, 56.9% win —\n"
        "  a fixed hold was capping exactly the trades that mattered most."
    ),

}

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT — 2 subplots: price + BB + MA50, RSI + volume
# ─────────────────────────────────────────────────────────────────────────────

def build_plot(ind, company_name, ticker, date_label, regime, lookback=120,
               rev_fires=None, mom_fires=None):
    """rev_fires / mom_fires: optional lists of bar indices (absolute, into ind
    arrays) where each signal fired historically — marked on the chart so the
    email shows past triggers within the visible window, not just today's."""
    n     = len(ind["close"])
    start = max(0, n - lookback)
    x     = np.arange(start, n)

    opn    = ind["open"][start:n]
    high   = ind["high"][start:n]
    low    = ind["low"][start:n]
    close  = ind["close"][start:n]
    bb_up  = ind["bb_up"][start:n]
    bb_low = ind["bb_low"][start:n]
    bb_mid = ind["bb_mid"][start:n]
    ma50   = ind["ma50"][start:n]
    rsi    = ind["rsi14"][start:n]
    vol    = ind["vol"][start:n]
    vol_ma = ind["vol_ma20"][start:n]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
    )
    regime_colour = {"UP": "#26a69a", "DOWN": "#ef5350", "RANGE": "#78909c"}
    fig.suptitle(
        f"{company_name} ({ticker})  —  {date_label}  |  Regime: {regime}",
        fontsize=11, fontweight="bold",
        color=regime_colour.get(regime, "black"),
    )

    # ── Candlesticks ─────────────────────────────────────────────────────
    up_col, dn_col = "#26a69a", "#ef5350"
    for xi, o, h, l, c in zip(x, opn, high, low, close):
        col = up_col if c >= o else dn_col
        ax1.vlines(xi, l, h, color=col, lw=0.7, zorder=2)               # wick
        body_lo, body_hi = min(o, c), max(o, c)
        if body_hi - body_lo < 1e-9:                                     # doji
            ax1.hlines(o, xi - 0.3, xi + 0.3, color=col, lw=1.0, zorder=3)
        else:
            ax1.add_patch(plt.Rectangle(
                (xi - 0.3, body_lo), 0.6, body_hi - body_lo,
                facecolor=col, edgecolor=col, lw=0.5, zorder=3))

    # ── BB + MA50 overlays ───────────────────────────────────────────────
    ax1.plot(x, bb_up,  color="#27ae60",  lw=0.9, ls="--", label="BB Upper")
    ax1.plot(x, bb_low, color="#e74c3c",  lw=0.9, ls="--", label="BB Lower")
    ax1.plot(x, bb_mid, color="steelblue",lw=0.7, ls=":",  alpha=0.7, label="BB Mid")
    ax1.plot(x, ma50,   color="#f39c12",  lw=1.1, ls="-",  label="MA50")
    ax1.fill_between(x, bb_low, bb_up, alpha=0.05, color="steelblue")

    # ── Past signal triggers in the visible window ───────────────────────
    price_span = np.nanmax(high) - np.nanmin(low)
    marker_off  = price_span * 0.03
    if rev_fires:
        rf = [i for i in rev_fires if start <= i < n]
        if rf:
            ax1.scatter(rf, [ind["low"][i] - marker_off for i in rf],
                        marker="^", s=80, color="#2ecc71", edgecolor="black",
                        lw=0.7, zorder=5, label="REV trigger")
    if mom_fires:
        mf = [i for i in mom_fires if start <= i < n]
        if mf:
            ax1.scatter(mf, [ind["low"][i] - marker_off for i in mf],
                        marker="^", s=80, color="#3498db", edgecolor="black",
                        lw=0.7, zorder=5, label="MOM trigger")

    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=7, ncol=6, framealpha=0.7)
    ax1.grid(alpha=0.25)

    # ── RSI + Volume ─────────────────────────────────────────────────────
    ax2.plot(x, rsi, color="darkorange", lw=1.2, label="RSI(14)")
    ax2.axhline(70, color="#e74c3c", ls="--", lw=0.8)
    ax2.axhline(30, color="#27ae60", ls="--", lw=0.8)
    ax2.axhline(50, color="gray",    ls=":",  lw=0.6, alpha=0.5)
    ax2.fill_between(x, rsi, 30, where=(rsi < 30), alpha=0.15, color="#27ae60")
    ax2.fill_between(x, rsi, 70, where=(rsi > 70), alpha=0.15, color="#e74c3c")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")

    # Volume bars on twin axis
    ax3 = ax2.twinx()
    bar_colors = [up_col if c >= o else dn_col for c, o in zip(close, opn)]
    ax3.bar(x, vol, color=bar_colors, alpha=0.3, width=0.8)
    ax3.plot(x, vol_ma, color="#78909c", lw=0.8, ls="--", alpha=0.7)
    ax3.set_ylabel("Volume", fontsize=8)
    ax3.tick_params(labelsize=7)

    ax2.set_xlabel("Bar index")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
# 6. EMAIL LOG — dedup same-day alerts
# ─────────────────────────────────────────────────────────────────────────────

def load_log():
    if not os.path.exists(LOG_PATH):
        return {}
    with open(LOG_PATH, "r") as f:
        content = f.read().strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}

def save_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

def load_mom_cross_log():
    if not os.path.exists(MOM_CROSS_LOG_PATH):
        return {}
    with open(MOM_CROSS_LOG_PATH, "r") as f:
        content = f.read().strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}

def save_mom_cross_log(log):
    with open(MOM_CROSS_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

def send_email(subject, body, attachments):
    """attachments: list of (filename, png_bytes)"""
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER
    msg.attach(MIMEText(body, "plain"))
    for fname, png_bytes in attachments:
        img = MIMEImage(png_bytes, name=fname)
        msg.attach(img)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(EMAIL_SENDER, EMAIL_PASSWORD)
            srv.send_message(msg)
        print(f"  Email sent: {subject}")
        return True
    except Exception as e:
        print(f"  Email failed: {e}")
        return False



# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log          = load_log()
    mom_cross_log = load_mom_cross_log()
    today_label  = None

    report_sections  = []
    plot_attachments = []

    for stock_name, cfg in STOCKS.items():
        print(f"\n── {stock_name} ──────────────────────────────────────")

        try:
            df, meta = fetch_and_update_csv(cfg["ticker"], cfg["csv_path"])
        except Exception as e:
            print(f"  Fetch failed: {e}")
            continue

        if df is None:
            print("  No data (market closed?)")
            continue

        MIN_ROWS_REQUIRED = 60  # MA50 needs 50, ADX needs 14+, plus a small buffer
        valid_rows = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(valid_rows) < MIN_ROWS_REQUIRED:
            print(f"  Skipping — only {len(valid_rows)} valid rows, need >= {MIN_ROWS_REQUIRED} "
                  f"for MA50/ADX (new/short-history stock? backfill more history first)")
            continue

        today_label = meta["date"]
        try:
            ind = compute_indicators(df)
        except Exception as e:
            print(f"  compute_indicators failed: {e} — skipping this stock, continuing with the rest")
            continue
        i   = len(ind["close"]) - 1  # today's bar

        # ── Regime (informational only — neither REV v2 nor MOM v2 gate on it) ──
        regime = get_regime(ind, i)

        signals_to_run = cfg["Signals"]  # "REV", "MOM", or "BOTH"

        # ── REV ──────────────────────────────────────────────────────────
        rev_fired = False
        rev_w_ret = None
        if signals_to_run in ("REV", "BOTH"):
            rev_fired = check_rev(ind, i, cfg)
            rev_w_ret = rev_window_return(ind, i, cfg["RevWindow"])

        # ── MOM v2 — with cross-event dedup (don't refire the same cross) ──
        mom_fired = False
        if signals_to_run in ("MOM", "BOTH"):
            mom_condition_true = check_mom(ind, i, cfg)
            if mom_condition_true:
                cross_id = mom_cross_id(ind, i)
                already_alerted = mom_cross_log.get(stock_name) == cross_id
                if not already_alerted:
                    mom_fired = True
                    mom_cross_log[stock_name] = cross_id

        # ── Print daily status ────────────────────────────────────────────
        print(f"  Regime    : {regime}   (informational — REV/MOM v2 do not gate on this)")
        print(f"  Close     : {ind['close'][i]:.2f}")
        if signals_to_run in ("REV", "BOTH"):
            print(f"  {cfg['RevWindow']}d ret (REV): {rev_w_ret:.1f}%")
        if signals_to_run in ("MOM", "BOTH"):
            print(f"  MOM: {ind['days_since_ma50_cross'][i]:.0f}d since MA50 cross  |  "
                  f"MA50 10d slope: {ind['ma50_slope_10'][i]:.2f}%  |  "
                  f"Vol(5d/20d): {ind['vol_expansion'][i]:.2f}x")
        print(f"  Vol ratio : {ind['vol_ratio'][i]:.2f}x  |  ADX: {ind['adx'][i]:.1f}  |  DI+: {ind['di_plus'][i]:.1f}  DI-: {ind['di_minus'][i]:.1f}")
        print(f"  REV fired : {'YES' if rev_fired else 'no'}  |  MOM fired: {'YES' if mom_fired else 'no'}")

        if not rev_fired and not mom_fired:
            continue

        # ── Build email section ───────────────────────────────────────────
        lines = [
            f"\n{'='*60}",
            f"{meta['company']} ({meta['ticker']})  —  {meta['date']}",
            f"{'='*60}",
            f"Regime   : {regime}  (informational only)",
            f"Close    : {ind['close'][i]:.2f}",
            f"Vol ratio: {ind['vol_ratio'][i]:.2f}x   ADX: {ind['adx'][i]:.1f}   DI+: {ind['di_plus'][i]:.1f}  DI-: {ind['di_minus'][i]:.1f}",
            f"MA50 slope (20d): {ind['ma50_slope'][i]:.2f}%",
        ]

        for sig_name, fired in [("REV", rev_fired), ("MOM", mom_fired)]:
            if not fired:
                continue
            log_key = f"{stock_name}_{sig_name}"
            already = log.get(log_key) == today_label
            tag = "  [already sent today]" if already else ""
            fmt_vals = dict(cfg)
            fmt_vals.update(MOM2_CROSS_DAYS=MOM2_CROSS_DAYS, MOM2_SLOPE_MIN=MOM2_SLOPE_MIN, MOM2_VOL_EXP=MOM2_VOL_EXP)
            desc = SIGNAL_DESCRIPTIONS[sig_name].format(**fmt_vals)
            lines.append(f"\n{'─'*40}")
            lines.append(f"SIGNAL: {sig_name}{tag}")
            lines.append(desc)
            if not already:
                log[log_key] = today_label

        report_sections.append("\n".join(lines))

        # ── Build chart (with past triggers marked in the window) ─────────
        signals_cfg = cfg["Signals"]
        hist_rev, hist_mom = [], []
        if signals_cfg in ("REV", "BOTH"):
            hist_rev = [k for k in range(len(ind["close"])) if check_rev(ind, k, cfg)]
        if signals_cfg in ("MOM", "BOTH"):
            seen_crosses = set()
            for k in range(len(ind["close"])):
                if check_mom(ind, k, cfg):
                    cid = mom_cross_id(ind, k)
                    if cid is not None and cid not in seen_crosses:
                        seen_crosses.add(cid)
                        hist_mom.append(k)   # first qualifying day per cross only
        png = build_plot(ind, meta["company"], meta["ticker"], meta["date"], regime,
                          rev_fires=hist_rev, mom_fires=hist_mom)
        plot_attachments.append((f"{stock_name}_{today_label}.png", png))

    # ── Save state ────────────────────────────────────────────────────────
    save_log(log)
    save_mom_cross_log(mom_cross_log)

    if not report_sections:
        print("\nNo signals today — no email sent.")
        return

    # ── Send email ────────────────────────────────────────────────────────
    n_stocks = len(report_sections)
    subject  = f"[Signal Scanner] {n_stocks} stock(s) — {today_label}"
    body = (
        f"DAILY SIGNAL SCAN  —  {today_label}\n"
        f"Strategy: REV (mean reversion, RANGE regime) + MOM (momentum, UP regime)\n"
        f"Hold: 30 days\n"
        f"Stocks scanned: {len(STOCKS)}\n"
        + "\n".join(report_sections)
    )
    send_email(subject, body, plot_attachments)


if __name__ == "__main__":
    main()
