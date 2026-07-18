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
    (MomRet/MomVol/MomADX below are UNCHANGED — momentum redesign pending)

  MOM signal  (fires in UP only):
    - 60-day return > ret_thresh  (real momentum)
    - Volume ratio  > vol_thresh  (participation behind the move)
    - ADX           > adx_floor   (confirmed trend strength)
    - DI+ > DI-                   (direction confirmed)
    → Hold 30 days, 25% stop loss

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
#   MomRet      — min 60d return to trigger MOM (positive %)   [unchanged, v1]
#   MomVol      — min volume ratio vs 20d avg to trigger MOM   [unchanged, v1]
#   MomADX      — min ADX to trigger MOM (confirms trend)      [unchanged, v1]
#
# REV params below come from a per-stock grid search (lookback window x
# return threshold x volume threshold) maximizing 25%-SL-capped forward
# return quality, cross-checked against a ground-truth local-minima audit
# for BSE/EDELWEISS/EICHER/INDOTHAI/ORIRAIL/TITAGARH. E2E/HINDZINC/PARAS had
# no combination clear a positive-return bar — left on a conservative
# fallback and flagged; don't expect REV to fire much on those three.

STOCKS = {
    "BSE": {
        "ticker":   "BSE.NS",
        "csv_path": "Data/bse.csv",
        "RevWindow": 20, "RevDD": -3,  "RevVol": 0.7,
        "MomRet": 10, "MomVol": 1.5, "MomADX": 30,
    },
    "CDSL": {
        "ticker":   "CDSL.NS",
        "csv_path": "Data/cdsl.csv",
        "RevWindow": 20, "RevDD": -3,  "RevVol": 0.7,
        "MomRet": 20, "MomVol": 1.2, "MomADX": 25,
    },
    "EDELWEISS": {
        "ticker":   "EDELWEISS.BO",
        "csv_path": "Data/edelweiss.csv",
        "RevWindow": 5,  "RevDD": -3,  "RevVol": 0.8,
        "MomRet": 10, "MomVol": 1.5, "MomADX": 30,
    },
    "EICHER": {
        "ticker":   "EICHERMOT.BO",
        "csv_path": "Data/eicher.csv",
        "RevWindow": 10, "RevDD": -3,  "RevVol": 0.8,
        "MomRet": 10, "MomVol": 1.2, "MomADX": 25,
    },
    "HINDCOPPER": {
        "ticker":   "HINDCOPPER.NS",
        "csv_path": "Data/hindcopper.csv",
        "RevWindow": 10, "RevDD": -3,  "RevVol": 0.7,
        "MomRet": 10, "MomVol": 1.5, "MomADX": 30,
    },
    "HINDZINC": {
        "ticker":   "HINDZINC.BO",
        "csv_path": "Data/hindzinc.csv",
        # No REV combo cleared a positive-return bar in tuning — conservative
        # fallback kept in place, flagged for manual review before trusting.
        "RevWindow": 20, "RevDD": -15, "RevVol": 1.5,
        "MomRet": 10, "MomVol": 1.0, "MomADX": 25,
    },
    "INDOTHAI": {
        "ticker":   "INDOTHAI.NS",
        "csv_path": "Data/indothai.csv",
        "RevWindow": 5,  "RevDD": -3,  "RevVol": 0.8,
        "MomRet": 10, "MomVol": 1.0, "MomADX": 30,
    },
    "ORIRAIL": {
        "ticker":   "ORIRAIL.BO",
        "csv_path": "Data/orirail.csv",
        "RevWindow": 5,  "RevDD": -3,  "RevVol": 0.8,
        "MomRet": 10, "MomVol": 1.0, "MomADX": 25,
    },
    "PARAS": {
        "ticker":   "PARAS.NS",
        "csv_path": "Data/paras.csv",
        # No REV combo cleared a positive-return bar in tuning — conservative
        # fallback kept in place, flagged for manual review before trusting.
        "RevWindow": 20, "RevDD": -15, "RevVol": 1.5,
        "MomRet": 10, "MomVol": 1.0, "MomADX": 30,
    },
    "RECLTD": {
        "ticker":   "RECLTD.NS",
        "csv_path": "Data/recltd.csv",
        "RevWindow": 15, "RevDD": -3,  "RevVol": 0.7,
        "MomRet": 10, "MomVol": 1.5, "MomADX": 30,
    },
    "SBIN": {
        "ticker":   "SBIN.NS",
        "csv_path": "Data/sbin.csv",
        "RevWindow": 20, "RevDD": -3,  "RevVol": 0.7,
        "MomRet": 10, "MomVol": 1.0, "MomADX": 25,
    },
    "TITAGARH": {
        "ticker":   "TITAGARH.NS",
        "csv_path": "Data/titagarh.csv",
        "RevWindow": 5,  "RevDD": -3,  "RevVol": 0.8,
        "MomRet": 10, "MomVol": 1.2, "MomADX": 30,
    },
    "TITAN": {
        "ticker":   "TITAN.NS",
        "csv_path": "Data/titan.csv",
        "RevWindow": 15, "RevDD": -3,  "RevVol": 0.7,
        "MomRet": 10, "MomVol": 1.0, "MomADX": 25,
    },
    "TRENT": {
        "ticker":   "TRENT.NS",
        "csv_path": "Data/trent.csv",
        "RevWindow": 10, "RevDD": -3,  "RevVol": 0.7,
        "MomRet": 10, "MomVol": 1.0, "MomADX": 30,
    },
    "DOLAT": {
        "ticker":   "DOLAT.BO",
        "csv_path": "Data/dolat.csv",
        "RevWindow": 15, "RevDD": -3,  "RevVol": 0.7,
        "MomRet": 15, "MomVol": 1.2, "MomADX": 25,
    },
    "E2E": {
        "ticker":   "E2ENETWORKS.NS",
        "csv_path": "Data/e2e.csv",
        # No REV combo cleared a positive-return bar in tuning — conservative
        # fallback kept in place, flagged for manual review before trusting.
        "RevWindow": 20, "RevDD": -15, "RevVol": 1.5,
        "MomRet": 15, "MomVol": 1.2, "MomADX": 25,
    },
    "EICHERMOT": {
        "ticker":   "EICHERMOT.NS",
        "csv_path": "Data/eichermot.csv",
        "RevWindow": 10, "RevDD": -3,  "RevVol": 0.7,
        "MomRet": 10, "MomVol": 1.2, "MomADX": 25,
    },
    "SARTHAK": {
        "ticker":   "SARTHAKGL.NS",
        "csv_path": "Data/sarthak.csv",
        "RevWindow": 5,  "RevDD": -6,  "RevVol": 1.2,
        "MomRet": 15, "MomVol": 1.2, "MomADX": 25,
    },
    # OGST removed — consistent underperformer on MOM (v1 finding).
    # NOTE: OGST's REV signal alone tested well in this session's audit
    # (+18.8% avg, 80% win, window=10/ret<-3/vol>0.7) — worth reconsidering
    # once MOM is rebuilt, since the original removal reason was MOM-specific.
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
    close = np.array(df["Close"], dtype=float)
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

    # ADX, DI+, DI-
    adxi    = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    adx     = np.array(adxi.adx())
    di_plus = np.array(adxi.adx_pos())
    di_minus= np.array(adxi.adx_neg())

    # Volume ratio vs 20d average
    vol_ma20  = pd.Series(vol).rolling(20).mean().values
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
        close=close, high=high, low=low, vol=vol,
        ma50=ma50, ma50_slope=ma50_slope, price_vs_ma50=price_vs_ma50,
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


def check_mom(ind, i, cfg):
    """
    MOM — Momentum entry. Fires in UP regime only.
    Conditions:
      1. 60d return above ret_thresh  → real medium-term momentum
      2. Volume ratio above vol_thresh → participation behind the move
      3. ADX above adx_floor          → confirmed trend strength
      4. DI+ > DI-                    → direction is upward
    """
    vals = [ind["ret60d"][i], ind["vol_ratio"][i], ind["adx"][i]]
    if any(np.isnan(v) for v in vals):
        return False
    return (ind["ret60d"][i]    > cfg["MomRet"] and
            ind["vol_ratio"][i]  > cfg["MomVol"] and
            ind["adx"][i]        > cfg["MomADX"] and
            ind["di_plus"][i]    > ind["di_minus"][i])


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
        "MOMENTUM ENTRY (UP regime)\n"
        "  Stock has outperformed over 60 days with volume participation\n"
        "  and a confirmed uptrend — trend continuation expected.\n"
        "  Conditions: 60d return > {MomRet}%  |  Volume > {MomVol}x avg  |  ADX > {MomADX}  |  DI+ > DI-\n"
        "  Strategy: Hold 30 days."
    ),

}

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT — 2 subplots: price + BB + MA50, RSI + volume
# ─────────────────────────────────────────────────────────────────────────────

def build_plot(ind, company_name, ticker, date_label, regime, lookback=120):
    n     = len(ind["close"])
    start = max(0, n - lookback)
    x     = np.arange(start, n)

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

    # ── Price + BB + MA50 ────────────────────────────────────────────────
    ax1.plot(x, close,  color="black",    lw=1.3, zorder=3, label="Close")
    ax1.plot(x, bb_up,  color="#27ae60",  lw=0.9, ls="--", label="BB Upper")
    ax1.plot(x, bb_low, color="#e74c3c",  lw=0.9, ls="--", label="BB Lower")
    ax1.plot(x, bb_mid, color="steelblue",lw=0.7, ls=":",  alpha=0.7, label="BB Mid")
    ax1.plot(x, ma50,   color="#f39c12",  lw=1.1, ls="-",  label="MA50")
    ax1.fill_between(x, bb_low, bb_up, alpha=0.06, color="steelblue")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=7, ncol=5, framealpha=0.7)
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
    bar_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(ind["close"][start:n], ind["close"][max(0,start-1):n-1])]
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
    log         = load_log()
    today_label = None

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

        today_label = meta["date"]
        ind = compute_indicators(df)
        i   = len(ind["close"]) - 1  # today's bar

        # ── Regime ───────────────────────────────────────────────────────
        regime = get_regime(ind, i)

        # ── Signals ──────────────────────────────────────────────────────
        # REV: no regime gate (v2 — dropped after ground-truth audit, see docstring)
        # MOM: still regime-gated to UP only — unchanged, momentum redesign pending
        rev_fired = check_rev(ind, i, cfg)
        mom_fired = (regime == "UP") and check_mom(ind, i, cfg)
        rev_w_ret = rev_window_return(ind, i, cfg["RevWindow"])

        # ── Print daily status ────────────────────────────────────────────
        print(f"  Regime    : {regime}")
        print(f"  Close     : {ind['close'][i]:.2f}")
        print(f"  {cfg['RevWindow']}d ret (REV): {rev_w_ret:.1f}%  |  60d ret (MOM): {ind['ret60d'][i]:.1f}%")
        print(f"  Vol ratio : {ind['vol_ratio'][i]:.2f}x  |  ADX: {ind['adx'][i]:.1f}  |  DI+: {ind['di_plus'][i]:.1f}  DI-: {ind['di_minus'][i]:.1f}")
        print(f"  REV fired : {'YES' if rev_fired else 'no'}  |  MOM fired: {'YES' if mom_fired else 'no'}")

        if not rev_fired and not mom_fired:
            continue

        # ── Build email section ───────────────────────────────────────────
        lines = [
            f"\n{'='*60}",
            f"{meta['company']} ({meta['ticker']})  —  {meta['date']}",
            f"{'='*60}",
            f"Regime   : {regime}",
            f"Close    : {ind['close'][i]:.2f}",
            f"{cfg['RevWindow']}d ret (REV): {rev_w_ret:.1f}%   60d ret (MOM): {ind['ret60d'][i]:.1f}%",
            f"Vol ratio: {ind['vol_ratio'][i]:.2f}x   ADX: {ind['adx'][i]:.1f}   DI+: {ind['di_plus'][i]:.1f}  DI-: {ind['di_minus'][i]:.1f}",
            f"MA50 slope (20d): {ind['ma50_slope'][i]:.2f}%",
        ]

        for sig_name, fired in [("REV", rev_fired), ("MOM", mom_fired)]:
            if not fired:
                continue
            log_key = f"{stock_name}_{sig_name}"
            already = log.get(log_key) == today_label
            tag = "  [already sent today]" if already else ""
            desc = SIGNAL_DESCRIPTIONS[sig_name].format(**cfg)
            lines.append(f"\n{'─'*40}")
            lines.append(f"SIGNAL: {sig_name}{tag}")
            lines.append(desc)
            if not already:
                log[log_key] = today_label

        report_sections.append("\n".join(lines))

        # ── Build chart ───────────────────────────────────────────────────
        png = build_plot(ind, meta["company"], meta["ticker"], meta["date"], regime)
        plot_attachments.append((f"{stock_name}_{today_label}.png", png))

    # ── Save state ────────────────────────────────────────────────────────
    save_log(log)

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
