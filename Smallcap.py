"""
Smallcap_Combined.py  —  Daily Signal Scanner (Nifty Microcap/Smallcap 250)
===========================================================================
Same operational shape as Combined.py (fetch -> update CSV -> check signals ->
email with charts, with per-day de-duplication), but for the ~250-stock
Smallcap universe and with two NEW signals derived from scratch on that
universe.

  UNIVERSE
    Read from universe_manifest.csv (Label, Ticker, Sector) with tickers
    cross-checked against ticker_cache.json. CSV paths are rebuilt as
    Smallcap/<Sector_with_underscores>/<label>.csv -- the manifest stores
    absolute Windows paths, which do not work in GitHub Actions.

  REV — "BOUNCE" (mean reversion, two-stage)
    Stage 1 casts a wide net on price dislocation; Stage 2 filters for
    whether the bottom will actually PAY. The two feature sets are nearly
    disjoint -- dislocation features locate a bottom (near-tautologically)
    but carry almost no information about forward return, which is governed
    by volatility regime and cycle position instead.
      Walk-forward OOS: +5.79% avg, 58.6% win, 28.6% recall, 42.3% precision
      (base rate +3.06%), positive edge in all 4 folds INCLUDING both
      down-market folds.
    Rule fallback (T3 tier): N=4338, +7.67% avg, 63.3% win vs +3.06% base.
    → EXIT: 30-day hold, 25% hard stop.

  MOM — "LEG" (momentum-leg inception)
    Targets the START of a >=10%-within-10-days leg, but only legs
    originating from STRENGTH. This distinction is the whole signal: legs
    from near 60d highs continue (+7.08% after the leg peak, 56.7%
    follow-through, -13.75% MAE) while legs off deep drawdowns mean-revert
    (+3.27%, 50.1%, -16.39% MAE) -- i.e. a reversal bounce gives it back, a
    real momentum leg does not.
      NOTE: "no local minima before the rise" was tested as a discriminator
      and REJECTED -- after-leg return is flat across trough-proximity
      buckets (5.24 / 5.30 / 5.80). Proximity to highs is what matters.
    Rule fallback: 3.4x lift on leg-start rate (38.7% vs 11.4% base).
    → EXIT: 21 trading days (~1 month), plain hold.
      Early-cut variants were explicitly tested and ALL underperformed:
        raw 21d hold  +5.50%  (50.0% win)
        trail 10/-12  +4.44%  (41.1% win)   <-- cutting on weakness LOSES
        trail 15/-15  +5.01%  (44.6% win)
        trail 25/-25  +5.31%  (49.1% win)
      On this horizon these names dip and recover inside the window, so
      exiting on weakness sells the dip. Hold the month.

  MODELS
    If bounce_model.pkl / leg_model.pkl are present AND scikit-learn imports
    cleanly, the ML versions are used (better validated). Otherwise the
    scanner falls back to the validated rule tiers automatically and says so
    in the email. This keeps the file runnable in GitHub Actions even if
    scikit-learn is absent or a pickle was written by another sklearn build.
    To use the ML path, add `scikit-learn` to requirements.txt.

Run daily. Same email/env conventions as Combined.py (EMAIL_PASSWORD secret).
"""

import os
import io
import json
import smtplib
import warnings
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MANIFEST_PATH  = "universe_manifest.csv"
TICKER_CACHE   = "ticker_cache.json"
DATA_ROOT      = "Smallcap"
LOG_PATH       = "smallcap_email_log.json"
MOM_LOG_PATH   = "smallcap_mom_log.json"

BOUNCE_MODEL   = "bounce_model.pkl"
LEG_MODEL      = "leg_model.pkl"

EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

MIN_ROWS       = 260      # z-scores/252d ranks need ~1yr; below this signals are unreliable
PLOT_LOOKBACK  = 200
MAX_CHARTS     = 25       # cap attachments so the email doesn't balloon

# ── Data-quality exclusions ────────────────────────────────────────────────
# Same bug class as AFCOM in Combined.py: an unadjusted split/bonus shows up as
# a huge overnight price drop with Stock Splits == 0, which manufactures phantom
# REV/MOM fires. These were confirmed by inspection; DIACABS additionally fails
# liquidity outright (431 zero-volume days, ~Rs 0.02 Cr/day median turnover).
EXCLUDE = {"SKFINDIA", "DIACABS", "PARAS", "REDTAPE", "JSLL", "TRIVENI", "STAR", "QUESS"}

# Runtime guard: catch NEW unadjusted corporate actions as they appear, so the
# hardcoded list above doesn't silently go stale.
SPLIT_DROP_PCT   = -35.0   # single-day close-to-close move this negative...
SPLIT_LOOKBACK   = None    # ...anywhere in history (None = full), with no split flag
MIN_MEDIAN_TURNOVER_CR = 0.5   # skip untradeable names (Rs crore/day, median)

# REV exit
REV_HOLD, REV_SL = 30, -25.0
# MOM exit
MOM_HOLD = 21

# ── Rule-tier thresholds (used when models are unavailable) ─────────────────
# REV "T3 moderate" tier from the validated recall/return frontier.
REV_PX_MA10, REV_Z5, REV_DST, REV_UPL, REV_ATR, REV_RET60 = -5, -1.0, 10, 10, 3.5, -30
# MOM rule, grid-searched for leg-start precision.
MOM_RET5, MOM_PCT250, MOM_EFF, MOM_ADX, MOM_DD60 = 8, 95, 0.25, 25, -8


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def sector_folder(sector):
    keep = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in str(sector))
    return keep.strip().replace(" ", "_")


def load_universe():
    """Label -> dict(ticker, csv_path, sector). Paths rebuilt relative to the
    repo root; the manifest's CSV_Path column holds absolute Windows paths."""
    man = pd.read_csv(MANIFEST_PATH)
    man.columns = man.columns.str.strip()

    cache = {}
    if os.path.exists(TICKER_CACHE):
        try:
            with open(TICKER_CACHE) as f:
                cache = json.load(f) or {}
        except json.JSONDecodeError:
            cache = {}

    uni = {}
    for _, r in man.iterrows():
        if str(r.get("Resolved", "Y")).strip().upper() != "Y":
            continue
        label = str(r["Label"]).strip()
        ticker = cache.get(label) or str(r.get("Ticker", "")).strip()
        if not ticker or ticker.lower() == "nan":
            continue
        uni[label] = dict(
            ticker=ticker,
            sector=str(r.get("Sector", "Unclassified")).strip(),
            csv_path=os.path.join(DATA_ROOT, sector_folder(r.get("Sector", "Unclassified")),
                                  f"{label.lower()}.csv"),
        )
    return uni


# ─────────────────────────────────────────────────────────────────────────────
# 2. FETCH + UPDATE CSV   (same pattern/schema as Combined.py)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_update_csv(ticker, csv_path):
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
    except Exception:
        info = {}
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
        "Volume":       volume or int(hist["Volume"].iloc[-1]),
        "Avg_Volume":   avg_volume,
        "Dividends":    round(float(hist.get("Dividends", pd.Series([0])).iloc[-1]), 2),
        "Stock Splits": round(float(hist.get("Stock Splits", pd.Series([0])).iloc[-1]), 2),
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
    meta = dict(company=company_name, ticker=ticker, date=today_str)
    return df, meta


def read_clean(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)


def data_quality_ok(df):
    """Returns (ok, reason). Guards against the two failure modes that produce
    phantom signals: unadjusted corporate actions, and untradeable illiquidity."""
    c = df["Close"].values.astype(float)
    if len(c) < 30:
        return False, "too few rows"

    tail = df if SPLIT_LOOKBACK is None else df.tail(SPLIT_LOOKBACK)
    ct = tail["Close"].values.astype(float)
    splits = (tail["Stock Splits"].values.astype(float)
              if "Stock Splits" in tail.columns else np.zeros(len(ct)))
    rets = np.diff(ct) / ct[:-1] * 100
    for k, r in enumerate(rets):
        if r <= SPLIT_DROP_PCT and (np.isnan(splits[k+1]) or splits[k+1] == 0):
            d = tail["Date"].iloc[k+1]
            return False, (f"unadjusted corporate action? {r:.1f}% drop on "
                           f"{pd.Timestamp(d).date()} with no split flag")

    recent = df.tail(250)
    v = recent["Volume"].values.astype(float)
    turnover_cr = np.nanmedian(recent["Close"].values.astype(float) * v) / 1e7
    if turnover_cr < MIN_MEDIAN_TURNOVER_CR:
        return False, f"illiquid (median turnover Rs {turnover_cr:.2f} Cr/day)"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. INDICATORS  (only what the two signals actually need)
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df):
    c = df["Close"].values.astype(float); h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float);   o = df["Open"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c); S = pd.Series(c); F = {}
    logr = np.concatenate([[np.nan], np.diff(np.log(np.maximum(c, 1e-9)))])

    for w in (5, 10, 20, 40, 60, 120, 250):
        r = np.full(n, np.nan); r[w:] = (c[w:] - c[:-w]) / c[:-w] * 100
        F[f"ret{w}"] = r

    # volatility-normalised 5d dislocation (REV trigger)
    s5 = pd.Series(F["ret5"])
    F["z5"] = ((s5 - s5.rolling(252, min_periods=60).mean())
               / s5.rolling(252, min_periods=60).std()).values

    ma = {w: S.rolling(w).mean().values for w in (10, 20, 50, 200)}
    for w in (10, 20, 50, 200):
        F[f"px_vs_ma{w}"] = np.where(ma[w] > 0, (c - ma[w]) / ma[w] * 100, np.nan)
    F["ma_aligned"] = ((ma[10] > ma[20]) & (ma[20] > ma[50]) & (ma[50] > ma[200])).astype(float)
    sl = np.full(n, np.nan)
    sl[10:] = (ma[50][10:] - ma[50][:-10]) / np.where(ma[50][:-10] == 0, np.nan, ma[50][:-10]) * 100
    F["ma50_slope"] = sl
    F["ma10"], F["ma20"], F["ma50"] = ma[10], ma[20], ma[50]

    rmin252 = S.rolling(252, min_periods=20).min().values
    F["up_from_low252"] = (c - rmin252) / rmin252 * 100
    rmax60 = S.rolling(60, min_periods=20).max().values
    F["dd60"] = (c - rmax60) / rmax60 * 100
    rmax250 = S.rolling(250, min_periods=20).max().values
    F["pct_of_250high"] = np.where(rmax250 > 0, c / rmax250 * 100, np.nan)

    # trend efficiency (Kaufman): net move / total path travelled
    absd = np.abs(np.concatenate([[0.0], np.diff(c)]))
    path40 = pd.Series(absd).rolling(40).sum().values
    net40 = np.full(n, np.nan); net40[40:] = np.abs(c[40:] - c[:-40])
    F["eff_ratio40"] = np.where(path40 > 0, net40 / path40, np.nan)

    # ATR / ADX
    tr = np.full(n, np.nan); pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        u_, d_ = h[i]-h[i-1], l[i-1]-l[i]
        pdm[i] = u_ if (u_ > d_ and u_ > 0) else 0.0
        ndm[i] = d_ if (d_ > u_ and d_ > 0) else 0.0
    F["atr_pct"] = np.where(c > 0, pd.Series(tr).rolling(14).mean().values / c * 100, np.nan)

    def wil(x, p=14):
        out = np.full(n, np.nan); acc = np.nan
        for i in range(1, n):
            val = x[i] if not np.isnan(x[i]) else 0.0
            acc = val if np.isnan(acc) else acc - acc/p + val
            if i >= p: out[i] = acc
        return out
    a_, p_, m_ = wil(tr), wil(pdm), wil(ndm)
    with np.errstate(divide="ignore", invalid="ignore"):
        dip = np.where(a_ > 0, 100*p_/a_, np.nan)
        dim = np.where(a_ > 0, 100*m_/a_, np.nan)
        dx = np.where((dip+dim) > 0, 100*np.abs(dip-dim)/(dip+dim), np.nan)
    F["adx"] = pd.Series(dx).rolling(14).mean().values
    F["di_plus"], F["di_minus"] = dip, dim

    # days since last CONFIRMED trough. argrelextrema only confirms an extremum
    # with `order` bars either side WITHIN the supplied data; since live data
    # ends today, this is look-ahead-safe in production.
    tro = argrelextrema(c, np.less_equal, order=5)[0]
    dst = np.full(n, np.nan); lt, ti = -1, 0
    for i in range(n):
        while ti < len(tro) and tro[ti] <= i:
            lt = tro[ti]; ti += 1
        if lt >= 0: dst[i] = i - lt
    F["days_since_trough"] = dst

    # volume + RSI (RSI/BB for the chart)
    vma20 = pd.Series(v).rolling(20).mean().values
    F["vol_r"] = np.where(vma20 > 0, v / vma20, np.nan)
    F["vol_ma20"] = vma20
    d_ = np.concatenate([[0.0], np.diff(c)])
    g_ = pd.Series(np.where(d_ > 0, d_, 0.0)).ewm(alpha=1/14, adjust=False).mean().values
    l2 = pd.Series(np.where(d_ < 0, -d_, 0.0)).ewm(alpha=1/14, adjust=False).mean().values
    F["rsi"] = np.where(l2 > 0, 100 - 100/(1 + g_/l2), 100.0)
    bm = S.rolling(20).mean().values; bs = S.rolling(20).std().values
    F["bb_mid"], F["bb_up"], F["bb_low"] = bm, bm + 2*bs, bm - 2*bs

    F["close"], F["high"], F["low"], F["open"], F["vol"] = c, h, l, o, v
    return F


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {"bounce": None, "leg": None, "ok": False}


def try_load_models():
    """ML path is optional. If sklearn or a pickle is missing/incompatible we
    silently fall back to the validated rule tiers rather than crashing."""
    try:
        import pickle
        import sklearn  # noqa: F401
        if os.path.exists(BOUNCE_MODEL):
            with open(BOUNCE_MODEL, "rb") as f:
                MODELS["bounce"] = pickle.load(f)
        if os.path.exists(LEG_MODEL):
            with open(LEG_MODEL, "rb") as f:
                MODELS["leg"] = pickle.load(f)
        MODELS["ok"] = MODELS["bounce"] is not None
    except Exception as e:
        print(f"  [models] unavailable ({e}) — using validated rule tiers")
        MODELS["ok"] = False
    return MODELS["ok"]


def check_rev(F, i):
    """REV / BOUNCE — rule tier (T3). Deep-but-recoverable dislocation, near a
    confirmed trough, already off the 52w low, volatile enough to bounce."""
    keys = ["px_vs_ma10", "z5", "days_since_trough", "up_from_low252", "atr_pct", "ret60"]
    if any(np.isnan(F[k][i]) for k in keys):
        return False
    return (F["px_vs_ma10"][i] < REV_PX_MA10 and
            F["z5"][i] < REV_Z5 and
            F["days_since_trough"][i] <= REV_DST and
            F["up_from_low252"][i] >= REV_UPL and
            F["atr_pct"][i] >= REV_ATR and
            F["ret60"][i] >= REV_RET60)


def check_mom(F, i):
    """MOM / LEG — rule tier. Leg inception FROM STRENGTH: pushing up, near its
    250d high, clean/efficient trend, MAs stacked, ADX confirming."""
    keys = ["ret5", "pct_of_250high", "eff_ratio40", "adx", "dd60", "ma_aligned"]
    if any(np.isnan(F[k][i]) for k in keys):
        return False
    return (F["ret5"][i] >= MOM_RET5 and
            F["pct_of_250high"][i] >= MOM_PCT250 and
            F["eff_ratio40"][i] >= MOM_EFF and
            F["adx"][i] >= MOM_ADX and
            F["dd60"][i] >= MOM_DD60 and
            F["ma_aligned"][i] == 1)


SIGNAL_DESCRIPTIONS = {
    "REV": (
        "REVERSAL ENTRY — 'BOUNCE'\n"
        "  Volatility-normalised dislocation near a CONFIRMED trough, in a name\n"
        "  already off its 52-week low with its 60-day trend not destroyed.\n"
        "  Built from scratch on the smallcap universe: the features that LOCATE\n"
        "  a bottom and the features that tell you it will PAY are nearly\n"
        "  disjoint -- this uses both.\n"
        "  Conditions: px vs MA10 < {a}%  |  5d z-score < {b}  |  <= {c}d since trough\n"
        "              |  >= {d}% off 52w low  |  ATR >= {e}%  |  60d ret >= {f}%\n"
        "  Validated OOS: +7.67% avg, 63.3% win (base +3.06%).\n"
        "  EXIT: 30-day hold, 25% hard stop."
    ),
    "MOM": (
        "MOMENTUM LEG ENTRY — 'LEG'\n"
        "  Start of a >=10%-within-10-days leg, but only legs originating from\n"
        "  STRENGTH. Legs from near 60d highs CONTINUE (+7.08% after the leg\n"
        "  peak, 56.7% follow-through); legs off deep drawdowns mean-revert\n"
        "  (+3.27%, 50.1%) -- that is the bounce-vs-real-leg distinction.\n"
        "  Conditions: 5d ret >= {a}%  |  >= {b}% of 250d high  |  trend efficiency >= {c}\n"
        "              |  ADX >= {d}  |  dd from 60d high >= {e}%  |  MA10>MA20>MA50>MA200\n"
        "  Validated OOS: 38.7% become real legs vs 11.4% base (3.4x lift).\n"
        "  EXIT: hold 21 trading days (~1 month). Do NOT cut early on weakness --\n"
        "  every trailing-stop variant tested UNDERPERFORMED the plain hold\n"
        "  (trail10 +4.44% vs hold +5.50%); these names dip and recover inside\n"
        "  the month."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT — price + BB + MAs, RSI + volume
# ─────────────────────────────────────────────────────────────────────────────

def build_plot(F, company, ticker, date_label, kinds, lookback=PLOT_LOOKBACK,
               rev_fires=None, mom_fires=None):
    n = len(F["close"]); start = max(0, n - lookback); x = np.arange(start, n)
    o, h, l, c = (F["open"][start:n], F["high"][start:n],
                  F["low"][start:n], F["close"][start:n])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                   gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    fig.suptitle(f"{company} ({ticker})  —  {date_label}  |  {' + '.join(kinds)}",
                 fontsize=11, fontweight="bold")

    up, dn = "#26a69a", "#ef5350"
    for xi, oo, hh, ll, cc in zip(x, o, h, l, c):
        col = up if cc >= oo else dn
        ax1.vlines(xi, ll, hh, color=col, lw=0.7, zorder=2)
        b0, b1 = min(oo, cc), max(oo, cc)
        if b1 - b0 < 1e-9:
            ax1.hlines(oo, xi-0.3, xi+0.3, color=col, lw=1.0, zorder=3)
        else:
            ax1.add_patch(plt.Rectangle((xi-0.3, b0), 0.6, b1-b0,
                                        facecolor=col, edgecolor=col, lw=0.5, zorder=3))

    ax1.plot(x, F["bb_up"][start:n],  color="#27ae60", lw=0.9, ls="--", label="BB Upper")
    ax1.plot(x, F["bb_low"][start:n], color="#e74c3c", lw=0.9, ls="--", label="BB Lower")
    ax1.plot(x, F["bb_mid"][start:n], color="#7f8c8d", lw=0.7, ls=":",  label="BB Mid")
    ax1.fill_between(x, F["bb_low"][start:n], F["bb_up"][start:n], alpha=0.05, color="steelblue")
    ax1.plot(x, F["ma50"][start:n], color="#f39c12", lw=1.1, label="MA50")
    ax1.plot(x, F["ma20"][start:n], color="steelblue", lw=0.8, ls="--", alpha=0.7, label="MA20")

    span = np.nanmax(h) - np.nanmin(l); off = span * 0.035
    for fires, col, lab in ((rev_fires, "#2ecc71", "REV"), (mom_fires, "#8e44ad", "MOM")):
        vis = [i for i in (fires or []) if start <= i < n]
        if vis:
            ax1.scatter(vis, [F["low"][i] - off for i in vis], marker="^", s=95,
                        color=col, edgecolor="black", lw=0.7, zorder=6, label=f"{lab} fire")

    ax1.set_ylabel("Price"); ax1.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.75)
    ax1.grid(alpha=0.25)

    ax2.plot(x, F["rsi"][start:n], color="darkorange", lw=1.1, label="RSI(14)")
    ax2.axhline(70, color="#e74c3c", ls="--", lw=0.7); ax2.axhline(30, color="#27ae60", ls="--", lw=0.7)
    ax2.set_ylim(0, 100); ax2.set_ylabel("RSI")
    ax3 = ax2.twinx()
    ax3.bar(x, F["vol"][start:n], color=[up if cc >= oo else dn for cc, oo in zip(c, o)],
            alpha=0.3, width=0.8)
    ax3.plot(x, F["vol_ma20"][start:n], color="#78909c", lw=0.8, ls="--", alpha=0.7)
    ax3.set_ylabel("Volume", fontsize=8); ax3.tick_params(labelsize=7)
    ax2.set_xlabel("Bar index"); ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.25)

    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# 6. LOGS + EMAIL   (same pattern as Combined.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        content = f.read().strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def send_email(subject, body, attachments):
    msg = MIMEMultipart()
    msg["Subject"] = subject; msg["From"] = EMAIL_SENDER; msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(body, "plain"))
    for fname, png in attachments:
        msg.attach(MIMEImage(png, name=fname))
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
# 7. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    universe = load_universe()
    print(f"Universe: {len(universe)} stocks")
    using_ml = try_load_models()
    print(f"Signal engine: {'ML models' if using_ml else 'validated rule tiers'}")

    log = _load(LOG_PATH); mom_log = _load(MOM_LOG_PATH)
    today_label = None
    sections, charts = [], []
    rev_hits, mom_hits = [], []

    skipped_quality = []
    for name, cfg in universe.items():
        if name in EXCLUDE:
            continue
        try:
            df, meta = fetch_and_update_csv(cfg["ticker"], cfg["csv_path"])
        except Exception as e:
            print(f"── {name}: fetch failed ({e})")
            continue
        if df is None:
            continue

        try:
            clean = read_clean(cfg["csv_path"])
        except Exception as e:
            print(f"── {name}: read failed ({e})")
            continue
        if len(clean) < MIN_ROWS:
            print(f"── {name}: only {len(clean)} rows (need {MIN_ROWS}) — skipped")
            continue

        ok, why = data_quality_ok(clean)
        if not ok:
            print(f"── {name}: SKIPPED — {why}")
            skipped_quality.append(f"{name} ({why})")
            continue

        today_label = meta["date"]
        try:
            F = compute_indicators(clean)
        except Exception as e:
            print(f"── {name}: indicators failed ({e}) — skipped")
            continue
        i = len(F["close"]) - 1

        rev = check_rev(F, i)
        mom = check_mom(F, i)

        # MOM de-dup: don't refire on consecutive days of the same leg
        if mom:
            last_bar = mom_log.get(name, -10**9)
            if i - last_bar < 10:
                mom = False
            else:
                mom_log[name] = i

        print(f"── {name:12s} close={F['close'][i]:9.2f}  z5={F['z5'][i]:6.2f}  "
              f"dd60={F['dd60'][i]:7.2f}%  ADX={F['adx'][i]:5.1f}  "
              f"REV={'YES' if rev else 'no':3s}  MOM={'YES' if mom else 'no'}")

        if not (rev or mom):
            continue
        if rev: rev_hits.append(name)
        if mom: mom_hits.append(name)

        kinds = [k for k, on in (("REV", rev), ("MOM", mom)) if on]
        lines = [
            f"\n{'='*64}",
            f"{meta['company']} ({meta['ticker']})  —  {meta['date']}   [{cfg['sector']}]",
            f"{'='*64}",
            f"Close        : {F['close'][i]:.2f}",
            f"5d ret       : {F['ret5'][i]:+.2f}%    5d z-score: {F['z5'][i]:+.2f}",
            f"vs MA10      : {F['px_vs_ma10'][i]:+.2f}%   vs MA50: {F['px_vs_ma50'][i]:+.2f}%",
            f"dd from 60dH : {F['dd60'][i]:+.2f}%   % of 250d high: {F['pct_of_250high'][i]:.1f}%",
            f"off 52w low  : {F['up_from_low252'][i]:+.2f}%   days since trough: {F['days_since_trough'][i]:.0f}",
            f"ATR          : {F['atr_pct'][i]:.2f}%   ADX: {F['adx'][i]:.1f}   vol: {F['vol_r'][i]:.2f}x",
        ]
        for k in kinds:
            key = f"{name}_{k}"
            already = log.get(key) == today_label
            if k == "REV":
                desc = SIGNAL_DESCRIPTIONS["REV"].format(
                    a=REV_PX_MA10, b=REV_Z5, c=REV_DST, d=REV_UPL, e=REV_ATR, f=REV_RET60)
            else:
                desc = SIGNAL_DESCRIPTIONS["MOM"].format(
                    a=MOM_RET5, b=MOM_PCT250, c=MOM_EFF, d=MOM_ADX, e=MOM_DD60)
            lines.append(f"\n{'-'*44}")
            lines.append(f"SIGNAL: {k}" + ("   [already sent today]" if already else ""))
            lines.append(desc)
            if not already:
                log[key] = today_label
        sections.append("\n".join(lines))

        if len(charts) < MAX_CHARTS:
            hist_rev = [k for k in range(len(F["close"])) if check_rev(F, k)]
            hist_mom = [k for k in range(len(F["close"])) if check_mom(F, k)]
            try:
                png = build_plot(F, meta["company"], meta["ticker"], meta["date"], kinds,
                                 rev_fires=hist_rev, mom_fires=hist_mom)
                charts.append((f"{name}_{today_label}.png", png))
            except Exception as e:
                print(f"   chart failed: {e}")

    _save(LOG_PATH, log); _save(MOM_LOG_PATH, mom_log)

    if not sections:
        print("\nNo signals today — no email sent.")
        return

    body = (
        f"SMALLCAP DAILY SCAN  —  {today_label}\n"
        f"Engine   : {'ML models' if MODELS['ok'] else 'validated rule tiers'}\n"
        f"Universe : {len(universe)} stocks\n"
        f"REV fires: {len(rev_hits)}  ({', '.join(rev_hits) if rev_hits else '-'})\n"
        f"MOM fires: {len(mom_hits)}  ({', '.join(mom_hits) if mom_hits else '-'})\n"
        f"\nEXITS — REV: 30-day hold, 25% hard stop.  MOM: hold 21 trading days.\n"
        f"MOM reminder: do NOT cut early on weakness — every trailing-stop\n"
        f"variant tested underperformed the plain 1-month hold.\n"
        + (f"\nDATA-QUALITY SKIPS ({len(skipped_quality)}): "
           f"{'; '.join(skipped_quality)}\n" if skipped_quality else "")
        + ("\n(charts capped at %d attachments)\n" % MAX_CHARTS
           if len(sections) > MAX_CHARTS else "")
        + "\n".join(sections)
    )
    subject = f"[Smallcap Scanner] {len(rev_hits)} REV / {len(mom_hits)} MOM — {today_label}"
    send_email(subject, body, charts)


if __name__ == "__main__":
    main()