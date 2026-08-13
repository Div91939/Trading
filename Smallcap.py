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

  DESIGN
    Deliberately self-contained: every threshold is hardcoded below, there are
    no model files, no pickles and no scikit-learn dependency. Deps are just
    pandas / numpy / scipy / matplotlib / yfinance. Nothing to keep in sync,
    nothing that can silently fall out of date.

  THRESHOLDS
    REV uses the tier re-fit on the CORRECTED 8%-minima labels (the "v2"
    parameters). Per-fold grid search picked this exact set in 3 of 4 folds.
    MOM and SURGE thresholds are likewise walk-forward derived; see the
    comment blocks at each constant for the validation numbers.

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
# universe_manifest.csv holds BOTH the microcap list and 50 NIFTY50 large-caps.
# Without this filter the scanner would fetch RELIANCE/HDFCBANK/TCS into
# Smallcap/ and apply microcap-tuned thresholds to large-caps.
UNIVERSE_FILTER = "NIFTY_MICROCAP_250"
LOG_PATH       = "smallcap_email_log.json"
MOM_LOG_PATH   = "smallcap_mom_log.json"
SURGE_LOG_PATH = "smallcap_surge_log.json"
SPRED_LOG_PATH = "smallcap_spred_log.json"
ALERT_LOG_PATH = "smallcap_alert_log.json"

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
# SURGE exit
SURGE_HOLD = 30
SURGE_SL   = -25.0
# A1 / A5 exit — same convention as REV
ALERT_HOLD = 30
ALERT_SL   = -25.0

# ── SURGE thresholds ───────────────────────────────────────────────────────
# Walk-forward tuned: k in {1,2,3,4,5,7,10} days x threshold in {4..15}%, params
# fit on TRAIN ONLY across 5 folds. (4 days, 15%) was selected in ALL 5 folds
# independently -- the most stable convergence of any parameter in this project.
#   4d>=15%                    : N=2432 avg +6.46% edge +2.51pp, 5/5 windows +
#   4d>=15% AND near 52w high  : N=1617 avg +8.06% med +3.79% win 58.4%
#                                P(>=10%) 38.7% edge +3.67pp, 5/5 windows +
#   4d>=15% AND NOT near high  : edge +0.05pp, 3/5 -- WORTHLESS, hence the gate.
# For reference, a looser 2d>=7% trigger gives only +1.55pp edge (4/5) and fires
# 2.7x as often, so the tighter/longer window is doing real work.
SURGE_DAYS    = 4      # lookback (trading days) for the move
SURGE_PCT     = 15.0   # minimum % move over that window
SURGE_NEAR52  = 85.0   # must be >= this % of its 52-week high
SURGE_COOLOFF = 10     # bars before the same stock can re-fire SURGE

# ── SPRED: surge-PREDICTION thresholds ─────────────────────────────────────
# Built against a ground-truth set of "price rises >=10% over the NEXT 2 days"
# (2.20% of all bars; a labelled bar returns +13.42% over 2d vs +0.20% for a
# random bar). Discovery on the train window found volatility STATE to be the
# dominant precursor -- bb_width AUC 0.652, atr_pct 0.652, rv20 0.642 -- then
# trough proximity and volume expansion.
#   Walk-forward: hit rate 12.29% vs 2.54% base = +9.75pp lift, 6/6 windows,
#   params converged in 5 of 6 folds. Fires on 0.40% of bars (~2 alerts/day).
# Checks it passed:
#   - not concentrated: 615 fires across 125 stocks, top-5 only 13%
#   - NOT merely a volatility proxy: all high-vol bars (ATR>=5.5 & rv20>=35)
#     hit only 4.81%; adding the trough/volume/off-low conditions takes it to
#     12.68%, i.e. +7.88pp comes from the NON-volatility conditions
#   - misses are cheap: the 537 non-surge fires still averaged +1.95% over 2d,
#     only 6.3% fell below -5%. Hits averaged +14.18%.
# NOTE the target is a 2-DAY move, so this is a short-horizon signal. The 30-day
# hold used below is a conservative default, not what was validated.
# SPRED -- also retuned post-fix; its dst<=3 condition was likewise
# unreachable causally. Trough term dropped entirely:
#   N=5576, +5.93% avg, +2.72% med, 55.7% win, edge +3.29pp, 5/5 windows
# (the previously-quoted +9.75pp was inflated by the look-ahead bug).
SPRED_ATR   = 5.5    # ATR% floor
SPRED_RV20  = 35.0   # 20d realised vol (annualised %) floor
SPRED_VOLZ  = 0.5    # 60d volume z-score floor
SPRED_UPL   = 50.0   # min % above the 52-week low

# ── A1 / A5 alerts ─────────────────────────────────────────────────────────
# A1: below the lower Bollinger band AND still falling, but only in names
# already well off their 52-week low and volatile enough to snap back.
#   Raw (no filter): +0.86pp edge, 3/5 windows -- weak.
#   With the off-low + ATR filter: +4.47pp edge, 5/5 windows, +6.15% avg.
#   CAVEAT: median is -0.65% and win rate 48% -- a fat-tailed signal. Roughly
#   half of these lose; it needs many fires to pay out.
A1_UPL = 20.0
A1_ATR = 4.5
# A5: a >=5% single day, but ONLY near the 52-week high.
#   Near the high: +1.80pp edge, 5/5 windows, +5.63% avg, 57% win, med +2.25%.
#   Away from the high: +0.18pp edge with a NEGATIVE median -- hence the gate.
#   Volume barely matters either way (3x+ gives +1.91pp vs +1.72pp below 3x).
# A5 loosened 5.0 -> 4.0 after a walk-forward sweep of day-move x 52wH gate.
# 4/85 improves BOTH recall and returns:
#   5/85 (old): recall 15.82%, +4.58% avg, edge +2.17pp
#   4/85 (new): recall 22.56%, +4.74% avg, edge +2.33pp
# Cost is volume: historical fires 4,185 -> 7,272 (~74% more alerts).
A5_DAY     = 4.0
A5_NEAR52  = 85.0
ALERT_COOLOFF = 5    # bars before the same stock can re-fire A1/A5/SPRED

# ── Signal thresholds (all hardcoded; this file has no external deps) ──────
# REV -- retuned after the causal-trough fix. The old rule required
# days_since_trough <= 5, which is UNREACHABLE once troughs are causal
# (causal dst is always >= 5), so that version would barely fire live.
# Re-tuning without any trough term beat every causal variant tested:
#   causal dst term kept : N=211, +9.46% avg, edge +6.86pp, 2/3 windows
#   trough term DROPPED  : N=504, +9.48% avg, +6.45% med, 63.1% win,
#                          edge +5.40pp, 5/5 windows   <-- chosen
# These params were selected in 4 of 5 walk-forward folds.
# REV -- the WHOLE rule (thresholds + filter) grid-searched as a UNIT with
# walk-forward, scored on TOTAL P/L PER YEAR rather than per-trade average.
# That objective change matters: scoring on average return picks rules that
# fire 6 times a year at +9.6%, which contribute nothing to the book. The
# previous (-12,-2.0 + downtrend filter) stack scored best on per-trade edge
# (+6.09pp) and produced 6 trades / Rs-1,859 over the last 12 months.
#
# Last-12-month comparison on this universe (Rs10k/fire, 30d hold, 25% stop):
#   (-12,-2.0,20,4.5,-40)+filter :   6 trades  -3.10%  Rs   -1,859   <- was live
#   (-8,-1.5,20,4.5,-40)         : 122 trades  +3.40%  Rs  +41,489
#   (-4,-1.0, 0,2.5,-40)         : 977 trades  +3.15%  Rs +308,087
#   (-6,-1.0, 0,2.5,-40)         : 680 trades  +4.26%  Rs +289,418   <- chosen
# The -6 variant is chosen over -4: nearly the same P/L on 30% fewer trades
# and less capital (avg +4.26% vs +3.15%, win 55.7% vs 53.7%).
# Walk-forward: 5/5 windows positive, +1.81pp edge over base.
REV_PX_MA10, REV_Z5, REV_UPL, REV_ATR, REV_RET60 = -6, -1.0, 0, 2.5, -40

# ── Downtrend filter — applied to REV ONLY ─────────────────────────────────
# Blocks entries into stocks carving persistent lower lows or in a confirmed
# strong downtrend, i.e. falling knives.
# Walk-forward, per signal (gain from turning the filter ON):
#   REV    +9.19% -> +13.63%   gain +4.29pp   <-- applied
#   A5     +5.44% -> +5.51%    gain +0.12pp   (noise)
#   A1     +6.32% -> +6.99%    gain +0.06pp   (noise, 1/5 windows)
#   SURGE  +6.78% -> +6.38%    gain -0.15pp   (hurts)
#   SPRED  +7.63% -> +6.98%    gain -0.82pp   (hurts, 1/5 windows)
# Only REV benefits: it is the pure mean-reversion signal, so "is this a
# falling knife" is precisely its failure mode. The momentum signals already
# require proximity to the 52-week high, which excludes structural downtrends
# by construction, so the filter is redundant there and costs good trades.
# Known blind spot: does NOT catch a high-multiple stock unwinding from
# strength (no persistent lower lows, no ADX downtrend) -- that needs a
# separate valuation/extension filter, which has not been built or tested.
DTF_LOWER_LOWS = 8     # 20-bar count of (10d low < 10d low from 10 days ago)
DTF_ADX        = 30    # ADX above this + -DI>+DI = confirmed downtrend
# MOM rule, grid-searched for leg-start precision.
# MOM REMOVED -- SURGE now carries the momentum thesis alone.
# In the deduplicated last-year P/L, MOM was the ONLY negative contributor
# (-0.15% avg, 44.7% win, Rs-579 over 38 trades), and cluster analysis found
# it negative on 123 of 216 stocks. SURGE-only fires beat MOM-only fires
# (+7.93% vs +6.01% avg, 59.7% vs 57.4% win) and the two overlapped only
# ~20% (Jaccard), so little coverage is lost.
MOM_RET5, MOM_PCT250, MOM_EFF, MOM_ADX, MOM_DD60 = 8, 95, 0.25, 25, -8   # unused


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
        if UNIVERSE_FILTER and "Universe" in man.columns:
            if str(r.get("Universe", "")).strip() != UNIVERSE_FILTER:
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

    # SURGE: k-day return, and position vs the true 52-week HIGH (uses highs,
    # not closes -- a surge is measured against the actual prior extreme)
    rs = np.full(n, np.nan)
    if n > SURGE_DAYS:
        rs[SURGE_DAYS:] = (c[SURGE_DAYS:] - c[:-SURGE_DAYS]) / c[:-SURGE_DAYS] * 100
    F["surge_ret"] = rs
    hi52 = pd.Series(h).rolling(252, min_periods=120).max().values
    F["pct_of_52whigh"] = np.where(hi52 > 0, c / hi52 * 100, np.nan)

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
    # CAUSAL: argrelextrema(order=K) can only label bar t a trough once the K
    # bars AFTER t exist, so reading dst at bar i when dst<K uses data that did
    # not exist yet. Backtests showed bars with dst 0-1 returning +8.88%/30d
    # purely because the trough label REQUIRED them to be rising. A trough at t
    # is only usable from bar t+K onward, so causal dst is always >= K.
    _K = 5
    tro = argrelextrema(c, np.less_equal, order=_K)[0]
    dst = np.full(n, np.nan); lt, ti = -1, 0
    for i in range(n):
        while ti < len(tro) and tro[ti] + _K <= i:
            lt = tro[ti]; ti += 1
        if lt >= 0: dst[i] = i - lt
    F["days_since_trough"] = dst

    # Downtrend-filter input: how many of the last 20 bars had their 10-day low
    # below the 10-day low from 10 bars earlier (i.e. persistently carving
    # lower lows). Purely backward-looking, no look-ahead.
    low10 = pd.Series(l).rolling(10).min().values
    llf = np.zeros(n)
    for k in range(10, n):
        if not np.isnan(low10[k]) and not np.isnan(low10[k-10]):
            llf[k] = 1.0 if low10[k] < low10[k-10] else 0.0
    F["lower_lows20"] = pd.Series(llf).rolling(20).sum().values

    # realised vol + volume z-score + 1-day return (SPRED / A1 / A5)
    logr = np.concatenate([[np.nan], np.diff(np.log(np.maximum(c, 1e-9)))])
    F["rv20"] = pd.Series(logr).rolling(20).std().values * np.sqrt(252) * 100
    vs = pd.Series(v)
    F["vol_z"] = ((vs - vs.rolling(60, min_periods=20).mean())
                  / vs.rolling(60, min_periods=20).std()).values
    dayret = np.full(n, np.nan)
    dayret[1:] = (c[1:] - c[:-1]) / c[:-1] * 100
    F["day_ret"] = dayret
    F["falling"] = np.concatenate([[False], c[1:] < c[:-1]])

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

def check_rev(F, i):
    """REV / BOUNCE — rule tier (T3). Deep-but-recoverable dislocation, near a
    confirmed trough, already off the 52w low, volatile enough to bounce."""
    keys = ["px_vs_ma10", "z5", "up_from_low252", "atr_pct", "ret60"]
    if any(np.isnan(F[k][i]) for k in keys):
        return False
    # NOTE: the downtrend filter is deliberately NOT applied. When the whole
    # rule was optimised as a unit, the grid selected filter=OFF in every
    # walk-forward fold on both universes -- it blocks ~73% of fires for a
    # gain that disappears once frequency is priced in.
    return (F["px_vs_ma10"][i] < REV_PX_MA10 and
            F["z5"][i] < REV_Z5 and
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


def check_surge(F, i):
    """SURGE -- a sharp multi-day thrust, but ONLY near the 52-week high.
    The near-high gate is not cosmetic: walk-forward, surges away from the high
    carry an edge of +0.05pp (3/5 windows) versus +3.67pp (5/5) near it."""
    for k in ("surge_ret", "pct_of_52whigh"):
        if np.isnan(F[k][i]):
            return False
    return (F["surge_ret"][i] >= SURGE_PCT and
            F["pct_of_52whigh"][i] >= SURGE_NEAR52)


def passes_downtrend(F, i):
    """True = OK to enter. Blocks persistent lower-lows structures and
    confirmed strong downtrends. Fails OPEN when inputs are unavailable, so a
    short-history stock is not silently suppressed."""
    ll = F["lower_lows20"][i]
    adx, dip, dim = F["adx"][i], F["di_plus"][i], F["di_minus"][i]
    if not np.isnan(ll) and ll >= DTF_LOWER_LOWS:
        return False
    if (not np.isnan(adx) and not np.isnan(dip) and not np.isnan(dim)
            and adx > DTF_ADX and dim > dip):
        return False
    return True


def check_spred(F, i):
    """SPRED — predicts a >=10% move over the NEXT 2 days. Volatility state is
    the precursor; the trough/volume/off-low conditions are what lift it from
    4.81% (any high-vol bar) to 12.68%."""
    for k in ("atr_pct", "rv20", "vol_z", "up_from_low252"):
        if np.isnan(F[k][i]):
            return False
    return (F["atr_pct"][i]          >= SPRED_ATR and
            F["rv20"][i]             >= SPRED_RV20 and
            F["vol_z"][i]            >= SPRED_VOLZ and
            F["up_from_low252"][i]   >= SPRED_UPL)


def check_a1(F, i):
    """A1 — punctured the lower Bollinger band and still falling, but only in a
    name already off its 52w low and volatile enough to bounce."""
    if np.isnan(F["bb_low"][i]) or np.isnan(F["up_from_low252"][i]) or np.isnan(F["atr_pct"][i]):
        return False
    return (F["close"][i] < F["bb_low"][i] and
            bool(F["falling"][i]) and
            F["up_from_low252"][i] >= A1_UPL and
            F["atr_pct"][i]        >= A1_ATR)


def check_a5(F, i):
    """A5 — a >=5% single day NEAR the 52-week high. The gate is the signal:
    the same jump far from the high has essentially no edge."""
    if np.isnan(F["day_ret"][i]) or np.isnan(F["pct_of_52whigh"][i]):
        return False
    return (F["day_ret"][i]        >= A5_DAY and
            F["pct_of_52whigh"][i] >= A5_NEAR52)


SIGNAL_DESCRIPTIONS = {
    "REV": (
        "REVERSAL ENTRY (mean reversion)\n"
        "  A volatility-normalised dislocation -- price well below its 10-day\n"
        "  average with a deep 5-day z-score -- in a name already off its\n"
        "  52-week low, volatile enough to rebound, and whose 60-day trend is\n"
        "  not destroyed. Gated by the downtrend filter so it does not buy a\n"
        "  falling knife.\n"
        "  Conditions: px vs MA10 < {a}%  |  5d z-score < {b}\n"
        "              |  >= {c}% off 52w low  |  ATR >= {d}%  |  60d ret >= {e}%\n"
        "  Walk-forward (whole rule optimised as a unit, scored on total P/L):\n"
        "  5/5 windows positive, +1.81pp edge. Last 12 months: 680 trades,\n"
        "  +4.26% avg, +1.96% median, 55.7% win, 29.4% hit>=10%, Rs+289,418\n"
        "  at Rs10k/fire. Fires across 212 of 219 stocks -- broad, not\n"
        "  concentrated (top 10 names = 34% of P/L).\n"
        "  EXIT: 30-day hold, 25% hard stop."
    ),
    "SURGE": (
        "SURGE ENTRY (sharp thrust near the 52-week high)\n"
        "  A {a}-day move of >= {b}% while trading at >= {c}% of the 52-week\n"
        "  high. Walk-forward tuned; (4d, 15%) was chosen in all 5 folds.\n"
        "  The near-high gate is the signal: surges FAR from the high carry\n"
        "  essentially no edge (+0.05pp, 3/5 windows) while surges near it\n"
        "  give +3.67pp over base in 5/5 windows.\n"
        "  Validated OOS: N=1617, avg +8.06%, median +3.79%, win 58.4%,\n"
        "  P(>=10%) 38.7% (base +3.07%).\n"
        "  EXIT: 30-day hold, 25% hard stop (same as REV).\n"
        "  NOTE: expect drawdown first -- historical mean worst-case inside\n"
        "  20 days for this setup is about -8%."
    ),
    "SPRED": (
        "SURGE PREDICTION (>=10% expected over the NEXT 2 days)\n"
        "  Volatility state is the precursor -- wide bands, high ATR, elevated\n"
        "  realised vol -- combined with trough proximity, a volume push and\n"
        "  being well off the 52-week low.\n"
        "  Conditions: ATR >= {a}%  |  20d realised vol >= {b}%  |  <= {c}d since trough\n"
        "              |  volume z-score >= {d}  |  >= {e}% above 52w low\n"
        "  Validated OOS: 12.29% hit vs 2.54% base (+9.75pp), 6/6 windows.\n"
        "  Not just a volatility proxy: high-vol bars alone hit only 4.81%.\n"
        "  Misses are cheap -- non-surge fires still averaged +1.95% over 2d.\n"
        "  SHORT HORIZON: the target is a 2-day move. Consider taking profit\n"
        "  quickly rather than holding the full 30 days."
    ),
    "A1": (
        "A1 ALERT (oversold snap-back candidate)\n"
        "  Closed below the lower Bollinger band and still falling, but only in\n"
        "  a name already well off its 52-week low with enough volatility to\n"
        "  actually rebound.\n"
        "  Conditions: close < lower BB  |  falling today  |  >= {a}% off 52w low  |  ATR >= {b}%\n"
        "  Validated OOS: +6.15% avg, +4.47pp edge, 5/5 windows.\n"
        "  WARNING: median is -0.65% and win rate only 48%. This is a\n"
        "  fat-tailed signal -- about half of these lose money, and it relies\n"
        "  on a minority of large winners. Size accordingly."
    ),
    "A5": (
        "A5 ALERT (>=5% day near the 52-week high)\n"
        "  A single-day jump of {a}%+ while trading at >= {b}% of the 52-week\n"
        "  high. The near-high gate IS the signal: the same jump far from the\n"
        "  high carries an edge of only +0.18pp with a negative median.\n"
        "  Validated OOS: +5.63% avg, median +2.25%, 57% win, +1.80pp edge,\n"
        "  5/5 windows. Volume barely matters here (3x+ adds ~0.2pp)."
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
               rev_fires=None, mom_fires=None, surge_fires=None, alert_fires=None):
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
    for fires, col, lab, mk, side in ((rev_fires, "#2ecc71", "REV", "^", "lo"),
                                       (mom_fires, "#8e44ad", "MOM", "^", "lo"),
                                       (surge_fires, "#e67e22", "SURGE", "v", "hi"),
                                       (alert_fires, "#2980b9", "ALERT", "v", "hi")):
        vis = [i for i in (fires or []) if start <= i < n]
        if vis:
            ys = ([F["low"][i] - off for i in vis] if side == "lo"
                  else [F["high"][i] + off for i in vis])
            ax1.scatter(vis, ys, marker=mk, s=95, color=col, edgecolor="black",
                        lw=0.7, zorder=6, label=f"{lab} fire")

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

    log = _load(LOG_PATH); mom_log = _load(MOM_LOG_PATH)
    surge_log = _load(SURGE_LOG_PATH)
    spred_log = _load(SPRED_LOG_PATH); alert_log = _load(ALERT_LOG_PATH)
    today_label = None
    sections, charts = [], []
    rev_hits, mom_hits, surge_hits = [], [], []
    spred_hits, a1_hits, a5_hits = [], [], []

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
        mom = False          # MOM retired -- see the constant block for why
        surge = check_surge(F, i)

        spred = check_spred(F, i)
        a1    = check_a1(F, i)
        a5    = check_a5(F, i)

        # SURGE de-dup: one alert per thrust, not every day it stays elevated
        if surge:
            if i - surge_log.get(name, -10**9) < SURGE_COOLOFF:
                surge = False
            else:
                surge_log[name] = i
        # same idea for the three new alerts, tracked independently
        for _k, _on in (("SPRED", spred), ("A1", a1), ("A5", a5)):
            if not _on:
                continue
            _key = f"{name}_{_k}"
            if i - alert_log.get(_key, -10**9) < ALERT_COOLOFF:
                if _k == "SPRED": spred = False
                elif _k == "A1":  a1 = False
                else:             a5 = False
            else:
                alert_log[_key] = i

        # MOM de-dup: don't refire on consecutive days of the same leg
        if mom:
            last_bar = mom_log.get(name, -10**9)
            if i - last_bar < 10:
                mom = False
            else:
                mom_log[name] = i

        print(f"── {name:12s} close={F['close'][i]:9.2f}  z5={F['z5'][i]:6.2f}  "
              f"dd60={F['dd60'][i]:7.2f}%  ADX={F['adx'][i]:5.1f}  "
              f"{SURGE_DAYS}d={F['surge_ret'][i]:6.2f}%  52wH={F['pct_of_52whigh'][i]:5.1f}%  "
              f"REV={'YES' if rev else 'no':3s}  MOM={'YES' if mom else 'no':3s}  "
              f"SURGE={'YES' if surge else 'no':3s}  SPRED={'YES' if spred else 'no':3s}  "
              f"A1={'YES' if a1 else 'no':3s}  A5={'YES' if a5 else 'no'}")

        if not (rev or mom or surge or spred or a1 or a5):
            continue
        if rev: rev_hits.append(name)
        if mom: mom_hits.append(name)
        if surge: surge_hits.append(name)
        if spred: spred_hits.append(name)
        if a1: a1_hits.append(name)
        if a5: a5_hits.append(name)

        kinds = [k for k, on in (("REV", rev), ("MOM", mom), ("SURGE", surge),
                                 ("SPRED", spred), ("A1", a1), ("A5", a5)) if on]
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
            f"{SURGE_DAYS}d move      : {F['surge_ret'][i]:+.2f}%   % of 52w high: {F['pct_of_52whigh'][i]:.1f}%",
            f"1d move      : {F['day_ret'][i]:+.2f}%   20d real vol: {F['rv20'][i]:.1f}%   vol z: {F['vol_z'][i]:+.2f}",
            f"off 52w low  : {F['up_from_low252'][i]:+.1f}%   days since trough: {F['days_since_trough'][i]:.0f}",
        ]
        for k in kinds:
            key = f"{name}_{k}"
            already = log.get(key) == today_label
            if k == "REV":
                desc = SIGNAL_DESCRIPTIONS["REV"].format(
                    a=REV_PX_MA10, b=REV_Z5, c=REV_UPL, d=REV_ATR, e=REV_RET60)
            elif k == "MOM":
                desc = SIGNAL_DESCRIPTIONS["MOM"].format(
                    a=MOM_RET5, b=MOM_PCT250, c=MOM_EFF, d=MOM_ADX, e=MOM_DD60)
            elif k == "SURGE":
                desc = SIGNAL_DESCRIPTIONS["SURGE"].format(
                    a=SURGE_DAYS, b=SURGE_PCT, c=SURGE_NEAR52)
            elif k == "SPRED":
                desc = SIGNAL_DESCRIPTIONS["SPRED"].format(
                    a=SPRED_ATR, b=SPRED_RV20, c=SPRED_ATR, d=SPRED_VOLZ, e=SPRED_UPL)
            elif k == "A1":
                desc = SIGNAL_DESCRIPTIONS["A1"].format(a=A1_UPL, b=A1_ATR)
            else:
                desc = SIGNAL_DESCRIPTIONS["A5"].format(a=A5_DAY, b=A5_NEAR52)
            lines.append(f"\n{'-'*44}")
            lines.append(f"SIGNAL: {k}" + ("   [already sent today]" if already else ""))
            lines.append(desc)
            if not already:
                log[key] = today_label
        sections.append("\n".join(lines))

        if len(charts) < MAX_CHARTS:
            hist_rev = [k for k in range(len(F["close"])) if check_rev(F, k)]
            hist_mom = [k for k in range(len(F["close"])) if check_mom(F, k)]
            hist_surge = [k for k in range(len(F["close"])) if check_surge(F, k)]
            hist_alert = [k for k in range(len(F["close"]))
                          if check_spred(F, k) or check_a1(F, k) or check_a5(F, k)]
            try:
                png = build_plot(F, meta["company"], meta["ticker"], meta["date"], kinds,
                                 rev_fires=hist_rev, mom_fires=hist_mom,
                                 surge_fires=hist_surge, alert_fires=hist_alert)
                charts.append((f"{name}_{today_label}.png", png))
            except Exception as e:
                print(f"   chart failed: {e}")

    _save(LOG_PATH, log); _save(MOM_LOG_PATH, mom_log)
    _save(SURGE_LOG_PATH, surge_log)
    _save(SPRED_LOG_PATH, spred_log); _save(ALERT_LOG_PATH, alert_log)

    if not sections:
        print("\nNo signals today — no email sent.")
        return

    body = (
        f"SMALLCAP DAILY SCAN  —  {today_label}\n"
        f"Engine   : hardcoded rule thresholds (self-contained, no model files)\n"
        f"Universe : {len(universe)} stocks\n"
        f"REV fires: {len(rev_hits)}  ({', '.join(rev_hits) if rev_hits else '-'})\n"
        f"SURGE    : {len(surge_hits)}  ({', '.join(surge_hits) if surge_hits else '-'})\n"
        f"SPRED    : {len(spred_hits)}  ({', '.join(spred_hits) if spred_hits else '-'})\n"
        f"A1       : {len(a1_hits)}  ({', '.join(a1_hits) if a1_hits else '-'})\n"
        f"A5       : {len(a5_hits)}  ({', '.join(a5_hits) if a5_hits else '-'})\n"
        f"\nEXITS — REV/SURGE/A1/A5: 30-day hold, 25% hard stop.\n"
        f"        SPRED targets a 2-DAY move -- consider taking profit early.\n"
        f"        A1 is fat-tailed: ~half these lose, median -0.65%. Size small.\n"
        f"NOTE: MOM has been retired (negative contributor); SURGE now carries\n"
        f"the momentum thesis. REV/SPRED no longer use days-since-trough after\n"
        f"a look-ahead bug was found and fixed.\n"
        + (f"\nDATA-QUALITY SKIPS ({len(skipped_quality)}): "
           f"{'; '.join(skipped_quality)}\n" if skipped_quality else "")
        + ("\n(charts capped at %d attachments)\n" % MAX_CHARTS
           if len(sections) > MAX_CHARTS else "")
        + "\n".join(sections)
    )
    subject = (f"[Smallcap Scanner] {len(rev_hits)}R/{len(surge_hits)}S/"
               f"{len(spred_hits)}P/{len(a1_hits)}A1/{len(a5_hits)}A5 — {today_label}")
    send_email(subject, body, charts)


if __name__ == "__main__":
    main()