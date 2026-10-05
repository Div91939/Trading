"""
US_Smallcap.py — US S&P 600 smallcap scanner (single self-contained file)
===========================================================================
US analogue of Smallcap.py, same operational shape: read the stored CSVs ->
fetch today's bar -> append in place -> score signals -> email alerts with
charts, plus a batched day-1 follow-up email.

One file, no project-local imports, no model files — the same self-contained
discipline as Smallcap.py. Only stdlib + yfinance/pandas/numpy/matplotlib
(no scipy: nothing here uses argrelextrema).

Universe is the CONTENTS OF THE CSV FOLDER, not a manifest: one flat
<TICKER>.csv in us_smallcap_data/. Drop a CSV in, it is scanned; delete it,
it is gone.

NO GRADING — deliberate. On US data the A/B/C grader fit to grading.py v2's
target (P(+10% in 10d)) orders correctly on that target but INVERTS on traded
P/L: it selects volatility (A-graded mean ATR 7.62% vs C 4.64%) while the exit
is path-dependent. Suppressing C-grade fires cost 22.7pp of return on deployed
capital in the 2026 walk-forward window (+55.9% -> +33.2%). Not wired in.
See experiment registry rows 63-68.

===========================================================================
CORPORATE ACTIONS — the one place this differs materially from Smallcap.py
===========================================================================
Smallcap.py stores RAW prices plus Dividends / Stock Splits columns and flags
unadjusted actions after the fact. The stored US history was built with
yfinance auto_adjust=True, i.e. already split- and dividend-adjusted.

That creates a failure mode Smallcap.py does not have: when a new split or
dividend occurs, yfinance retroactively re-adjusts the ENTIRE history, so
appending one freshly-adjusted bar onto a series adjusted at an earlier date
puts a silent discontinuity in the middle of the file. Every indicator
downstream then reads a phantom gap.

Two automatic guards:
  1. ACTION DETECTION — a non-zero split or dividend on a fetched bar triggers
     a full re-fetch and rewrite of that ticker, not an append.
  2. DRIFT CHECK — the last CHECK_BARS stored closes are compared against a
     fresh fetch; divergence beyond DRIFT_TOL triggers the same rewrite. This
     catches retroactive adjustments that arrive with no visible action flag.
Re-fetches are counted and named in the email header, so a silent rewrite
never goes unnoticed.

===========================================================================
RUNNING IT
===========================================================================
Manual trigger via workflow_dispatch (see us_signals.yml); the schedule is
left commented out there, matching daily_signals.yml.

A same-trading-day guard stops a second run re-sending the whole scan — the
CSVs still refresh, but no duplicate email. Set FORCE_RUN=1 to override.

===========================================================================
HOW THE THRESHOLDS WERE DERIVED  (read before trusting any of them)
===========================================================================
Data      : 602 iShares IJR (S&P SmallCap 600) constituents, 5y daily OHLCV
            via yfinance, 2021-09-29 -> 2026-09-29, split/dividend adjusted.
            INDV and IVT excluded (stale quotes: 27.8% and 0.6% of bars have
            High==Low on sub-1000-share volume, producing phantom 572% and
            238% one-day moves). 572 usable after the 1,240-bar floor.
Validation: 5 expanding-window walk-forward folds, non-overlapping 130-bar
            OOS blocks, universal thresholds only (no per-stock fitting).
Exit      : the live rule from Smallcap.py's MOM block — 25% hard stop on the
            intraday low, or the close one bar after an order=1 local max at
            least 5% above entry is confirmed, or a 120-bar cap.
Objective : portfolio return on FINITE capital (20 concurrent slots), not
            total P/L and not per-trade average. Total P/L is monotonic in
            trade count and collapsed the grid to its loosest corner for 6 of
            6 signals; per-trade percent is anti-monotonic and made parameter
            stability worse. Percent only fixes it when the denominator is
            capital over time.
Benchmark : 20 slots filled with NO signal earns +15.4%/yr. Anything below
            that is worse than nothing. Fold 3 loses -21.5%/yr unconditionally,
            so 4/5 positive folds is the ceiling for anything long-only here.

Survivorship is the largest effect measured and is NOT fixed: the universe is
current index members, and the benchmark alone pays -2.13% per trade in the
bottom 5y-return quintile against +3.50% in the top, with 42.8% of its trades
landing in that top quintile. Each signal below carries exQ5 (its
within-quintile edge over the benchmark across the bottom four quintiles) and
q>0 (quintiles beaten). Those are the trustworthy numbers. Point-in-time index
membership is the fix and has not been sourced.

  SPRED  exQ5 +3.19pp  5/5 quintiles  +57.1%/yr  5/5 folds   <- best
  SURGE  exQ5 +1.80pp  4/5 quintiles  +32.3%/yr  4/5 folds
  A5     exQ5 +1.28pp  5/5 quintiles  +27.8%/yr  3/5 folds
  REV    exQ5 +1.10pp  4/5 quintiles  +28.8%/yr  4/5 folds
  MOM    exQ5 +0.75pp  3/5 quintiles  +19.5%/yr  4/5 folds   <- shipped at 2.0/gate off;
                                                     Smallcap.py's 3.0/gate-90 measured
                                                     exQ5 -0.41pp, 1 of 5 quintiles here
  A1     exQ5 not run  --             +17.2%/yr  4/5 folds   <- ~benchmark

THE POSITION GATES DO NOT TRANSFER. Every "near the 52w/250d high" and "off
the 52w low" gate that is load-bearing on Indian data wants to be looser or
OFF here: MOM near250 90->OFF, SURGE near52 85->OFF, A5 near52 85->OFF,
SPRED upl 50->25, A1 upl 20->0, REV unchanged. Checked against survivorship
rather than taken at face value — gate-off wins on the survivorship-robust
metric too for SURGE (+1.80 vs +1.59pp) and MOM (+0.75 vs -0.41pp). MOM's live
near250=90 gate is the worst configuration measured anywhere in this project:
exQ5 -0.41pp, beating the benchmark in 1 of 5 quintiles.
===========================================================================
"""

import os
import io
import json
import smtplib
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = "us_smallcap_data"
MIN_ROWS  = 1240          # 250d/252d windows need this much; below it, unreliable
PLOT_LOOKBACK = 200

# Stale-quote contaminated — see the header block.
EXCLUDE = {"INDV", "IVT"}

# Runtime data-quality guards (US analogues of the Indian split/turnover guards)
FLAT_BAR_PCT_MAX   = 2.0     # max % of bars with High==Low before a ticker is dropped
MIN_MEDIAN_DOLLAR_VOL = 1_000_000   # median close*volume, USD/day

# Exits — unchanged from Smallcap.py's validated MOM rule, applied to all
# signals here because that is the rule every number above was measured under.
EXIT_STOP_PCT   = -25.0
EXIT_TARGET_PCT = 5.0
EXIT_MAXHOLD    = 120

SLOTS = 20               # concurrent book capacity the thresholds were fit for


# ── SPRED — strongest signal on US smallcap ────────────────────────────────
# Rule shape unchanged: volatility state + volume expansion + off-the-low.
#   live (Indian) (5.5, 35.0, 0.5, 50.0) : PORT_PCT +?  exQ5 +2.27pp, 4/5 quintiles
#   US retuned    (5.5, 45.0, 0.0, 25.0) : PORT_PCT +57.1%/yr, 5/5 folds,
#                                          exQ5 +3.19pp, 5/5 quintiles,
#                                          avg +3.63%/trade, win 72.5%, n=811
# Only 31.6% of its trades land in the top 5y-return quintile, BELOW the
# benchmark's 42.8% — it under-samples the eventual winners and still wins,
# which is the strongest evidence of a real edge anywhere in this study.
# Changes from live: realised-vol floor TIGHTENED 35 -> 45, volume z-score
# gate turned OFF, off-the-low gate LOOSENED 50 -> 25.
SPRED_ATR  = 5.5
SPRED_RV20 = 45.0
SPRED_VOLZ = 0.0
SPRED_UPL  = 25.0

# ── SURGE ──────────────────────────────────────────────────────────────────
# Rule shape unchanged: a k-day thrust, optionally gated on proximity to the
# 52-week high.
#   live (Indian) (4d, 15%, near85) : PORT_PCT +?  exQ5 +1.49pp, only 2/5 quintiles
#   US retuned    (5d, 10%, OFF)    : PORT_PCT +32.3%/yr, 4/5 folds,
#                                     exQ5 +1.80pp, 4/5 quintiles, avg +2.72%
# The 52w-high gate that is worth +3.67pp vs +0.05pp on Indian data is not
# worth keeping here: with it, exQ5 falls to +1.59pp and the bottom quintile
# goes to -1.05pp. Window lengthened 4 -> 5 days, threshold loosened 15 -> 10%.
SURGE_DAYS   = 5
SURGE_PCT    = 10.0
SURGE_NEAR52 = 0.0        # 0 = gate disabled

# ── REV ────────────────────────────────────────────────────────────────────
# Rule shape unchanged (px_vs_ma10 / z5 / off-low / ATR / ret60 floor).
#   live (Indian) (-6, -1.0, 0, 2.5, -40)
#   US retuned    (-6, -0.5, 0, 3.5, -40) : PORT_PCT +28.8%/yr, 4/5 folds,
#                                           exQ5 +1.10pp, 4/5 quintiles,
#                                           avg +2.31%, win 75.9%, n=643
# Only the dislocation z-score and the ATR floor move: the z-score threshold
# LOOSENS (-1.0 -> -0.5) and the volatility floor TIGHTENS (2.5 -> 3.5),
# consistent with US smallcaps having a median ATR14 of 3.24% vs the much
# higher Indian microcap level — the same absolute floor sits in a different
# part of the distribution.
# REV takes only 24.2% of its trades in the top 5y-return quintile, the lowest
# of any signal here, so its (modest) edge is the least survivorship-flattered.
REV_PX_MA10, REV_Z5, REV_UPL, REV_ATR, REV_RET60 = -6, -0.5, 0, 3.5, -40

# ── A5 ─────────────────────────────────────────────────────────────────────
# Rule shape unchanged: a large single day, optionally gated near the 52w high.
# THIS ONE IS UNRESOLVED — the two criteria disagree, so the choice below is a
# judgement call, not a validated result:
#   (6.0, OFF)    PORT_PCT +27.8%/yr, exQ5 +1.28pp, 5/5 quintiles   <- shipped
#   (6.0, near85) PORT_PCT lower,     exQ5 +1.88pp, 3/5 quintiles
# Gate-off is more CONSISTENT (beats the benchmark in every quintile) while
# gate-on has larger MAGNITUDE but leans harder on the survivors (42.6% of its
# trades in the top quintile vs 30.1%). Consistency was preferred because the
# survivorship bias is unquantified. Revisit once point-in-time membership
# exists — this is the threshold most likely to flip.
# Day threshold tightened 4.0 -> 6.0, reflecting that a 4% day is a much more
# common event in this universe.
A5_DAY     = 6.0
A5_NEAR52  = 0.0          # 0 = gate disabled

# ── MOM ───────────────────────────────────────────────────────────────────
# SET TO 2.0% WITH NO POSITION GATE — the pair the US walk-forward selected.
# This DIVERGES from Smallcap.py / Combined.py, which both run 3.0% with
# pct_of_250high >= 90. Stated by value rather than "old/new" because those
# labels are ambiguous: Smallcap.py's 3.0/90 rule is itself the *newer* MOM
# within this project (it replaced the retired LEG/MA50-cross definition in
# Sept 2026), while 2.0/no-gate is newer only as a US retune.
#
#   2.0, gate OFF   <- SET BELOW
#       +19.5%/yr against a +15.4%/yr do-nothing benchmark, exQ5 +0.75pp,
#       positive in 3 of 5 return quintiles. Weakest of the five validated
#       signals, but the best of the two MOM pairs on the US walk-forward.
#   3.0, gate 90    <- what Smallcap.py and Combined.py run
#       exQ5 -0.41pp, beating the benchmark in 1 of 5 quintiles — the worst
#       configuration measured anywhere in the US study.
#
# WHAT DROPPING THE GATE ACTUALLY DOES — measured across all 572 tickers over
# the last 200 bars, and visible in the comparison charts:
#   3.0/90 fires 783 times; 2.0/off fires 4,358. Of those, 3,575 (82%) are
#   fires the gated rule would have blocked, and the median stock spends 62%
#   of its bars below the 90% line. 309 of 572 tickers produce NO gated fires
#   at all in that window.
#   The two rules are not one signal at two sensitivities. The gated version
#   fires on continuation near the highs; the ungated version fires on that
#   PLUS every sharp bounce inside a drawdown. Mean 2-day move on gated fires
#   is 15-25%, on gate-blocked fires 7-12% — the gate was selecting the
#   violent ones, not merely thinning the count.
# Consequence to watch: at 4,358 vs 783 fires, MOM will dominate a 20-slot
# book and displace SPRED, which is the only signal here with a
# survivorship-robust edge (exQ5 +3.19pp, 5/5 quintiles). If capital is
# slot-constrained, either set ENABLED["MOM"]=False or put MOM last in
# SIGNAL_PRIORITY (it already is).
# To match the Indian books instead, set 3.0 / 90.0.
MOM_DAYPCT  = 2.0    # each of 2 consecutive days must be >= this (%)
MOM_NEAR250 = 0.0    # % of the 250-day high required; 0 = gate disabled

# ── A1 ─────────────────────────────────────────────────────────────────────
# Rule shape unchanged: below the lower Bollinger band and still falling.
#   live (Indian) (20.0, 4.5)
#   US retuned    (0.0, 4.5) : PORT_PCT +17.2%/yr, 4/5 folds
# DOES NOT CLEAR THE BENCHMARK MEANINGFULLY (+17.2% vs +15.4% do-nothing), and
# the survivorship decomposition was not run for it. Constants are provided for
# completeness; treat A1 as UNVALIDATED on US data and do not allocate slots to
# it until the quintile test is run. On Indian data A1 was already flagged as
# fat-tailed (median -0.65%, win 48%) and needing many fires to pay out — a
# profile that interacts badly with a 20-slot constraint.
A1_UPL = 0.0
A1_ATR = 4.5

# ── Downtrend filter ───────────────────────────────────────────────────────
# Kept at the Indian values and NOT applied to any signal, matching
# Smallcap.py's own conclusion (the grid selected filter=OFF in every fold on
# both universes). Retained so the filter can be tested on US data without
# re-deriving the inputs; it has NOT been walk-forward tested here.
DTF_LOWER_LOWS = 8
DTF_ADX        = 30


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────
def read_clean(csv_path):
    """Read one ticker CSV. Drops the all-NaN phantom first bar that yfinance's
    period='5y' boundary produces (24 of 602 files had one)."""
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        return None
    df = df.dropna(subset=["Close"])
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def data_quality_ok(df, ticker=""):
    """US analogue of Smallcap.py's split/turnover guard. The failure mode here
    is not unadjusted corporate actions but STALE QUOTES on a thin listing:
    High==Low on negligible volume, which manufactures phantom gaps."""
    if ticker in EXCLUDE:
        return False, "excluded (stale quotes)"
    if len(df) < MIN_ROWS:
        return False, f"only {len(df)} rows"
    flat_pct = 100.0 * (df["High"] == df["Low"]).mean()
    if flat_pct > FLAT_BAR_PCT_MAX:
        return False, f"{flat_pct:.1f}% flat bars (stale quotes)"
    dv = (df["Close"] * df["Volume"]).median()
    if not np.isfinite(dv) or dv < MIN_MEDIAN_DOLLAR_VOL:
        return False, f"median $vol {dv:,.0f} below floor"
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# 2. INDICATORS  (ported from Smallcap.py; ADX/DI/ATR vectorised)
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(df):
    c = df["Close"].values.astype(float); h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float);   o = df["Open"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c); S = pd.Series(c); F = {}

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
    F["ma_aligned"] = ((ma[10] > ma[20]) & (ma[20] > ma[50])
                       & (ma[50] > ma[200])).astype(float)
    sl = np.full(n, np.nan)
    sl[10:] = (ma[50][10:] - ma[50][:-10]) / np.where(
        ma[50][:-10] == 0, np.nan, ma[50][:-10]) * 100
    F["ma50_slope"] = sl
    F["ma10"], F["ma20"], F["ma50"] = ma[10], ma[20], ma[50]

    rmin252 = S.rolling(252, min_periods=20).min().values
    F["up_from_low252"] = (c - rmin252) / rmin252 * 100
    rmax60 = S.rolling(60, min_periods=20).max().values
    F["dd60"] = (c - rmax60) / rmax60 * 100
    rmax250 = S.rolling(250, min_periods=20).max().values
    F["pct_of_250high"] = np.where(rmax250 > 0, c / rmax250 * 100, np.nan)

    # SURGE: k-day return + position vs the true 52-week HIGH (uses highs)
    rs = np.full(n, np.nan)
    if n > SURGE_DAYS:
        rs[SURGE_DAYS:] = (c[SURGE_DAYS:] - c[:-SURGE_DAYS]) / c[:-SURGE_DAYS] * 100
    F["surge_ret"] = rs
    hi52 = pd.Series(h).rolling(252, min_periods=120).max().values
    F["pct_of_52whigh"] = np.where(hi52 > 0, c / hi52 * 100, np.nan)

    # trend efficiency (Kaufman)
    absd = np.abs(np.concatenate([[0.0], np.diff(c)]))
    path40 = pd.Series(absd).rolling(40).sum().values
    net40 = np.full(n, np.nan); net40[40:] = np.abs(c[40:] - c[:-40])
    F["eff_ratio40"] = np.where(path40 > 0, net40 / path40, np.nan)

    # ATR / ADX / DI
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum.reduce([h[1:] - l[1:],
                                np.abs(h[1:] - c[:-1]),
                                np.abs(l[1:] - c[:-1])])
    upm = np.zeros(n); dnm = np.zeros(n)
    upm[1:] = h[1:] - h[:-1]
    dnm[1:] = l[:-1] - l[1:]
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
    ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    F["atr_pct"] = np.where(c > 0,
                            pd.Series(tr).rolling(14).mean().values / c * 100, np.nan)

    def wil(x, p=14):
        out = np.full(n, np.nan); acc = np.nan
        for i in range(1, n):
            val = x[i] if not np.isnan(x[i]) else 0.0
            acc = val if np.isnan(acc) else acc - acc / p + val
            if i >= p:
                out[i] = acc
        return out
    a_, p_, m_ = wil(tr), wil(pdm), wil(ndm)
    with np.errstate(divide="ignore", invalid="ignore"):
        dip = np.where(a_ > 0, 100 * p_ / a_, np.nan)
        dim = np.where(a_ > 0, 100 * m_ / a_, np.nan)
        dx = np.where((dip + dim) > 0, 100 * np.abs(dip - dim) / (dip + dim), np.nan)
    F["adx"] = pd.Series(dx).rolling(14).mean().values
    F["di_plus"], F["di_minus"] = dip, dim

    # Downtrend-filter input (purely backward-looking)
    low10 = pd.Series(l).rolling(10).min().values
    llf = np.zeros(n); ok = ~np.isnan(low10)
    for k in range(10, n):
        if ok[k] and ok[k - 10]:
            llf[k] = 1.0 if low10[k] < low10[k - 10] else 0.0
    F["lower_lows20"] = pd.Series(llf).rolling(20).sum().values

    # realised vol + volume z-score + 1-day return (SPRED / A1 / A5)
    logr = np.concatenate([[np.nan], np.diff(np.log(np.maximum(c, 1e-9)))])
    F["rv20"] = pd.Series(logr).rolling(20).std().values * np.sqrt(252) * 100
    F["rv5"] = pd.Series(logr).rolling(5).std().values * np.sqrt(252) * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        F["rv_ratio"] = F["rv5"] / F["rv20"]
    vs = pd.Series(v)
    F["vol_z"] = ((vs - vs.rolling(60, min_periods=20).mean())
                  / vs.rolling(60, min_periods=20).std()).values
    dayret = np.full(n, np.nan); dayret[1:] = (c[1:] - c[:-1]) / c[:-1] * 100
    F["day_ret"] = dayret
    F["ret1"] = dayret
    F["falling"] = np.concatenate([[False], c[1:] < c[:-1]])

    # volume + RSI + Bollinger (A1 trigger, and the chart)
    vma20 = pd.Series(v).rolling(20).mean().values
    F["vol_r"] = np.where(vma20 > 0, v / vma20, np.nan)
    F["vol_ma20"] = vma20
    d_ = np.concatenate([[0.0], np.diff(c)])
    g_ = pd.Series(np.where(d_ > 0, d_, 0.0)).ewm(alpha=1/14, adjust=False).mean().values
    l2 = pd.Series(np.where(d_ < 0, -d_, 0.0)).ewm(alpha=1/14, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        F["rsi"] = np.where(l2 > 0, 100 - 100 / (1 + g_ / l2), 100.0)
    bm = S.rolling(20).mean().values; bs = S.rolling(20).std().values
    F["bb_mid"], F["bb_up"], F["bb_low"] = bm, bm + 2 * bs, bm - 2 * bs
    F["bb_width"] = np.where(bm > 0, (F["bb_up"] - F["bb_low"]) / bm * 100, np.nan)
    bwid = F["bb_up"] - F["bb_low"]
    F["pctB"] = np.where(bwid > 0, (c - F["bb_low"]) / bwid, np.nan)

    # REBOUND trigger input, kept ready for a future refit on US data
    hi20 = pd.Series(c).rolling(20).max().values
    F["dd_20d"] = (c / hi20 - 1) * 100
    down1 = (dayret < 0).astype(float)
    cd = np.zeros(n)
    for i in range(1, n):
        cd[i] = cd[i - 1] + 1 if (not np.isnan(down1[i]) and down1[i]) else 0
    F["consec_down"] = cd
    rng_hl = h - l
    F["clspos"] = np.divide(c - l, rng_hl, out=np.full(n, np.nan), where=rng_hl > 0)
    gap_ = np.full(n, np.nan); gap_[1:] = (o[1:] / c[:-1] - 1) * 100
    F["gap"] = gap_
    ll14 = pd.Series(l).rolling(14).min().values
    hh14 = pd.Series(h).rolling(14).max().values
    F["stoch_k"] = 100 * (c - ll14) / np.where((hh14 - ll14) > 0, hh14 - ll14, np.nan)

    F["close"], F["high"], F["low"], F["open"], F["vol"] = c, h, l, o, v
    return F


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
def check_spred(F, i):
    """SPRED — strongest signal on this universe (exQ5 +3.19pp, 5/5 quintiles,
    5/5 folds). Volatility state + off-the-low. The volume z-score gate that
    Indian data needed is OFF here; the realised-vol floor does that work."""
    for k in ("atr_pct", "rv20", "vol_z", "up_from_low252"):
        if np.isnan(F[k][i]):
            return False
    return (F["atr_pct"][i]        >= SPRED_ATR and
            F["rv20"][i]           >= SPRED_RV20 and
            F["vol_z"][i]          >= SPRED_VOLZ and
            F["up_from_low252"][i] >= SPRED_UPL)


def check_surge(F, i):
    """SURGE — a 5-day thrust >= 10%. The 52w-high gate is DISABLED on US data
    (SURGE_NEAR52=0): keeping it costs exQ5 edge and turns the bottom
    5y-return quintile negative. That is the opposite of the Indian result,
    where the gate was the signal."""
    for k in ("surge_ret", "pct_of_52whigh"):
        if np.isnan(F[k][i]):
            return False
    if SURGE_NEAR52 > 0 and F["pct_of_52whigh"][i] < SURGE_NEAR52:
        return False
    return F["surge_ret"][i] >= SURGE_PCT


def check_rev(F, i):
    """REV / BOUNCE — dislocation below MA10, volatility-normalised, off the
    52w low, volatile enough to snap back. Least survivorship-flattered signal
    here (only 24.2% of its trades in the top 5y-return quintile).
    Downtrend filter deliberately NOT applied, matching Smallcap.py."""
    keys = ("px_vs_ma10", "z5", "up_from_low252", "atr_pct", "ret60")
    if any(np.isnan(F[k][i]) for k in keys):
        return False
    return (F["px_vs_ma10"][i]     <  REV_PX_MA10 and
            F["z5"][i]             <  REV_Z5 and
            F["up_from_low252"][i] >= REV_UPL and
            F["atr_pct"][i]        >= REV_ATR and
            F["ret60"][i]          >= REV_RET60)


def check_a5(F, i):
    """A5 — a >= 6% single day. Gate DISABLED by default; see the A5 constants
    block, this is the one threshold where the two validation criteria
    disagree and the choice is a judgement call."""
    if np.isnan(F["day_ret"][i]) or np.isnan(F["pct_of_52whigh"][i]):
        return False
    if A5_NEAR52 > 0 and F["pct_of_52whigh"][i] < A5_NEAR52:
        return False
    return F["day_ret"][i] >= A5_DAY


def check_mom(F, i):
    """MOM — two consecutive days each up >= 2%. The near-250d-high gate is
    DISABLED: at the live value of 90 this is the worst configuration measured
    in this study (exQ5 -0.41pp, 1/5 quintiles). Weakest signal shipped —
    first candidate for removal under slot pressure."""
    if i < 1:
        return False
    if any(np.isnan(F[k][i]) for k in ("day_ret", "pct_of_250high")):
        return False
    if np.isnan(F["day_ret"][i - 1]):
        return False
    if MOM_NEAR250 > 0 and F["pct_of_250high"][i] < MOM_NEAR250:
        return False
    return (F["day_ret"][i]     >= MOM_DAYPCT and
            F["day_ret"][i - 1] >= MOM_DAYPCT)


def check_a1(F, i):
    """A1 — below the lower Bollinger band and still falling.
    UNVALIDATED on US data: +17.2%/yr vs a +15.4% do-nothing benchmark, and the
    survivorship decomposition has not been run. Do not allocate slots to it
    on the strength of these constants alone."""
    if any(np.isnan(F[k][i]) for k in ("bb_low", "up_from_low252", "atr_pct")):
        return False
    return (F["close"][i] < F["bb_low"][i] and
            bool(F["falling"][i]) and
            F["up_from_low252"][i] >= A1_UPL and
            F["atr_pct"][i]        >= A1_ATR)


def passes_downtrend(F, i):
    """True = OK to enter. NOT applied to any signal above (kept for testing).
    Fails OPEN when inputs are unavailable."""
    ll = F["lower_lows20"][i]
    adx, dip, dim = F["adx"][i], F["di_plus"][i], F["di_minus"][i]
    if not np.isnan(ll) and ll >= DTF_LOWER_LOWS:
        return False
    if (not np.isnan(adx) and not np.isnan(dip) and not np.isnan(dim)
            and adx > DTF_ADX and dim > dip):
        return False
    return True


SIGNAL_CHECKS = {
    "SPRED": check_spred,
    "SURGE": check_surge,
    "REV":   check_rev,
    "A5":    check_a5,
    "MOM":   check_mom,
    "A1":    check_a1,
}

# Priority order for slot allocation when several signals fire on the same bar
# and the book is at capacity. Ordered by survivorship-robust edge (exQ5), not
# by headline return. A1 last because it is unvalidated here.
SIGNAL_PRIORITY = ["SPRED", "SURGE", "A5", "REV", "MOM", "A1"]

SIGNAL_DESCRIPTIONS = {
    "SPRED": ("Volatility state + off-the-low. Strongest signal on US smallcap:\n"
              "  exQ5 +3.19pp, positive in 5/5 return quintiles, 5/5 walk-forward\n"
              "  folds, +3.63% avg per trade. Under-samples the 5-year winners\n"
              "  (31.6% of trades in the top quintile vs 42.8% for a random bar),\n"
              "  which is the strongest evidence of real edge in this study."),
    "SURGE": ("A 5-day thrust >=10%. The 52-week-high gate is OFF — the reverse\n"
              "  of the Indian result, where surges away from the high had no edge.\n"
              "  exQ5 +1.80pp, 4/5 quintiles, 4/5 folds."),
    "REV":   ("Mean reversion: dislocated below MA10, volatility-normalised, off\n"
              "  the 52w low. exQ5 +1.10pp, 4/5 quintiles. Modest, but the least\n"
              "  survivorship-flattered signal here (24.2% of trades in the top\n"
              "  5y-return quintile, the lowest of any signal)."),
    "A5":    ("A >=6% single day, gate off by default. exQ5 +1.28pp and positive\n"
              "  in all 5 quintiles, but only 3/5 folds. The gate-on variant has a\n"
              "  larger edge (+1.88pp) in fewer quintiles (3/5) — UNRESOLVED."),
    "MOM":   ("Two consecutive days each up >=2%, no position gate. Weakest signal\n"
              "  shipped: +19.5%/yr vs a +15.4% do-nothing benchmark, exQ5 +0.75pp,\n"
              "  3/5 quintiles. The live Indian config (3%/near-90) is the worst\n"
              "  configuration measured anywhere in this study (exQ5 -0.41pp)."),
    "A1":    ("Below the lower Bollinger band and still falling. UNVALIDATED on US\n"
              "  data — does not clear the do-nothing benchmark meaningfully and the\n"
              "  survivorship test has not been run. Constants for completeness."),
}


# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
NAME_CACHE     = "us_ticker_names.json"
LOG_PATH       = "us_scanner_log.json"
LAST_RUN_KEY   = "_last_scan_date"   # same-trading-day guard, see run_guard()
PENDING_PATH   = "us_pending_followups.json"

EMAIL_SENDER   = "tradingscript1357@gmail.com"
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

MAX_CHARTS     = 25                  # cap attachments so the email stays sendable

BATCH          = 100                 # tickers per yfinance download call
CHECK_BARS     = 5                   # stored closes compared against a fresh fetch
FETCH_PERIOD   = "1mo"               # window pulled per run: long enough to cover
                                     # a missed week and to overlap CHECK_BARS

DRIFT_TOL      = 0.005               # 0.5% — beyond this, full re-fetch
FULL_PERIOD    = "5y"                # window used when rebuilding a ticker

# Which signals to scan. All six are wired; the two flagged below are ON only
# because switching them off silently would hide a decision from you.
#   MOM — runs at 2.0% with no position gate (diverges from Smallcap.py's
#         3.0/gate-90). +19.5%/yr against a +15.4%/yr benchmark, exQ5 +0.75pp,
#         3 of 5 quintiles. Weakest of the five validated signals, and at this
#         setting it fires ~5.6x more often than the gated version.
#   A1  — UNVALIDATED on US data: +17.2%/yr, does not clear the benchmark
#         meaningfully, survivorship decomposition never run.
# Under a slot-constrained book every MOM or A1 fire displaces a possible
# SPRED fire, so consider turning both off once you are allocating capital.
ENABLED = {"SPRED": True, "SURGE": True, "A5": True,
           "REV": True, "MOM": True, "A1": True}

# Signals that register a day-1 follow-up. Smallcap.py follows REV/MOM/REBOUND;
# here the set is the signals whose next-bar behaviour is worth a second look.
FOLLOWUP_SIGNALS = {"SPRED", "SURGE", "REV", "A5", "MOM"}

# Distinct two-char codes for the subject line — S/A alone collide
# (SPRED vs SURGE, A5 vs A1).
CODE = {"SPRED": "SP", "SURGE": "SU", "A5": "A5", "REV": "R", "MOM": "M", "A1": "A1"}

PLOT_STYLE = {"SPRED": ("#2980b9", "v", "hi"), "SURGE": ("#e67e22", "v", "hi"),
              "A5": ("#e91e63", "v", "hi"), "REV": ("#2ecc71", "^", "lo"),
              "MOM": ("#8e44ad", "^", "lo"), "A1": ("#f1c40f", "s", "lo")}


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIVERSE  (the folder IS the manifest)
# ─────────────────────────────────────────────────────────────────────────────
def load_universe():
    """{TICKER: csv_path} for every CSV in DATA_ROOT, minus the exclusions."""
    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(f"data folder not found: {DATA_ROOT}")
    uni = {}
    for fn in sorted(os.listdir(DATA_ROOT)):
        if not fn.endswith(".csv"):
            continue
        t = fn[:-4].strip().upper()
        if t in EXCLUDE:
            continue
        uni[t] = os.path.join(DATA_ROOT, fn)
    return uni


def load_names(tickers):
    """Company names, cached. yfinance .info is slow, so it is called once per
    ticker ever and the result is committed alongside the CSVs."""
    cache = {}
    if os.path.exists(NAME_CACHE):
        try:
            with open(NAME_CACHE) as f:
                cache = json.load(f) or {}
        except json.JSONDecodeError:
            cache = {}
    missing = [t for t in tickers if t not in cache]
    for t in missing:
        try:
            info = yf.Ticker(t).info or {}
            cache[t] = info.get("longName") or info.get("shortName") or t
        except Exception:
            cache[t] = t
    if missing:
        with open(NAME_CACHE, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        print(f"name cache: added {len(missing)}")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# 2. FETCH + APPEND IN PLACE
# ─────────────────────────────────────────────────────────────────────────────
COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def download_batch(tickers):
    """Recent bars for a batch of tickers in one yfinance call.

    Returns {ticker: DataFrame} indexed by date. `actions=True` is required:
    update_csv's first guard looks for the Dividends / Stock Splits columns to
    decide whether the stored adjusted series is still the same series, and
    without them a split would silently corrupt the file instead of triggering
    a rebuild. A ticker that comes back empty is simply absent from the dict,
    which update_csv reports as 'nodata' rather than failing the run.
    """
    out = {}
    tickers = list(tickers)
    if not tickers:
        return out
    raw = yf.download(tickers, period=FETCH_PERIOD, interval="1d",
                      auto_adjust=True, actions=True, group_by="ticker",
                      progress=False, threads=True)
    if raw is None or len(raw) == 0:
        return out
    # one ticker comes back with flat columns, several with a MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        have = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t not in have:
                continue
            d = raw[t].dropna(how="all")
            if len(d):
                out[t] = d
    else:
        d = raw.dropna(how="all")
        if len(d):
            out[tickers[0]] = d
    return out


def _frame_to_rows(fresh):
    """A yfinance frame -> the row dicts update_csv works in.

    Dates become dd-mm-YYYY to match the stored schema, prices are rounded to
    4dp and volume to int. 4dp rather than 2dp deliberately: the drift check
    trips at 0.5% divergence, and on a sub-$5 stock 2dp rounding is itself a
    ~0.5% error, so it would cause spurious full re-fetches. Rows with any
    missing price are dropped — a half-formed bar is worse than no bar.
    """
    rows = []
    if fresh is None or len(fresh) == 0:
        return rows
    d = fresh.copy()
    d.columns = [str(c).strip() for c in d.columns]
    idx = pd.to_datetime(d.index, errors="coerce")
    for ts, r in zip(idx, d.to_dict("records")):
        if pd.isna(ts):
            continue
        try:
            o, h, l, c = (float(r["Open"]), float(r["High"]),
                          float(r["Low"]), float(r["Close"]))
        except (KeyError, TypeError, ValueError):
            continue
        if any(pd.isna(x) for x in (o, h, l, c)):
            continue
        v = r.get("Volume", 0)
        try:
            v = 0 if pd.isna(v) else int(v)
        except (TypeError, ValueError):
            v = 0
        rows.append({"Date": ts.strftime("%d-%m-%Y"),
                     "Open": round(o, 4), "High": round(h, 4),
                     "Low": round(l, 4), "Close": round(c, 4),
                     "Volume": v})
    return rows


def full_refetch(ticker, csv_path):
    """Rebuild one ticker's whole history from scratch. Returns (ok, n_bars).

    Called when a corporate action or a drifting tail means yfinance is no
    longer serving the same adjusted series the file was built from — an
    append would then splice two incompatible price scales together.

    The MIN_ROWS // 2 floor is the safety catch: a throttled or failed fetch
    returns a short frame, and writing that would truncate a good 5-year file
    down to a stub that data_quality_ok then rejects for the rest of time. On
    a short result the existing file is left untouched and the caller keeps
    using it.
    """
    try:
        d = yf.Ticker(ticker).history(period=FULL_PERIOD, interval="1d",
                                      auto_adjust=True)
    except Exception:
        return False, 0
    rows = _frame_to_rows(d)
    if len(rows) < MIN_ROWS // 2:
        return False, len(rows)
    out = pd.DataFrame(rows, columns=COLS)
    out["_d"] = pd.to_datetime(out["Date"], format="%d-%m-%Y", errors="coerce")
    out = (out.dropna(subset=["_d"]).sort_values("_d")
              .drop_duplicates(subset="Date", keep="last")
              .drop(columns="_d"))
    out[COLS].to_csv(csv_path, index=False)
    return True, len(out)



def update_csv(ticker, csv_path, fresh):
    """Append today's bar(s) to the stored CSV, in place, same schema.

    Returns (df, status) where status is one of:
      'appended'  new bar(s) added
      'updated'   today's existing bar refreshed in place
      'nochange'  nothing new
      'refetch'   history rewritten (corporate action or drift)
      'nodata'    nothing usable came back
    """
    if fresh is None or fresh.empty:
        return None, "nodata"

    # ── guard 1: corporate action on any fetched bar -> rebuild ──────────
    action = False
    for col in ("Stock Splits", "Dividends"):
        if col in fresh.columns:
            v = pd.to_numeric(fresh[col], errors="coerce").fillna(0.0)
            if (v.abs() > 0).any():
                action = True
    if action:
        ok, n = full_refetch(ticker, csv_path)
        if ok:
            return pd.read_csv(csv_path), "refetch"

    if not os.path.exists(csv_path):
        ok, n = full_refetch(ticker, csv_path)
        return (pd.read_csv(csv_path), "refetch") if ok else (None, "nodata")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLS].dropna(how="all").drop_duplicates(subset="Date", keep="last")

    # ── guard 2: drift check against the stored tail -> rebuild ──────────
    stored = df.set_index("Date")["Close"].astype(float)
    newrows = _frame_to_rows(fresh)
    overlap = [r for r in newrows if r["Date"] in stored.index][-CHECK_BARS:]
    if overlap:
        drift = max(abs(r["Close"] / stored[r["Date"]] - 1)
                    for r in overlap if stored[r["Date"]] > 0)
        if drift > DRIFT_TOL:
            ok, n = full_refetch(ticker, csv_path)
            if ok:
                print(f"   {ticker}: {100*drift:.2f}% drift on the stored tail "
                      f"-> full re-fetch ({n} bars)")
                return pd.read_csv(csv_path), "refetch"

    have = set(df["Date"].values)
    added, updated = 0, 0
    for r in newrows:
        if r["Date"] in have:
            i = df.index[df["Date"] == r["Date"]][0]
            before = df.loc[i, COLS[1:]].astype(float).values.copy()
            for c in COLS[1:]:
                df.loc[i, c] = r[c]
            if not np.allclose(before, df.loc[i, COLS[1:]].astype(float).values,
                               equal_nan=True):
                updated += 1
        else:
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
            added += 1

    if added or updated:
        df["_d"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
        df = df.dropna(subset=["_d"]).sort_values("_d").drop(columns="_d")
        df[COLS].to_csv(csv_path, index=False)
    return df[COLS], ("appended" if added else ("updated" if updated else "nochange"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHART  (styling fixed — matches the Combined.py build_plot template)
# ─────────────────────────────────────────────────────────────────────────────
def build_plot(F, company, ticker, date_label, kinds, lookback=PLOT_LOOKBACK,
               fires=None):
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
            ax1.hlines(oo, xi - 0.3, xi + 0.3, color=col, lw=1.0, zorder=3)
        else:
            ax1.add_patch(plt.Rectangle((xi - 0.3, b0), 0.6, b1 - b0,
                                        facecolor=col, edgecolor=col, lw=0.5, zorder=3))

    ax1.plot(x, F["bb_up"][start:n], color="#27ae60", lw=0.9, ls="--", label="BB Upper")
    ax1.plot(x, F["bb_low"][start:n], color="#e74c3c", lw=0.9, ls="--", label="BB Lower")
    ax1.plot(x, F["bb_mid"][start:n], color="#7f8c8d", lw=0.7, ls=":", label="BB Mid")
    ax1.fill_between(x, F["bb_low"][start:n], F["bb_up"][start:n],
                     alpha=0.05, color="steelblue")
    ax1.plot(x, F["ma50"][start:n], color="#f39c12", lw=1.1, label="MA50")
    ax1.plot(x, F["ma20"][start:n], color="steelblue", lw=0.8, ls="--",
             alpha=0.7, label="MA20")

    span = np.nanmax(h) - np.nanmin(l); off = (span or 1.0) * 0.035
    for sig, bars in (fires or {}).items():
        col, mk, side = PLOT_STYLE.get(sig, ("#555555", "o", "lo"))
        vis = [i for i in bars if start <= i < n]
        if not vis:
            continue
        ys = ([F["low"][i] - off for i in vis] if side == "lo"
              else [F["high"][i] + off for i in vis])
        ax1.scatter(vis, ys, marker=mk, s=95, color=col, edgecolor="black",
                    lw=0.7, zorder=6, label=f"{sig} fire")

    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.75)
    ax1.grid(alpha=0.25)

    ax2.plot(x, F["rsi"][start:n], color="darkorange", lw=1.1, label="RSI(14)")
    ax2.axhline(70, color="#e74c3c", ls="--", lw=0.7)
    ax2.axhline(30, color="#27ae60", ls="--", lw=0.7)
    ax2.set_ylim(0, 100); ax2.set_ylabel("RSI")
    ax3 = ax2.twinx()
    ax3.bar(x, F["vol"][start:n],
            color=[up if cc >= oo else dn for cc, oo in zip(c, o)],
            alpha=0.3, width=0.8)
    ax3.plot(x, F["vol_ma20"][start:n], color="#78909c", lw=0.8, ls="--", alpha=0.7)
    ax3.set_ylabel("Volume", fontsize=8); ax3.tick_params(labelsize=7)
    ax2.set_xlabel("Bar index"); ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOGS + EMAIL
# ─────────────────────────────────────────────────────────────────────────────
def fmt_list(tickers, names, width=96, indent=9, namelen=24):
    """'TICKER (Company Name)' comma list, wrapped. Names are what the email is
    read for — a bare ticker means looking it up before you can judge it."""
    if not tickers:
        return "-"
    parts = []
    for t in tickers:
        nm = str(names.get(t, "")).strip()
        parts.append(f"{t} ({nm[:namelen]})" if nm and nm != t else t)
    lines, cur = [], ""
    for p in parts:
        cand = p if not cur else f"{cur}, {p}"
        if len(cand) > width and cur:
            lines.append(cur); cur = p
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return ("\n" + " " * indent).join(lines)


def _load(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        txt = f.read().strip()
    if not txt:
        return default
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return default


def _save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def already_ran(log, today_label):
    """True if a scan email has already gone out for this trading day.

    The cron fires once, but a manual workflow_dispatch on the same day — or
    any re-run — would otherwise re-send the whole scan. The pending-append
    guard already stops duplicate follow-up entries; this stops the duplicate
    email. Set FORCE_RUN=1 to override."""
    if os.environ.get("FORCE_RUN") == "1":
        print("FORCE_RUN=1 — same-day guard bypassed")
        return False
    return log.get(LAST_RUN_KEY) == today_label


def send_email(subject, body, attachments):
    if not EMAIL_PASSWORD:
        print(f"  EMAIL_PASSWORD unset — not sending: {subject}")
        return False
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
# 5. DAY-1 FOLLOW-UP
# ─────────────────────────────────────────────────────────────────────────────
# When a signal in FOLLOWUP_SIGNALS fires, an entry is recorded. On the next
# run — one bar later in THAT ticker's own series, so weekends and holidays
# skip themselves via bar-index math — a day-1 update is built showing the
# price change since the fire and whether the condition still holds. Entries
# more than one bar old are dropped silently: a missed run is not retried and
# there is no late send. Everything is batched into ONE email.
def load_pending():
    raw = _load(PENDING_PATH, [])
    if not isinstance(raw, list):
        return []
    seen, out = set(), []
    for e in raw:
        k = (e.get("ticker"), e.get("signal"), e.get("fire_bar"))
        if k in seen:
            continue
        seen.add(k); out.append(e)
    return out


def process_followups(pending, resolved, ticker, F, i, company, date_label):
    keep = []
    for e in pending:
        if e["ticker"] != ticker:
            keep.append(e); continue
        age = i - e["fire_bar"]
        if age == 1:
            entry = float(e["entry_close"]); now = float(F["close"][i])
            pct = (now / entry - 1) * 100 if entry else float("nan")
            sig = e["signal"]
            still = SIGNAL_CHECKS[sig](F, i) if sig in SIGNAL_CHECKS else False
            try:
                png = build_plot(F, company, ticker, date_label, [sig],
                                 fires={sig: [e["fire_bar"]]})
            except Exception:
                png = None
            body = (f"{ticker} — {company}\n"
                    f"Signal        : {sig}\n"
                    f"Fired on      : {e['fire_date']}  (close {entry:.2f})\n"
                    f"Now ({date_label}): close {now:.2f}\n"
                    f"Change        : {pct:+.2f}%\n"
                    f"Condition still holds today: {'YES' if still else 'no'}\n")
            resolved.append(dict(ticker=ticker, signal=sig, body=body, png=png,
                                 image_name=f"{ticker}_{sig}_day1.png"))
        elif age > 1:
            pass          # stale — dropped silently, per design
        else:
            keep.append(e)
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    uni = load_universe()
    print(f"Universe: {len(uni)} tickers from {DATA_ROOT}/")
    names = load_names(list(uni))

    log = _load(LOG_PATH, {})
    pending = load_pending()
    resolved, sections, charts = [], [], []
    hits = {s: [] for s in ENABLED}
    skipped, refetched, stale = [], [], []
    today_label = None

    # ── PASS 1: fetch and append every CSV before scoring anything ───────
    # Scoring is per-stock and needs no cross-sectional state, but keeping
    # the two passes split means a fetch failure part-way through cannot
    # leave half the universe scored against yesterday's disk.
    tick = list(uni)
    fresh = {}
    for b in range(0, len(tick), BATCH):
        chunk = tick[b:b + BATCH]
        try:
            fresh.update(download_batch(chunk))
        except Exception as e:
            print(f"  batch {b//BATCH+1} failed: {e}")
        print(f"  fetched {min(b+BATCH, len(tick))}/{len(tick)}")

    status_count = {}
    for t, path in uni.items():
        try:
            _, st = update_csv(t, path, fresh.get(t))
        except Exception as e:
            st = "error"
            print(f"── {t}: update failed ({e})")
        status_count[st] = status_count.get(st, 0) + 1
        if st == "refetch":
            refetched.append(t)
    print("CSV update: " + ", ".join(f"{k}={v}" for k, v in sorted(status_count.items())))

    # ── Establish the reference trading day as the MODAL last-bar date
    # across the universe. Anchoring on whichever ticker happens to be
    # scored first would flag the whole universe stale if that one ticker
    # were itself behind. ─────────────────────────────────────────────────
    last_dates = {}
    for t, path in uni.items():
        try:
            d = pd.read_csv(path, usecols=["Date"])
            if len(d):
                last_dates[t] = str(d["Date"].iloc[-1]).strip()
        except Exception:
            pass
    if last_dates:
        today_label = pd.Series(list(last_dates.values())).mode().iloc[0]
        print(f"reference trading day (modal last bar): {today_label}")

    if already_ran(log, today_label):
        print(f"\nScan already sent for {today_label} — nothing further to do. "
              f"(CSVs were still refreshed above; re-run with FORCE_RUN=1 to re-send.)")
        return

    # ── PASS 2: score ────────────────────────────────────────────────────
    for t, path in uni.items():
        df = read_clean(path)
        if df is None or len(df) < MIN_ROWS:
            skipped.append(f"{t} (only {0 if df is None else len(df)} rows)")
            continue
        ok, why = data_quality_ok(df, t)
        if not ok:
            skipped.append(f"{t} ({why})")
            continue
        try:
            F = compute_indicators(df)
        except Exception as e:
            skipped.append(f"{t} (indicators failed: {e})")
            continue

        i = len(F["close"]) - 1
        date_label = pd.Timestamp(df["Date"].iloc[i]).strftime("%d-%m-%Y")
        company = names.get(t, t)

        # A ticker whose last bar is behind the universe's modal day did not
        # update. It is still scored — the signals are per-stock — but it is
        # named in the email header so a quietly-dead feed cannot hide.
        if today_label and date_label != today_label:
            stale.append(f"{t} ({date_label})")

        # day-1 follow-ups resolve BEFORE today's own signal check
        pending = process_followups(pending, resolved, t, F, i, company, date_label)

        fired = [s for s in SIGNAL_PRIORITY
                 if ENABLED.get(s) and SIGNAL_CHECKS[s](F, i)]

        # register today's fires for tomorrow, guarded against a double run
        for s in fired:
            if s not in FOLLOWUP_SIGNALS:
                continue
            if any(e["ticker"] == t and e["signal"] == s and e["fire_bar"] == i
                   for e in pending):
                continue
            pending.append(dict(ticker=t, signal=s, fire_bar=i,
                                fire_date=date_label,
                                entry_close=float(F["close"][i])))

        if not fired:
            continue
        for s in fired:
            hits[s].append(t)
            log[f"{t}_{s}"] = date_label

        # Alert body: four fields only. Diagnostics live in the chart.
        sections.append("\n".join([
            f"\n{t} — {company}",
            f"  Signal    : {', '.join(fired)}",
            f"  1d change : {F['day_ret'][i]:+.2f}%",
            f"  5d change : {F['ret5'][i]:+.2f}%",
        ]))

        if len(charts) < MAX_CHARTS:
            try:
                hist = {s: [k for k in range(len(F["close"]))
                            if SIGNAL_CHECKS[s](F, k)] for s in fired}
                charts.append((f"{t}_{date_label}.png",
                               build_plot(F, company, t, date_label, fired, fires=hist)))
            except Exception as e:
                print(f"   {t}: chart failed ({e})")

    # NOTE: the same-day guard (LAST_RUN_KEY) is deliberately NOT written here.
    # It is set only after the scan email is confirmed sent, at the bottom of
    # main() — otherwise a send failure would arm the guard and a re-run would
    # silently skip, needing FORCE_RUN to recover.
    _save(LOG_PATH, log)
    _save(PENDING_PATH, pending)

    # ── follow-up email, independent of whether anything fired today ─────
    if resolved:
        body = (f"DAY-1 FOLLOW-UP  —  {today_label}\n{len(resolved)} item(s)\n"
                + "\n".join(f"\n{'='*60}\n{r['body']}" for r in resolved))
        atts = [(r["image_name"], r["png"]) for r in resolved if r["png"]]
        send_email(f"[US Day-1 Follow-up] {len(resolved)} item(s) — {today_label}",
                   body, atts)

    if not sections:
        print("\nNo signals today — no daily scan email sent.")
        return

    off = [s for s, on in ENABLED.items() if not on]
    header = (
        f"US SMALLCAP DAILY SCAN  —  {today_label}\n"
        f"Engine   : hardcoded rule thresholds from US_Smallcap.py "
        f"(self-contained, no model files)\n"
        f"Grading  : OFF — drop-C cost 22.7pp of return on deployed capital "
        f"in walk-forward on this universe\n"
        + f"Universe : {len(uni)} tickers"
        + (f", {len(off)} signal(s) disabled: {', '.join(off)}" if off else "")
        + "\n"
        + (f"Re-fetched (corporate action / drift): {len(refetched)}"
           f" — {', '.join(refetched[:12])}{' ...' if len(refetched) > 12 else ''}\n"
           if refetched else "")
        + (f"STALE last bar: {len(stale)} — {', '.join(stale[:12])}"
           f"{' ...' if len(stale) > 12 else ''}\n" if stale else "")
        + (f"Skipped on data quality: {len(skipped)}"
           f" — {', '.join(skipped[:8])}{' ...' if len(skipped) > 8 else ''}\n"
           if skipped else "")
    )
    for s in SIGNAL_PRIORITY:
        if not ENABLED.get(s):
            continue
        h = hits[s]
        header += f"{s:6s}: {len(h):3d}  {fmt_list(h, names)}\n"

    multi = {}
    for s in SIGNAL_PRIORITY:
        for t in hits.get(s, []):
            multi.setdefault(t, []).append(s)
    conf = {t: v for t, v in multi.items() if len(v) > 1}
    if conf:
        header += "Confluence (2+):\n"
        for t, v in list(conf.items())[:20]:
            nm = str(names.get(t, "")).strip()
            header += f"         {t} ({nm[:34]}) — {'+'.join(v)}\n" if nm and nm != t \
                      else f"         {t} — {'+'.join(v)}\n"

    if len(sections) > MAX_CHARTS:
        header += f"\n(charts capped at {MAX_CHARTS} attachments)\n"

    counts = "/".join(f"{len(hits[s])}{CODE[s]}" for s in SIGNAL_PRIORITY
                      if ENABLED.get(s))
    ok = send_email(f"[US Scanner] {counts} — {today_label}",
                    header + "\n".join(sections), charts)

    # Arm the same-day guard only on a confirmed send, so a failed email can be
    # recovered by simply re-running the workflow.
    if ok:
        log[LAST_RUN_KEY] = today_label
        _save(LOG_PATH, log)
    else:
        print("scan email not sent — same-day guard left unarmed so a re-run retries")


if __name__ == "__main__":
    main()
