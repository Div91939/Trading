"""
US_Scanner.py — US S&P 600 smallcap daily scanner (single self-contained file)
=============================================================================
Built to the same operational shape as Smallcap.py, the Indian scanner that
has been running reliably: fetch one ticker at a time, append to the stored
CSV, score, e-mail. Stdlib plus yfinance / pandas / numpy / matplotlib.

WHY THE FETCH LOOKS LIKE THIS
-----------------------------
One `yf.Ticker(t).history()` call per ticker, exactly as Smallcap.py does it.
An earlier version of this scanner used batched `yf.download()` with
MultiIndex unpacking, corporate-action detection and a drift check against
the stored tail. That machinery was never verified against live Yahoo and it
is where every failure happened. It is all gone. The per-ticker call is the
one that is proven in production here daily, a failure is isolated to its own
ticker, and there is no column-shape guessing.

The cost of that choice, stated plainly: the stored history was built with
auto_adjust=True, so it is already split- and dividend-adjusted. If a ticker
splits, yfinance re-adjusts its whole history and the newly appended bars sit
on a different price scale from the stored ones. Smallcap.py has the same
exposure and handles it by being re-backfilled when it matters. If a split
shows up, re-run the backfill for that ticker rather than trusting the file.
`FETCH_DAYS` bars are pulled each run and any bar whose date is already
stored is overwritten in place, so a late correction to the last few sessions
does get picked up.

UNIVERSE is the contents of the CSV folder, not a manifest: one flat
<TICKER>.csv in us_smallcap_data/. Drop one in and it is scanned; delete it
and it is gone.

NO ATTACHMENT CAP. Every fire that produces a chart is attached. Gmail
rejects a message over ~25 MB, so if a day ever fires wide enough to hit
that, the send fails loudly rather than silently truncating; the count is
printed before sending so it is visible in the run log.

NO GRADING — deliberate, not an omission. On US data the A/B/C grader orders
correctly on its own 10-day target but inverts on traded P/L, and suppressing
C-grade fires cost 22.7pp of return on deployed capital in the 2026
walk-forward window. See experiment registry rows 63-68.

THRESHOLDS are the US-retuned ones, not the Indian values. Every position
gate re-derived looser or off on US data (registry rows 57-62): MOM near250
90 -> off, SURGE near52 85 -> off, A5 near52 85 -> off, SPRED up-from-low
50 -> 25, A1 up-from-low 20 -> 0. SPRED is the strongest signal measured
here (+57.1%/yr, 5/5 folds); MOM and A1 are the weakest and are the first
two to switch off under slot pressure.
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
DATA_ROOT     = "us_smallcap_data"
MIN_ROWS      = 1240      # the 250d/252d windows need this much to be meaningful
PLOT_LOOKBACK = 200

NAME_CACHE   = "us_ticker_names.json"
LOG_PATH     = "us_scanner_log.json"
PENDING_PATH = "us_pending_followups.json"

EMAIL_SENDER   = "tradingscript1357@gmail.com"
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# Bars pulled per ticker per run. 1 would match Smallcap.py exactly, but a
# missed run would then leave a permanent hole in the series and every
# rolling window downstream would silently span it. 7 covers a missed week.
FETCH_DAYS = 7

COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Stale-quote contaminated: INDV had 27.8% flat High==Low bars and a phantom
# 572% move, IVT a phantom 238% move. Both manufacture false signals.
EXCLUDE = {"INDV", "IVT"}

# Runtime data-quality guards
FLAT_BAR_PCT_MAX      = 2.0          # max % of High==Low bars before dropping
MIN_MEDIAN_DOLLAR_VOL = 1_000_000    # median close*volume, USD/day

# ── Signal thresholds — US-retuned, see the header ─────────────────────────
SPRED_ATR, SPRED_RV20, SPRED_VOLZ, SPRED_UPL = 5.5, 45.0, 0.0, 25.0
SURGE_DAYS, SURGE_PCT, SURGE_NEAR52 = 5, 10.0, 0.0
REV_PX_MA10, REV_Z5, REV_UPL, REV_ATR, REV_RET60 = -6, -0.5, 0, 3.5, -40
A5_DAY, A5_NEAR52 = 6.0, 0.0
MOM_DAYPCT, MOM_NEAR250 = 2.0, 0.0
A1_UPL, A1_ATR = 0.0, 4.5

# Downtrend filter: computed, deliberately NOT applied to any signal, matching
# Smallcap.py where it is applied to REV only and earned nothing here.
DTF_LOWER_LOWS, DTF_ADX = 8, 30

ENABLED = {"SPRED": True, "SURGE": True, "A5": True,
           "REV": True, "MOM": True, "A1": True}

# Signals that register a day-1 follow-up.
FOLLOWUP_SIGNALS = {"SPRED", "SURGE", "REV", "A5", "MOM"}

# Distinct two-char subject codes — S/A alone collide (SPRED vs SURGE, A5 vs A1).
CODE = {"SPRED": "SP", "SURGE": "SU", "A5": "A5", "REV": "R", "MOM": "M", "A1": "A1"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIVERSE + NAMES  (the folder IS the manifest)
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
# 2. READ + DATA QUALITY
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
# 3. FETCH + APPEND   (one ticker at a time, the Smallcap.py pattern)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_and_update_csv(ticker, csv_path):
    """Pull the last FETCH_DAYS sessions for one ticker and merge them into the
    stored CSV. Returns (df, date_label) or (None, None).

    Deliberately boring. No batching, no MultiIndex unpacking, no corporate-
    action branch: a single-ticker history() call returns flat columns, so
    there is no shape to guess at, and anything that goes wrong takes down one
    ticker instead of a batch of a hundred.

    Bars already present are overwritten in place, so a late correction to a
    recent session is picked up. Prices are stored at 4dp: the backfill wrote
    full float64, and 2dp would be a ~0.5% error on a sub-$5 stock.
    """
    hist = yf.Ticker(ticker).history(period=f"{FETCH_DAYS}d", interval="1d",
                                     auto_adjust=True)
    if hist is None or len(hist) == 0:
        return None, None

    rows = []
    for ts, r in zip(pd.to_datetime(hist.index), hist.to_dict("records")):
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
        rows.append({"Date": ts.strftime("%d-%m-%Y"), "Open": round(o, 4),
                     "High": round(h, 4), "Low": round(l, 4),
                     "Close": round(c, 4), "Volume": v})
    if not rows:
        return None, None

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[COLS].dropna(how="all").drop_duplicates(subset="Date", keep="last")
    else:
        df = pd.DataFrame(columns=COLS)

    have = {d: k for k, d in enumerate(df["Date"].astype(str).values)}
    added = updated = 0
    for r in rows:
        if r["Date"] in have:
            k = df.index[have[r["Date"]]]
            for col in COLS[1:]:
                df.loc[k, col] = r[col]
            updated += 1
        else:
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
            added += 1

    df["_d"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    df = (df.dropna(subset=["_d"]).sort_values("_d")
            .drop_duplicates(subset="Date", keep="last").drop(columns="_d"))
    if added or updated:
        df[COLS].to_csv(csv_path, index=False)
    return df[COLS].reset_index(drop=True), rows[-1]["Date"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. INDICATORS
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
# 5. SIGNALS
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


PLOT_STYLE = {"SPRED": ("#2980b9", "v", "hi"), "SURGE": ("#e67e22", "v", "hi"),
              "A5": ("#e91e63", "v", "hi"), "REV": ("#2ecc71", "^", "lo"),
              "MOM": ("#8e44ad", "^", "lo"), "A1": ("#f1c40f", "s", "lo")}


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
# 6. DAY-1 FOLLOW-UP
# ─────────────────────────────────────────────────────────────────────────────
# When a FOLLOWUP_SIGNALS signal fires, an entry is recorded here. On the next
# run that is exactly one bar later in THAT ticker's own series (bar-index
# math, so weekends and holidays look after themselves), a section is built
# showing the move since the fire and whether the condition still holds. All
# sections go out in ONE batched e-mail. An entry more than one bar old is
# dropped silently — a missed run is not retried late. Same design as
# Smallcap.py.
def load_pending():
    if not os.path.exists(PENDING_PATH):
        return []
    try:
        with open(PENDING_PATH) as f:
            raw = json.loads(f.read().strip() or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    # collapse any (ticker, signal, fire_bar) duplicates already in the file
    seen, out = set(), []
    for e in raw:
        key = (e.get("ticker"), e.get("signal"), e.get("fire_bar"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def process_followups(pending, resolved, ticker, F, i, company, date_label):
    """Resolve this ticker's due follow-ups BEFORE today's own check. Appends
    to `resolved`; returns the pending list with this ticker's done/stale
    entries removed."""
    keep = []
    for e in pending:
        if e.get("ticker") != ticker:
            keep.append(e)
            continue
        age = i - int(e.get("fire_bar", -99))
        if age != 1:
            if age < 1:
                keep.append(e)      # not due yet
            continue                # age > 1: stale, dropped
        sig = e.get("signal")
        entry = float(e.get("entry_close") or 0.0)
        now = float(F["close"][i])
        pct = (now / entry - 1) * 100 if entry else float("nan")
        still = bool(SIGNAL_CHECKS[sig](F, i)) if sig in SIGNAL_CHECKS else False
        try:
            png = build_plot(F, company, ticker, date_label, [sig],
                             fires={sig: [int(e["fire_bar"])]})
        except Exception as ex:
            print(f"   {ticker}: follow-up chart failed ({ex})")
            png = None
        body = (f"{company} ({ticker})\n"
                f"  Signal        : {sig}\n"
                f"  Fired on      : {e.get('fire_date')}  (close {entry:.2f})\n"
                f"  Now ({date_label}): close {now:.2f}\n"
                f"  Change        : {pct:+.2f}%\n"
                f"  Still holds   : {'YES' if still else 'no'}\n")
        resolved.append(dict(name=f"{ticker}_{sig}_day1.png", body=body, png=png))
    return keep



# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    universe = load_universe()
    print(f"Universe: {len(universe)} tickers from {DATA_ROOT}/")

    log = _load(LOG_PATH, {})
    pending = load_pending()
    names = load_names(list(universe))

    resolved, sections, charts = [], [], []
    hits = {s: [] for s in ENABLED if ENABLED[s]}
    skipped, fetch_failed = [], []
    today_label = None

    # ── PASS 1: bring every CSV up to date on disk ───────────────────────
    # Split from scoring so a fetch dying half way through cannot leave part
    # of the universe scored against yesterday's data.
    fresh_dates = {}
    for n, (ticker, path) in enumerate(sorted(universe.items()), 1):
        try:
            df, date_label = fetch_and_update_csv(ticker, path)
            if df is None:
                fetch_failed.append(f"{ticker} (no data)")
            else:
                fresh_dates[ticker] = date_label
        except Exception as e:
            fetch_failed.append(f"{ticker} ({type(e).__name__})")
            print(f"── {ticker}: fetch failed ({type(e).__name__}: {e})")
        if n % 50 == 0:
            print(f"  fetched {n}/{len(universe)}  "
                  f"({len(fresh_dates)} ok, {len(fetch_failed)} failed)", flush=True)
    print(f"fetch done: {len(fresh_dates)} updated, {len(fetch_failed)} failed")
    if not fresh_dates:
        print("NOTHING fetched — aborting before the scan rather than e-mailing "
              "a scan of stale data. Check the fetch errors above.")
        return

    # The reference trading day is the MODAL last-bar date across the universe.
    # Anchoring on whichever ticker is scanned first would call the whole
    # universe stale whenever that one ticker happened to be behind.
    tally = {}
    for d in fresh_dates.values():
        tally[d] = tally.get(d, 0) + 1
    today_label = max(tally, key=tally.get)
    print(f"reference trading day: {today_label}  "
          f"({tally[today_label]}/{len(fresh_dates)} tickers)")

    # ── PASS 2: score ────────────────────────────────────────────────────
    for ticker, path in sorted(universe.items()):
        try:
            clean = read_clean(path)
        except Exception as e:
            print(f"── {ticker}: read failed ({e})")
            continue
        if clean is None or len(clean) < MIN_ROWS:
            skipped.append(f"{ticker} (only {0 if clean is None else len(clean)} rows)")
            continue
        ok, why = data_quality_ok(clean, ticker)
        if not ok:
            skipped.append(f"{ticker} ({why})")
            continue
        try:
            F = compute_indicators(clean)
        except Exception as e:
            print(f"── {ticker}: indicators failed ({e})")
            continue
        i = len(F["close"]) - 1
        company = names.get(ticker, ticker)

        # resolve yesterday's fires for this ticker before checking today's
        pending = process_followups(pending, resolved, ticker, F, i,
                                    company, today_label)

        fired = []
        for s in SIGNAL_PRIORITY:
            if not ENABLED.get(s):
                continue
            try:
                if SIGNAL_CHECKS[s](F, i):
                    fired.append(s)
            except Exception as e:
                print(f"── {ticker}: {s} check failed ({e})")
        if not fired:
            continue

        for s in fired:
            hits[s].append(ticker)
            log[f"{ticker}_{s}"] = today_label

        # register for tomorrow's follow-up, guarded against a double run
        for s in fired:
            if s not in FOLLOWUP_SIGNALS:
                continue
            if any(e.get("ticker") == ticker and e.get("signal") == s
                   and e.get("fire_bar") == i for e in pending):
                continue
            pending.append(dict(ticker=ticker, signal=s, fire_bar=i,
                                fire_date=today_label,
                                entry_close=float(F["close"][i])))

        sections.append("\n".join([
            f"\n{ticker} — {company}",
            f"  Signal    : {', '.join(fired)}",
            f"  Close     : {F['close'][i]:.2f}",
            f"  1d change : {F['day_ret'][i]:+.2f}%",
            f"  5d change : {F['ret5'][i]:+.2f}%",
            f"  ATR%      : {F['atr_pct'][i]:.2f}   52wH: {F['pct_of_52whigh'][i]:.1f}%",
        ]))

        # every fire gets a chart — no cap, see the header
        try:
            hist = {s: [k for k in range(len(F["close"])) if SIGNAL_CHECKS[s](F, k)]
                    for s in fired}
            charts.append((f"{ticker}_{today_label}.png",
                           build_plot(F, company, ticker, today_label, fired, fires=hist)))
        except Exception as e:
            print(f"   {ticker}: chart failed ({e})")

    log["_last_scan_date"] = today_label
    _save(LOG_PATH, log)
    _save(PENDING_PATH, pending)

    # ── day-1 follow-up e-mail, independent of today's fires ─────────────
    if resolved:
        body = (f"US DAY-1 FOLLOW-UP  —  {today_label}\n{len(resolved)} item(s)\n"
                + "\n".join(f"\n{'='*62}\n{r['body']}" for r in resolved))
        atts = [(r["name"], r["png"]) for r in resolved if r["png"]]
        print(f"follow-up e-mail: {len(resolved)} item(s), {len(atts)} chart(s)")
        send_email(f"[US Day-1 Follow-up] {len(resolved)} item(s) — {today_label}",
                   body, atts)

    if not sections:
        print("No signals today — no scan e-mail sent.")
        return

    counts = "/".join(f"{len(hits[s])}{CODE[s]}" for s in SIGNAL_PRIORITY
                      if ENABLED.get(s))
    head = [f"US SMALLCAP DAILY SCAN  —  {today_label}",
            "Engine   : hardcoded thresholds in US_Scanner.py (self-contained)",
            "Grading  : OFF — drop-C cost 22.7pp of return in walk-forward here",
            f"Universe : {len(universe)} tickers, {len(fresh_dates)} fetched OK"]
    if fetch_failed:
        head.append(f"Fetch failed ({len(fetch_failed)}): "
                    + ", ".join(fetch_failed[:12])
                    + (" ..." if len(fetch_failed) > 12 else ""))
    if skipped:
        head.append(f"Skipped on data quality ({len(skipped)}): "
                    + ", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else ""))
    for s in SIGNAL_PRIORITY:
        if not ENABLED.get(s):
            continue
        head.append(f"{s:6}: {len(hits[s]):3}  "
                    + (fmt_list(hits[s], names) if hits[s] else "-"))

    conf = {}
    for s in SIGNAL_PRIORITY:
        for t in hits.get(s, []):
            conf.setdefault(t, []).append(s)
    multi = {t: v for t, v in conf.items() if len(v) > 1}
    if multi:
        head.append("Confluence (2+):")
        for t, v in multi.items():
            head.append(f"         {t} ({names.get(t, t)[:24]}) — {'+'.join(v)}")

    print(f"scan e-mail: {len(sections)} fire(s), {len(charts)} chart(s)")
    send_email(f"[US Scanner] {counts} — {today_label}",
               "\n".join(head) + "\n" + "\n".join(sections), charts)


if __name__ == "__main__":
    main()
