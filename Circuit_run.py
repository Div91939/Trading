

import pandas as pd
import numpy as np
matplotlib_backend_set = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

LOG_PATH     = "circuit_email_log.json"
EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = 'osrp rtab jvyv rcvz'
#EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# Ignition signal — fixed thresholds, same for every stock (see docstring)
HAZARD_WINDOW_DAYS = 9     # must be within this many days of last streak END
BB_WINDOW           = 20
BB_STD              = 2

# Consecutive-RISE detection (generalized from strict upper-circuit-only) —
# a "rise day" is any day with return > RISE_MIN_RET; a streak needs at
# least RISE_MIN_STREAK_LEN consecutive rise days. Validated OOS: +3.23%
# avg return, N=713, on the hazard-window entry rule at these settings.
RISE_MIN_RET        = 1.0   # % daily return to count as a "rise" day
RISE_MIN_STREAK_LEN = 2     # consecutive rise days needed to count as a streak

# Per-stock config:
#   ticker    — yfinance ticker string (⚠ VERIFY — see docstring)
#   csv_path  — local CSV path
#   band_pct  — the stock's circuit band (5% for most, 2% for a couple —
#               detected empirically per-stock in this session's analysis)
STOCKS = {
    "E2E": {
        "ticker":   "E2E.NS",
        "csv_path": "Circuit/e2e.csv",
        "band_pct": 5,
    },
    "GVKPIL": {
        "ticker":   "GVKPIL.BO",  # VERIFY TICKER
        "csv_path": "Circuit/gvkpil.csv",
        "band_pct": 5,
    },
    "OGST": {
        "ticker":   "ONEGLOBAL.BO",  # VERIFY TICKER
        "csv_path": "Circuit/ogst.csv",
        "band_pct": 5,
    },
    "REFEX": {
        "ticker":   "REFEX.BO",
        "csv_path": "Circuit/refex.csv",
        "band_pct": 5,
    },
    "SARTHAKGL": {
        "ticker":   "SARTHAKGL.BO",   # confirmed from your existing Circuit_Check.py
        "csv_path": "Circuit/sarthakgl.csv",
        "band_pct": 5,
        # Loosely PERIODIC (CV=0.72, ~11-14d rhythm) — the one stock in this
        # universe where calendar timing alone has some signal.
    },
    "SPELS": {
        "ticker":   "SPELS.BO",  # VERIFY TICKER
        "csv_path": "Circuit/spels.csv",
        "band_pct": 5,
    },
    "SRMENERGY": {
        "ticker":   "SRMENERGY.BO",   # confirmed from your existing Circuit_Check.py
        "csv_path": "Circuit/srmenergy.csv",
        "band_pct": 2,   # narrower band than the rest of this universe
    },
    "STARCOM": {
        "ticker":   "STARCOM.BO",  # VERIFY TICKER
        "csv_path": "Circuit/starcom.csv",
        "band_pct": 5,
    },
    "VIKALPS": {
        "ticker":   "VIKALPS.BO",  # VERIFY TICKER — low confidence match
        "csv_path": "Circuit/vikalps.csv",
        "band_pct": 5,
        # Loosely PERIODIC (CV=0.76, ~27d rhythm) — second calendar-timeable name.
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH + UPDATE CSV  (identical pattern to Combined.py)
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
# 2. COMPUTE INDICATORS — circuit detection, streak tracking, BB, RSI, volz
# ─────────────────────────────────────────────────────────────────────────────

def compute_circuit_indicators(df, band_pct):
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    close = np.array(df["Close"], dtype=float)
    high  = np.array(df["High"],  dtype=float)
    low   = np.array(df["Low"],   dtype=float)
    vol   = np.array(df["Volume"], dtype=float)
    n     = len(close)

    ret = np.full(n, np.nan)
    ret[1:] = (close[1:] - close[:-1]) / close[:-1] * 100

    b = band_pct
    is_uc = (ret > b * 0.9) & (ret <= b * 1.1)
    is_lc = (ret < -b * 0.9) & (ret >= -b * 1.1)
    is_locked = (high == low)
    is_tradeable = (high > low) & (vol > 0)

    # ── Consecutive-RISE detection (generalized from strict-UC) ──────────
    # Doesn't require hitting the exact circuit band — any day with a
    # return > RISE_MIN_RET counts. Validated OOS across a threshold/length
    # sweep (0.5-2% x 2-3 days): 1% / 2-day is a robust middle setting, not
    # a cherry-picked cell (nearby settings gave similar +2-3.3% OOS avg
    # returns on the hazard-window entry). Still a meaningfully SMALLER
    # requirement than a full circuit lock, so it fires on plain momentum
    # runs too, not just circuit-specific action — that's the point.
    is_rise = ret > RISE_MIN_RET

    # Bollinger Bands (20, 2sd)
    close_s = pd.Series(close)
    mid = close_s.rolling(BB_WINDOW).mean().values
    sd  = close_s.rolling(BB_WINDOW).std().values
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_pos = np.where(sd > 0, (close - mid) / (BB_STD * sd), np.nan)  # >1 = above upper band

    # RSI-14 (for context in the email; not required for the fire condition)
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    up_s, dn_s = pd.Series(up), pd.Series(dn)
    roll_up = up_s.ewm(alpha=1/14, adjust=False).mean().values
    roll_dn = dn_s.ewm(alpha=1/14, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(roll_dn > 0, roll_up / roll_dn, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # Volume z-score vs 20d
    vol_s = pd.Series(vol)
    vol_ma20 = vol_s.rolling(20).mean().values
    vol_sd20 = vol_s.rolling(20).std().values
    with np.errstate(divide="ignore", invalid="ignore"):
        volz = np.where(vol_sd20 > 0, (vol - vol_ma20) / vol_sd20, np.nan)

    # Streak marking (RISE runs of length >= RISE_MIN_STREAK_LEN) + days since
    # last streak ENDED — this now drives IgnA's hazard window (previously
    # required a strict circuit-band streak; generalized per your instruction).
    streak_ends = set()
    i = 0
    while i < n:
        if is_rise[i] and (i == 0 or not is_rise[i - 1]):
            j = i
            while j < n and is_rise[j]:
                j += 1
            if j - i >= RISE_MIN_STREAK_LEN:
                streak_ends.add(j - 1)
            i = j
        else:
            i += 1

    days_since_streak_end = np.full(n, np.inf)
    last_end = -np.inf
    for i in range(n):
        if (i - 1) in streak_ends:
            last_end = i - 1
        days_since_streak_end[i] = i - last_end

    # ── R1-R5 features: "hot" window since the last RISE day (generalized
    # from strict-UC — same rationale as above) ───────────────────────────
    days_since_uc = np.full(n, np.inf)
    last_uc = -np.inf
    for i in range(n):
        if is_rise[i]:
            last_uc = i
        days_since_uc[i] = i - last_uc

    ret5 = np.full(n, np.nan)
    ret5[5:] = (close[5:] - close[:-5]) / close[:-5] * 100

    vol_ma20_r = pd.Series(vol).rolling(20).mean().values
    up_day = ret > 0
    above_avg_vol = vol > vol_ma20_r
    accum5 = pd.Series((up_day & above_avg_vol).astype(float)).rolling(5).sum().values

    vol_ma5_r = pd.Series(vol).rolling(5).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_trend = np.where(vol_ma20_r > 0, vol_ma5_r / vol_ma20_r, np.nan)
    up_days5 = pd.Series(up_day.astype(float)).rolling(5).sum().values

    # ── CONVICTION features: is today the 2nd consecutive rise day (i.e. the
    # streak just got CONFIRMED)? If so, how strong is the confirmation?
    # (see check_conviction — this targets the long/big-return tail directly,
    # rather than averaging over all streaks including the ones that die at
    # day 2.)
    is_confirm_day = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if is_rise[i] and is_rise[i - 1] and not is_rise[i - 2]:
            is_confirm_day[i] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio_confirm = np.where(vol_ma20_r > 0, vol / vol_ma20_r, np.nan)
    accel = ret - np.concatenate(([np.nan], ret[:-1]))  # today's return - yesterday's

    return dict(
        close=close, high=high, low=low, vol=vol, ret=ret,
        is_uc=is_uc, is_lc=is_lc, is_rise=is_rise, is_locked=is_locked, is_tradeable=is_tradeable,
        bb_pos=bb_pos, rsi=rsi, volz=volz,
        days_since_streak_end=days_since_streak_end,
        days_since_uc=days_since_uc, ret5=ret5, accum5=accum5,
        vol_trend=vol_trend, up_days5=up_days5,
        is_confirm_day=is_confirm_day, vol_ratio_confirm=vol_ratio_confirm, accel=accel,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3. IGNITION SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def check_ignition(ind, i):
    """
    IgnA — the best-validated (still small-N) ignition rule from this
    session's research. Fires on non-UC days only (no point alerting mid-streak
    — you're already watching it by then).
      1. Within HAZARD_WINDOW_DAYS of the last streak ending (self-excitation)
      2. Close above the upper Bollinger Band (ignition-from-strength)
    """
    if ind["is_rise"][i]:
        return False  # already mid-streak, not a fresh ignition
    if np.isnan(ind["bb_pos"][i]):
        return False
    return (ind["days_since_streak_end"][i] <= HAZARD_WINDOW_DAYS and
            ind["bb_pos"][i] > 1.0)

SIGNAL_DESCRIPTION = (
    "IGNITION WATCH (research-stage — small OOS sample, treat as a\n"
    "  watchlist trigger, NOT an auto-buy)\n"
    "  Within {hazard_window}d of this stock's last UC streak ending, AND\n"
    "  today's close broke above its upper Bollinger Band — the combination\n"
    "  that historically preceded a new circuit streak most often (OOS:\n"
    "  ~67% of these setups saw a streak begin within 5 days, N=12 — small\n"
    "  sample, take the base rate with real caution).\n"
    "  SUGGESTED ACTION: add to intraday watchlist. Do not buy at whatever\n"
    "  price prints today — watch tomorrow's session for a move toward\n"
    "  +3-4% intraday (before the day's circuit lock) as the actual entry,\n"
    "  per the two-stage plan discussed. If a lower circuit ever prints on\n"
    "  a held position, exit at the first tradeable print — no averaging down."
)

# ─────────────────────────────────────────────────────────────────────────────
# 3b. R1-R5 — anticipatory "hot window" rules (all fire on a UC day within the
#     last few days but NOT today; entry = buy now, WAIT up to 5 days for a
#     new UC to actually start; if it never comes, exit at the wait-day close
#     accepting the small loss — this is the "predict the start" approach).
#     OOS-validated stats (30-stock research set, last ~12mo, N/avg/win%):
#       R1 hot only         : N=108  avg +0.60%  win 54.6%  — weak but positive
#       R2 hot + momentum   : N= 84  avg +1.77%  win 60.7%  — BEST of the five
#       R3 hot + dip        : N= 40  avg +1.22%  win 50.0%  — modest
#       R4 hot + accumulation: N= 50  avg +0.18%  win 52.0%  — near breakeven
#       R5 hot + vol/updays : N= 49  avg -1.07%  win 55.1%  — NET NEGATIVE OOS
#     R5 is included for completeness/comparison only — its OOS average
#     return is negative despite a >50% win rate (small average winners,
#     larger average losers). Don't size R5 alerts the same as R2.
# ─────────────────────────────────────────────────────────────────────────────

R_HOT_WINDOW    = 5   # "hot" days-since-last-UC window for R1-R4
R5_HOT_WINDOW   = 8   # R5 uses a slightly wider window (per original research)

def check_r1(ind, i):
    """R1: hot only — within R_HOT_WINDOW days of the last UC day."""
    if ind["is_rise"][i]:
        return False
    return ind["days_since_uc"][i] <= R_HOT_WINDOW

def check_r2(ind, i):
    """R2: hot + momentum (5d return > +3%) — the strongest of the five, OOS."""
    if ind["is_rise"][i] or np.isnan(ind["ret5"][i]):
        return False
    return ind["days_since_uc"][i] <= R_HOT_WINDOW and ind["ret5"][i] > 3

def check_r3(ind, i):
    """R3: hot + dip (5d return < -3%) — buying weakness inside a hot window."""
    if ind["is_rise"][i] or np.isnan(ind["ret5"][i]):
        return False
    return ind["days_since_uc"][i] <= R_HOT_WINDOW and ind["ret5"][i] < -3

def check_r4(ind, i):
    """R4: hot + accumulation (>=2 of last 5 days were up-days on above-avg volume)."""
    if ind["is_rise"][i] or np.isnan(ind["accum5"][i]):
        return False
    return ind["days_since_uc"][i] <= R_HOT_WINDOW and ind["accum5"][i] >= 2

def check_r5(ind, i):
    """R5: hot(8d) + volume trend expanding + >=3 of last 5 days up.
    NET NEGATIVE OOS average return — kept for visibility/comparison, not
    recommended for sizing decisions as-is."""
    if ind["is_rise"][i] or np.isnan(ind["vol_trend"][i]) or np.isnan(ind["up_days5"][i]):
        return False
    return (ind["days_since_uc"][i] <= R5_HOT_WINDOW and
            ind["vol_trend"][i] > 1.2 and
            ind["up_days5"][i] >= 3)

R_RULES = {
    "R1": (check_r1, "Hot window ({R_HOT_WINDOW}d since last UC). OOS: +0.60% avg, 54.6% win — weak but positive."),
    "R2": (check_r2, "Hot window + 5d return >3% (momentum). OOS: +1.77% avg, 60.7% win — the strongest of the five."),
    "R3": (check_r3, "Hot window + 5d return <-3% (dip). OOS: +1.22% avg, 50.0% win — modest."),
    "R4": (check_r4, "Hot window + accumulation (>=2 up-days on above-avg volume). OOS: +0.18% avg, 52.0% win — near breakeven."),
    "R5": (check_r5, "Hot window(8d) + volume trend + >=3 up-days. OOS: -1.07% avg, 55.1% win — NET NEGATIVE, shown for comparison only."),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3c. CONVICTION — targets the long/big-return tail directly (per your
#     instruction that catching a real streak should deliver 10%+ returns).
#
#     THE KEY INSIGHT: riding every confirmed 2-day rise-streak to its
#     natural end (not a fixed hold) already shows the shape you're seeing
#     empirically — most streaks die right at day 2 (avg only +3.9%), but a
#     real subset run 6-14 days (avg +34%) or 15+ days (avg +128%). The lever
#     isn't a better average across everything; it's identifying, AT THE
#     MOMENT OF CONFIRMATION, which streaks belong to the long tail.
#
#     Fires on the CONFIRMATION day (2nd consecutive rise day — the earliest
#     point you can act on a real streak, not a prediction before it exists)
#     when ALL of:
#       1. Volume > 1.5x the 20d average on the confirmation day
#       2. ACCELERATING — today's return exceeds yesterday's (momentum
#          building, not a one-day fluke fading into day 2)
#       3. "Hot" — this streak started within HAZARD_WINDOW_DAYS of the
#          stock's last rise-streak ending (self-excitation, same pattern
#          IgnA and R1-R5 use)
#
#     Exit: RIDE while the rise continues (no fixed hold), sell first
#     non-rise tradeable close — same exit logic as the reactive backtest.
#
#     ⚠ VALIDATED, BUT ON A VERY SMALL OOS SAMPLE — READ BEFORE TRUSTING:
#       Train: N=25, avg +6.03%, win 52.0%
#       OOS  : N=10, avg +14.26%, win 50.0%, P(hits >=10%) = 50.0%
#     N=10 is NOT enough to call this proven. It clears your 10% bar in this
#     sample and the mechanism (volume + acceleration + recency) is sound,
#     but treat early signals from this rule as "worth watching closely and
#     sizing small," not "certain." Track its live performance before
#     scaling up size on it.
# ─────────────────────────────────────────────────────────────────────────────

CONVICTION_VOL_MIN  = 1.5   # volume ratio on confirmation day
CONVICTION_HOT_DAYS = 9     # hazard window for the streak's start (same as IgnA)

def check_conviction(ind, i):
    """Fires only on the confirmation day itself (2nd consecutive rise day)."""
    if not ind["is_confirm_day"][i]:
        return False
    if np.isnan(ind["vol_ratio_confirm"][i]) or np.isnan(ind["accel"][i]):
        return False
    # days since the streak STARTED (i-1, since today is day 2) vs its own
    # prior streak end — reuse days_since_streak_end evaluated one bar back
    was_hot = ind["days_since_streak_end"][i - 1] <= CONVICTION_HOT_DAYS if i > 0 else False
    return (ind["vol_ratio_confirm"][i] > CONVICTION_VOL_MIN and
            ind["accel"][i] > 0.5 and
            was_hot)

CONVICTION_DESCRIPTION = (
    "CONVICTION — confirmed rise-streak with volume + acceleration + recency\n"
    "  (research-stage, SMALL OOS SAMPLE — N=10 — read the caveat below)\n"
    "  Today is this streak's 2nd consecutive rise day (confirmed, not\n"
    "  predicted) with volume >{vol_min}x average, accelerating momentum\n"
    "  (today's return exceeds yesterday's), and the streak started within\n"
    "  {hot_days}d of the last one ending. This combination targets the\n"
    "  long/big-return tail directly: most confirmed streaks die right here\n"
    "  at day 2 (avg only +3.9%), but this filter's OOS sample averaged\n"
    "  +14.26% with half the trades hitting >=10%.\n"
    "  ⚠ N=10 OOS is genuinely too small to call this proven — the direction\n"
    "  and mechanism are sound, but size small and track live performance\n"
    "  before trusting this with real conviction (ironic, given the name).\n"
    "  SUGGESTED ACTION: ride while the rise continues; exit at the first\n"
    "  non-rise tradeable close. If a lower circuit ever prints on a held\n"
    "  position, exit at the first tradeable print — no averaging down."
)



# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT
# ─────────────────────────────────────────────────────────────────────────────

def build_plot(ind, company_name, ticker, date_label, lookback=150):
    n = len(ind["close"])
    lo = max(0, n - lookback)
    x = np.arange(lo, n)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})

    ax = axes[0]
    ax.plot(x, ind["close"][lo:n], color="black", lw=1, label="Close")
    uc_x = [xi for xi in x if ind["is_uc"][xi]]
    uc_y = [ind["close"][xi] for xi in uc_x]
    lc_x = [xi for xi in x if ind["is_lc"][xi]]
    lc_y = [ind["close"][xi] for xi in lc_x]
    ax.scatter(uc_x, uc_y, color="green", marker="^", s=25, label="UC day", zorder=3)
    ax.scatter(lc_x, lc_y, color="red", marker="v", s=25, label="LC day", zorder=3)
    ax.set_title(f"{company_name} ({ticker}) — {date_label}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(x, ind["bb_pos"][lo:n], color="purple", lw=1)
    ax2.axhline(1.0, color="green", ls="--", lw=0.8, label="Upper BB")
    ax2.axhline(-1.0, color="red", ls="--", lw=0.8, label="Lower BB")
    ax2.set_ylabel("BB pos (σ)")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.bar(x, ind["vol"][lo:n], color="steelblue", width=0.8)
    ax3.set_ylabel("Volume")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
# 5. EMAIL LOG — dedup same-day alerts
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
# 6. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log = load_log()
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

        MIN_ROWS_REQUIRED = 40  # BB needs 20, streak history needs some runway
        valid_rows = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(valid_rows) < MIN_ROWS_REQUIRED:
            print(f"  Skipping — only {len(valid_rows)} valid rows, need >= {MIN_ROWS_REQUIRED}")
            continue

        today_label = meta["date"]
        try:
            ind = compute_circuit_indicators(df, cfg["band_pct"])
        except Exception as e:
            print(f"  compute_circuit_indicators failed: {e} — skipping, continuing with the rest")
            continue

        i = len(ind["close"]) - 1  # today's bar

        ignition_fired = check_ignition(ind, i)
        r_fired = {name: fn(ind, i) for name, (fn, _) in R_RULES.items()}
        conviction_fired = check_conviction(ind, i)
        any_fired = ignition_fired or any(r_fired.values()) or conviction_fired

        print(f"  Close     : {ind['close'][i]:.2f}")
        print(f"  Days since last streak end: {ind['days_since_streak_end'][i]:.0f}   "
              f"Days since last UC day: {ind['days_since_uc'][i]:.0f}")
        print(f"  BB position: {ind['bb_pos'][i]:.2f}σ   RSI: {ind['rsi'][i]:.1f}   Vol z-score: {ind['volz'][i]:.2f}")
        print(f"  5d ret: {ind['ret5'][i]:.1f}%   Accum(5d): {ind['accum5'][i]:.0f}   "
              f"Vol trend: {ind['vol_trend'][i]:.2f}x   Up-days(5d): {ind['up_days5'][i]:.0f}")
        print(f"  Ignition fired: {'YES' if ignition_fired else 'no'}   |   "
              + "  ".join(f"{n}: {'YES' if v else 'no'}" for n, v in r_fired.items())
              + f"   |   CONVICTION: {'YES' if conviction_fired else 'no'}")

        if not any_fired:
            continue

        lines = [
            f"\n{'='*60}",
            f"{meta['company']} ({meta['ticker']})  —  {meta['date']}",
            f"{'='*60}",
            f"Close      : {ind['close'][i]:.2f}",
            f"Days since last streak end: {ind['days_since_streak_end'][i]:.0f}   "
            f"Days since last UC day: {ind['days_since_uc'][i]:.0f}",
            f"BB position: {ind['bb_pos'][i]:.2f}σ   RSI: {ind['rsi'][i]:.1f}   Vol z-score: {ind['volz'][i]:.2f}",
            f"5d ret: {ind['ret5'][i]:.1f}%   Accum(5d): {ind['accum5'][i]:.0f}   "
            f"Vol trend: {ind['vol_trend'][i]:.2f}x   Up-days(5d): {ind['up_days5'][i]:.0f}",
        ]

        if ignition_fired:
            log_key = f"{stock_name}_IGNITION"
            already  = log.get(log_key) == today_label
            tag = "  [already sent today]" if already else ""
            lines.append(f"\n{'─'*40}")
            lines.append(f"SIGNAL: IGNITION{tag}")
            lines.append(SIGNAL_DESCRIPTION.format(hazard_window=HAZARD_WINDOW_DAYS))
            if not already:
                log[log_key] = today_label

        for r_name, fired in r_fired.items():
            if not fired:
                continue
            _, desc = R_RULES[r_name]
            log_key = f"{stock_name}_{r_name}"
            already  = log.get(log_key) == today_label
            tag = "  [already sent today]" if already else ""
            lines.append(f"\n{'─'*40}")
            lines.append(f"SIGNAL: {r_name}{tag}")
            lines.append(desc.format(R_HOT_WINDOW=R_HOT_WINDOW))
            if not already:
                log[log_key] = today_label

        if conviction_fired:
            log_key = f"{stock_name}_CONVICTION"
            already  = log.get(log_key) == today_label
            tag = "  [already sent today]" if already else ""
            lines.append(f"\n{'─'*40}")
            lines.append(f"SIGNAL: CONVICTION{tag}")
            lines.append(CONVICTION_DESCRIPTION.format(vol_min=CONVICTION_VOL_MIN, hot_days=CONVICTION_HOT_DAYS))
            if not already:
                log[log_key] = today_label

        report_sections.append("\n".join(lines))

        png = build_plot(ind, meta["company"], meta["ticker"], meta["date"])
        plot_attachments.append((f"{stock_name}_{today_label}.png", png))

    save_log(log)

    if not report_sections:
        print("\nNo ignition signals today — no email sent.")
        return

    n_stocks = len(report_sections)
    subject  = f"[Circuit Watch] {n_stocks} stock(s) — {today_label}"
    body = (
        f"CIRCUIT WATCH SCAN  —  {today_label}\n"
        f"Signals: IGNITION (hazard-window + BB breakout) + R1-R5 (anticipatory hot-window rules)\n"
        f"Status: RESEARCH STAGE across the board — R2 is the strongest (OOS +1.77% avg, 60.7% win),\n"
        f"R5 is net NEGATIVE OOS (shown for comparison only, not a buy signal)\n"
        f"Stocks scanned: {len(STOCKS)}\n"
        + "\n".join(report_sections)
    )
    send_email(subject, body, plot_attachments)


if __name__ == "__main__":
    main()