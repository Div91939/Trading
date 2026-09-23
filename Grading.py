"""
grading.py — A/B/C conviction grading for REV (and optionally REBOUND).
=======================================================================
Imported by BOTH Combined.py and Smallcap.py so the two scanners cannot
drift apart. Pure pandas/numpy, no model files — same self-contained
philosophy as the REBOUND coefficients.

v2 (2026-09-23) — REBUILT ON OUTCOMES, NOT ON INDICATOR EXTENT
--------------------------------------------------------------
v1 graded by correlating factors with exit P/L. v2 instead asks a sharper
question: across 30,580 REV/REBOUND fires, what separated the ones that
gained MORE THAN 10% in 10 trading days from the ones that gained less
than 5%? Factors had to hold their direction in all five years.

What actually separates them (REV):
    market volatility        d=0.45  <- by far the largest single factor
    market 10-day return     d=-0.33    big winners come while the market
                                        itself is falling and jumpy
    stock ATR                d=0.25
    fire-day return          d=-0.25    a harder fall on the day
    drawdown from 60d high   d=-0.22
    own 10-day swing size    d=0.22
    past responsiveness      d=-0.16  <- see below

THE COUNTER-INTUITIVE ONE
    resp_excess = this stock's average prior 10-day outcome on this signal,
    minus the stock's own ordinary 10-day drift. It carries a NEGATIVE
    weight: stocks that responded well to the signal in the past produce
    SMALLER moves next time. Responsiveness mean-reverts rather than
    persisting, which is also why every per-stock tuning attempt in this
    project has failed out of sample. Do not "fix" the sign.

VERIFICATION
    Graded on the 10-day outcome, then scored on the ACTUAL exit-rule P/L
    (-25% stop, or the 2nd close after the next >=+5% local max):
        Smallcap REV   A +4.27%  B +2.79%  C +0.88%   (win 76.3 / 74.0 / 66.5)
        Combined REV   A +6.52%  B +2.37%  C +1.22%   (win 83.8 / 71.9 / 64.9)
    Walk-forward, ratings assigned from prior years only: P(>10% in 10d)
    rises from 7.0% at the bottom to 26.1% at the top, a 3.7x lift, and the
    grade beats a volatility-only ranking by 2.4x on volatility-adjusted
    return — so it is not merely selecting volatile stocks.

    REBOUND does NOT grade. On the real exit P/L its B and C bands are
    indistinguishable on Smallcap, and on Combined the grade inverts
    (C +5.58% vs A +1.96%). GRADE_REBOUND is therefore False.

KNOWN LIMITS
    - 2025 inverts: yearly rank-correlation +0.378 / -0.114 / +0.186. The
      grade was actively wrong for a year.
    - Only three bands are real. A 1-10 scale was tested and the middle
      ratings do not order reliably (rating 9 underperformed rating 3), so
      the output stays A/B/C.
    - High grades arrive in CLUSTERS: the dominant factor is market-wide,
      so when volatility spikes much of the universe grades A on the same
      day, in a falling market.
    - Fitted weights drift. Refit periodically with fit_grades_v2.py in the
      research repo, same as the REBOUND regression.
"""

import os
import glob
import numpy as np
import pandas as pd

GRADE_FITTED_ON = "2026-09-23"
GRADE_VERSION = "v2-outcome"

# ── switches ────────────────────────────────────────────────────────────────
GRADE_REV      = True    # grade REV fires
GRADE_REBOUND  = False   # OFF — REBOUND does not grade, see header
GRADE_DROP_C   = True    # suppress C-graded fires from alerts entirely

SMALLCAP_ROOT = "Smallcap"
FWD_H = 10               # horizon the grade was fitted on (trading days)
GRADE_SPEC_SMALLCAP = {
    "REV": {
        "cuts": [-0.161289, -0.045382],
        "feats": {
            "mkt_vol20": {"w": 0.2325, "med": 21.206091,
                           "q": [4.636417, 9.775356, 11.906622, 12.900467, 13.932914, 14.999176, 15.652291, 17.222488, 19.383312, 20.254461, 21.206091, 21.869289, 22.871919, 25.655819, 26.631542, 27.515748, 28.448292, 29.496507, 30.139327, 32.54509, 43.289941]},
            "dd60": {"w": -0.1313, "med": -20.494067,
                      "q": [-53.962983, -36.212693, -32.790712, -30.46064, -28.455339, -26.771122, -25.293639, -24.087481, -22.819417, -21.663695, -20.494067, -19.396087, -18.40834, -17.419437, -16.519102, -15.53083, -14.563255, -13.61242, -12.505584, -11.217804, -7.810712]},
            "day_ret": {"w": -0.1101, "med": -2.742732,
                         "q": [-25.204093, -7.650196, -6.210205, -5.398306, -4.996021, -4.697792, -4.217397, -3.817888, -3.428918, -3.073733, -2.742732, -2.407933, -2.072914, -1.722702, -1.370534, -0.97926, -0.509889, 0.0, 0.676358, 1.91407, 13.515331]},
            "move_vs_own_vol": {"w": -0.1009, "med": -0.302923,
                                 "q": [-4.40202, -0.961824, -0.768878, -0.657403, -0.579401, -0.515127, -0.462103, -0.421386, -0.378223, -0.336704, -0.302923, -0.272268, -0.236706, -0.197239, -0.155026, -0.1117, -0.056557, 0.0, 0.078132, 0.220321, 1.278873]},
            "ret60": {"w": -0.0936, "med": -9.149964,
                       "q": [-39.987554, -32.376371, -28.481858, -25.499737, -22.947348, -20.730313, -18.634048, -16.462555, -13.978171, -11.651466, -9.149964, -6.39694, -3.309814, -0.284324, 3.027021, 7.035927, 11.276976, 16.728345, 24.885369, 41.05265, 180.873373]},
        },
    },
    "REBOUND": {
        "cuts": [-0.181758, -0.018757],
        "feats": {
            "mkt_vol20": {"w": 0.2059, "med": 23.134373,
                           "q": [4.636417, 9.606195, 12.421247, 13.784613, 15.095315, 16.242776, 18.112637, 19.84749, 20.902696, 21.777149, 23.134373, 25.848675, 26.685304, 27.6675, 28.538442, 29.496507, 30.169678, 31.481642, 32.629974, 34.667957, 43.838014]},
            "ret60": {"w": -0.1252, "med": -3.351042,
                       "q": [-70.215463, -36.957841, -32.067084, -28.20261, -25.030148, -21.803151, -18.633275, -15.162802, -11.391304, -7.532019, -3.351042, 0.894245, 5.20743, 10.041348, 15.561049, 21.345658, 28.687901, 38.185941, 50.870619, 70.055601, 436.538462]},
            "ma50_slope": {"w": -0.1116, "med": -0.169792,
                            "q": [-23.011136, -7.585767, -6.271356, -5.340757, -4.614739, -3.92842, -3.207036, -2.553554, -1.780852, -0.957011, -0.169792, 0.562149, 1.346474, 2.162181, 3.004512, 4.002106, 5.107447, 6.344123, 7.990435, 10.745975, 32.380279]},
            "px_vs_ma200": {"w": -0.1001, "med": 6.683302,
                             "q": [-61.596915, -36.167418, -30.409719, -25.857227, -21.25826, -16.420418, -11.714851, -7.301638, -3.032114, 1.573449, 6.683302, 11.77851, 16.710439, 22.261404, 28.082908, 34.102095, 41.015116, 49.220096, 60.81487, 79.653315, 279.061609]},
            "up_from_low252": {"w": -0.0676, "med": 77.625123,
                                "q": [0.0, 0.0, 3.901395, 8.415489, 14.344704, 22.89678, 32.402272, 41.980512, 53.788017, 65.696425, 77.625123, 92.422686, 108.459154, 125.943954, 146.737096, 169.810007, 196.136252, 228.680967, 285.643357, 408.657386, 2687.528604]},
        },
    },
}
GRADE_SPEC_COMBINED = {
    "REV": {
        "cuts": [-0.063973, 0.022459],
        "feats": {
            "mkt_vol20": {"w": 0.2446, "med": 21.343525,
                           "q": [5.580679, 9.311465, 12.107456, 13.821674, 14.673863, 15.308773, 16.679987, 18.597003, 19.728493, 20.575722, 21.343525, 21.869289, 23.121458, 25.452022, 26.631542, 27.538338, 28.465241, 29.496507, 30.47922, 32.65541, 38.108281]},
            "resp_excess": {"w": -0.1129, "med": 0.917158,
                             "q": [-18.136021, -9.304613, -4.597946, -2.795722, -1.960708, -1.319767, -0.858946, -0.413563, -0.037559, 0.515694, 0.917158, 1.22981, 1.731611, 2.388056, 3.117005, 3.859245, 4.273732, 4.794116, 5.698808, 8.614757, 28.326157]},
            "prior_resp": {"w": -0.1114, "med": 3.375056,
                            "q": [-13.891696, -5.036821, -2.045356, -0.456448, 0.571272, 1.326947, 1.875031, 2.158101, 2.42454, 2.766789, 3.375056, 3.676347, 4.203506, 4.828706, 5.616208, 6.466058, 7.187842, 8.227907, 9.855976, 11.964631, 32.175722]},
            "atr_pct": {"w": -0.06, "med": 5.270463,
                         "q": [3.500514, 3.69781, 3.861119, 4.067049, 4.231955, 4.370008, 4.553243, 4.744161, 4.923554, 5.094562, 5.270463, 5.448271, 5.661434, 5.937201, 6.230107, 6.527051, 6.786624, 7.157292, 7.702534, 8.580804, 14.835229]},
        },
    },
    "REBOUND": {
        "cuts": [-0.014997, 0.014758],
        "feats": {
            "gap": {"w": -0.0768, "med": 0.070811,
                     "q": [-79.483857, -4.99763, -4.558094, -3.297186, -2.101275, -1.427281, -0.785988, -0.347893, -0.031296, 0.0, 0.070811, 0.26138, 0.502869, 0.797333, 1.112847, 1.525275, 1.980899, 2.473436, 3.440453, 4.656929, 23.193121]},
            "consec_down": {"w": 0.0641, "med": 1.0,
                             "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 24.0]},
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# MARKET CONTEXT
# ─────────────────────────────────────────────────────────────────────────────
# Built from the Smallcap folder for BOTH universes — that is the backdrop
# the grade was fitted against. Fails open: if the folder is unreadable the
# scanner grades nothing and alerts every fire.

def build_market_context(root=SMALLCAP_ROOT):
    files = glob.glob(os.path.join(root, "*", "*.csv"))
    if not files:
        return None
    px = {}
    for p in files:
        nm = os.path.splitext(os.path.basename(p))[0].upper()
        try:
            d = pd.read_csv(p)
            d.columns = d.columns.str.strip()
            d["Date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y", errors="coerce")
            d = d.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date")
            if len(d) >= 300:
                px[nm] = d.set_index("Date")["Close"].astype(float)
        except Exception:
            continue
    if len(px) < 20:
        return None
    PX = pd.DataFrame(px).sort_index()
    ret = PX.pct_change()
    mkt = ret.mean(axis=1)
    idx = (1 + mkt).cumprod()
    return {
        "mkt_vol20": mkt.rolling(20).std() * np.sqrt(252) * 100,
        "mkt_10d": idx.pct_change(10) * 100,
        "mkt_20d": idx.pct_change(20) * 100,
        "mkt_dd20": (idx / idx.rolling(20).max() - 1) * 100,
    }


def _at(series, date):
    if series is None or len(series) == 0:
        return np.nan
    try:
        s = series.loc[:date]
        return float(s.iloc[-1]) if len(s) else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# STOCK-LEVEL HISTORY FACTORS
# ─────────────────────────────────────────────────────────────────────────────

def stock_history_factors(ind, i, fire_mask):
    """base_10d / base_10d_vol / prior_resp / resp_excess / move_vs_own_vol.

    prior_resp only counts fires whose 10-bar window had already CLOSED
    before bar i, so nothing here can see the future. `fire_mask` is this
    signal's boolean fire history for this stock; pass None to skip the
    history terms (they then fall back to their stored medians)."""
    c = np.asarray(ind["close"], float)
    o = np.asarray(ind["open"], float)
    out = {}
    lo = max(0, i - 250)
    seg = c[lo:i + 1]
    if len(seg) > FWD_H + 20:
        r10 = seg[FWD_H:] / seg[:-FWD_H] - 1.0
        r10 = r10[np.isfinite(r10)] * 100
        out["base_10d"] = float(np.mean(r10)) if len(r10) else np.nan
        out["base_10d_vol"] = float(np.std(r10)) if len(r10) else np.nan
    else:
        out["base_10d"] = out["base_10d_vol"] = np.nan

    dr = ind.get("day_ret", ind.get("ret1"))
    d0 = float(dr[i]) if dr is not None and np.isfinite(dr[i]) else np.nan
    out["move_vs_own_vol"] = (d0 / out["base_10d_vol"]
                              if np.isfinite(d0) and out["base_10d_vol"] and
                              np.isfinite(out["base_10d_vol"]) and out["base_10d_vol"] > 0 else np.nan)

    prior = []
    if fire_mask is not None:
        for j in np.where(np.asarray(fire_mask, bool)[:i])[0]:
            if j + 1 + FWD_H >= i or j + 1 >= len(c):
                continue                      # window not closed before today
            ent = o[j + 1]
            if np.isfinite(ent) and ent > 0:
                prior.append((c[j + 1 + FWD_H] / ent - 1) * 100)
    out["prior_resp"] = float(np.mean(prior)) if len(prior) >= 3 else np.nan
    out["resp_excess"] = (out["prior_resp"] - out["base_10d"]
                          if np.isfinite(out["prior_resp"]) and np.isfinite(out["base_10d"])
                          else np.nan)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def _feature(name, ind, i, ctx, date, hist):
    """Key names differ slightly between the two scanners, so both
    spellings are accepted."""
    def k(*names):
        for n in names:
            if n in ind:
                v = ind[n][i]
                return float(v) if np.isfinite(v) else np.nan
        return np.nan

    if name in ("mkt_vol20", "mkt_10d", "mkt_20d", "mkt_dd20"):
        return _at(ctx.get(name), date) if ctx else np.nan
    if name in hist:
        return hist[name]
    if name == "day_ret":
        return k("day_ret", "ret1")
    if name == "ret60":
        return k("ret60", "ret60d")
    if name == "px_vs_ma200":
        return k("px_vs_ma200", "price_vs_ma200")
    if name == "px_vs_ma10":
        return k("px_vs_ma10", "price_vs_ma10")
    if name == "dd60":
        v = k("dd60")
        if np.isfinite(v):
            return v
        c = np.asarray(ind["close"], float)
        hi = np.nanmax(c[max(0, i - 59):i + 1])
        return (c[i] / hi - 1) * 100 if hi > 0 else np.nan
    return k(name)


def _pctile(q, x):
    if not np.isfinite(x):
        return 0.5
    return float(np.interp(x, q, np.linspace(0, 1, len(q))))


# ─────────────────────────────────────────────────────────────────────────────
# GRADING
# ─────────────────────────────────────────────────────────────────────────────

def grade(universe, signal, ind, i, date, ctx, fire_mask=None):
    """Returns (grade, score, detail): grade is 'A' / 'B' / 'C', or
    (None, nan, reason) when this signal is not graded."""
    spec = GRADE_SPEC_SMALLCAP if universe == "Smallcap" else GRADE_SPEC_COMBINED
    if signal not in spec:
        return None, np.nan, "no spec"
    if signal == "REV" and not GRADE_REV:
        return None, np.nan, "off"
    if signal == "REBOUND" and not GRADE_REBOUND:
        return None, np.nan, "off"
    if ctx is None:
        return None, np.nan, "no market context"

    s = spec[signal]
    hist = stock_history_factors(ind, i, fire_mask)
    score, bits = 0.0, []
    for f, e in s["feats"].items():
        v = _feature(f, ind, i, ctx, date, hist)
        if not np.isfinite(v):
            v = e["med"]
        p = _pctile(e["q"], v)
        score += e["w"] * p
        bits.append(f"{f}={v:.2f}")
    lo, hi = s["cuts"]
    g = "A" if score > hi else ("B" if score > lo else "C")
    return g, score, "  ".join(bits)


def should_alert(g):
    """False only when the fire is graded C and dropping C is switched on."""
    return not (GRADE_DROP_C and g == "C")
