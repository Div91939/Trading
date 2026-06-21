import pandas as pd
import numpy as np
import ta
import smtplib
import os
import json
import yfinance as yf
from email.mime.text import MIMEText

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOG_PATH       = "email_log.json"
EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'osrp rtab jvyv rcvz')
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# ─────────────────────────────────────────────
# SIGNAL LIBRARY — same E1-E8 / X1-X8 definitions used across every stock
# Each function takes the indicator dict for ONE stock and a bar index i
# ─────────────────────────────────────────────
def e1_roc5_rsi(ind, i):
    return (not np.isnan(ind['roc5'][i]) and ind['roc5'][i] <= -8
            and not np.isnan(ind['rsi14'][i]) and ind['rsi14'][i] < 45)

def e2_willr_rsi(ind, i):
    return (not np.isnan(ind['willr'][i]) and ind['willr'][i] < -90
            and not np.isnan(ind['rsi14'][i]) and ind['rsi14'][i] < 45)

def e3_mtf_rsi(ind, i):
    return (not np.isnan(ind['rsi7'][i]) and not np.isnan(ind['rsi14'][i]) and not np.isnan(ind['rsi21'][i])
            and ind['rsi7'][i] < 35 and ind['rsi14'][i] < 40 and ind['rsi21'][i] < 45)

def e4_stoch_willr(ind, i):
    return (not np.isnan(ind['stoch'][i]) and ind['stoch'][i] < 15
            and not np.isnan(ind['willr'][i]) and ind['willr'][i] < -85)

def e5_doji_bblow(ind, i):
    body = abs(ind['close'][i] - ind['open_'][i])
    rng  = ind['high'][i] - ind['low'][i]
    return (rng > 0 and body / rng < 0.1
            and not np.isnan(ind['bb_low'][i]) and ind['close'][i] <= ind['bb_low'][i] * 1.02)

def e6_macd_turn_rsi(ind, i):
    if i < 1:
        return False
    macd_h = ind['macd_h']
    return (not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1])
            and macd_h[i] > macd_h[i-1] and macd_h[i-1] < 0
            and not np.isnan(ind['rsi14'][i]) and ind['rsi14'][i] < 45)

def e7_bblow_rsi(ind, i):
    return (not np.isnan(ind['bb_low'][i]) and ind['close'][i] < ind['bb_low'][i]
            and not np.isnan(ind['rsi14'][i]) and ind['rsi14'][i] < 40)

def e8_ema50_bounce(ind, i):
    ema50 = ind['ema50']
    return (not np.isnan(ema50[i]) and abs(ind['close'][i] / ema50[i] - 1) < 0.015
            and ind['close'][i] >= ema50[i]
            and not np.isnan(ind['rsi14'][i]) and 35 <= ind['rsi14'][i] <= 50)

def x1_rsi_overbought(ind, i):
    return not np.isnan(ind['rsi14'][i]) and ind['rsi14'][i] > 70

def x2_bb_upper(ind, i):
    return not np.isnan(ind['bb_up'][i]) and ind['close'][i] > ind['bb_up'][i]

def x3_macd_turn_neg(ind, i):
    if i < 1:
        return False
    macd_h = ind['macd_h']
    return (not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1])
            and macd_h[i] < macd_h[i-1] and macd_h[i-1] > 0)

def x4_willr_high(ind, i):
    return not np.isnan(ind['willr'][i]) and ind['willr'][i] > -10

def x5_stoch_high(ind, i):
    return not np.isnan(ind['stoch'][i]) and ind['stoch'][i] > 85

def x6_roc5_surge(ind, i):
    return not np.isnan(ind['roc5'][i]) and ind['roc5'][i] > 12

def x7_rsi7_high(ind, i):
    return not np.isnan(ind['rsi7'][i]) and ind['rsi7'][i] > 75

def x8_bbup_volfade(ind, i):
    return (not np.isnan(ind['bb_up'][i]) and ind['close'][i] > ind['bb_up'][i]
            and not np.isnan(ind['vol_ma20'][i]) and ind['vol_ma20'][i] > 0
            and ind['vol'][i] < 0.9 * ind['vol_ma20'][i])

ENTRY_FUNCS = {
    "E1_ROC5_RSI":     e1_roc5_rsi,
    "E2_WILLR_RSI":    e2_willr_rsi,
    "E3_MTF_RSI":      e3_mtf_rsi,
    "E4_STOCH_WILLR":  e4_stoch_willr,
    "E5_DOJI_BBLOW":   e5_doji_bblow,
    "E6_MACD_TURN_RSI":e6_macd_turn_rsi,
    "E7_BBLOW_RSI":    e7_bblow_rsi,
    "E8_EMA50_BOUNCE": e8_ema50_bounce,
}

EXIT_FUNCS = {
    "X1_RSI_OVERBOUGHT":   x1_rsi_overbought,
    "X2_BB_UPPER":         x2_bb_upper,
    "X3_MACD_TURN_NEG":    x3_macd_turn_neg,
    "X4_WILLR_HIGH":       x4_willr_high,
    "X5_STOCH_HIGH":       x5_stoch_high,
    "X6_ROC5_SURGE":       x6_roc5_surge,
    "X7_RSI7_HIGH":        x7_rsi7_high,
    "X8_BBUP_VOLFADE":     x8_bbup_volfade,
}

ENTRY_DESCRIPTIONS = {
    "E1_ROC5_RSI":      "5-day rate of change <= -8% AND RSI(14) < 45. Sharp short-term drop with confirming oversold momentum.",
    "E2_WILLR_RSI":     "Williams %R < -90 AND RSI(14) < 45. Price near the bottom of its 14-day range with RSI confirmation.",
    "E3_MTF_RSI":       "RSI(7) < 35 AND RSI(14) < 40 AND RSI(21) < 45. Three timeframes all confirming deep oversold.",
    "E4_STOCH_WILLR":   "Stochastic %K < 15 AND Williams %R < -85. Double oversold across momentum oscillators.",
    "E5_DOJI_BBLOW":    "Doji-style candle (small body) forming at/near the lower Bollinger Band.",
    "E6_MACD_TURN_RSI": "MACD histogram turning up from negative territory while RSI(14) < 45.",
    "E7_BBLOW_RSI":     "Close below the lower Bollinger Band AND RSI(14) < 40. Double oversold confirmation.",
    "E8_EMA50_BOUNCE":  "Price within 1.5% of EMA50 from above, with RSI(14) between 35-50 — a trend-pullback bounce.",
}

EXIT_DESCRIPTIONS = {
    "X1_RSI_OVERBOUGHT": "RSI(14) crosses above 70 — momentum has run hot.",
    "X2_BB_UPPER":       "Close breaks above the upper Bollinger Band — price at statistical extension.",
    "X3_MACD_TURN_NEG":  "MACD histogram turns down from positive territory — momentum rolling over. (Generally the weakest exit historically — fires early, caps gains.)",
    "X4_WILLR_HIGH":     "Williams %R rises above -10 — price near the top of its 14-day range.",
    "X5_STOCH_HIGH":     "Stochastic %K rises above 85 — short-term overbought.",
    "X6_ROC5_SURGE":     "5-day rate of change exceeds +12% — sharp momentum surge, take profit into strength.",
    "X7_RSI7_HIGH":      "RSI(7) rises above 75 — fast oscillator overbought.",
    "X8_BBUP_VOLFADE":   "Close above upper BB AND volume fading below its 20-day average — move has run out of fuel. Strongest exit across most stocks tested.",
}

# ─────────────────────────────────────────────
# STOCK UNIVERSE
# Each stock: ticker, CSV path, and curated entry/exit pairs
# (best-recommended pairs per stock from backtesting; not all 64 combos)
# ─────────────────────────────────────────────
STOCKS = {
    "BSE": {
        "ticker": "BSE.ns",
        "csv_path": "Data/bse.csv",
        "pairs": [("E2_WILLR_RSI", "X2_BB_UPPER"),
                  ("E3_MTF_RSI",   "X8_BBUP_VOLFADE"),
                  ("E4_STOCH_WILLR","X8_BBUP_VOLFADE")],
    },
    "EDELWEISS": {
        "ticker": "EDELWEISS.BO",
        "csv_path": "Data/edelweiss.csv",
        "pairs": [("E1_ROC5_RSI",      "X3_MACD_TURN_NEG"),
                  ("E6_MACD_TURN_RSI", "X3_MACD_TURN_NEG"),
                  ("E7_BBLOW_RSI",     "X2_BB_UPPER")],
    },
    "EICHER": {
        "ticker": "EICHERMOT.BO",
        "csv_path": "Data/eicher.csv",
        "pairs": [("E3_MTF_RSI",       "X8_BBUP_VOLFADE"),
                  ("E6_MACD_TURN_RSI", "X8_BBUP_VOLFADE"),
                  ("E7_BBLOW_RSI",     "X2_BB_UPPER")],
    },
    "HINDCOPPER": {
        "ticker": "HINDCOPPER.NS",
        "csv_path": "Data/hindcopper.csv",
        "pairs": [("E1_ROC5_RSI",   "X1_RSI_OVERBOUGHT"),
                  ("E2_WILLR_RSI",  "X1_RSI_OVERBOUGHT"),
                  ("E6_MACD_TURN_RSI","X2_BB_UPPER")],
    },
    "ORIRAIL": {
        "ticker": "ORIENTRAIL.NS",
        "csv_path": "Data/orirail.csv",
        "pairs": [("E4_STOCH_WILLR", "X8_BBUP_VOLFADE"),
                  ("E2_WILLR_RSI",   "X8_BBUP_VOLFADE"),
                  ("E1_ROC5_RSI",    "X1_RSI_OVERBOUGHT")],
    },
    "INDOTHAI": {
        "ticker": "INDOTHAI.NS",
        "csv_path": "Data/indothai.csv",
        "pairs": [("E3_MTF_RSI", "X8_BBUP_VOLFADE"),
                  ("E3_MTF_RSI", "X5_STOCH_HIGH"),
                  ("E3_MTF_RSI", "X6_ROC5_SURGE")],
    },
    "PARAS": {
        "ticker": "PARAS.NS",
        "csv_path": "Data/paras.csv",
        "pairs": [("E3_MTF_RSI",      "X4_WILLR_HIGH"),
                  ("E4_STOCH_WILLR",  "X2_BB_UPPER"),
                  ("E3_MTF_RSI",      "X2_BB_UPPER")],
    },
    "HINDZINC": {
        "ticker": "HINDZINC.BO",
        "csv_path": "Data/hindzinc.csv",
        "pairs": [("E3_MTF_RSI",   "X1_RSI_OVERBOUGHT"),
                  ("E2_WILLR_RSI", "X1_RSI_OVERBOUGHT"),
                  ("E7_BBLOW_RSI", "X1_RSI_OVERBOUGHT")],
    },
    # SARTHAKGL intentionally omitted — backtesting found no statistically
    # significant entry-exit pair for this stock (test-period was a clean
    # uptrend with no pullbacks for a mean-reversion strategy to catch).
}

# ─────────────────────────────────────────────
# 1. FETCH + APPEND TODAY'S BAR FOR ONE STOCK
# ─────────────────────────────────────────────
def fetch_and_update_csv(ticker, csv_path):
    stock = yf.Ticker(ticker)
    info  = stock.info

    company_name = info.get('longName') or info.get('shortName') or ticker
    avg_volume    = info.get('averageDailyVolume10Day') or 0
    volume        = info.get('volume') or 0

    hist_today = stock.history(period="1d", interval="1d")
    if hist_today.empty:
        return None, None  # market closed / no data

    today_str = hist_today.index[-1].strftime('%d-%m-%Y')

    new_row = {
        'Date':         today_str,
        'Open':         round(float(hist_today['Open'].iloc[-1]),  2),
        'High':         round(float(hist_today['High'].iloc[-1]),  2),
        'Low':          round(float(hist_today['Low'].iloc[-1]),   2),
        'Close':        round(float(hist_today['Close'].iloc[-1]), 2),
        'Volume':       volume,
        'Avg_Volume':   avg_volume,
        'Dividends':    round(float(hist_today['Dividends'].iloc[-1]), 2),
        'Stock Splits': round(float(hist_today['Stock Splits'].iloc[-1]), 2),
    }

    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame([new_row])
        df.to_csv(csv_path, index=False)
        df = pd.read_csv(csv_path)
        meta = {
            'company': company_name, 'ticker': ticker, 'date': today_str,
            'volume': volume, 'avg_volume': avg_volume,
        }
        return df, meta

    df = pd.read_csv(csv_path)
    df = df.dropna(how='all').drop_duplicates(subset='Date', keep='last')
    df.columns = df.columns.str.strip()

    if today_str in df['Date'].values:
        idx = df.index[df['Date'] == today_str][0]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg_Volume']:
            df.loc[idx, col] = new_row[col]
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)

    meta = {
        'company': company_name, 'ticker': ticker, 'date': today_str,
        'volume': volume, 'avg_volume': avg_volume,
    }
    return df, meta

# ─────────────────────────────────────────────
# 2. COMPUTE INDICATOR SET FOR ONE STOCK
# ─────────────────────────────────────────────
def compute_indicators(df):
    close  = np.array(df['Close'])
    high   = np.array(df['High'])
    low    = np.array(df['Low'])
    open_  = np.array(df['Open'])
    vol    = np.array(df['Volume'], dtype=float)

    rsi14 = np.array(ta.momentum.RSIIndicator(df["Close"], window=14).rsi())
    rsi7  = np.array(ta.momentum.RSIIndicator(df["Close"], window=7).rsi())
    rsi21 = np.array(ta.momentum.RSIIndicator(df["Close"], window=21).rsi())

    bb     = ta.volatility.BollingerBands(df["Close"], window=25, window_dev=2)
    bb_up  = np.array(bb.bollinger_hband())
    bb_low = np.array(bb.bollinger_lband())

    stoch = np.array(ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3).stoch())
    willr = np.array(ta.momentum.WilliamsRIndicator(
        df["High"], df["Low"], df["Close"], lbp=14).williams_r())

    macd_i = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    macd_h = np.array(macd_i.macd_diff())

    ema50    = np.array(ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator())
    vol_ma20 = np.array(df['Volume'].rolling(20).mean())
    roc5     = np.array(df['Close'].pct_change(5) * 100)

    return {
        'close': close, 'high': high, 'low': low, 'open_': open_, 'vol': vol,
        'rsi14': rsi14, 'rsi7': rsi7, 'rsi21': rsi21,
        'bb_up': bb_up, 'bb_low': bb_low,
        'stoch': stoch, 'willr': willr, 'macd_h': macd_h,
        'ema50': ema50, 'vol_ma20': vol_ma20, 'roc5': roc5,
    }

# ─────────────────────────────────────────────
# 3. EMAIL LOG (dedup so we don't re-send same-day alerts)
# ─────────────────────────────────────────────
def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            content = f.read().strip()
            if not content:
                return {}
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {}
    return {}

def save_log(log):
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f)

def send_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From']    = EMAIL_SENDER
    msg['To']      = EMAIL_RECEIVER
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"Email sent: {subject}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

# ─────────────────────────────────────────────
# 4. MAIN LOOP — scan every stock for today's signals
# ─────────────────────────────────────────────
def main():
    log = load_log()
    today_date_label = None
    report_sections = []

    for stock_name, cfg in STOCKS.items():
        print(f"\n--- {stock_name} ---")
        try:
            df, meta = fetch_and_update_csv(cfg["ticker"], cfg["csv_path"])
        except Exception as e:
            print(f"Failed to fetch/update {stock_name}: {e}")
            continue

        if df is None:
            print(f"No data for {stock_name} (market may be closed).")
            continue

        today_date_label = meta['date']
        ind = compute_indicators(df)
        i_today = len(ind['close']) - 1

        # which entries/exits are actually used by this stock's curated pairs
        used_entries = sorted({p[0] for p in cfg["pairs"]})
        used_exits   = sorted({p[1] for p in cfg["pairs"]})

        fired_entries = [name for name in used_entries if ENTRY_FUNCS[name](ind, i_today)]
        fired_exits   = [name for name in used_exits   if EXIT_FUNCS[name](ind, i_today)]

        for name in used_entries:
            print(f"  ENTRY {name}: {'TRIGGERED' if name in fired_entries else 'no'}")
        for name in used_exits:
            print(f"  EXIT  {name}: {'TRIGGERED' if name in fired_exits else 'no'}")

        if not fired_entries and not fired_exits:
            continue

        lines = [f"\n{'='*60}", f"{meta['company']} ({meta['ticker']})  --  {meta['date']}", f"{'='*60}"]
        lines.append(
            f"Close: {ind['close'][i_today]:.2f}  RSI14: {ind['rsi14'][i_today]:.2f}  "
            f"WillR: {ind['willr'][i_today]:.2f}  Stoch: {ind['stoch'][i_today]:.2f}  "
            f"BB Up: {ind['bb_up'][i_today]:.2f}  BB Low: {ind['bb_low'][i_today]:.2f}"
        )

        for name in fired_entries:
            log_key = f"{stock_name}_ENTRY_{name}"
            tag = " [already sent today]" if log.get(log_key) == today_date_label else ""
            lines.append(f"\nENTRY: {name}{tag}")
            lines.append(f"  {ENTRY_DESCRIPTIONS.get(name, '')}")
            recommended_exits = sorted({p[1] for p in cfg["pairs"] if p[0] == name})
            lines.append(f"  Recommended exit(s): {', '.join(recommended_exits)}")
            if not tag:
                log[log_key] = today_date_label

        for name in fired_exits:
            log_key = f"{stock_name}_EXIT_{name}"
            tag = " [already sent today]" if log.get(log_key) == today_date_label else ""
            lines.append(f"\nEXIT: {name}{tag}")
            lines.append(f"  {EXIT_DESCRIPTIONS.get(name, '')}")
            if not tag:
                log[log_key] = today_date_label

        report_sections.append("\n".join(lines))

    save_log(log)

    if not report_sections:
        print("\nNo signals across any stock today — no email sent.")
        return

    full_body = (
        f"DAILY SIGNAL SCAN — {today_date_label}\n"
        f"Stocks scanned: {len(STOCKS)}\n"
        + "\n".join(report_sections)
    )
    send_email(f"[Daily Scan] {len(report_sections)} stock(s) with signals — {today_date_label}", full_body)

if __name__ == "__main__":
    main()