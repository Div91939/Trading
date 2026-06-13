import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import smtplib
import os
import json
import yfinance as yf
from email.mime.text import MIMEText

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKER         = "HINDZINC.BO"
CSV_PATH = "Data/eicher.csv"
LOG_PATH = "email_log.json"
EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'osrp rtab jvyv rcvz')
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# ─────────────────────────────────────────────
# BACKTEST STATS (Test period: Jun 2025 - Jun 2026)
# Base: mean +1.84%, std 7.96%
# Only signals with p < 0.05 (statistically significant)
# ─────────────────────────────────────────────
BACKTEST = {
    "MACD_TURN_RSI":  {"N": 7,  "Mean": "+8.88%", "Win": "100%", "Sharpe": "1.39", "p": "0.027"},
    "RSI14_SUB35":    {"N": 6,  "Mean": "+7.33%", "Win":  "83%", "Sharpe": "1.17", "p": "0.085"},
    "BB_SQUEEZE_BREAK":{"N":13, "Mean": "-2.49%", "Win":  "39%", "Sharpe":"-0.36", "p": "0.048"},
}

SIGNAL_DESCRIPTIONS = {
    "MACD_TURN_RSI":   "MACD histogram turning up from negative while RSI < 45. Momentum shifting from negative to positive at oversold levels. Best signal on HindZinc — 100% win rate across 7 test triggers, mean return +8.88%, Sharpe 1.39.",
    "RSI14_SUB35":     "RSI(14) below 35. Classic oversold reading — 83% win rate on HindZinc in test period, mean return +7.33%.",
    "BB_SQUEEZE_BREAK":"Bollinger Band squeeze (bandwidth compressed) followed by price breakout with volume. WARNING — this signal was bearish on HindZinc in testing (win rate 39%, mean -2.49%). Treat as a SHORT signal or avoid.",
}

# ─────────────────────────────────────────────
# 1. FETCH FROM YAHOO FINANCE
# ─────────────────────────────────────────────
stock = yf.Ticker(TICKER)
info  = stock.info

company_name = info.get('longName') or info.get('shortName') or TICKER
avg_volume   = info.get('averageDailyVolume10Day') or 0
volume       = info.get('volume') or 0

hist_today = stock.history(period="1d", interval="1d")
if hist_today.empty:
    raise ValueError("No data returned from yfinance. Market may be closed.")

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

# ─────────────────────────────────────────────
# 2. APPEND OR UPDATE CSV
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna(how='all').drop_duplicates(subset='Date', keep='last')
df.columns = df.columns.str.strip()

if today_str in df['Date'].values:
    idx = df.index[df['Date'] == today_str][0]
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg_Volume']:
        df.loc[idx, col] = new_row[col]
else:
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv(CSV_PATH, index=False)
df = pd.read_csv(CSV_PATH)

# ─────────────────────────────────────────────
# 3. COMPUTE INDICATORS
# ─────────────────────────────────────────────
close  = np.array(df['Close'])
high   = np.array(df['High'])
low    = np.array(df['Low'])
open_  = np.array(df['Open'])
vol    = np.array(df['Volume'], dtype=float)

rsi    = np.array(ta.momentum.RSIIndicator(df["Close"], window=14).rsi())

bb     = ta.volatility.BollingerBands(df["Close"], window=25, window_dev=2)
bb_up  = np.array(bb.bollinger_hband())
bb_mav = np.array(bb.bollinger_mavg())
bb_low = np.array(bb.bollinger_lband())
bb_pct = np.array(bb.bollinger_pband())
bb_wid = np.array(bb.bollinger_wband())
bb_wid_ma = pd.Series(bb_wid).rolling(20).mean().values

macd_i = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
macd_h = np.array(macd_i.macd_diff())

ema20    = np.array(ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator())
ema50    = np.array(ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator())
vol_ma20 = np.array(df['Volume'].rolling(20).mean())

# ─────────────────────────────────────────────
# 4. SIGNAL CHECKS
# ─────────────────────────────────────────────
def check_macd_turn_rsi(i):
    if i < 1: return False
    return (not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1])
            and macd_h[i] > macd_h[i-1]
            and macd_h[i-1] < 0
            and not np.isnan(rsi[i])
            and rsi[i] < 45)

def check_rsi14_sub35(i):
    return not np.isnan(rsi[i]) and rsi[i] < 35

def check_bb_squeeze_break(i):
    if i < 1: return False
    return (not np.isnan(bb_wid_ma[i])
            and bb_wid[i] < bb_wid_ma[i]
            and close[i] > close[i-1]
            and not np.isnan(vol_ma20[i])
            and vol[i] > 1.5 * vol_ma20[i])

# ─────────────────────────────────────────────
# SIGNALS LIST
# Ranked by test-period Sharpe
# Note: BB_SQUEEZE_BREAK is bearish on this stock
# ─────────────────────────────────────────────
SIGNALS = [
    {"name": "MACD_TURN_RSI",   "check": check_macd_turn_rsi,   "color": "green"},
    {"name": "RSI14_SUB35",     "check": check_rsi14_sub35,     "color": "red"},
    {"name": "BB_SQUEEZE_BREAK","check": check_bb_squeeze_break,"color": "orange"},
]

# ─────────────────────────────────────────────
# 5. TODAY'S SIGNAL DETECTION
# ─────────────────────────────────────────────
i_today = len(close) - 1

triggered = []
for sig in SIGNALS:
    if sig["check"](i_today):
        triggered.append(sig)
        print(f"Signal triggered: {sig['name']}")
    else:
        print(f"No signal:        {sig['name']}")

# ─────────────────────────────────────────────
# 6. EMAIL
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

def build_email_body(sig_name, d):
    desc  = SIGNAL_DESCRIPTIONS.get(sig_name, "")
    stats = BACKTEST.get(sig_name, {})
    return f"""
{'='*55}
{d['company']} ({d['ticker']})
Signal: {sig_name}
Date:   {d['date']}
{'='*55}

WHAT THIS SIGNAL MEANS
{desc}

TODAY'S READINGS
  Close      : {d['close']:.2f}
  RSI (14)   : {d['rsi']:.2f}
  BB Low     : {d['bb_low']:.2f}
  BB Upper   : {d['bb_up']:.2f}
  MACD Hist  : {d['macd_h']:.4f}
  Volume     : {d['vol']:,}
  Avg Volume : {d['avg_vol']:,}

BACKTEST PERFORMANCE (Test: Jun 2025 - Jun 2026)
  Triggers (N)    : {stats.get('N',      '-')}
  Mean 10d return : {stats.get('Mean',   '-')}
  Win rate        : {stats.get('Win',    '-')}
  Sharpe          : {stats.get('Sharpe', '-')}
  p-value         : {stats.get('p',      '-')}
  Base mean return: +1.84%

{'='*55}
"""

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

signal_data = {
    'company': company_name, 'ticker': TICKER, 'date': today_str,
    'close':   close[i_today], 'rsi': rsi[i_today],
    'bb_low':  bb_low[i_today], 'bb_up': bb_up[i_today],
    'macd_h':  macd_h[i_today] if not np.isnan(macd_h[i_today]) else 0.0,
    'vol':     volume, 'avg_vol': avg_volume,
}

log = load_log()
if not triggered:
    print("\nNo signals today — no emails sent.")
else:
    print(f"\n{len(triggered)} signal(s) triggered. Sending emails...")
    for sig in triggered:
        if log.get(sig['name']) == today_str:
            print(f"Already emailed {sig['name']} today — skipping.")
            continue
        if send_email(f"[HINDZINC] {sig['name']} — {today_str}",
                      build_email_body(sig['name'], signal_data)):
            log[sig['name']] = today_str
            save_log(log)

# ─────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────
n = len(close)
x = np.arange(n)

sig_arrays = {sig["name"]: np.array([sig["check"](i) for i in range(n)]) for sig in SIGNALS}

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1, 1]},
                                     sharex=True)
fig.suptitle(f"{company_name} ({TICKER})  --  {today_str}", fontsize=13, fontweight='bold')

ax1.plot(x, close,  color='black',     lw=1.5, label='Close')
ax1.plot(x, bb_up,  color='green',     lw=0.8, ls='--', label='BB Upper')
ax1.plot(x, bb_low, color='red',       lw=0.8, ls='--', label='BB Lower')
ax1.plot(x, bb_mav, color='steelblue', lw=0.8, ls='--', label='BB Mid')
ax1.plot(x, ema20,  color='orange',    lw=0.8, label='EMA20')
ax1.plot(x, ema50,  color='purple',    lw=0.8, label='EMA50')
ax1.fill_between(x, bb_low, bb_up, alpha=0.05, color='steelblue')
for sig in SIGNALS:
    arr = sig_arrays[sig["name"]]
    ax1.scatter(x, np.where(arr, close - close * 0.012, np.nan),
                color=sig["color"], marker='^', s=55, zorder=5, label=sig["name"])
ax1.set_ylabel("Price (Rs)")
ax1.legend(loc='upper left', fontsize=6, ncol=3)
ax1.grid(alpha=0.3)

ax2.plot(x, rsi, color='darkorange', lw=1.2, label='RSI(14)')
ax2.axhline(70, color='red',   ls='--', lw=0.8)
ax2.axhline(35, color='green', ls='--', lw=0.8)
ax2.axhline(50, color='gray',  ls=':',  lw=0.6)
ax2.fill_between(x, rsi, 35, where=(rsi < 35), alpha=0.2, color='green')
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI")
ax2.legend(fontsize=7)
ax2.grid(alpha=0.3)

vol_colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
ax3.bar(x, vol, color=vol_colors, alpha=0.7, width=0.8)
ax3.plot(x, vol_ma20, color='blue', lw=1, label='Vol MA20')
ax3.set_ylabel("Volume")
ax3.set_xlabel("Bar index")
ax3.legend(fontsize=7)
ax3.grid(alpha=0.3)

plt.tight_layout()
#plt.show()
