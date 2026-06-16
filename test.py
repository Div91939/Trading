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
TICKER         = "ORIRAIL.BO"
CSV_PATH       = r'C:\\Users\\fourt\\OneDrive\\Desktop\\Finance\\Finance_Codes\\orirail.csv'
LOG_PATH       = r'C:\\Users\\fourt\\OneDrive\\Desktop\\Finance\\Finance_Codes\\email_log.json'
EMAIL_PASSWORD = "osrp rtab jvyv rcvz"
EMAIL_ = "divyanshdewan@gmail.com"

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
rsi7   = np.array(ta.momentum.RSIIndicator(df["Close"], window=7).rsi())
rsi21  = np.array(ta.momentum.RSIIndicator(df["Close"], window=21).rsi())

bb     = ta.volatility.BollingerBands(df["Close"], window=25, window_dev=2)
bb_up  = np.array(bb.bollinger_hband())
bb_mav = np.array(bb.bollinger_mavg())
bb_low = np.array(bb.bollinger_lband())
bb_pct = np.array(bb.bollinger_pband())

stoch_k = np.array(ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3).stoch())
willr   = np.array(ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=14).williams_r())
roc5    = np.array(ta.momentum.ROCIndicator(df["Close"], window=5).roc())

ema20    = np.array(ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator())
ema50    = np.array(ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator())
vol_ma20 = np.array(df['Volume'].rolling(20).mean())


x=[i for i in range(len(close))]

fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [4, 1]},
                                     sharex=True)
ax1.plot(x,close, color='black')
ax1.plot(x,bb_up, color='green')
ax1.plot(x,bb_low, color='red')
ax1.plot(x,bb_mav)
ax2.plot(x,stoch_k, color='purple')
ax2.axhline(70, color='red',   ls='--', lw=0.8)
ax2.axhline(30, color='green', ls='--', lw=0.8)

'''
# ─────────────────────────────────────────────
# 4. SIGNAL CHECKS
# ─────────────────────────────────────────────
def check_roc_extreme_drop(i):
    return (not np.isnan(roc5[i]) and roc5[i] < -8 and rsi[i] < 45)

def check_willr_deep_oversold(i):
    return (not np.isnan(willr[i]) and willr[i] < -90 and rsi[i] < 45)

def check_mtf_rsi_oversold(i):
    return (not np.isnan(rsi7[i]) and not np.isnan(rsi[i]) and not np.isnan(rsi21[i])
            and rsi7[i] < 30 and rsi[i] < 40 and rsi21[i] < 45)

def check_stoch_willr_oversold(i):
    return (not np.isnan(stoch_k[i]) and not np.isnan(willr[i])
            and stoch_k[i] < 15 and willr[i] < -85 and rsi[i] < 45)

def check_doji_at_bottom(i):
    if np.isnan(bb_pct[i]) or np.isnan(vol_ma20[i]): return False
    day_range = high[i] - low[i]
    if day_range == 0: return False
    body = abs(close[i] - open_[i])
    return (body / day_range) < 0.20 and bb_pct[i] < 0.25 and vol[i] > vol_ma20[i]

# ─────────────────────────────────────────────
# SIGNALS LIST  (top 5 by backtest score)
# ─────────────────────────────────────────────
SIGNALS = [
    {"name": "ROC_EXTREME_DROP",     "check": check_roc_extreme_drop,     "color": "red"},
    {"name": "WILLR_DEEP_OVERSOLD",  "check": check_willr_deep_oversold,  "color": "darkorange"},
    {"name": "MTF_RSI_OVERSOLD",     "check": check_mtf_rsi_oversold,     "color": "purple"},
    {"name": "STOCH_WILLR_OVERSOLD", "check": check_stoch_willr_oversold, "color": "blue"},
    {"name": "DOJI_AT_BOTTOM",       "check": check_doji_at_bottom,       "color": "cyan"},
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
            return json.load(f)
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
  Close      : {d['close']}
  RSI        : {d['rsi']:.2f}
  BB Low     : {d['bb_low']:.2f}
  BB Upper   : {d['bb_up']:.2f}
  Volume     : {d['vol']}
  Avg Volume : {d['avg_vol']}

HISTORICAL PERFORMANCE (backtested on {d['ticker']})
  Occurrences   : {stats.get('N',     '-')}
  Win Rate 3d   : {stats.get('W3',    '-')}
  Win Rate 10d  : {stats.get('W10',   '-')}
  Avg Return 10d: {stats.get('Avg10', '-')}
  Edge vs Base  : {stats.get('Edge',  '-')}
  (Base win rate for this stock: 54.8%)

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
    'close': close[i_today], 'rsi': rsi[i_today],
    'bb_low': bb_low[i_today], 'bb_up': bb_up[i_today],
    'vol': volume, 'avg_vol': avg_volume,
}

log = load_log()
if not triggered:
    print("\nNo signals today — no emails sent.")
else:
    print(f"\n{len(triggered)} signal(s) triggered. Sending emails...")
    for sig in triggered:
        subject = f"[ORIRAIL] {sig['name']} — {today_str}"
        body    = build_email_body(sig['name'], signal_data)
        if send_email(subject, body):
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
fig.suptitle(f"{company_name} ({TICKER})  —  {today_str}", fontsize=13, fontweight='bold')

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
ax1.set_ylabel("Price (₹)")
ax1.legend(loc='upper left', fontsize=6, ncol=3)
ax1.grid(alpha=0.3)

ax2.plot(x, rsi, color='darkorange', lw=1.2, label='RSI(14)')
ax2.axhline(70, color='red',   ls='--', lw=0.8)
ax2.axhline(30, color='green', ls='--', lw=0.8)
ax2.axhline(50, color='gray',  ls=':',  lw=0.6)
ax2.fill_between(x, rsi, 70, where=(rsi > 70), alpha=0.2, color='red')
ax2.fill_between(x, rsi, 30, where=(rsi < 30), alpha=0.2, color='green')
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
'''

