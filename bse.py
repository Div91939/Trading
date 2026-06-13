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
TICKER         = "BSE.ns"
DATA_DIR       = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Desktop', 'Finance', 'Finance_Codes')
CSV_PATH = "Data/bse.csv"
LOG_PATH       = "email_log.json"
EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'osrp rtab jvyv rcvz')
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# ─────────────────────────────────────────────
# BACKTEST STATS (Test period: Jun 2025 - Jun 2026)
# Base: mean +2.57%, std 8.29%
# Significant signals only (p < 0.05 marked with *)
# ─────────────────────────────────────────────
BACKTEST = {
    "CCI_MFI":      {"N":  8, "Mean": "+12.91%", "Win": "88%", "Sharpe": "1.61", "p": "0.008"},
    "RSI14_SUB35":  {"N":  7, "Mean": "+11.02%", "Win": "86%", "Sharpe": "1.24", "p": "0.046"},
    "STOCH_SUB10":  {"N":  7, "Mean": "+11.02%", "Win": "86%", "Sharpe": "1.24", "p": "0.046"},
    "MTF_RSI":      {"N":  9, "Mean":  "+8.80%", "Win": "89%", "Sharpe": "1.18", "p": "0.038"},
    "WILLR_DEEP":   {"N": 18, "Mean":  "+8.71%", "Win": "94%", "Sharpe": "1.06", "p": "0.007"},
    "STOCH_WILLR":  {"N": 22, "Mean":  "+8.64%", "Win": "96%", "Sharpe": "1.13", "p": "0.002"},
    "CCI_RSI_100":  {"N": 22, "Mean":  "+7.05%", "Win": "82%", "Sharpe": "0.84", "p": "0.024"},
}

SIGNAL_DESCRIPTIONS = {
    "CCI_MFI":     "CCI below -100 AND MFI below 40. CCI measures how far price deviated from its average — below -100 is extreme. MFI confirms real money flow is negative. Best signal on BSE — 88% win rate, mean +12.91%, Sharpe 1.61, p=0.008.",
    "RSI14_SUB35": "RSI(14) below 35. Classic oversold — 86% win rate, mean +11.02% on BSE in test period.",
    "STOCH_SUB10": "Stochastic K below 10. Extreme short-term oversold — price near the absolute bottom of its recent 14-day range. 86% win rate, mean +11.02% on BSE.",
    "MTF_RSI":     "RSI(7) below 30 AND RSI(14) below 40 AND RSI(21) below 45. Three timeframes all confirming deep oversold simultaneously. 89% win rate, mean +8.80%, p=0.038.",
    "WILLR_DEEP":  "Williams %R below -90 AND RSI below 45. Price near the bottom of its 14-day high-low range with RSI confirmation. 94% win rate across 18 triggers, mean +8.71%, p=0.007.",
    "STOCH_WILLR": "Stochastic K below 15 AND Williams %R below -85 AND RSI below 45. Triple oversold across three oscillators. Highest-N significant signal on BSE — 96% win rate across 22 triggers, p=0.002.",
    "CCI_RSI_100": "CCI below -100 AND RSI below 40. Double oversold confirmation. 82% win rate across 22 triggers, mean +7.05%, p=0.024.",
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

rsi14  = np.array(ta.momentum.RSIIndicator(df["Close"], window=14).rsi())
rsi7   = np.array(ta.momentum.RSIIndicator(df["Close"], window=7).rsi())
rsi21  = np.array(ta.momentum.RSIIndicator(df["Close"], window=21).rsi())

bb     = ta.volatility.BollingerBands(df["Close"], window=25, window_dev=2)
bb_up  = np.array(bb.bollinger_hband())
bb_mav = np.array(bb.bollinger_mavg())
bb_low = np.array(bb.bollinger_lband())

stoch_k = np.array(ta.momentum.StochasticOscillator(
    df["High"], df["Low"], df["Close"], window=14, smooth_window=3).stoch())
willr   = np.array(ta.momentum.WilliamsRIndicator(
    df["High"], df["Low"], df["Close"], lbp=14).williams_r())
cci     = np.array(ta.trend.CCIIndicator(
    df["High"], df["Low"], df["Close"], window=20).cci())
mfi     = np.array(ta.volume.MFIIndicator(
    df["High"], df["Low"], df["Close"], df["Volume"].astype(float),
    window=14).money_flow_index())

ema20    = np.array(ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator())
ema50    = np.array(ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator())
vol_ma20 = np.array(df['Volume'].rolling(20).mean())

# ─────────────────────────────────────────────
# 4. SIGNAL CHECKS
# ─────────────────────────────────────────────
def check_cci_mfi(i):
    return (not np.isnan(cci[i]) and not np.isnan(mfi[i])
            and cci[i] < -100 and mfi[i] < 40)

def check_rsi14_sub35(i):
    return not np.isnan(rsi14[i]) and rsi14[i] < 35

def check_stoch_sub10(i):
    return not np.isnan(stoch_k[i]) and stoch_k[i] < 10

def check_mtf_rsi(i):
    return (not np.isnan(rsi7[i]) and not np.isnan(rsi14[i]) and not np.isnan(rsi21[i])
            and rsi7[i] < 30 and rsi14[i] < 40 and rsi21[i] < 45)

def check_willr_deep(i):
    return (not np.isnan(willr[i]) and not np.isnan(rsi14[i])
            and willr[i] < -90 and rsi14[i] < 45)

def check_stoch_willr(i):
    return (not np.isnan(stoch_k[i]) and not np.isnan(willr[i]) and not np.isnan(rsi14[i])
            and stoch_k[i] < 15 and willr[i] < -85 and rsi14[i] < 45)

def check_cci_rsi_100(i):
    return (not np.isnan(cci[i]) and not np.isnan(rsi14[i])
            and cci[i] < -100 and rsi14[i] < 40)

# ─────────────────────────────────────────────
# SIGNALS LIST
# Ordered by Sharpe ratio (test period)
# ─────────────────────────────────────────────
SIGNALS = [
    {"name": "CCI_MFI",      "check": check_cci_mfi,      "color": "red"},
    {"name": "RSI14_SUB35",  "check": check_rsi14_sub35,  "color": "darkorange"},
    {"name": "STOCH_SUB10",  "check": check_stoch_sub10,  "color": "purple"},
    {"name": "MTF_RSI",      "check": check_mtf_rsi,      "color": "blue"},
    {"name": "WILLR_DEEP",   "check": check_willr_deep,   "color": "green"},
    {"name": "STOCH_WILLR",  "check": check_stoch_willr,  "color": "teal"},
    {"name": "CCI_RSI_100",  "check": check_cci_rsi_100,  "color": "cyan"},
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
  Close        : {d['close']:.2f}
  RSI (14)     : {d['rsi14']:.2f}
  RSI (7)      : {d['rsi7']:.2f}
  RSI (21)     : {d['rsi21']:.2f}
  Stochastic K : {d['stoch']:.2f}
  Williams R   : {d['willr']:.2f}
  CCI          : {d['cci']:.2f}
  MFI          : {d['mfi']:.2f}
  BB Low       : {d['bb_low']:.2f}
  BB Upper     : {d['bb_up']:.2f}
  Volume       : {d['vol']:,}
  Avg Volume   : {d['avg_vol']:,}

BACKTEST PERFORMANCE (Test: Jun 2025 - Jun 2026)
  Triggers (N)    : {stats.get('N',      '-')}
  Mean 10d return : {stats.get('Mean',   '-')}
  Win rate        : {stats.get('Win',    '-')}
  Sharpe          : {stats.get('Sharpe', '-')}
  p-value         : {stats.get('p',      '-')}
  Base mean return: +2.57%

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
    'close':   close[i_today],
    'rsi14':   rsi14[i_today],
    'rsi7':    rsi7[i_today],
    'rsi21':   rsi21[i_today],
    'stoch':   stoch_k[i_today],
    'willr':   willr[i_today],
    'cci':     cci[i_today],
    'mfi':     mfi[i_today],
    'bb_low':  bb_low[i_today], 'bb_up': bb_up[i_today],
    'vol': volume, 'avg_vol': avg_volume,
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
        if send_email(f"[BSE] {sig['name']} — {today_str}",
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
                color=sig["color"], marker='^', s=40, zorder=5, label=sig["name"])
ax1.set_ylabel("Price (Rs)")
ax1.legend(loc='upper left', fontsize=6, ncol=4)
ax1.grid(alpha=0.3)

ax2.plot(x, rsi14, color='darkorange', lw=1.2, label='RSI(14)')
ax2.axhline(70, color='red',   ls='--', lw=0.8)
ax2.axhline(35, color='green', ls='--', lw=0.8)
ax2.axhline(50, color='gray',  ls=':',  lw=0.6)
ax2.fill_between(x, rsi14, 35, where=(rsi14 < 35), alpha=0.2, color='green')
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
