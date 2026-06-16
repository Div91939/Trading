<<<<<<< HEAD
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
TICKER         = "EDELWEISS.BO"
CSV_PATH = "C:\\Users\\fourt\\OneDrive\\Desktop\\Finance\\Trading\\Data\\edelweiss.csv"
LOG_PATH = "email_log.json"
EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'osrp rtab jvyv rcvz')
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# ─────────────────────────────────────────────
# BACKTEST STATS  |  Base win rate: 36.7%
# ─────────────────────────────────────────────
BACKTEST = {
    "ADX_TREND_EXHAUSTION": {"N":  5, "W3": "100%", "W10":  "80%", "Avg10": "+10.5%", "Edge": "+43.3%"},
    "MTF_RSI_OVERSOLD":     {"N": 19, "W3":  "89%", "W10":  "74%", "Avg10":  "+8.8%", "Edge": "+37.0%"},
    "ATR_SPIKE_OVERSOLD":   {"N": 26, "W3":  "92%", "W10":  "73%", "Avg10":  "+7.7%", "Edge": "+36.4%"},
    "CCI_RSI_OVERSOLD":     {"N": 21, "W3":  "90%", "W10":  "71%", "Avg10":  "+7.4%", "Edge": "+34.7%"},
    "WILLR_DEEP_OVERSOLD":  {"N": 21, "W3":  "86%", "W10":  "71%", "Avg10":  "+6.2%", "Edge": "+34.7%"},
}

SIGNAL_DESCRIPTIONS = {
    "ADX_TREND_EXHAUSTION": "ADX was above 30 (confirmed downtrend) but is now falling, while RSI<45 and -DI dominant. The downtrend is losing force before reversing. 100% 3-day win rate, +10.5% avg 10d return on Edelweiss.",
    "MTF_RSI_OVERSOLD":     "RSI7<30 + RSI14<40 + RSI21<45 simultaneously. Three timeframes all agreeing the stock is deeply oversold — a high-conviction multi-timeframe exhaustion signal. 89% 3-day win rate on Edelweiss.",
    "ATR_SPIKE_OVERSOLD":   "Volatility spike (ATR>3% of price) at oversold RSI inside lower BB. The sell-off burned itself out — 92% 3-day win rate on Edelweiss.",
    "CCI_RSI_OVERSOLD":     "CCI < -100 AND RSI < 40. Double oversold confirmation across two independent indicator families. 90% 3-day win rate on Edelweiss.",
    "WILLR_DEEP_OVERSOLD":  "Williams %R < -90 with RSI < 45. Momentum deeply oversold on short-term oscillator with RSI confirmation. 86% 3-day win rate on Edelweiss.",
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
rsi7   = np.array(ta.momentum.RSIIndicator(df["Close"], window=7).rsi())
rsi21  = np.array(ta.momentum.RSIIndicator(df["Close"], window=21).rsi())

bb     = ta.volatility.BollingerBands(df["Close"], window=25, window_dev=2)
bb_up  = np.array(bb.bollinger_hband())
bb_mav = np.array(bb.bollinger_mavg())
bb_low = np.array(bb.bollinger_lband())
bb_pct = np.array(bb.bollinger_pband())

atr     = np.array(ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range())
atr_pct = np.where(close > 0, atr / close * 100, np.nan)

cci = np.array(ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=20).cci())

adx_obj = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
adx     = np.array(adx_obj.adx())
adx_neg = np.array(adx_obj.adx_neg())
adx_pos = np.array(adx_obj.adx_pos())

willr = np.array(ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=14).williams_r())

ema20  = np.array(ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator())
ema50  = np.array(ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator())

vol_ma20 = np.array(df['Volume'].rolling(20).mean())

# ─────────────────────────────────────────────
# 4. SIGNAL CHECKS
# ─────────────────────────────────────────────
def check_adx_trend_exhaustion(i):
    if i < 1: return False
    return (not np.isnan(adx[i]) and not np.isnan(adx[i-1])
            and adx[i-1] > 30 and adx[i] < adx[i-1]
            and rsi[i] < 45
            and not np.isnan(adx_neg[i]) and not np.isnan(adx_pos[i])
            and adx_neg[i] > adx_pos[i])

def check_mtf_rsi_oversold(i):
    return (not np.isnan(rsi7[i]) and not np.isnan(rsi[i]) and not np.isnan(rsi21[i])
            and rsi7[i] < 30 and rsi[i] < 40 and rsi21[i] < 45)

def check_atr_spike_oversold(i):
    return (not np.isnan(atr_pct[i]) and not np.isnan(bb_pct[i])
            and atr_pct[i] > 3 and rsi[i] < 40 and bb_pct[i] < 0.25)

def check_cci_rsi_oversold(i):
    return (not np.isnan(cci[i]) and cci[i] < -100 and rsi[i] < 40)

def check_willr_deep_oversold(i):
    return (not np.isnan(willr[i]) and willr[i] < -90 and rsi[i] < 45)

# ─────────────────────────────────────────────
# SIGNALS LIST  (top 5 by backtest score)
# ─────────────────────────────────────────────
SIGNALS = [
    {"name": "ADX_TREND_EXHAUSTION", "check": check_adx_trend_exhaustion, "color": "red"},
    {"name": "MTF_RSI_OVERSOLD",     "check": check_mtf_rsi_oversold,     "color": "darkorange"},
    {"name": "ATR_SPIKE_OVERSOLD",   "check": check_atr_spike_oversold,   "color": "purple"},
    {"name": "CCI_RSI_OVERSOLD",     "check": check_cci_rsi_oversold,     "color": "blue"},
    {"name": "WILLR_DEEP_OVERSOLD",  "check": check_willr_deep_oversold,  "color": "cyan"},
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
  (Base win rate for this stock: 36.7%)

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
        subject = f"[EDELWEISS] {sig['name']} — {today_str}"
        body    = build_email_body(sig['name'], signal_data)
        if send_email(subject, body):
            log[f"EDELWEISS_{sig['name']}"] = today_str
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
=======
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
TICKER         = "EDELWEISS.BO"
CSV_PATH = "Data/edelweiss.csv"
LOG_PATH = "email_log.json"
EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'osrp rtab jvyv rcvz')
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# ─────────────────────────────────────────────
# BACKTEST STATS  |  Base win rate: 36.7%
# ─────────────────────────────────────────────
BACKTEST = {
    "ADX_TREND_EXHAUSTION": {"N":  5, "W3": "100%", "W10":  "80%", "Avg10": "+10.5%", "Edge": "+43.3%"},
    "MTF_RSI_OVERSOLD":     {"N": 19, "W3":  "89%", "W10":  "74%", "Avg10":  "+8.8%", "Edge": "+37.0%"},
    "ATR_SPIKE_OVERSOLD":   {"N": 26, "W3":  "92%", "W10":  "73%", "Avg10":  "+7.7%", "Edge": "+36.4%"},
    "CCI_RSI_OVERSOLD":     {"N": 21, "W3":  "90%", "W10":  "71%", "Avg10":  "+7.4%", "Edge": "+34.7%"},
    "WILLR_DEEP_OVERSOLD":  {"N": 21, "W3":  "86%", "W10":  "71%", "Avg10":  "+6.2%", "Edge": "+34.7%"},
}

SIGNAL_DESCRIPTIONS = {
    "ADX_TREND_EXHAUSTION": "ADX was above 30 (confirmed downtrend) but is now falling, while RSI<45 and -DI dominant. The downtrend is losing force before reversing. 100% 3-day win rate, +10.5% avg 10d return on Edelweiss.",
    "MTF_RSI_OVERSOLD":     "RSI7<30 + RSI14<40 + RSI21<45 simultaneously. Three timeframes all agreeing the stock is deeply oversold — a high-conviction multi-timeframe exhaustion signal. 89% 3-day win rate on Edelweiss.",
    "ATR_SPIKE_OVERSOLD":   "Volatility spike (ATR>3% of price) at oversold RSI inside lower BB. The sell-off burned itself out — 92% 3-day win rate on Edelweiss.",
    "CCI_RSI_OVERSOLD":     "CCI < -100 AND RSI < 40. Double oversold confirmation across two independent indicator families. 90% 3-day win rate on Edelweiss.",
    "WILLR_DEEP_OVERSOLD":  "Williams %R < -90 with RSI < 45. Momentum deeply oversold on short-term oscillator with RSI confirmation. 86% 3-day win rate on Edelweiss.",
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
rsi7   = np.array(ta.momentum.RSIIndicator(df["Close"], window=7).rsi())
rsi21  = np.array(ta.momentum.RSIIndicator(df["Close"], window=21).rsi())

bb     = ta.volatility.BollingerBands(df["Close"], window=25, window_dev=2)
bb_up  = np.array(bb.bollinger_hband())
bb_mav = np.array(bb.bollinger_mavg())
bb_low = np.array(bb.bollinger_lband())
bb_pct = np.array(bb.bollinger_pband())

atr     = np.array(ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range())
atr_pct = np.where(close > 0, atr / close * 100, np.nan)

cci = np.array(ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=20).cci())

adx_obj = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
adx     = np.array(adx_obj.adx())
adx_neg = np.array(adx_obj.adx_neg())
adx_pos = np.array(adx_obj.adx_pos())

willr = np.array(ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=14).williams_r())

ema20  = np.array(ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator())
ema50  = np.array(ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator())

vol_ma20 = np.array(df['Volume'].rolling(20).mean())

# ─────────────────────────────────────────────
# 4. SIGNAL CHECKS
# ─────────────────────────────────────────────
def check_adx_trend_exhaustion(i):
    if i < 1: return False
    return (not np.isnan(adx[i]) and not np.isnan(adx[i-1])
            and adx[i-1] > 30 and adx[i] < adx[i-1]
            and rsi[i] < 45
            and not np.isnan(adx_neg[i]) and not np.isnan(adx_pos[i])
            and adx_neg[i] > adx_pos[i])

def check_mtf_rsi_oversold(i):
    return (not np.isnan(rsi7[i]) and not np.isnan(rsi[i]) and not np.isnan(rsi21[i])
            and rsi7[i] < 30 and rsi[i] < 40 and rsi21[i] < 45)

def check_atr_spike_oversold(i):
    return (not np.isnan(atr_pct[i]) and not np.isnan(bb_pct[i])
            and atr_pct[i] > 3 and rsi[i] < 40 and bb_pct[i] < 0.25)

def check_cci_rsi_oversold(i):
    return (not np.isnan(cci[i]) and cci[i] < -100 and rsi[i] < 40)

def check_willr_deep_oversold(i):
    return (not np.isnan(willr[i]) and willr[i] < -90 and rsi[i] < 45)

# ─────────────────────────────────────────────
# SIGNALS LIST  (top 5 by backtest score)
# ─────────────────────────────────────────────
SIGNALS = [
    {"name": "ADX_TREND_EXHAUSTION", "check": check_adx_trend_exhaustion, "color": "red"},
    {"name": "MTF_RSI_OVERSOLD",     "check": check_mtf_rsi_oversold,     "color": "darkorange"},
    {"name": "ATR_SPIKE_OVERSOLD",   "check": check_atr_spike_oversold,   "color": "purple"},
    {"name": "CCI_RSI_OVERSOLD",     "check": check_cci_rsi_oversold,     "color": "blue"},
    {"name": "WILLR_DEEP_OVERSOLD",  "check": check_willr_deep_oversold,  "color": "cyan"},
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
  (Base win rate for this stock: 36.7%)

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
        subject = f"[EDELWEISS] {sig['name']} — {today_str}"
        body    = build_email_body(sig['name'], signal_data)
        if send_email(subject, body):
            log[f"EDELWEISS_{sig['name']}"] = today_str
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
>>>>>>> 336b0b05b4c127b872c16ac6802f9136a8a7a32f
