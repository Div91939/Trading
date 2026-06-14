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
TICKER         = "HINDCOPPER.BO"
CSV_PATH       = "Data/hindcopper.csv"
LOG_PATH       = "email_log.json"
EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'osrp rtab jvyv rcvz')
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# ─────────────────────────────────────────────
# BACKTEST STATS (Test period: Jun 2025 - Jun 2026)
# Base: Win rate 55.4%, GBM mean +2.92%, Std 12.13%
# ─────────────────────────────────────────────
BACKTEST = {
    "RSI_BULL_DIV3":    {"N":  8, "W10": "88%", "Mean": "+8.4%",  "Sharpe": "1.05", "Edge": "+32.6", "p": "0.175"},
    "FOUR_DAY_REV":     {"N":  5, "W10": "80%", "Mean": "+8.4%",  "Sharpe": "0.64", "Edge": "+24.6", "p": "0.498"},
    "RSI14_SUB35":      {"N":  8, "W10": "88%", "Mean": "+7.6%",  "Sharpe": "0.93", "Edge": "+32.6", "p": "0.034"},
    "BB_BELOW_LOWER":   {"N":  9, "W10": "67%", "Mean": "+7.2%",  "Sharpe": "0.84", "Edge": "+11.6", "p": "0.210"},
    "CCI_MFI":          {"N":  2, "W10":"100%", "Mean": "+7.0%",  "Sharpe": "—",    "Edge": "+44.6", "p": "—"   },
    "MACD_TURN_EMAS":   {"N": 18, "W10": "67%", "Mean": "+5.7%",  "Sharpe": "0.67", "Edge": "+11.6", "p": "0.447"},
    "STOCH_SUB10":      {"N":  6, "W10": "83%", "Mean": "+5.2%",  "Sharpe": "0.77", "Edge": "+27.6", "p": "0.697"},
    "MTF_RSI":          {"N": 12, "W10": "75%", "Mean": "+4.5%",  "Sharpe": "0.54", "Edge": "+19.6", "p": "0.087"},
}

SIGNAL_DESCRIPTIONS = {
    "RSI_BULL_DIV3":  "Price lower than 3 days ago but RSI is higher — sellers losing momentum before price shows it. Best signal on HindCopper — 88% win rate, +8.4% mean return in test period.",
    "FOUR_DAY_REV":   "4 consecutive down closes followed by a green candle today. Extended seller exhaustion — 80% win rate, +8.4% mean return on HindCopper in testing.",
    "RSI14_SUB35":    "RSI(14) below 35. Classic oversold — 88% win rate, +7.6% mean, statistically significant (p=0.034) on HindCopper.",
    "BB_BELOW_LOWER": "Price below the lower Bollinger Band with RSI < 45. Trading outside 2 standard deviations — statistically extreme, tends to snap back. 67% win rate, +7.2% mean.",
    "CCI_MFI":        "CCI below -100 AND MFI below 30 simultaneously. Oversold on both price deviation and volume flow. Very rare (2 test triggers) — treat with caution but historically perfect on this stock.",
    "MACD_TURN_EMAS": "MACD histogram turning up from negative territory while price is below EMA20 and EMA50. Momentum shifting while price still looks weak. 67% win rate, +5.7% mean, highest N (18) — most reliable trigger count.",
    "STOCH_SUB10":    "Stochastic oscillator below 10 with RSI below 35. Deeply oversold on both momentum measures — 83% win rate, +5.2% mean return.",
    "MTF_RSI":        "RSI7 < 30 AND RSI14 < 40 AND RSI21 < 45 simultaneously. Three timeframes all confirming deep oversold — 75% win rate, +4.5% mean.",
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
close = np.array(df['Close'])
high  = np.array(df['High'])
low   = np.array(df['Low'])
open_ = np.array(df['Open'])
vol   = np.array(df['Volume'], dtype=float)

rsi14 = np.array(ta.momentum.RSIIndicator(df["Close"], window=14).rsi())
rsi7  = np.array(ta.momentum.RSIIndicator(df["Close"], window=7).rsi())
rsi21 = np.array(ta.momentum.RSIIndicator(df["Close"], window=21).rsi())

bb      = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
bb_up   = np.array(bb.bollinger_hband())
bb_mav  = np.array(bb.bollinger_mavg())
bb_low  = np.array(bb.bollinger_lband())
bb_pct  = np.array(bb.bollinger_pband())

macd_i  = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
macd_h  = np.array(macd_i.macd_diff())

ema20   = np.array(ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator())
ema50   = np.array(ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator())

cci     = np.array(ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=20).cci())
mfi     = np.array(ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=14).money_flow_index())
stk     = np.array(ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3).stoch())

vol_ma20 = np.array(df['Volume'].rolling(20).mean())

# ─────────────────────────────────────────────
# 4. SIGNAL CHECKS
# ─────────────────────────────────────────────
def check_rsi_bull_div3(i):
    if i < 3: return False
    return (not np.isnan(rsi14[i]) and not np.isnan(rsi14[i-3])
            and close[i] < close[i-3]
            and rsi14[i] > rsi14[i-3]
            and rsi14[i] < 45)

def check_four_day_rev(i):
    if i < 4: return False
    four_down   = all(close[i-k] < close[i-k-1] for k in range(1, 5))
    green_today = close[i] > open_[i]
    return four_down and green_today

def check_rsi14_sub35(i):
    return not np.isnan(rsi14[i]) and rsi14[i] < 35

def check_bb_below_lower(i):
    return (not np.isnan(bb_pct[i])
            and bb_pct[i] < 0
            and rsi14[i] < 45)

def check_cci_mfi(i):
    return (not np.isnan(cci[i]) and not np.isnan(mfi[i])
            and cci[i] < -100 and mfi[i] < 30)

def check_macd_turn_emas(i):
    if i < 1: return False
    return (not np.isnan(ema20[i]) and not np.isnan(ema50[i])
            and not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1])
            and close[i] < ema20[i] and close[i] < ema50[i]
            and macd_h[i] > macd_h[i-1] and macd_h[i-1] < 0)

def check_stoch_sub10(i):
    return (not np.isnan(stk[i])
            and stk[i] < 10
            and rsi14[i] < 35)

def check_mtf_rsi(i):
    return (not np.isnan(rsi7[i]) and not np.isnan(rsi21[i])
            and rsi7[i] < 30 and rsi14[i] < 40 and rsi21[i] < 45)

# ─────────────────────────────────────────────
# SIGNALS LIST — ranked by test-period mean return
# ─────────────────────────────────────────────
SIGNALS = [
    {"name": "RSI_BULL_DIV3",  "check": check_rsi_bull_div3,  "color": "red"},
    {"name": "FOUR_DAY_REV",   "check": check_four_day_rev,   "color": "darkorange"},
    {"name": "RSI14_SUB35",    "check": check_rsi14_sub35,    "color": "blue"},
    {"name": "BB_BELOW_LOWER", "check": check_bb_below_lower, "color": "purple"},
    {"name": "CCI_MFI",        "check": check_cci_mfi,        "color": "cyan"},
    {"name": "MACD_TURN_EMAS", "check": check_macd_turn_emas, "color": "green"},
    {"name": "STOCH_SUB10",    "check": check_stoch_sub10,    "color": "magenta"},
    {"name": "MTF_RSI",        "check": check_mtf_rsi,        "color": "brown"},
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
  Win Rate 10d    : {stats.get('W10',    '-')}
  Mean 10d return : {stats.get('Mean',   '-')}
  Sharpe          : {stats.get('Sharpe', '-')}
  Edge vs base    : {stats.get('Edge',   '-')}
  p-value         : {stats.get('p',      '-')}
  Base win rate   : 55.4%  |  Base mean: +2.92%

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
    'close':   close[i_today],  'rsi': rsi14[i_today],
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
        if log.get(f"HINDCOPPER_{sig['name']}") == today_str:
            print(f"Already emailed {sig['name']} today — skipping.")
            continue
        if send_email(f"[HINDCOPPER] {sig['name']} — {today_str}",
                      build_email_body(sig['name'], signal_data)):
            log[f"HINDCOPPER_{sig['name']}"] = today_str
            save_log(log)

# ─────────────────────────────────────────────
# 7. PLOT — 3 clean subplots
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
