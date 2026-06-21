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
CSV_PATH       = "C:\\Users\\fourt\\OneDrive\\Desktop\\Finance\\Trading\\Data\\orirail.csv"
LOG_PATH       = "email_log.json"
EMAIL_SENDER   = "divyanshdewan@gmail.com"
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'osrp rtab jvyv rcvz')
EMAIL_RECEIVER = "divyanshdewan@gmail.com"

# ─────────────────────────────────────────────
# ENTRY BACKTEST STATS
# Base win rate: 54.8%  |  Test: Jun 2025 – Jun 2026
# ─────────────────────────────────────────────
ENTRY_BACKTEST = {
    "ROC_EXTREME_DROP":     {"N": 15, "W3": "100%", "W10":  "87%", "Avg10": "+23.9%", "Edge": "+31.9%"},
    "WILLR_DEEP_OVERSOLD":  {"N": 35, "W3": "100%", "W10":  "83%", "Avg10": "+15.3%", "Edge": "+28.1%"},
    "MTF_RSI_OVERSOLD":     {"N": 42, "W3": "100%", "W10":  "79%", "Avg10": "+15.5%", "Edge": "+23.8%"},
    "STOCH_WILLR_OVERSOLD": {"N": 53, "W3":  "98%", "W10":  "79%", "Avg10": "+13.2%", "Edge": "+24.5%"},
    "DOJI_AT_BOTTOM":       {"N":  4, "W3": "100%", "W10":  "75%", "Avg10": "+18.9%", "Edge": "+20.2%"},
}

# ─────────────────────────────────────────────
# EXIT SIGNAL DEFINITIONS
# ─────────────────────────────────────────────
EXIT_DESCRIPTIONS = {
    "RSI_OVERBOUGHT":      "RSI(14) > 70. Classic overbought exit — momentum exhausted.",
    "BB_UPPER_TOUCH":      "Close crosses above BB upper band. Price extended beyond 2-std envelope.",
    "MACD_HIST_TURNS_NEG": "MACD histogram flips negative from positive. Momentum reversing — earliest exit, lowest returns.",
    "WILLR_OVERBOUGHT":    "Williams %R > −10. Momentum fully recovered to overbought extreme.",
    "STOCH_OVERBOUGHT":    "Stochastic K > 85. Short-term overbought — pairs well with momentum entries.",
    "ROC5_SURGE":          "5-day ROC > +12%. Rapid price surge signals exhaustion of the bounce.",
    "RSI7_OVERBOUGHT":     "RSI(7) > 75. Short-term RSI overbought — quicker exit, lower avg return.",
    "BB_UPPER_VOL_FADE":   "Close above BB upper AND volume ratio < 0.8. Distribution: price extended on falling volume.",
}

# ─────────────────────────────────────────────
# ENTRY → EXIT PAIR STATS  (from 3yr backtest)
# Sorted best-to-worst by AvgRet within each entry
# ─────────────────────────────────────────────
PAIRS = {
    "ROC_EXTREME_DROP": [
        {"exit": "BB_UPPER_VOL_FADE",   "N": 16,  "AvgRet": "+26.1%", "WinR": "100%", "Hold": "21d", "Sharpe": "1.62"},
        {"exit": "RSI_OVERBOUGHT",       "N": 26,  "AvgRet": "+20.9%", "WinR": "100%", "Hold": "22d", "Sharpe": "3.92"},
        {"exit": "BB_UPPER_TOUCH",       "N": 28,  "AvgRet": "+20.2%", "WinR": "100%", "Hold": "19d", "Sharpe": "2.51"},
        {"exit": "WILLR_OVERBOUGHT",     "N": 36,  "AvgRet": "+11.3%", "WinR":  "94%", "Hold": "15d", "Sharpe": "1.70"},
        {"exit": "STOCH_OVERBOUGHT",     "N": 39,  "AvgRet": "+11.1%", "WinR":  "95%", "Hold": "15d", "Sharpe": "1.76"},
    ],
    "WILLR_DEEP_OVERSOLD": [
        {"exit": "BB_UPPER_VOL_FADE",   "N": 30,  "AvgRet": "+25.3%", "WinR":  "97%", "Hold": "20d", "Sharpe": "1.54"},
        {"exit": "RSI_OVERBOUGHT",       "N": 48,  "AvgRet": "+18.1%", "WinR": "100%", "Hold": "18d", "Sharpe": "3.34"},
        {"exit": "BB_UPPER_TOUCH",       "N": 54,  "AvgRet": "+17.4%", "WinR":  "96%", "Hold": "17d", "Sharpe": "2.14"},
        {"exit": "WILLR_OVERBOUGHT",     "N": 64,  "AvgRet":  "+9.9%", "WinR":  "91%", "Hold": "15d", "Sharpe": "1.37"},
        {"exit": "STOCH_OVERBOUGHT",     "N": 72,  "AvgRet":  "+9.5%", "WinR":  "92%", "Hold": "14d", "Sharpe": "1.41"},
    ],
    "MTF_RSI_OVERSOLD": [
        {"exit": "BB_UPPER_VOL_FADE",   "N": 26,  "AvgRet": "+19.2%", "WinR": "100%", "Hold": "21d", "Sharpe": "2.11"},
        {"exit": "BB_UPPER_TOUCH",       "N": 55,  "AvgRet": "+19.1%", "WinR": "100%", "Hold": "19d", "Sharpe": "2.46"},
        {"exit": "RSI_OVERBOUGHT",       "N": 51,  "AvgRet": "+18.4%", "WinR": "100%", "Hold": "19d", "Sharpe": "3.40"},
        {"exit": "WILLR_OVERBOUGHT",     "N": 63,  "AvgRet": "+10.2%", "WinR":  "89%", "Hold": "15d", "Sharpe": "1.47"},
        {"exit": "STOCH_OVERBOUGHT",     "N": 74,  "AvgRet":  "+9.8%", "WinR":  "91%", "Hold": "15d", "Sharpe": "1.54"},
    ],
    "STOCH_WILLR_OVERSOLD": [
        {"exit": "BB_UPPER_VOL_FADE",   "N": 39,  "AvgRet": "+24.1%", "WinR":  "97%", "Hold": "21d", "Sharpe": "1.60"},
        {"exit": "BB_UPPER_TOUCH",       "N": 79,  "AvgRet": "+17.9%", "WinR":  "97%", "Hold": "18d", "Sharpe": "2.25"},
        {"exit": "RSI_OVERBOUGHT",       "N": 72,  "AvgRet": "+17.2%", "WinR": "100%", "Hold": "19d", "Sharpe": "2.83"},
        {"exit": "WILLR_OVERBOUGHT",     "N": 90,  "AvgRet": "+10.0%", "WinR":  "93%", "Hold": "15d", "Sharpe": "1.48"},
        {"exit": "STOCH_OVERBOUGHT",     "N": 105, "AvgRet":  "+9.3%", "WinR":  "93%", "Hold": "14d", "Sharpe": "1.46"},
    ],
    "DOJI_AT_BOTTOM": [
        {"exit": "RSI_OVERBOUGHT",       "N":  4,  "AvgRet": "+22.0%", "WinR": "100%", "Hold": "21d", "Sharpe": "3.30"},
        {"exit": "BB_UPPER_TOUCH",       "N":  4,  "AvgRet": "+16.3%", "WinR": "100%", "Hold": "16d", "Sharpe": "5.06"},
        {"exit": "WILLR_OVERBOUGHT",     "N":  7,  "AvgRet": "+11.0%", "WinR": "100%", "Hold": "13d", "Sharpe": "2.06"},
        {"exit": "STOCH_OVERBOUGHT",     "N":  8,  "AvgRet": "+10.7%", "WinR": "100%", "Hold": "13d", "Sharpe": "2.07"},
    ],
}

SIGNAL_DESCRIPTIONS = {
    "ROC_EXTREME_DROP":     "5-day return worse than -8% with RSI < 45. Panic selling into oversold territory — 100% 3-day win rate on Orirail historically. Stocks rarely fall more than 8% in 5 days without a bounce.",
    "WILLR_DEEP_OVERSOLD":  "Williams %R < -90 with RSI < 45. 100% 3-day win rate on Orirail across 35 historical triggers. Extremely reliable oversold signal on this stock.",
    "MTF_RSI_OVERSOLD":     "RSI7<30 + RSI14<40 + RSI21<45 simultaneously. Three timeframes all confirming deep oversold — 100% 3-day win rate, +15.5% avg 10d return on Orirail.",
    "STOCH_WILLR_OVERSOLD": "Stochastic < 15 AND Williams %R < -85 AND RSI < 45. Triple momentum oversold — 98% 3-day win rate on Orirail across 53 triggers. Highest-N reliable signal on this stock.",
    "DOJI_AT_BOTTOM":       "Body < 20% of day's range near lower BB with above-average volume. Indecision at bottom with volume confirms buyers stepping in. 100% 3-day win rate on Orirail.",
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

# RSI family
rsi    = np.array(ta.momentum.RSIIndicator(df["Close"], window=14).rsi())
rsi7   = np.array(ta.momentum.RSIIndicator(df["Close"], window=7).rsi())
rsi21  = np.array(ta.momentum.RSIIndicator(df["Close"], window=21).rsi())

# Bollinger Bands
bb     = ta.volatility.BollingerBands(df["Close"], window=25, window_dev=2)
bb_up  = np.array(bb.bollinger_hband())
bb_mav = np.array(bb.bollinger_mavg())
bb_low = np.array(bb.bollinger_lband())
bb_pct = np.array(bb.bollinger_pband())

# Momentum
stoch   = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
stoch_k = np.array(stoch.stoch())
stoch_d = np.array(stoch.stoch_signal())
willr   = np.array(ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=14).williams_r())
roc5    = np.array(ta.momentum.ROCIndicator(df["Close"], window=5).roc())

# MACD
macd_i  = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
macd_h  = np.array(macd_i.macd_diff())

# Trend
ema20    = np.array(ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator())
ema50    = np.array(ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator())
vol_ma20 = np.array(df['Volume'].rolling(20).mean())
vol_r    = vol / np.where(vol_ma20 > 0, vol_ma20, np.nan)

# ─────────────────────────────────────────────
# 4. ENTRY SIGNAL CHECKS
# ─────────────────────────────────────────────
def check_roc_extreme_drop(i):
    return not np.isnan(roc5[i]) and roc5[i] < -8 and rsi[i] < 45

def check_willr_deep_oversold(i):
    return not np.isnan(willr[i]) and willr[i] < -90 and rsi[i] < 45

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
# 5. EXIT SIGNAL CHECKS
# ─────────────────────────────────────────────
def check_rsi_overbought(i):
    return not np.isnan(rsi[i]) and rsi[i] > 70

def check_bb_upper_touch(i):
    return not np.isnan(bb_up[i]) and close[i] > bb_up[i]

def check_macd_hist_turns_neg(i):
    if i < 1: return False
    return (not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1])
            and macd_h[i] < macd_h[i-1] and macd_h[i-1] > 0)

def check_willr_overbought(i):
    return not np.isnan(willr[i]) and willr[i] > -10

def check_stoch_overbought(i):
    return not np.isnan(stoch_k[i]) and stoch_k[i] > 85

def check_roc5_surge(i):
    return not np.isnan(roc5[i]) and roc5[i] > 12

def check_rsi7_overbought(i):
    return not np.isnan(rsi7[i]) and rsi7[i] > 75

def check_bb_upper_vol_fade(i):
    return (not np.isnan(bb_up[i]) and not np.isnan(vol_r[i])
            and close[i] > bb_up[i] and vol_r[i] < 0.8)

# ─────────────────────────────────────────────
# SIGNALS LISTS
# ─────────────────────────────────────────────
ENTRY_SIGNALS = [
    {"name": "ROC_EXTREME_DROP",     "check": check_roc_extreme_drop,     "color": "red"},
    {"name": "WILLR_DEEP_OVERSOLD",  "check": check_willr_deep_oversold,  "color": "darkorange"},
    {"name": "MTF_RSI_OVERSOLD",     "check": check_mtf_rsi_oversold,     "color": "purple"},
    {"name": "STOCH_WILLR_OVERSOLD", "check": check_stoch_willr_oversold, "color": "blue"},
    {"name": "DOJI_AT_BOTTOM",       "check": check_doji_at_bottom,       "color": "cyan"},
]

EXIT_SIGNALS = [
    {"name": "RSI_OVERBOUGHT",      "check": check_rsi_overbought,      "color": "red"},
    {"name": "BB_UPPER_TOUCH",      "check": check_bb_upper_touch,      "color": "green"},
    {"name": "MACD_HIST_TURNS_NEG", "check": check_macd_hist_turns_neg, "color": "gray"},
    {"name": "WILLR_OVERBOUGHT",    "check": check_willr_overbought,    "color": "darkorange"},
    {"name": "STOCH_OVERBOUGHT",    "check": check_stoch_overbought,    "color": "blue"},
    {"name": "ROC5_SURGE",          "check": check_roc5_surge,          "color": "gold"},
    {"name": "RSI7_OVERBOUGHT",     "check": check_rsi7_overbought,     "color": "pink"},
    {"name": "BB_UPPER_VOL_FADE",   "check": check_bb_upper_vol_fade,   "color": "black"},
]

# ─────────────────────────────────────────────
# 6. TODAY'S SIGNAL DETECTION
# ─────────────────────────────────────────────
i_today = len(close) - 1

triggered_entries = []
for sig in ENTRY_SIGNALS:
    if sig["check"](i_today):
        triggered_entries.append(sig)
        print(f"ENTRY triggered:  {sig['name']}")
    else:
        print(f"No entry:         {sig['name']}")

triggered_exits = []
for sig in EXIT_SIGNALS:
    if sig["check"](i_today):
        triggered_exits.append(sig)
        print(f"EXIT triggered:   {sig['name']}")
    else:
        print(f"No exit:          {sig['name']}")

# ─────────────────────────────────────────────
# 7. EMAIL
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

def build_pairs_block(sig_name):
    pairs = PAIRS.get(sig_name, [])
    if not pairs:
        return "  No pair data available.\n"
    lines = [f"  {'Exit Signal':<22} {'N':>4}  {'AvgRet':>7}  {'WinR':>5}  {'Hold':>5}  {'Sharpe':>6}"]
    lines.append("  " + "-" * 58)
    for p in pairs:
        lines.append(f"  {p['exit']:<22} {p['N']:>4}  {p['AvgRet']:>7}  {p['WinR']:>5}  {p['Hold']:>5}  {p['Sharpe']:>6}")
    return "\n".join(lines) + "\n"

def build_email_body(sig_name, d):
    desc  = SIGNAL_DESCRIPTIONS.get(sig_name, "")
    stats = ENTRY_BACKTEST.get(sig_name, {})
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
  RSI (7)    : {d['rsi7']:.2f}
  RSI (21)   : {d['rsi21']:.2f}
  Williams %R: {d['willr']:.2f}
  Stoch K    : {d['stoch_k']:.2f}
  ROC (5d)   : {d['roc5']:.2f}%
  MACD Hist  : {d['macd_h']:.4f}
  BB Low     : {d['bb_low']:.2f}
  BB Upper   : {d['bb_up']:.2f}
  Volume     : {d['vol']:,}
  Avg Volume : {d['avg_vol']:,}

ENTRY BACKTEST PERFORMANCE (Test: Jun 2025 – Jun 2026)
  Triggers (N)    : {stats.get('N',     '-')}
  Win Rate 3d     : {stats.get('W3',    '-')}
  Win Rate 10d    : {stats.get('W10',   '-')}
  Avg Return 10d  : {stats.get('Avg10', '-')}
  Edge vs Base    : {stats.get('Edge',  '-')}
  Base win rate   : 54.8%

RECOMMENDED EXIT PAIRS (3yr backtest, ranked by avg return)
{build_pairs_block(sig_name)}
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
    'company':  company_name,
    'ticker':   TICKER,
    'date':     today_str,
    'close':    close[i_today],
    'rsi':      rsi[i_today]     if not np.isnan(rsi[i_today])     else 0.0,
    'rsi7':     rsi7[i_today]    if not np.isnan(rsi7[i_today])    else 0.0,
    'rsi21':    rsi21[i_today]   if not np.isnan(rsi21[i_today])   else 0.0,
    'willr':    willr[i_today]   if not np.isnan(willr[i_today])   else 0.0,
    'stoch_k':  stoch_k[i_today] if not np.isnan(stoch_k[i_today]) else 0.0,
    'roc5':     roc5[i_today]    if not np.isnan(roc5[i_today])    else 0.0,
    'macd_h':   macd_h[i_today]  if not np.isnan(macd_h[i_today])  else 0.0,
    'bb_low':   bb_low[i_today],
    'bb_up':    bb_up[i_today],
    'vol':      volume,
    'avg_vol':  avg_volume,
}

log = load_log()
if not triggered_entries:
    print("\nNo entry signals today — no emails sent.")
else:
    print(f"\n{len(triggered_entries)} entry signal(s) triggered. Sending emails...")
    for sig in triggered_entries:
        if log.get(sig['name']) == today_str:
            print(f"Already emailed {sig['name']} today — skipping.")
            continue
        if send_email(f"[ORIRAIL] {sig['name']} — {today_str}",
                      build_email_body(sig['name'], signal_data)):
            log[sig['name']] = today_str
            save_log(log)

# ─────────────────────────────────────────────
# 8. PLOT
# ─────────────────────────────────────────────
n = len(close)
x = np.arange(n)

entry_arrays = {sig["name"]: np.array([sig["check"](i) for i in range(n)]) for sig in ENTRY_SIGNALS}
exit_arrays  = {sig["name"]: np.array([sig["check"](i) for i in range(n)]) for sig in EXIT_SIGNALS}

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1, 1]},
                                     sharex=True)
fig.suptitle(f"{company_name} ({TICKER})  --  {today_str}", fontsize=13, fontweight='bold')

# Price + Bollinger + EMAs
ax1.plot(x, close,  color='black',     lw=1.5, label='Close')
ax1.plot(x, bb_up,  color='green',     lw=0.8, ls='--', label='BB Upper')
ax1.plot(x, bb_low, color='red',       lw=0.8, ls='--', label='BB Lower')
ax1.plot(x, bb_mav, color='steelblue', lw=0.8, ls='--', label='BB Mid')
ax1.plot(x, ema20,  color='orange',    lw=0.8, label='EMA20')
ax1.plot(x, ema50,  color='purple',    lw=0.8, label='EMA50')
ax1.fill_between(x, bb_low, bb_up, alpha=0.05, color='steelblue')

# Entry markers (▲ below price)
for sig in ENTRY_SIGNALS:
    arr = entry_arrays[sig["name"]]
    ax1.scatter(x, np.where(arr, close - close * 0.015, np.nan),
                color=sig["color"], marker='^', s=55, zorder=5, label=f"▲ {sig['name']}")

# Exit markers (▼ above price) — top 3 exits only to keep chart readable
top_exits = ["RSI_OVERBOUGHT", "BB_UPPER_TOUCH", "BB_UPPER_VOL_FADE"]
for sig in EXIT_SIGNALS:
    if sig["name"] not in top_exits:
        continue
    arr = exit_arrays[sig["name"]]
    ax1.scatter(x, np.where(arr, close + close * 0.015, np.nan),
                color=sig["color"], marker='v', s=45, zorder=5,
                alpha=0.7, label=f"▼ {sig['name']}")

ax1.set_ylabel("Price (Rs)")
ax1.legend(loc='upper left', fontsize=6, ncol=3)
ax1.grid(alpha=0.3)

# RSI panel
ax2.plot(x, rsi,  color='darkorange', lw=1.2, label='RSI(14)')
ax2.plot(x, rsi7, color='steelblue',  lw=0.8, ls='--', label='RSI(7)', alpha=0.7)
ax2.axhline(70, color='red',   ls='--', lw=0.8)
ax2.axhline(35, color='green', ls='--', lw=0.8)
ax2.axhline(50, color='gray',  ls=':',  lw=0.6)
ax2.fill_between(x, rsi, 35, where=(rsi < 35), alpha=0.2, color='green')
ax2.fill_between(x, rsi, 70, where=(rsi > 70), alpha=0.2, color='red')
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI")
ax2.legend(fontsize=7)
ax2.grid(alpha=0.3)

# Volume panel
vol_colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
ax3.bar(x, vol, color=vol_colors, alpha=0.7, width=0.8)
ax3.plot(x, vol_ma20, color='blue', lw=1, label='Vol MA20')
ax3.set_ylabel("Volume")
ax3.set_xlabel("Bar index")
ax3.legend(fontsize=7)
ax3.grid(alpha=0.3)

plt.tight_layout()
#plt.show()