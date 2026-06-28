import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOOKBACK = 120   # bars of price history to display
import yfinance as yf

# ─────────────────────────────────────────────
# SIGNAL LIBRARY — E1-E8 entries / X1-X8 exits
# Same definitions used for every stock
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
    "E1_ROC5_RSI":      e1_roc5_rsi,
    "E2_WILLR_RSI":     e2_willr_rsi,
    "E3_MTF_RSI":       e3_mtf_rsi,
    "E4_STOCH_WILLR":   e4_stoch_willr,
    "E5_DOJI_BBLOW":    e5_doji_bblow,
    "E6_MACD_TURN_RSI": e6_macd_turn_rsi,
    "E7_BBLOW_RSI":     e7_bblow_rsi,
    "E8_EMA50_BOUNCE":  e8_ema50_bounce,
}

EXIT_FUNCS = {
    "X1_RSI_OVERBOUGHT": x1_rsi_overbought,
    "X2_BB_UPPER":       x2_bb_upper,
    "X3_MACD_TURN_NEG":  x3_macd_turn_neg,
    "X4_WILLR_HIGH":     x4_willr_high,
    "X5_STOCH_HIGH":     x5_stoch_high,
    "X6_ROC5_SURGE":     x6_roc5_surge,
    "X7_RSI7_HIGH":      x7_rsi7_high,
    "X8_BBUP_VOLFADE":   x8_bbup_volfade,
}

# Distinct marker colors so entries/exits are visually separable on a busy chart
ENTRY_COLORS = {
    "E1_ROC5_RSI":      "#1f77b4",
    "E2_WILLR_RSI":     "#2ca02c",
    "E3_MTF_RSI":       "#9467bd",
    "E4_STOCH_WILLR":   "#8c564b",
    "E5_DOJI_BBLOW":    "#e377c2",
    "E6_MACD_TURN_RSI": "#17becf",
    "E7_BBLOW_RSI":     "#bcbd22",
    "E8_EMA50_BOUNCE":  "#1a55ff",
}
EXIT_COLORS = {
    "X1_RSI_OVERBOUGHT": "#d62728",
    "X2_BB_UPPER":       "#ff7f0e",
    "X3_MACD_TURN_NEG":  "#7f0000",
    "X4_WILLR_HIGH":     "#ff1493",
    "X5_STOCH_HIGH":     "#b22222",
    "X6_ROC5_SURGE":     "#ff4500",
    "X7_RSI7_HIGH":      "#cc6600",
    "X8_BBUP_VOLFADE":   "#990000",
}

# ─────────────────────────────────────────────
# STOCK UNIVERSE
# ─────────────────────────────────────────────
STOCKS = {
    "BSE":        {"ticker": "BSE.NS",         "csv_path": "Data/bse.csv"},
    "EDELWEISS":  {"ticker": "EDELWEISS.BO",   "csv_path": "Data/edelweiss.csv"},
    "EICHER":     {"ticker": "EICHERMOT.BO",   "csv_path": "Data/eicher.csv"},
    "HINDCOPPER": {"ticker": "HINDCOPPER.NS",  "csv_path": "Data/hindcopper.csv"},
    "HINDZINC":   {"ticker": "HINDZINC.BO",    "csv_path": "Data/hindzinc.csv"},
    "INDOTHAI":   {"ticker": "INDOTHAI.NS",    "csv_path": "Data/indothai.csv"},
    "ORIRAIL":    {"ticker": "ORIRAIL.BO",     "csv_path": "Data/orirail.csv"},
    "PARAS":      {"ticker": "PARAS.NS",       "csv_path": "Data/paras.csv"},
    "DOLAT":      {"ticker": "DOLAT.BO",       "csv_path": "Data/dolat.csv"},
    "TITAN":      {"ticker": "TITAN.NS",       "csv_path": "Data/titan.csv"},
    "TRENT":      {"ticker": "TRENT.NS",       "csv_path": "Data/trent.csv"},
    "SBIN":       {"ticker": "SBIN.NS",        "csv_path": "Data/sbin.csv"},
    "CDSL":       {"ticker": "CDSL.NS",        "csv_path": "Data/cdsl.csv"},
    "RECLTD":     {"ticker": "RECLTD.NS",      "csv_path": "Data/recltd.csv"},
    "TITAGARH":   {"ticker": "TITAGARH.NS",    "csv_path": "Data/titagarh.csv"},
}

# ─────────────────────────────────────────────
# 1. FETCH + APPEND TODAY'S BAR FOR ONE STOCK
# ─────────────────────────────────────────────
def fetch_and_update_csv(ticker, csv_path):
    stock = yf.Ticker(ticker)
    info  = stock.info

    company_name = info.get('longName') or info.get('shortName') or ticker
    avg_volume   = info.get('averageDailyVolume10Day') or 0
    volume       = info.get('volume') or 0

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
        meta = {'company': company_name, 'ticker': ticker, 'date': today_str,
                'volume': volume, 'avg_volume': avg_volume}
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

    meta = {'company': company_name, 'ticker': ticker, 'date': today_str,
            'volume': volume, 'avg_volume': avg_volume}
    return df, meta

# ─────────────────────────────────────────────
# 2. COMPUTE INDICATORS FOR ONE STOCK
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
    bb_mav = np.array(bb.bollinger_mavg())
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
        'bb_up': bb_up, 'bb_mav': bb_mav, 'bb_low': bb_low,
        'stoch': stoch, 'willr': willr, 'macd_h': macd_h,
        'ema50': ema50, 'vol_ma20': vol_ma20, 'roc5': roc5,
    }

# ─────────────────────────────────────────────
# 3. PLOT — price+BB subplot (with signal markers), RSI subplot
# Signals are computed over the FULL history (so MACD/EMA warm-up etc.
# is correct) but only markers within the last `lookback` bars are drawn.
# ─────────────────────────────────────────────
def build_plot(ind, company_name, ticker, date_label, lookback=LOOKBACK):
    n = len(ind['close'])
    start = max(0, n - lookback)
    x = np.arange(start, n)

    # Find every entry/exit occurrence within the visible window
    entry_hits = {name: [] for name in ENTRY_FUNCS}
    exit_hits  = {name: [] for name in EXIT_FUNCS}
    for i in range(start, n):
        for name, fn in ENTRY_FUNCS.items():
            if fn(ind, i):
                entry_hits[name].append(i)
        for name, fn in EXIT_FUNCS.items():
            if fn(ind, i):
                exit_hits[name].append(i)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={'height_ratios': [2, 1]},
                                    sharex=True)
    fig.suptitle(f"{company_name} ({ticker}) - {date_label}", fontsize=12, fontweight='bold')

    close = ind['close']
    ax1.plot(x, close[start:n], color='black', lw=1.3, label='Close', zorder=3)
    ax1.plot(x, ind['bb_up'][start:n], color='green', lw=0.8, ls='--', label='BB Upper')
    ax1.plot(x, ind['bb_low'][start:n], color='red', lw=0.8, ls='--', label='BB Lower')
    ax1.plot(x, ind['bb_mav'][start:n], color='steelblue', lw=0.8, ls='--', label='BB Mid')
    ax1.fill_between(x, ind['bb_low'][start:n], ind['bb_up'][start:n], alpha=0.05, color='steelblue')

    # Entry markers: triangle-up below price
    for name, idxs in entry_hits.items():
        if not idxs:
            continue
        ys = [close[i] * 0.985 for i in idxs]
        ax1.scatter(idxs, ys, marker='^', s=70, color=ENTRY_COLORS[name],
                    label=f"Entry {name}", zorder=5, edgecolors='black', linewidths=0.4)

    # Exit markers: triangle-down above price
    for name, idxs in exit_hits.items():
        if not idxs:
            continue
        ys = [close[i] * 1.015 for i in idxs]
        ax1.scatter(idxs, ys, marker='v', s=70, color=EXIT_COLORS[name],
                    label=f"Exit {name}", zorder=5, edgecolors='black', linewidths=0.4)

    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left', fontsize=6, ncol=3)
    ax1.grid(alpha=0.3)

    ax2.plot(x, ind['rsi14'][start:n], color='darkorange', lw=1.2, label='RSI(14)')
    ax2.axhline(70, color='red', ls='--', lw=0.8)
    ax2.axhline(35, color='green', ls='--', lw=0.8)
    ax2.axhline(50, color='gray', ls=':', lw=0.6)
    ax2.fill_between(x, ind['rsi14'][start:n], 35,
                      where=(ind['rsi14'][start:n] < 35), alpha=0.2, color='green')
    ax2.fill_between(x, ind['rsi14'][start:n], 70,
                      where=(ind['rsi14'][start:n] > 70), alpha=0.2, color='red')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Bar index")
    ax2.legend(loc='upper left', fontsize=7)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# 4. MAIN LOOP — one plot per stock, shown interactively
# ─────────────────────────────────────────────
def main():
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

        ind = compute_indicators(df)
        i_today = len(ind['close']) - 1

        fired_entries = [name for name, fn in ENTRY_FUNCS.items() if fn(ind, i_today)]
        fired_exits   = [name for name, fn in EXIT_FUNCS.items()  if fn(ind, i_today)]

        for name in ENTRY_FUNCS:
            print(f"  ENTRY {name}: {'TRIGGERED' if name in fired_entries else 'no'}")
        for name in EXIT_FUNCS:
            print(f"  EXIT  {name}: {'TRIGGERED' if name in fired_exits else 'no'}")

        fig = build_plot(ind, meta['company'], meta['ticker'], meta['date'])
        plt.show()

if __name__ == "__main__":
    main()