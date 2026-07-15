import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import os
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit only this section
# ═══════════════════════════════════════════════════════════════════════════════

LOOKBACK = 320   # number of recent bars to display on the chart

STOCKS = {

    # ── Template — copy/paste and fill in for each stock ──────────────────────
    #
    # "LABEL": {
    #     "ticker":      "TICKER.NS",          # yfinance ticker
    #     "csv_path":    "Data/label.csv",      # local CSV path
    #
    #     # ── Bollinger Band settings ──
    #     "bb_window":   20,                    # rolling window for BB
    #     "bb_std":      2.0,                   # standard deviations
    #
    #     # ── RSI settings ──
    #     "rsi_window":  14,                    # RSI period
    #
    #     # ── Entry signal thresholds ──
    #     "rsi_entry":          30,             # RSI < this  → entry
    #     "bb_entry_mult":      1.0,            # close < bb_lower * this → entry
    #                                           #   1.0  = exactly at band
    #                                           #   1.02 = up to 2% above band
    #
    #     # ── Exit signal thresholds ──
    #     "rsi_exit":           70,             # RSI > this  → exit
    #     "bb_exit_mult":       1.0,            # close > bb_upper * this → exit
    # },

    "SARTHAK": {
        "ticker":         "SARTHAKGL.BO",
        "csv_path":       "C:\\Users\\Divyansh\\OneDrive\\Desktop\\IISER\\Trading_git\\Training_Testing\\sarthakgl_3y.csv",
        "bb_window":      20,
        "bb_std":         2.0,
        "rsi_window":     14,
        "rsi_entry":      35,
        "bb_entry_mult":  1.0,
        "rsi_exit":       70,
        "bb_exit_mult":   1.0,
    },
    "SRMENERGY": {
        "ticker":         "SRMENERGY.BO",
        "csv_path":       "C:\\Users\\Divyansh\\OneDrive\\Desktop\\srmenergy.csv",
        "bb_window":      20,
        "bb_std":         2.0,
        "rsi_window":     14,
        "rsi_entry":      35,
        "bb_entry_mult":  1.0,
        "rsi_exit":       70,
        "bb_exit_mult":   1.0,
    },

    # ── Add more stocks below ──────────────────────────────────────────────────

    # "BSE": {
    #     "ticker":         "BSE.NS",
    #     "csv_path":       "Data/bse.csv",
    #     "bb_window":      25,
    #     "bb_std":         2.0,
    #     "rsi_window":     14,
    #     "rsi_entry":      35,
    #     "bb_entry_mult":  1.02,    # close < bb_lower * 1.02  (slightly above band)
    #     "rsi_exit":       65,
    #     "bb_exit_mult":   0.98,    # close > bb_upper * 0.98  (slightly below band)
    # },

}

# ═══════════════════════════════════════════════════════════════════════════════
#  END OF CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
# SIGNAL DEFINITIONS (RSI + BB only)
# ─────────────────────────────────────────────

def entry_rsi(ind, i, cfg):
    """RSI drops below rsi_entry threshold."""
    v = ind['rsi'][i]
    return not np.isnan(v) and v < cfg['rsi_entry']

def entry_bb(ind, i, cfg):
    """Close falls below bb_lower * bb_entry_mult."""
    bl = ind['bb_low'][i]
    return not np.isnan(bl) and ind['close'][i] < bl * cfg['bb_entry_mult']

def exit_rsi(ind, i, cfg):
    """RSI rises above rsi_exit threshold."""
    v = ind['rsi'][i]
    return not np.isnan(v) and v > cfg['rsi_exit']

def exit_bb(ind, i, cfg):
    """Close rises above bb_upper * bb_exit_mult."""
    bu = ind['bb_up'][i]
    return not np.isnan(bu) and ind['close'][i] > bu * cfg['bb_exit_mult']


ENTRY_FUNCS = {
    "E_RSI": entry_rsi,
    "E_BB":  entry_bb,
}

EXIT_FUNCS = {
    "X_RSI": exit_rsi,
    "X_BB":  exit_bb,
}

ENTRY_COLORS = {
    "E_RSI": "#2196F3",   # blue
    "E_BB":  "#4CAF50",   # green
}

EXIT_COLORS = {
    "X_RSI": "#F44336",   # red
    "X_BB":  "#FF9800",   # orange
}


# ─────────────────────────────────────────────
# FETCH + UPDATE CSV
# ─────────────────────────────────────────────

def fetch_and_update_csv(ticker, csv_path):
    stock = yf.Ticker(ticker)
    info  = stock.info

    company_name = info.get('longName') or info.get('shortName') or ticker
    avg_volume   = info.get('averageDailyVolume10Day') or 0
    volume       = info.get('volume') or 0

    hist_today = stock.history(period="1d", interval="1d")
    if hist_today.empty:
        return None, None

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
    else:
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
        'company':    company_name,
        'ticker':     ticker,
        'date':       today_str,
        'volume':     volume,
        'avg_volume': avg_volume,
    }
    return df, meta


# ─────────────────────────────────────────────
# COMPUTE INDICATORS  (per-stock params from cfg)
# ─────────────────────────────────────────────

def compute_indicators(df, cfg):
    close = np.array(df['Close'], dtype=float)
    high  = np.array(df['High'],  dtype=float)
    low   = np.array(df['Low'],   dtype=float)
    open_ = np.array(df['Open'],  dtype=float)
    vol   = np.array(df['Volume'],dtype=float)

    rsi = np.array(
        ta.momentum.RSIIndicator(df["Close"], window=cfg['rsi_window']).rsi()
    )

    bb     = ta.volatility.BollingerBands(
        df["Close"], window=cfg['bb_window'], window_dev=cfg['bb_std']
    )
    bb_up  = np.array(bb.bollinger_hband())
    bb_mav = np.array(bb.bollinger_mavg())
    bb_low = np.array(bb.bollinger_lband())

    return {
        'close': close, 'high': high, 'low': low, 'open_': open_, 'vol': vol,
        'rsi':   rsi,
        'bb_up': bb_up, 'bb_mav': bb_mav, 'bb_low': bb_low,
    }


# ─────────────────────────────────────────────
# BUILD PLOT
# Top subplot  : Price + Bollinger Bands + entry/exit markers
# Bottom subplot: RSI + thresholds + entry/exit markers
# ─────────────────────────────────────────────

def build_plot(ind, cfg, company_name, ticker, date_label, lookback=LOOKBACK):
    n     = len(ind['close'])
    start = max(0, n - lookback)
    x     = np.arange(start, n)

    # ── Collect signal hits in the visible window ──────────────────────────────
    entry_hits = {name: [] for name in ENTRY_FUNCS}
    exit_hits  = {name: [] for name in EXIT_FUNCS}

    for i in range(start, n):
        for name, fn in ENTRY_FUNCS.items():
            if fn(ind, i, cfg):
                entry_hits[name].append(i)
        for name, fn in EXIT_FUNCS.items():
            if fn(ind, i, cfg):
                exit_hits[name].append(i)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True
    )
    fig.suptitle(
        f"{company_name} ({ticker})  —  {date_label}"
        f"\nBB({cfg['bb_window']}, {cfg['bb_std']})  |  RSI({cfg['rsi_window']})"
        f"  |  Entry: RSI<{cfg['rsi_entry']}, Close<{cfg['bb_entry_mult']}×BBlow"
        f"  |  Exit: RSI>{cfg['rsi_exit']}, Close>{cfg['bb_exit_mult']}×BBup",
        fontsize=10, fontweight='bold'
    )

    close  = ind['close']
    bb_up  = ind['bb_up']
    bb_low = ind['bb_low']
    bb_mav = ind['bb_mav']
    rsi    = ind['rsi']

    # ── Subplot 1: Price + BB ──────────────────────────────────────────────────
    ax1.plot(x, close[start:n],  color='black',     lw=1.3,  label='Close',    zorder=3)
    ax1.plot(x, bb_up[start:n],  color='#27ae60',   lw=0.9,  ls='--', label=f'BB Upper')
    ax1.plot(x, bb_low[start:n], color='#e74c3c',   lw=0.9,  ls='--', label='BB Lower')
    ax1.plot(x, bb_mav[start:n], color='steelblue', lw=0.7,  ls=':',  label='BB Mid',  alpha=0.7)
    ax1.fill_between(x, bb_low[start:n], bb_up[start:n], alpha=0.06, color='steelblue')

    # Entry markers — triangle-up, below the bar
    for name, idxs in entry_hits.items():
        if not idxs:
            continue
        ys = [close[i] * 0.984 for i in idxs]
        ax1.scatter(idxs, ys, marker='^', s=80,
                    color=ENTRY_COLORS[name], zorder=5,
                    edgecolors='black', linewidths=0.5,
                    label=f"Entry {name}")

    # Exit markers — triangle-down, above the bar
    for name, idxs in exit_hits.items():
        if not idxs:
            continue
        ys = [close[i] * 1.016 for i in idxs]
        ax1.scatter(idxs, ys, marker='v', s=80,
                    color=EXIT_COLORS[name], zorder=5,
                    edgecolors='black', linewidths=0.5,
                    label=f"Exit {name}")

    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left', fontsize=7, ncol=4, framealpha=0.7)
    ax1.grid(alpha=0.25)

    # ── Subplot 2: RSI ────────────────────────────────────────────────────────
    ax2.plot(x, rsi[start:n], color='darkorange', lw=1.2, label=f"RSI({cfg['rsi_window']})")

    # Dynamic thresholds from config
    ax2.axhline(cfg['rsi_exit'],  color='#e74c3c', ls='--', lw=0.9,
                label=f"Overbought ({cfg['rsi_exit']})")
    ax2.axhline(cfg['rsi_entry'], color='#27ae60', ls='--', lw=0.9,
                label=f"Oversold ({cfg['rsi_entry']})")
    ax2.axhline(50, color='gray', ls=':', lw=0.6, alpha=0.6)

    ax2.fill_between(x, rsi[start:n], cfg['rsi_entry'],
                     where=(rsi[start:n] < cfg['rsi_entry']),
                     alpha=0.18, color='#27ae60')
    ax2.fill_between(x, rsi[start:n], cfg['rsi_exit'],
                     where=(rsi[start:n] > cfg['rsi_exit']),
                     alpha=0.18, color='#e74c3c')

    # Mirror RSI entry markers on the RSI subplot
    rsi_entry_idxs = entry_hits.get("E_RSI", [])
    if rsi_entry_idxs:
        ax2.scatter(rsi_entry_idxs,
                    [rsi[i] for i in rsi_entry_idxs],
                    marker='^', s=70,
                    color=ENTRY_COLORS["E_RSI"], zorder=5,
                    edgecolors='black', linewidths=0.5)

    rsi_exit_idxs = exit_hits.get("X_RSI", [])
    if rsi_exit_idxs:
        ax2.scatter(rsi_exit_idxs,
                    [rsi[i] for i in rsi_exit_idxs],
                    marker='v', s=70,
                    color=EXIT_COLORS["X_RSI"], zorder=5,
                    edgecolors='black', linewidths=0.5)

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Bar index")
    ax2.legend(loc='upper left', fontsize=7, framealpha=0.7)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    for stock_name, cfg in STOCKS.items():
        print(f"\n{'─'*50}")
        print(f"  {stock_name}  |  BB({cfg['bb_window']}, {cfg['bb_std']})  "
              f"RSI({cfg['rsi_window']})")
        print(f"  Entry: RSI<{cfg['rsi_entry']}  |  "
              f"Close < {cfg['bb_entry_mult']}×BB_lower")
        print(f"  Exit : RSI>{cfg['rsi_exit']}  |  "
              f"Close > {cfg['bb_exit_mult']}×BB_upper")

        try:
            df, meta = fetch_and_update_csv(cfg['ticker'], cfg['csv_path'])
        except Exception as e:
            print(f"  ✗ Failed to fetch/update: {e}")
            continue

        if df is None:
            print("  ✗ No data returned (market may be closed).")
            continue

        ind = compute_indicators(df, cfg)
        i_today = len(ind['close']) - 1

        # Print today's signal status
        for name, fn in ENTRY_FUNCS.items():
            fired = fn(ind, i_today, cfg)
            print(f"  ENTRY {name}: {'✓ TRIGGERED' if fired else '—'}")
        for name, fn in EXIT_FUNCS.items():
            fired = fn(ind, i_today, cfg)
            print(f"  EXIT  {name}: {'✓ TRIGGERED' if fired else '—'}")

        fig = build_plot(ind, cfg, meta['company'], meta['ticker'], meta['date'])
        plt.show()


if __name__ == "__main__":
    main()