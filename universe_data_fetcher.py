"""
universe_data_fetcher.py
=========================
Expands your stock universe WITHOUT manually typing ticker/path entries for
every stock. Point it at one or more NSE index universes (Nifty 50, Next 50,
Midcap 150, ...), and it will:

  1. Pull the constituent list + sector/industry directly from NSE's own
     published index CSVs (no manual stock-name typing, sectors come for free)
  2. Auto-resolve each symbol to the correct yfinance ticker (.NS vs .BO) by
     actually trying both and checking which one returns data — cached so the
     trial-and-error only happens once per stock, ever
  3. Bulk-download 5 years of daily OHLCV per resolved ticker
  4. Save each CSV under Data/<Sector>/<Label>.csv, sector-segregated
  5. Write a manifest CSV (Label, Ticker, Sector, Universe, CSV path, Resolved)
     you can use later to auto-generate Combined.py's STOCKS dict, instead of
     typing entries by hand

USAGE
-----
Edit only the UNIVERSES_TO_FETCH list below, then run:
    python universe_data_fetcher.py

Add a new universe by adding one line to UNIVERSES_TO_FETCH — no per-stock
editing required.

NOTES
-----
- NSE's archive CSVs occasionally reject requests without a browser-like
  User-Agent header — this is handled below. If NSE blocks/changes the URL
  format entirely, the script tells you clearly rather than silently
  producing an empty universe.
- Nifty index constituents are reconstituted periodically (semi-annually).
  Pulling live from NSE each run (rather than a hardcoded list baked into
  this script) means you always get the current membership, not a stale
  snapshot from whenever this script was written.
- Rate-limited (SLEEP_BETWEEN_CALLS) to avoid yfinance/Yahoo throttling on
  a 50-200 ticker run. A full Nifty50+Next50+Midcap150 pull is ~250 tickers
  x 2 API calls each (resolve + history) — expect this to take a while.
"""

import pandas as pd
import numpy as np
import os
import io
import json
import time
import requests
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG — edit only this section
# ═══════════════════════════════════════════════════════════════════════════

# Which universes to pull. Add/remove lines here — never touch per-stock config.
UNIVERSES_TO_FETCH = [
    "NIFTY50",
    # "NIFTY_NEXT_50",
    # "NIFTY_MIDCAP_50",
    # "NIFTY_MIDCAP_150",
]

DATA_ROOT           = "Data"                    # sector-segregated CSVs land here
TICKER_CACHE_PATH   = "ticker_cache.json"        # resolved .NS/.BO decisions, persisted
MANIFEST_PATH       = "universe_manifest.csv"    # Label/Ticker/Sector/Universe/CSV/Resolved
HISTORY_PERIOD      = "5y"                       # yfinance period for the backfill
SLEEP_BETWEEN_CALLS = 1.0                        # seconds — avoid throttling on bulk runs

# NSE's official index constituent CSVs. Each has Symbol / Company Name / Industry
# columns already — this is where sector segregation comes from, free.
INDEX_CSV_URLS = {
    "NIFTY50":           "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY_NEXT_50":     "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "NIFTY_MIDCAP_50":   "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap50list.csv",
    "NIFTY_MIDCAP_150":  "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
}

# NSE rejects requests that don't look like a browser
NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,*/*",
}

# ═══════════════════════════════════════════════════════════════════════════
#  END OF CONFIG
# ═══════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────
# 1. PULL INDEX CONSTITUENTS FROM NSE  (symbol + sector, no manual typing)
# ─────────────────────────────────────────────────────────────────────────

def fetch_index_constituents(universe_key):
    """Returns a DataFrame with columns: Symbol, Company, Sector — pulled
    live from NSE's own index CSV. Raises with a clear message if NSE
    blocks/changes the format, rather than failing silently."""
    url = INDEX_CSV_URLS.get(universe_key)
    if url is None:
        raise ValueError(f"Unknown universe '{universe_key}'. "
                          f"Known: {list(INDEX_CSV_URLS)}")

    resp = requests.get(url, headers=NSE_HEADERS, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"NSE returned HTTP {resp.status_code} for {universe_key} ({url}). "
            f"NSE sometimes blocks non-browser requests or moves this file — "
            f"if this persists, download the CSV manually from nseindia.com's "
            f"'Indices' > 'Historical Data' page and load it with pd.read_csv() instead."
        )

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = df.columns.str.strip()

    # NSE's column naming has some variance across index files historically —
    # normalize defensively rather than assuming exact names.
    symbol_col  = next((c for c in df.columns if c.lower() == "symbol"), None)
    company_col = next((c for c in df.columns if "company" in c.lower()), None)
    sector_col  = next((c for c in df.columns if "industry" in c.lower()
                         or "sector" in c.lower()), None)

    if symbol_col is None:
        raise RuntimeError(
            f"Couldn't find a 'Symbol' column in the {universe_key} CSV "
            f"(columns found: {list(df.columns)}). NSE may have changed the format."
        )

    out = pd.DataFrame({
        "Symbol":  df[symbol_col].astype(str).str.strip(),
        "Company": df[company_col].astype(str).str.strip() if company_col else "",
        "Sector":  df[sector_col].astype(str).str.strip() if sector_col else "Unclassified",
    })
    out["Universe"] = universe_key
    return out


# ─────────────────────────────────────────────────────────────────────────
# 2. AUTO-RESOLVE TICKER SUFFIX  (.NS vs .BO — trial and error, cached)
# ─────────────────────────────────────────────────────────────────────────

def load_ticker_cache():
    if not os.path.exists(TICKER_CACHE_PATH):
        return {}
    with open(TICKER_CACHE_PATH, "r") as f:
        content = f.read().strip()
    return json.loads(content) if content else {}


def save_ticker_cache(cache):
    with open(TICKER_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def resolve_ticker(symbol, cache):
    """Tries SYMBOL.NS first (more liquid on average for NSE-listed names),
    falls back to SYMBOL.BO. Returns the working ticker string, or None if
    neither returns data. Cached so this trial-and-error runs once per
    symbol ever, not on every script run."""
    if symbol in cache:
        return cache[symbol]  # may be None (previously confirmed unresolved)

    for suffix in (".NS", ".BO"):
        candidate = f"{symbol}{suffix}"
        try:
            hist = yf.Ticker(candidate).history(period="5d")
            if not hist.empty:
                cache[symbol] = candidate
                return candidate
        except Exception:
            pass  # try the next suffix
        time.sleep(0.3)  # small gap between the two trial calls

    cache[symbol] = None
    return None


# ─────────────────────────────────────────────────────────────────────────
# 3. BULK 5-YEAR BACKFILL
# ─────────────────────────────────────────────────────────────────────────

def fetch_5y_history(ticker):
    """Returns a DataFrame in the same schema used elsewhere in the project
    (Date in dd-mm-yyyy, Open/High/Low/Close/Volume/Dividends/Stock Splits).
    Avg_Volume is left blank on backfill — yfinance only exposes a *current*
    10-day average via .info, not a historical daily series, and downstream
    signal code (Combined.py) already computes its own rolling vol_ratio from
    actual Volume rather than trusting this column (documented data-bug note:
    Avg_Volume as a static snapshot is not a real time series)."""
    hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD, interval="1d")
    if hist.empty:
        return None

    hist = hist.reset_index()
    date_col = "Date" if "Date" in hist.columns else hist.columns[0]

    df = pd.DataFrame({
        "Date":         pd.to_datetime(hist[date_col]).dt.strftime("%d-%m-%Y"),
        "Open":         hist["Open"].round(2),
        "High":         hist["High"].round(2),
        "Low":          hist["Low"].round(2),
        "Close":        hist["Close"].round(2),
        "Volume":       hist["Volume"].astype("int64", errors="ignore"),
        "Avg_Volume":   np.nan,   # see docstring — not a reliable yfinance field historically
        "Dividends":    hist.get("Dividends", 0.0).round(2),
        "Stock Splits": hist.get("Stock Splits", 0.0).round(2),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────
# 4. MAIN — pull universes, resolve tickers, backfill, save sector-segregated
# ─────────────────────────────────────────────────────────────────────────

def sanitize_folder_name(name):
    keep = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name)
    return keep.strip().replace(" ", "_")


def main():
    ticker_cache = load_ticker_cache()
    manifest_rows = []

    # ── 1. Collect all constituents across requested universes ─────────────
    all_constituents = []
    for universe_key in UNIVERSES_TO_FETCH:
        print(f"\n── Fetching {universe_key} constituent list from NSE ──")
        try:
            const_df = fetch_index_constituents(universe_key)
            print(f"  Got {len(const_df)} symbols, "
                  f"{const_df['Sector'].nunique()} sectors")
            all_constituents.append(const_df)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            continue

    if not all_constituents:
        print("\nNo universes fetched successfully — nothing to do. Check the "
              "error(s) above (likely an NSE request-blocking issue).")
        return

    universe_df = pd.concat(all_constituents, ignore_index=True)
    universe_df = universe_df.drop_duplicates(subset="Symbol", keep="first")
    print(f"\nTotal unique symbols across all requested universes: {len(universe_df)}")

    # ── 2. Resolve tickers + backfill each ──────────────────────────────────
    for _, row in universe_df.iterrows():
        symbol, sector, universe = row["Symbol"], row["Sector"], row["Universe"]
        label = symbol  # used as filename stem; STOCKS dict key would use this too

        print(f"\n── {symbol}  ({sector}) ──")
        ticker = resolve_ticker(symbol, ticker_cache)
        if ticker is None:
            print("  ✗ Could not resolve to a working .NS or .BO ticker — skipping")
            manifest_rows.append(dict(Label=label, Ticker=None, Sector=sector,
                                       Universe=universe, CSV_Path=None, Resolved="N"))
            continue
        print(f"  Resolved ticker: {ticker}")

        sector_folder = os.path.join(DATA_ROOT, sanitize_folder_name(sector))
        csv_path = os.path.join(sector_folder, f"{label.lower()}.csv")

        try:
            df = fetch_5y_history(ticker)
        except Exception as e:
            print(f"  ✗ History fetch failed: {e}")
            manifest_rows.append(dict(Label=label, Ticker=ticker, Sector=sector,
                                       Universe=universe, CSV_Path=None, Resolved="N"))
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        if df is None or df.empty:
            print("  ✗ No historical data returned")
            manifest_rows.append(dict(Label=label, Ticker=ticker, Sector=sector,
                                       Universe=universe, CSV_Path=None, Resolved="N"))
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        os.makedirs(sector_folder, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {len(df)} rows -> {csv_path}")

        manifest_rows.append(dict(Label=label, Ticker=ticker, Sector=sector,
                                   Universe=universe, CSV_Path=csv_path, Resolved="Y"))

        save_ticker_cache(ticker_cache)  # persist incrementally, not just at the end
        time.sleep(SLEEP_BETWEEN_CALLS)

    # ── 3. Manifest + summary ───────────────────────────────────────────────
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    n_ok   = (manifest_df["Resolved"] == "Y").sum()
    n_fail = (manifest_df["Resolved"] == "N").sum()
    print(f"\n{'='*60}")
    print(f"DONE. {n_ok} resolved & saved, {n_fail} failed.")
    print(f"Manifest written to {MANIFEST_PATH} — use this to generate")
    print(f"Combined.py's STOCKS dict entries (Label/Ticker/Sector/CSV_Path).")
    if n_fail:
        print(f"\nFailed symbols:")
        print(manifest_df[manifest_df["Resolved"] == "N"][["Label", "Sector"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
