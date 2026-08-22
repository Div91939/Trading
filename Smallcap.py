--- repo/Combined.py	2026-08-22 12:20:25.412045859 +0000
+++ Combined_NEW.py	2026-08-22 18:28:04.350485910 +0000
@@ -99,6 +99,33 @@
 REV_ATR     = 3.5     # min ATR% — needs enough volatility to rebound
 REV_RET60   = -40.0   # 60-day return floor — excludes total breakdowns
 
+# ── REBOUND: fitted logistic score -- DISABLED BY DEFAULT ON THIS UNIVERSE ──
+# Same signal added to Smallcap.py, ported here for consistency, but the
+# corrected walk-forward evidence on THIS 30-stock universe does not clear
+# the bar to run live:
+#
+#   5-window walk-forward (2023-08 -> 2026-08, AFCOM/PARAS excluded from
+#   training even though they are not currently gated out of live REV --
+#   see the note further down):
+#     window   n_test  signals   AUC    P/L mean   P/L median   win%
+#     W1         1554      747  0.517    +12.75%      +9.30%    63.2%
+#     W2         2088       63  0.558     +2.23%     -16.63%    30.2%
+#     W3         2632      192  0.662     +0.83%      -8.88%    37.5%
+#     W4         1992      152  0.658     -1.57%      -9.15%    23.0%
+#     W5         2531       32  0.640     +6.81%      -3.58%    43.8%
+#
+#   Median P/L is NEGATIVE in 4 of 5 windows and win rate is below 50% in
+#   3 of 5. On the Smallcap universe (227 names) the identical construction
+#   holds up fine; on this 30-stock universe it does not. Read this as a
+#   small-sample problem, not evidence the underlying idea is wrong -- 30
+#   stocks isn't enough for a 25-feature fitted score to generalise the way
+#   it does on 227. Left wired in and ready to use, but gated off until
+#   there's a better reason to trust it here.
+#
+# To turn it on: set REBOUND_ENABLED = True. Everything else (constants,
+# check_rebound, logging, email/plot wiring) is already in place below.
+REBOUND_ENABLED = False
+
 # Per-stock config:
 #   ticker      — yfinance ticker string
 #   csv_path    — local CSV path
@@ -117,6 +144,56 @@
 # no combination clear a positive-return bar — left on a conservative
 # fallback and flagged; don't expect REV to fire much on those three.
 
+REBOUND_TRIGGER_DD20 = -5.0
+REBOUND_TRAIL   = 0.25
+REBOUND_MAXHOLD = 60
+REBOUND_COOLOFF = 5
+REBOUND_LOG_PATH = "combined_rebound_log.json"
+
+# Feature keys use ind[] naming (price_vs_ma10 not px_vs_ma10, etc.) to match
+# this file's existing compute_indicators() convention rather than Smallcap.py's.
+REBOUND_FEATS = ["ret1", "ret5", "ret10", "ret20d", "ret60d", "z5",
+                 "price_vs_ma10", "price_vs_ma20", "price_vs_ma50", "ma50_slope",
+                 "up_from_low252", "dd_20d", "atr_pct", "rv20", "rv5", "rv_ratio",
+                 "bb_width", "pctB", "rsi14", "vol_ratio", "vol_z",
+                 "consec_down", "clspos", "gap", "stoch_k"]
+
+REBOUND_SCALER_MEAN = {
+    "ret1": -0.65612329, "ret5": -2.73586171, "ret10": -4.01312565,
+    "ret20d": -4.02546823, "ret60d": 4.26466177, "z5": -0.49617815,
+    "price_vs_ma10": -2.44878087, "price_vs_ma20": -3.63667015, "price_vs_ma50": -3.06687789,
+    "ma50_slope": 0.45110424, "up_from_low252": 92.45784644, "dd_20d": -11.63268406,
+    "atr_pct": 4.77923722, "rv20": 46.61896169, "rv5": 41.88272303,
+    "rv_ratio": 0.91308612, "bb_width": 21.39502873, "pctB": 0.28885649,
+    "rsi14": 43.48797990, "vol_ratio": 0.92306671, "vol_z": -0.11063111,
+    "consec_down": 1.47916798, "clspos": 0.41362415, "gap": 0.17858727,
+    "stoch_k": 29.52459123,
+}
+REBOUND_SCALER_SCALE = {
+    "ret1": 2.93464906, "ret5": 5.90835414, "ret10": 8.27067161,
+    "ret20d": 13.62750026, "ret60d": 32.32508159, "z5": 0.81151178,
+    "price_vs_ma10": 4.14323708, "price_vs_ma20": 6.14418610, "price_vs_ma50": 12.35119176,
+    "ma50_slope": 5.18336851, "up_from_low252": 121.59496602, "dd_20d": 6.59037496,
+    "atr_pct": 1.93899131, "rv20": 20.09222619, "rv5": 27.29822819,
+    "rv_ratio": 0.40249642, "bb_width": 13.98064528, "pctB": 0.21705860,
+    "rsi14": 9.31022160, "vol_ratio": 0.95350291, "vol_z": 0.92655801,
+    "consec_down": 1.78373979, "clspos": 2.63537699, "gap": 1.96365344,
+    "stoch_k": 20.55871288,
+}
+REBOUND_COEF = {
+    "ret1": -0.05334801, "ret5": -0.18344443, "ret10": -0.06435885,
+    "ret20d": +0.03379477, "ret60d": -0.04991313, "z5": +0.11054393,
+    "price_vs_ma10": -0.14510386, "price_vs_ma20": +0.36385225, "price_vs_ma50": -0.04934408,
+    "ma50_slope": -0.02357428, "up_from_low252": +0.10372912, "dd_20d": -0.31614346,
+    "atr_pct": +0.27153541, "rv20": -0.13440608, "rv5": +0.05732904,
+    "rv_ratio": +0.04727853, "bb_width": -0.03045100, "pctB": -0.00630711,
+    "rsi14": +0.09044068, "vol_ratio": +0.06231850, "vol_z": -0.01716173,
+    "consec_down": -0.06081502, "clspos": +0.02722152, "gap": +0.00263720,
+    "stoch_k": +0.05711946,
+}
+REBOUND_INTERCEPT = -1.36375382
+REBOUND_THRESHOLD = 0.27904374
+
 STOCKS = {
     "ADANIENT": {
         "ticker":   "ADANIENT.NS",
@@ -461,6 +538,37 @@
         tr_v3[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
     atr_pct = np.where(close > 0, pd.Series(tr_v3).rolling(14).mean().values / close * 100, np.nan)
 
+    # ── extra fields needed only by REBOUND (everything above this line is
+    #    unchanged from the original file) ──────────────────────────────────
+    ret1 = np.full(n, np.nan); ret1[1:] = (close[1:]-close[:-1])/close[:-1]*100
+    ret5 = ret5_v3
+    ret10 = np.full(n, np.nan); ret10[10:] = (close[10:]-close[:-10])/close[:-10]*100
+    ma20_own = pd.Series(close).rolling(20).mean().values
+    price_vs_ma20 = np.where(ma20_own > 0, (close-ma20_own)/ma20_own*100, np.nan)
+    hi20 = pd.Series(close).rolling(20).max().values
+    dd_20d = (close/hi20-1)*100
+    logr_reb = np.concatenate([[np.nan], np.diff(np.log(np.maximum(close, 1e-9)))])
+    rv20_reb = pd.Series(logr_reb).rolling(20).std().values*np.sqrt(252)*100
+    rv5_reb = pd.Series(logr_reb).rolling(5).std().values*np.sqrt(252)*100
+    with np.errstate(divide="ignore", invalid="ignore"):
+        rv_ratio = rv5_reb/rv20_reb
+    bwid = bb_up-bb_low
+    bb_width = np.where(ma20_own > 0, bwid/ma20_own*100, np.nan)
+    pctB = np.where(bwid > 0, (close-bb_low)/bwid, np.nan)
+    vs_reb = pd.Series(vol)
+    vol_z = ((vs_reb-vs_reb.rolling(60, min_periods=20).mean())
+            / vs_reb.rolling(60, min_periods=20).std()).values
+    down1 = (ret1 < 0).astype(float)
+    consec_down = np.zeros(n)
+    for _i in range(1, n):
+        consec_down[_i] = consec_down[_i-1]+1 if (not np.isnan(down1[_i]) and down1[_i]) else 0
+    rng_hl = high-low
+    clspos = np.divide(close-low, rng_hl, out=np.full(n, np.nan), where=rng_hl > 0)
+    gap = np.full(n, np.nan); gap[1:] = (opn[1:]/close[:-1]-1)*100
+    ll14 = pd.Series(low).rolling(14).min().values
+    hh14 = pd.Series(high).rolling(14).max().values
+    stoch_k = 100*(close-ll14)/np.where((hh14-ll14) > 0, hh14-ll14, np.nan)
+
     return dict(
         close=close, high=high, low=low, vol=vol, open=opn,
         ma50=ma50, ma50_slope=ma50_slope, price_vs_ma50=price_vs_ma50,
@@ -472,6 +580,11 @@
         rsi14=rsi14, bb_up=bb_up, bb_low=bb_low, bb_mid=bb_mid,
         price_vs_ma10=price_vs_ma10, z5=z5,
         up_from_low252=up_from_low252, atr_pct=atr_pct,
+        # ── REBOUND-only fields (see the constants block above check_rev) ──
+        ret1=ret1, ret5=ret5, ret10=ret10, price_vs_ma20=price_vs_ma20,
+        dd_20d=dd_20d, rv20=rv20_reb, rv5=rv5_reb, rv_ratio=rv_ratio,
+        bb_width=bb_width, pctB=pctB, vol_z=vol_z, consec_down=consec_down,
+        clspos=clspos, gap=gap, stoch_k=stoch_k,
     )
 
 # ─────────────────────────────────────────────────────────────────────────────
@@ -575,6 +688,36 @@
     return int(i - d)  # bar index of the actual cross
 
 
+def rebound_probability(ind, i):
+    """Plain logistic-regression arithmetic -- no sklearn at runtime.
+    Returns NaN if any input feature is unavailable (e.g. still in warmup)."""
+    if any(np.isnan(ind[k][i]) for k in REBOUND_FEATS):
+        return np.nan
+    z = REBOUND_INTERCEPT
+    for k in REBOUND_FEATS:
+        x = (ind[k][i] - REBOUND_SCALER_MEAN[k]) / REBOUND_SCALER_SCALE[k]
+        z += REBOUND_COEF[k] * x
+    return 1.0 / (1.0 + np.exp(-z))
+
+
+def check_rebound(ind, i):
+    """REBOUND -- fitted logistic score on top of an observable >=5% fall
+    from the 20-day high. GATED OFF BY DEFAULT on this universe -- see the
+    REBOUND_ENABLED note in the constants block for the walk-forward
+    evidence (median P/L negative in 4/5 windows here, unlike Smallcap where
+    the identical construction holds up). Always returns (False, nan) while
+    REBOUND_ENABLED is False, regardless of the underlying score, so flipping
+    the flag is the only thing that changes behaviour."""
+    if not REBOUND_ENABLED:
+        return False, np.nan
+    if ind["dd_20d"][i] > REBOUND_TRIGGER_DD20:
+        return False, np.nan
+    p = rebound_probability(ind, i)
+    if np.isnan(p):
+        return False, np.nan
+    return (p >= REBOUND_THRESHOLD), p
+
+
 SIGNAL_DESCRIPTIONS = {
     "REV": (
         "REVERSAL ENTRY (v3 — universal, no per-stock tuning)\n"
@@ -601,6 +744,15 @@
         "  (vs +9.95% on a fixed 40-day hold), median +4.18%, 56.9% win —\n"
         "  a fixed hold was capping exactly the trades that mattered most."
     ),
+    "REBOUND": (
+        "REBOUND (fitted logistic score) -- GATED OFF BY DEFAULT\n"
+        "  Triggers on an observable >=5% fall from the 20-day high; scores\n"
+        "  P(closes >=10% above next open within 10 bars) via a 25-feature\n"
+        "  logistic regression, hardcoded as arithmetic (no model file).\n"
+        "  On THIS universe the corrected walk-forward showed negative median\n"
+        "  P/L in 4/5 windows -- see REBOUND_ENABLED in the constants block.\n"
+        "  Validated and live on Smallcap.py's universe; not here yet."
+    ),
 
 }
 
@@ -609,10 +761,11 @@
 # ─────────────────────────────────────────────────────────────────────────────
 
 def build_plot(ind, company_name, ticker, date_label, regime, lookback=120,
-               rev_fires=None, mom_fires=None):
-    """rev_fires / mom_fires: optional lists of bar indices (absolute, into ind
-    arrays) where each signal fired historically — marked on the chart so the
-    email shows past triggers within the visible window, not just today's."""
+               rev_fires=None, mom_fires=None, rebound_fires=None):
+    """rev_fires / mom_fires / rebound_fires: optional lists of bar indices
+    (absolute, into ind arrays) where each signal fired historically — marked
+    on the chart so the email shows past triggers within the visible window,
+    not just today's."""
     n     = len(ind["close"])
     start = max(0, n - lookback)
     x     = np.arange(start, n)
@@ -676,6 +829,12 @@
             ax1.scatter(mf, [ind["low"][i] - marker_off for i in mf],
                         marker="^", s=80, color="#3498db", edgecolor="black",
                         lw=0.7, zorder=5, label="MOM trigger")
+    if rebound_fires:
+        bf = [i for i in rebound_fires if start <= i < n]
+        if bf:
+            ax1.scatter(bf, [ind["high"][i] + marker_off for i in bf],
+                        marker="v", s=80, color="#9b59b6", edgecolor="black",
+                        lw=0.7, zorder=5, label="REBOUND trigger")
 
     ax1.set_ylabel("Price")
     ax1.legend(loc="upper left", fontsize=7, ncol=6, framealpha=0.7)
@@ -746,6 +905,22 @@
     with open(MOM_CROSS_LOG_PATH, "w") as f:
         json.dump(log, f, indent=2)
 
+def load_rebound_log():
+    if not os.path.exists(REBOUND_LOG_PATH):
+        return {}
+    with open(REBOUND_LOG_PATH, "r") as f:
+        content = f.read().strip()
+    if not content:
+        return {}
+    try:
+        return json.loads(content)
+    except json.JSONDecodeError:
+        return {}
+
+def save_rebound_log(log):
+    with open(REBOUND_LOG_PATH, "w") as f:
+        json.dump(log, f, indent=2)
+
 def send_email(subject, body, attachments):
     """attachments: list of (filename, png_bytes)"""
     msg = MIMEMultipart()
@@ -775,6 +950,7 @@
 def main():
     log          = load_log()
     mom_cross_log = load_mom_cross_log()
+    rebound_log   = load_rebound_log()
     today_label  = None
 
     report_sections  = []
@@ -831,6 +1007,17 @@
                     mom_fired = True
                     mom_cross_log[stock_name] = cross_id
 
+        # ── REBOUND — runs on every stock regardless of the Signals config
+        # (it is independent of REV/MOM). Returns (False, nan) unconditionally
+        # while REBOUND_ENABLED is False, so this is inert until that flag
+        # is flipped. Bar-based cooloff, same idea as MOM's cross dedup. ────
+        rebound_fired, rebound_p = check_rebound(ind, i)
+        if rebound_fired:
+            if i - rebound_log.get(stock_name, -10**9) < REBOUND_COOLOFF:
+                rebound_fired = False
+            else:
+                rebound_log[stock_name] = i
+
         # ── Print daily status ────────────────────────────────────────────
         print(f"  Regime    : {regime}   (informational — REV/MOM v2 do not gate on this)")
         print(f"  Close     : {ind['close'][i]:.2f}")
@@ -841,9 +1028,11 @@
                   f"MA50 10d slope: {ind['ma50_slope_10'][i]:.2f}%  |  "
                   f"Vol(5d/20d): {ind['vol_expansion'][i]:.2f}x")
         print(f"  Vol ratio : {ind['vol_ratio'][i]:.2f}x  |  ADX: {ind['adx'][i]:.1f}  |  DI+: {ind['di_plus'][i]:.1f}  DI-: {ind['di_minus'][i]:.1f}")
-        print(f"  REV fired : {'YES' if rev_fired else 'no'}  |  MOM fired: {'YES' if mom_fired else 'no'}")
+        print(f"  REV fired : {'YES' if rev_fired else 'no'}  |  MOM fired: {'YES' if mom_fired else 'no'}  |  "
+              f"REBOUND: {'YES' if rebound_fired else ('off' if not REBOUND_ENABLED else 'no')}"
+              + (f" (p={rebound_p:.2f})" if not np.isnan(rebound_p) else ""))
 
-        if not rev_fired and not mom_fired:
+        if not rev_fired and not mom_fired and not rebound_fired:
             continue
 
         # ── Build email section ───────────────────────────────────────────
@@ -856,8 +1045,14 @@
             f"Vol ratio: {ind['vol_ratio'][i]:.2f}x   ADX: {ind['adx'][i]:.1f}   DI+: {ind['di_plus'][i]:.1f}  DI-: {ind['di_minus'][i]:.1f}",
             f"MA50 slope (20d): {ind['ma50_slope'][i]:.2f}%",
         ]
+        if rebound_fired:
+            lines.append(
+                f"REBOUND prob: {rebound_p:.3f}  (threshold {REBOUND_THRESHOLD:.3f})   "
+                f"dd_20d: {ind['dd_20d'][i]:+.2f}%   [fitted score, gated on this "
+                f"universe -- see REBOUND_ENABLED in the constants block]"
+            )
 
-        for sig_name, fired in [("REV", rev_fired), ("MOM", mom_fired)]:
+        for sig_name, fired in [("REV", rev_fired), ("MOM", mom_fired), ("REBOUND", rebound_fired)]:
             if not fired:
                 continue
             log_key = f"{stock_name}_{sig_name}"
@@ -888,13 +1083,17 @@
                     if cid is not None and cid not in seen_crosses:
                         seen_crosses.add(cid)
                         hist_mom.append(k)   # first qualifying day per cross only
+        hist_rebound = ([k for k in range(len(ind["close"])) if check_rebound(ind, k)[0]]
+                        if REBOUND_ENABLED else [])
         png = build_plot(ind, meta["company"], meta["ticker"], meta["date"], regime,
-                          rev_fires=hist_rev, mom_fires=hist_mom)
+                          rev_fires=hist_rev, mom_fires=hist_mom,
+                          rebound_fires=hist_rebound)
         plot_attachments.append((f"{stock_name}_{today_label}.png", png))
 
     # ── Save state ────────────────────────────────────────────────────────
     save_log(log)
     save_mom_cross_log(mom_cross_log)
+    save_rebound_log(rebound_log)
 
     if not report_sections:
         print("\nNo signals today — no email sent.")
@@ -908,6 +1107,9 @@
         f"Strategy: REV (mean reversion, RANGE regime) + MOM (momentum, UP regime)\n"
         f"Hold: 30 days\n"
         f"Stocks scanned: {len(STOCKS)}\n"
+        + (f"REBOUND: gated OFF on this universe (see constants block) -- "
+           f"validated and live on Smallcap.py instead\n" if not REBOUND_ENABLED
+           else f"REBOUND: ENABLED\n")
         + "\n".join(report_sections)
     )
     send_email(subject, body, plot_attachments)
