# Technical Indicators

Complete reference for every feature emitted by `src/twii_forecast/features.py`
(`build_features`). **All features are strictly causal**: the row at time `t` uses
only observations at times `≤ t`. Every rolling op uses `min_periods = window` (never
centred) and every EMA uses `adjust=False`, so each value is a pure trailing recursion.
This is what `tests/test_leakage.py` enforces via truncation-invariance.

Notation: `O, H, L, C, V` = open, high, low, close, volume on day `t`;
`C₋₁ = C.shift(1)` = previous close. Window constants come from `config.py`.

Every indicator below is listed — **including the ones later dropped by the
temporal-consistency AUC feature selection** (see [Feature selection](#feature-selection)
at the end). Documenting a feature here does not mean it survived selection.

---

## 1. Price-shape / intraday

### `amplitude` — normalized daily range
```
amplitude = (H − L) / C₋₁
```
The day's high-low range scaled by the prior close. A unitless proxy for **intraday
volatility / trading activity**. Larger on turbulent days, compressed on quiet ones.

### `difference` — normalized open-to-close move
```
difference = (C − O) / C₋₁
```
The intraday directional move (close minus open) scaled by the prior close. This is the
**paper's top SHAP feature** — it captures whether buyers or sellers won the session,
normalized so it is comparable across price levels.

### `h_minus_l` — raw daily range
```
h_minus_l = H − L
```
The un-normalized high-low spread in index points. Same information as `amplitude` but
on the raw point scale (and therefore non-stationary as the index level drifts).

### `c_minus_o` — raw intraday move
```
c_minus_o = C − O
```
Un-normalized close-minus-open in index points. The raw counterpart of `difference`.

---

## 2. First differences

### `delta_c` — close-to-close change
```
delta_c = C − C₋₁
```
First difference of the close in index points. The single-day price change; the building
block of momentum and the `delta_c_ma_*` smooths below.

> An internal helper `intraday = C − O` (the same quantity as `c_minus_o`) is computed
> here as the basis for the `intraday_ma_*` features; it is not emitted as its own column.

---

## 3. Daily return / momentum

### `daily_return` — log return
```
daily_return = ln(C / C₋₁)
```
The one-day log return. Stationary and additive across time, this is the canonical
"how much did we move today" signal and the same family as the prediction **target**
(next-day log return).

### `momentum` — n-day price change
```
momentum = C − C.shift(MOMENTUM_WINDOW)        # MOMENTUM_WINDOW = 10
```
Price now minus price 10 trading days ago, in index points. A classic **trend-strength /
momentum** measure: positive in uptrends, negative in downtrends.

### `sign_delta_open` — open-gap direction
```
sign_delta_open = sign(O − O.shift(1))
```
The sign (−1 / 0 / +1) of today's open versus yesterday's open. A coarse categorical
indicator of **overnight gap direction**, discarding magnitude.

---

## 4. Short-window deltas (`INTRADAY_MA_WINDOWS = 5, 10, 20`)

For each window `n ∈ {5, 10, 20}`:

### `intraday_ma_{n}` — smoothed intraday move
```
intraday_ma_n = mean over last n days of (C − O)
```
Rolling mean of the intraday (close−open) move. Smooths the day-to-day noise in
`c_minus_o` to reveal a **persistent intraday bias** over the last `n` sessions.
→ `intraday_ma_5`, `intraday_ma_10`, `intraday_ma_20`.

### `delta_c_ma_{n}` — smoothed close change
```
delta_c_ma_n = mean over last n days of delta_c
```
Rolling mean of the close-to-close change. A short-horizon **drift / average daily move**
estimate.
→ `delta_c_ma_5`, `delta_c_ma_10`, `delta_c_ma_20`.

---

## 5. Moving averages & EMAs of close (`MA_WINDOWS = 5, 10, 20, 60`)

For each window `n ∈ {5, 10, 20, 60}`:

### `ma_{n}` — simple moving average
```
ma_n = mean over last n days of C
```
The trailing simple moving average of close. Equal-weighted **trend level** at horizon
`n`; short windows track price tightly, long windows lag and define the broader regime.
→ `ma_5`, `ma_10`, `ma_20`, `ma_60`.

### `ema_{n}` — exponential moving average
```
ema_n = EMA(C, span = n, adjust = False)
```
Exponentially-weighted moving average (recent days weighted more heavily). Reacts faster
than the equal-weighted `ma_n` to recent price changes.
→ `ema_5`, `ema_10`, `ema_20`, `ema_60`.

---

## 6. MACD (`MACD_FAST = 12`, `MACD_SLOW = 26`)

### `macd` — moving-average convergence/divergence
```
macd = EMA(C, span = 12) − EMA(C, span = 26)
```
The difference between a fast and a slow EMA. A standard **trend / momentum** oscillator:
positive when the fast average is above the slow (recent strength), negative when below.
(Only the MACD line is emitted; no signal line or histogram.)

---

## 7. Oscillators

### `rsi` — Relative Strength Index (`RSI_WINDOW = 14`)
```
avg_gain = mean over 14 days of max(ΔC, 0)
avg_loss = mean over 14 days of max(−ΔC, 0)
RS  = avg_gain / avg_loss
RSI = 100 − 100 / (1 + RS)
```
Wilder-style RSI computed causally with trailing rolling means, bounded in **[0, 100]**.
Measures the ratio of recent gains to losses: conventionally >70 = overbought,
<30 = oversold. Edge cases are handled explicitly — all gains → RSI = 100, all losses →
RSI = 0.

### `williams_r` — Williams %R (`WILLIAMS_WINDOW = 14`)
```
williams_r = (HighestHigh₁₄ − C) / (HighestHigh₁₄ − LowestLow₁₄) × −100
```
A momentum oscillator bounded in **[−100, 0]** showing where the close sits within the
14-day high-low range. Near 0 = close near the period high (strong); near −100 = close
near the period low (weak). Inverse-scaled cousin of the stochastic oscillator.

---

## 8. Bollinger bands (`BB_WINDOW = 21`, `BB_STD = 2.0`)

```
mid     = mean over 21 days of C
sd      = std  over 21 days of C
bb_high = mid + 2.0 × sd
bb_low  = mid − 2.0 × sd
```

### `bb_high` — upper band
The 21-day moving average plus 2 standard deviations. An adaptive **upper volatility
envelope**; closes approaching it suggest a stretched/overbought level.

### `bb_low` — lower band
The 21-day moving average minus 2 standard deviations. The **lower volatility envelope**;
closes approaching it suggest an oversold level. The gap between the bands widens with
volatility and narrows in quiet regimes. (The middle band itself is not emitted — it is
identical to `ma_20`-style smoothing and would be redundant.)

---

## 9. Lagged returns (`LAG_RETURNS = 0, 1, 2, 3, 5`)

For each lag `k ∈ {0, 1, 2, 3, 5}`:
```
ret_lag_k = daily_return.shift(k)
```
The log return from `k` days ago, giving the model an explicit short **autocorrelation**
window. `ret_lag_0` is today's return; `ret_lag_1…5` are the returns 1, 2, 3 and 5 days
back. Lets a tree split on recent return sequences (e.g. reversal vs continuation).
→ `ret_lag_0`, `ret_lag_1`, `ret_lag_2`, `ret_lag_3`, `ret_lag_5`.

---

## 10. Volume block — *only if the data-quality gate passes*

This block is emitted only when `volume_ok` is true (the runtime gate in `data.py`
requires fewer than `VOLUME_DEGENERATE_THRESHOLD = 5%` of rows to be zero/NaN volume).
On data where TWII volume is degenerate these columns are absent entirely.

### `delta_v` — volume change
```
delta_v = V − V₋₁
```
First difference of volume. A proxy for **changes in participation / conviction** behind
a move.

### `delta_v_ma_{n}` — smoothed volume change (`n ∈ {5, 10, 20}`)
```
delta_v_ma_n = mean over last n days of delta_v
```
Rolling mean of the volume change, smoothing the noisy day-to-day `delta_v` into a
short-horizon **volume-trend** signal.
→ `delta_v_ma_5`, `delta_v_ma_10`, `delta_v_ma_20`.

---

## Feature selection

All of the above are *candidate* features. Before training, `feature_selection.py` runs a
**temporal-consistency analysis (TCA)** on the **training set only**: the train period is
cut into trimonthly blocks (`TCA_BLOCK_MONTHS = 3`) and a classifier tries to tell pairs
of blocks apart from each feature's distribution. A feature whose aggregated AUC-ROC is
high is one whose distribution **drifts over time** (not temporally stable); a feature
near AUC ≈ 0.5 is stable across regimes.

Features are **kept only if their aggregated AUC ≤ `TCA_TAU = 0.7`** (`TCA_AGG = "mean"`).
The non-stationary, point-scale level features (e.g. raw `ma_*`, `ema_*`, `bb_high`,
`bb_low`, `h_minus_l`, `momentum`) are the typical casualties — their absolute level
trends with the index, so they drift and get dropped — while the normalized/stationary
features (returns, `amplitude`, `difference`, oscillators) tend to survive. The exact
surviving set is data-dependent; see the run report / notebook for the current selection.
