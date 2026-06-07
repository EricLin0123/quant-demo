# quant-demo — presentation outline

11 slides. Built for a ~7-minute demo. Each slide: the image to show (from `reports/`),
the bullets to put on screen, and a tight spoken script. Numbers are out-of-sample,
net of the real Taiwan cost stack, over **Sep 2016 → Jun 2026**.

---

## Slide 1 — Title / Overview

**Image:** none (title slide)

**On-slide**
- *A laptop-scale ML quant research loop — TWSE Top-50*
- Data → Alpha → Model → Backtest → Portfolio → Monitoring
- Goal: methodological rigor, not a Sharpe-of-3 fantasy

**Script**
> This is an end-to-end machine-learning quant pipeline on the 50 largest Taiwan-listed
> companies. The point isn't to find deployable alpha on free daily data — it's to stand up
> the *whole* loop honestly: leakage controls, real transaction costs, drift monitoring. The
> headline story is a twist: the flagship strategy *loses money*, and I'll show that the fix
> was portfolio construction, not the model.

---

## Slide 2 — Setup & the pipeline

**Image:** none (or a simple stage-flow diagram if you have time)

**On-slide**
- Universe: TWSE Top-50 (`.TW`); 49 with full history. Benchmark: **0050.TW**
- Label: 10-day forward return → cross-sectional rank · rebalance every 5 days
- Costs modeled explicitly: 0.1425% fee each side + 0.30% sell tax (~0.6% round-trip)
- Discipline: all logic in modules; notebooks only orchestrate

**Script**
> Daily adjusted prices, 2014 onward. The benchmark is 0050 — the cap-weighted Top-50 ETF, so
> we're competing against the index this universe *is*. We predict a 10-day forward return,
> turned into a daily cross-sectional rank. Every cost is the real Taiwan retail stack. And the
> code is structured so the pipeline reproduces every number from a cold cache — production
> logic lives in modules, the notebooks just wire stages together.

---

## Slide 3 — Stage 1 · Data ingestion

**Image:** `reports/price_overview_2330.png`

**On-slide**
- Adjusted OHLCV via yfinance, parquet-cached & immutable
- **Concentration:** TSMC + the chip cluster dominate the cap-weighted index
- Named limitation: survivorship bias (today's Top-50 list)

**Script**
> Here's TSMC, the single name that drives this universe. That concentration matters later:
> 0050 is cap-weighted, so beating it means beating a portfolio that's effectively a leveraged
> bet on the semiconductor winners. Two honest caveats I keep visible — survivorship bias from
> using today's constituent list, and the chip-cluster concentration, which is exactly why the
> next stage neutralizes by sector.

---

## Slide 4 — Stage 2 · Feature / alpha engineering

**Image:** none (list the feature families)

**On-slide**
- ~20 point-in-time features: momentum (12-1, 6-1), reversal, realized vol, liquidity/Amihud, trend (dist-from-SMA), RSI, skew, beta
- **Cross-sectional z-score each day** — "how this stock looks vs its peers today"
- Forward target never forward-filled; warmup + unlabeled rows dropped

**Script**
> About twenty features, every one strictly point-in-time. The conceptual core is the last
> line: each day I z-score every feature *across names*, so an absolute number becomes a
> relative view — how cheap, how strong this stock is versus its peers right now. That's the
> form a cross-sectional model can actually trade.

---

## Slide 5 — Stage 3 · Model + purged validation

**Image:** `reports/cv_fold_map.png`

**On-slide**
- LightGBM, shallow & regularized (low signal-to-noise punishes capacity)
- **Purged + embargoed walk-forward** — the only evaluation used
- A 10-day label overlaps 9 of its 10 days with neighbors → naive CV leaks the future

**Script**
> The model is a deliberately small LightGBM — shallow trees, strong regularization. The slide
> that separates the methodical from the rest is *this* one: walk-forward validation where I
> purge any training row whose 10-day label window overlaps the test window, plus an embargo
> gap. Each row here is one model trained only on the blue, tested only on the orange, with the
> overlap dropped. Skip this and the IC roughly doubles — that inflation would be a bug, not alpha.

---

## Slide 6 — Stage 4 · Signal quality (IC)

**Image:** `reports/rolling_ic.png`

**On-slide**
- Mean daily IC ≈ **+0.043**, consistently positive
- Lead with IC, not PnL — with ~50 names PnL is high-variance
- Overlapping labels inflate the naive ICIR; reported honestly

**Script**
> Before any equity curve, I check the Information Coefficient — the daily rank correlation
> between prediction and realized return. Mean IC is about 0.04, positive across most of the
> decade. With only 50 names the portfolio PnL is noisy, so IC is the honest read of the
> signal: it's modest, but it's real. **Remember this — the signal works.** What comes next is
> entirely about how we turn it into a book.

---

## Slide 7 — Stage 5–6 · Portfolio, costs… and the loss

**Image:** `reports/equity_gross_net.png`

**On-slide**
- Flagship book: dollar-neutral, sector-neutral quintile **long-short**
- Own vectorized engine; costs charged on turnover
- **Gross drifts up (+1.5 Sharpe). Net collapses to −0.67.** The gap *is* the cost.

**Script**
> The flagship strategy is a market-neutral long-short — long the top quintile, short the
> bottom. Blue is gross, red is net of costs. Gross looks fine. But this book churns ~70% every
> five days, and at 0.6% round-trip in Taiwan that bleeds nearly 10% a year. Net Sharpe is
> **minus 0.67.** This is the most important honesty chart in the deck — the drop between the
> lines is real money.

---

## Slide 8 — Stage 7 · Monitoring / drift

**Image:** `reports/monitoring.png`

**On-slide**
- Three rings: rolling-IC (performance), per-feature PSI (input drift), realized-vol regime
- Evidently HTML report as a real artifact (`reports/drift.html`)
- In markets the label is delayed → you can't lean on performance alone

**Script**
> Closing the loop: the same three-layer monitoring I'd run in any production ML system.
> Rolling IC for performance decay, PSI per feature for covariate shift, and a volatility-regime
> flag so bad stretches are explainable, not mysterious. In markets the label arrives late, so
> the input-drift ring is what warns you *before* performance confirms the damage.

---

## Slide 9 — The verdict: original strategy vs 0050

**Image:** `reports/equity_vs_benchmark.png`

**On-slide**
- Long-short (red): **−0.67** Sharpe — loses outright
- Long-only top-quintile (blue): **+1.01** — *still under* the benchmark
- 0050 (green): **+1.22**, +24%/yr
- Signal is good; the **construction** is losing

**Script**
> Put it next to the benchmark. The long-short, in red, just sinks. Even dropping the shorts —
> the blue long-only book — only gets us to +1.01 Sharpe, *below* 0050's +1.22. So we have a
> genuinely predictive signal that we're turning into a portfolio that can't beat the index.
> That's the problem statement. The question isn't "find a better model" — it's "build a better book."

---

## Slide 10 — The fix: one signal, six books

**Image:** `reports/strategy_compare.png`

**On-slide**
- Same `pred`, same 50 names — only the score→weight mapping changes
- **Rank-weighted tilt:** +1.32 Sharpe, shallowest drawdown (−32%), lowest turnover (31%) → deploy this
- **Slow rebalance (20d):** +1.34 Sharpe — free Sharpe from paying less in costs
- Concentration buys *return* not Sharpe; inverse-vol *underperforms* (honest negative)

**Script**
> Same signal, same names — I only change how scores become weights. Two books clear the
> benchmark. Slowing the rebalance from 5 to 20 days lifts Sharpe to 1.34 purely by paying less
> in costs. But the one I'd deploy is the rank-weighted tilt: weight all fifty names by their
> score instead of a hard top-ten cutoff. It beats 0050 on Sharpe *and* drawdown at a third of
> the turnover. And I keep the failures in — concentration buys return at the cost of a 53%
> drawdown, and inverse-vol actually underperforms because de-risking steers away from the
> high-vol winners carrying the index.

---

## Slide 11 — Conclusion

**Image:** none (or reuse `strategy_compare.png` small)

**On-slide**
- The signal was never the problem — the **book** was
- Beat 0050 net of costs by changing construction, not the model
- Honest read: modest, defensible outperformance — not a deployable edge
- The transferable asset is the *loop*: leakage control, real costs, monitoring

**Script**
> The takeaway: a modest signal lost money under one construction and beat the benchmark under
> another — without touching the model. The honest framing is that this is defensible
> outperformance on free daily data, not a deployable edge. What transfers to real infrastructure
> is the discipline — purged validation, real costs, drift monitoring — and the instinct to fix
> the portfolio before reaching for a fancier model.

---

### Notes
- Slides 4 and 11 are text-only; if you want every slide to carry an image, slide 4 can reuse the
  PSI-bar panel cropped from `monitoring.png` (it lists the feature set), and slide 11 a faded
  `strategy_compare.png`.
- To go *even* shorter (≈8 slides): merge 3+4 into one "Data & Features" slide, and 7+9 into one
  "Result: the book loses to costs and to 0050" slide.
