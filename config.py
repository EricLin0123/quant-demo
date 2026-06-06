"""Central configuration for the quant-demo pipeline.

Every tunable lives here so `run_pipeline.py` is reproducible from a single
source of truth. Stage 1 (data ingestion) only consumes the universe, the
date range, and the cache path; the rest is declared up-front for later stages.
"""

from __future__ import annotations

from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "cache"
REPORTS_DIR = ROOT / "reports"
PRICES_CACHE = CACHE_DIR / "prices.parquet"
INDEX_CACHE = CACHE_DIR / "index.parquet"
FEATURES_CACHE = CACHE_DIR / "features.parquet"

# --- date range & universe -------------------------------------------------
START, END = "2014-01-01", "2024-12-31"
UNIVERSE = "TWSE_TOP50"          # see TWSE_TOP50 list below (.TW suffix)
INDEX_PROXY = "^TWII"            # TAIEX, for the beta feature; 0050.TW = ETF proxy

# --- labelling & rebalancing (used from Stage 2 onward) --------------------
LABEL_HORIZON = 10               # trading days forward
REBALANCE_FREQ = 5               # trade every 5 days
N_SPLITS = 8                     # walk-forward folds
EMBARGO_DAYS = 5
N_QUANTILES = 5                  # quintile long-short (top/bottom 10 of 50 names)
COST_BPS = 30                    # round-trip cost; TW has ~0.3% sell tax + fees
SECTOR_NEUTRAL = True
MAX_WEIGHT = 0.10                # per-name cap; guardrail vs single-name concentration
SEED = 42

# --- ingestion hygiene -----------------------------------------------------
# Drop any ticker missing more than this fraction of the common trading
# calendar (e.g. a name that only listed partway through the window).
MAX_MISSING_FRAC = 0.10

# Hardcoded top-50 TWSE universe (Yahoo Finance `.TW` tickers).
#
# Snapshot of the ~50 largest TWSE names (approx. the 0050.TW / Yuanta Taiwan
# Top 50 constituents). Kept hardcoded so the run is reproducible.
#
# KNOWN LIMITATION — survivorship bias: this is *today's* ranking, so names
# that dropped out of the top 50 over 2014-2024 are excluded, which inflates
# historical returns. Production would use a point-in-time TWSE constituent
# history (e.g. TEJ). See plan.md §2.
#
# CONCENTRATION: TSMC (2330) and the semiconductor cluster dominate TWSE market
# cap, so this cross-section is chip-heavy — motivates sector-neutralization.
TWSE_TOP50 = [
    "2330.TW",  # TSMC — semiconductors
    "2317.TW",  # Hon Hai (Foxconn) — electronics manufacturing
    "2454.TW",  # MediaTek — semiconductors
    "2308.TW",  # Delta Electronics — electronics
    "2382.TW",  # Quanta Computer — electronics
    "2891.TW",  # CTBC Financial — financials
    "2881.TW",  # Fubon Financial — financials
    "2882.TW",  # Cathay Financial — financials
    "2412.TW",  # Chunghwa Telecom — telecom
    "2303.TW",  # UMC — semiconductors
    "3711.TW",  # ASE Technology — semiconductors
    "2886.TW",  # Mega Financial — financials
    "2884.TW",  # E.Sun Financial — financials
    "1303.TW",  # Nan Ya Plastics — materials
    "1301.TW",  # Formosa Plastics — materials
    "2002.TW",  # China Steel — materials
    "3008.TW",  # Largan Precision — optics/electronics
    "2885.TW",  # Yuanta Financial — financials
    "2892.TW",  # First Financial — financials
    "2880.TW",  # Hua Nan Financial — financials
    "2890.TW",  # SinoPac Financial — financials
    "2883.TW",  # China Development Financial — financials
    "2887.TW",  # Taishin Financial — financials
    "5880.TW",  # Taiwan Cooperative Financial — financials
    "2207.TW",  # Hotai Motor — autos
    "2357.TW",  # Asustek — electronics
    "2379.TW",  # Realtek — semiconductors
    "3034.TW",  # Novatek — semiconductors
    "3037.TW",  # Unimicron — electronics
    "3045.TW",  # Taiwan Mobile — telecom
    "4904.TW",  # Far EasTone — telecom
    "2395.TW",  # Advantech — electronics
    "3231.TW",  # Wistron — electronics
    "2474.TW",  # Catcher Technology — electronics
    "1216.TW",  # Uni-President — consumer staples
    "1326.TW",  # Formosa Chemicals & Fibre — materials
    "2327.TW",  # Yageo — electronics components
    "6505.TW",  # Formosa Petrochemical — energy
    "9910.TW",  # Feng Tay — consumer/footwear
    "2912.TW",  # President Chain Store (7-Eleven) — consumer staples
    "1101.TW",  # Taiwan Cement — materials
    "2603.TW",  # Evergreen Marine — shipping
    "2609.TW",  # Yang Ming Marine — shipping
    "2615.TW",  # Wan Hai Lines — shipping
    "5871.TW",  # Chailease Holding — financials
    "6669.TW",  # Wiwynn — electronics
    "3661.TW",  # Alchip — semiconductors
    "2376.TW",  # Gigabyte — electronics
    "2356.TW",  # Inventec — electronics
    "2105.TW",  # Cheng Shin Rubber (Maxxis) — autos/tyres
]

assert len(TWSE_TOP50) == 50, f"expected 50 tickers, got {len(TWSE_TOP50)}"

# Coarse GICS-style sector buckets, used from Stage 2 onward for the `sector`
# column and (Stage 5) sector-neutralization. Deliberately coarse: with only 50
# names, finer buckets would leave singleton groups that can't be neutralized.
SECTORS = {
    "2330.TW": "Semiconductors", "2454.TW": "Semiconductors", "2303.TW": "Semiconductors",
    "3711.TW": "Semiconductors", "2379.TW": "Semiconductors", "3034.TW": "Semiconductors",
    "3661.TW": "Semiconductors",
    "2317.TW": "Electronics", "2308.TW": "Electronics", "2382.TW": "Electronics",
    "3008.TW": "Electronics", "3037.TW": "Electronics", "2357.TW": "Electronics",
    "2395.TW": "Electronics", "3231.TW": "Electronics", "2474.TW": "Electronics",
    "2327.TW": "Electronics", "2376.TW": "Electronics", "2356.TW": "Electronics",
    "6669.TW": "Electronics",
    "2891.TW": "Financials", "2881.TW": "Financials", "2882.TW": "Financials",
    "2886.TW": "Financials", "2884.TW": "Financials", "2885.TW": "Financials",
    "2892.TW": "Financials", "2880.TW": "Financials", "2890.TW": "Financials",
    "2883.TW": "Financials", "2887.TW": "Financials", "5880.TW": "Financials",
    "5871.TW": "Financials",
    "2412.TW": "Telecom", "3045.TW": "Telecom", "4904.TW": "Telecom",
    "1303.TW": "Materials", "1301.TW": "Materials", "2002.TW": "Materials",
    "1326.TW": "Materials", "1101.TW": "Materials",
    "2207.TW": "Autos", "2105.TW": "Autos",
    "1216.TW": "ConsumerStaples", "9910.TW": "ConsumerStaples", "2912.TW": "ConsumerStaples",
    "6505.TW": "Energy",
    "2603.TW": "Shipping", "2609.TW": "Shipping", "2615.TW": "Shipping",
}

assert set(SECTORS) == set(TWSE_TOP50), "SECTORS must cover exactly the universe"
