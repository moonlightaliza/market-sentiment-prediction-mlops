
# ════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & CONFIGURATION
# ════════════════════════════════════════════════════════════

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ── Data Handling ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Visualisation (optional — for quick sanity checks) ────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️  matplotlib not found — skipping plots (install with: pip install matplotlib)")

print(f"✅ pandas  {pd.__version__}")
print(f"✅ numpy   {np.__version__}")
print("✅ All imports successful")



# ════════════════════════════════════════════════════════════
# SECTION 2 — PROJECT CONFIGURATION
# ════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  ALL CONFIGURABLE SETTINGS LIVE HERE — change once, affects entire notebook
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Input file paths
PRICE_CSV          = os.path.join(BASE_DIR, 'data', 'raw', 'yahoo_prices.csv')
NEWS_CSV           = os.path.join(BASE_DIR, 'data', 'sentiment_labeled', 'news_sentiment.csv')
REDDIT_CSV         = os.path.join(BASE_DIR, 'data', 'sentiment_labeled', 'reddit_sentiment.csv')
TWITTER_CSV        = os.path.join(BASE_DIR, 'data', 'sentiment_labeled', 'twitter_sentiment.csv')


# Output paths
OUTPUT_DIR         = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_CSV         = os.path.join(OUTPUT_DIR, 'timeseries_dataset.csv')
FEATURE_JSON       = os.path.join(OUTPUT_DIR, 'feature_columns.json')

# Time-series parameters
RESAMPLE_FREQ      = '15min'    # 15-minute windows (pandas >= 2.2 prefers '15min' over '15T')
SENTIMENT_LAG_PERIODS = [1, 2, 3]  # how many lags to create
ROLLING_WINDOW     = 3          # periods for moving averages

# Sentiment label → numeric mapping (Mobeen's labels)
SENTIMENT_MAP = {'positive': 1, 'neutral': 0, 'negative': -1}

# Classification threshold (future price change > threshold → UP)
TARGET_THRESHOLD   = 0.0

# ── Create output directory if it does not exist ──────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📂 Project Configuration")
print("=" * 50)
print(f"  Price data     : {PRICE_CSV}")
print(f"  News sentiment : {NEWS_CSV}")
print(f"  Reddit sent.   : {REDDIT_CSV}")
print(f"  Twitter sent.  : {TWITTER_CSV}")
print(f"  Output CSV     : {OUTPUT_CSV}")
print(f"  Resample freq  : {RESAMPLE_FREQ}")
print(f"  Output dir ✅  : {OUTPUT_DIR}")



# ════════════════════════════════════════════════════════════
# SECTION 3 — HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  Reusable helper functions
#  These will also be called later by the FastAPI real-time function
# ─────────────────────────────────────────────────────────────────────────────

def safe_load_csv(filepath: str, required_cols: list = None) -> pd.DataFrame:
    """
    Safely load a CSV file with error handling.
    Returns an empty DataFrame (with required_cols) if the file is missing or empty.
    """
    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found: {filepath} — skipping")
        return pd.DataFrame(columns=required_cols) if required_cols else pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"  ⚠️  Empty file: {filepath} — skipping")
            return pd.DataFrame(columns=required_cols) if required_cols else pd.DataFrame()
        print(f"  ✅ Loaded {filepath:<45} → {len(df):>6,} rows, {df.shape[1]} cols")
        return df
    except Exception as e:
        print(f"  ❌ Error loading {filepath}: {e}")
        return pd.DataFrame(columns=required_cols) if required_cols else pd.DataFrame()


def normalise_timestamp(df: pd.DataFrame, col: str = 'timestamp') -> pd.DataFrame:
    """
    Parse timestamp column to UTC-aware datetime and set as index.
    Handles mixed formats gracefully.
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    dropped = df[col].isna().sum()
    if dropped > 0:
        print(f"  ⚠️  Dropped {dropped} rows with unparseable timestamps")
    df = df.dropna(subset=[col])
    df = df.set_index(col).sort_index()
    return df


def map_sentiment_labels(df: pd.DataFrame,
                         label_col: str = 'sentiment',
                         score_col: str = 'vader_compound') -> pd.Series:
    """
    Returns a unified numeric sentiment score Series.

    Priority:
      1. If vader_compound exists and is numeric → use it directly
      2. If sentiment label column exists → map positive/neutral/negative → +1/0/-1
      3. Otherwise return 0 (neutral)
    """
    # Try numeric score column first (Mobeen's VADER compound scores)
    if score_col in df.columns:
        numeric = pd.to_numeric(df[score_col], errors='coerce')
        if numeric.notna().sum() > 0:
            return numeric.fillna(0.0)

    # Fall back to label mapping
    if label_col in df.columns:
        mapped = df[label_col].str.strip().str.lower().map(SENTIMENT_MAP)
        return mapped.fillna(0).astype(float)

    # Default
    return pd.Series(0.0, index=df.index)


def resample_sentiment(df: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    Resample sentiment data to fixed-frequency windows.
    Returns avg_sentiment and sentiment_count per window.
    """
    if df.empty:
        return pd.DataFrame(columns=['avg_sentiment', 'sentiment_count'])

    resampled = df['sentiment_score'].resample(freq).agg(
        avg_sentiment='mean',
        sentiment_count='count'
    )
    # Drop windows with zero posts
    resampled = resampled[resampled['sentiment_count'] > 0]
    return resampled


def resample_prices(df: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    Resample OHLCV price data to fixed-frequency windows.
    """
    agg_dict = {
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum'
    }
    # Only aggregate columns that actually exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    resampled = df.groupby('ticker').resample(freq).agg(agg_dict)
    resampled = resampled.reset_index(level=0)   # bring ticker back as column
    resampled = resampled.dropna(subset=['close'])
    return resampled


print("✅ Helper functions defined:")
print("   • safe_load_csv()       — load CSV with error handling")
print("   • normalise_timestamp() — parse & set datetime index")
print("   • map_sentiment_labels()— numeric score from label or compound")
print("   • resample_sentiment()  — 15-min sentiment aggregation")
print("   • resample_prices()     — 15-min OHLCV aggregation")



# ════════════════════════════════════════════════════════════
# SECTION 4 — LOAD & COMBINE ALL SENTIMENT SOURCES
# ════════════════════════════════════════════════════════════

print("─" * 60)
print("STEP 1 — Loading sentiment data (labeled by Mobeen)")
print("─" * 60)

sentiment_frames = []

# ── 1a. News Sentiment ────────────────────────────────────────────────────────
print("\n📰 News Sentiment:")
news_df = safe_load_csv(NEWS_CSV)

if not news_df.empty:
    news_df = normalise_timestamp(news_df)
    news_df['sentiment_score'] = map_sentiment_labels(news_df)
    news_df['source_type'] = 'news'
    sentiment_frames.append(news_df[['sentiment_score', 'source_type']])
    print(f"     Date range: {news_df.index.min()} → {news_df.index.max()}")

# ── 1b. Reddit Sentiment ──────────────────────────────────────────────────────
print("\n👾 Reddit Sentiment:")
reddit_df = safe_load_csv(REDDIT_CSV)

if not reddit_df.empty:
    reddit_df = normalise_timestamp(reddit_df)
    reddit_df['sentiment_score'] = map_sentiment_labels(reddit_df)
    reddit_df['source_type'] = 'reddit'
    sentiment_frames.append(reddit_df[['sentiment_score', 'source_type']])
    print(f"     Date range: {reddit_df.index.min()} → {reddit_df.index.max()}")

# ── 1c. Twitter Sentiment ─────────────────────────────────────────────────────
print("\n🐦 Twitter Sentiment:")
twitter_df = safe_load_csv(TWITTER_CSV)

if not twitter_df.empty:
    twitter_df = normalise_timestamp(twitter_df)
    twitter_df['sentiment_score'] = map_sentiment_labels(twitter_df)
    twitter_df['source_type'] = 'twitter'
    sentiment_frames.append(twitter_df[['sentiment_score', 'source_type']])
    print(f"     Date range: {twitter_df.index.min()} → {twitter_df.index.max()}")

# ── 1d. Combine all sources ───────────────────────────────────────────────────
if not sentiment_frames:
    raise RuntimeError("❌ No sentiment files could be loaded. Check paths in Cell 2.")

all_sentiment = pd.concat(sentiment_frames, axis=0).sort_index()

print(f"\n{'─'*60}")
print(f"✅ Combined sentiment rows : {len(all_sentiment):,}")
print(f"   Sources loaded          : {all_sentiment['source_type'].unique().tolist()}")
print(f"   Score range             : [{all_sentiment['sentiment_score'].min():.3f}, {all_sentiment['sentiment_score'].max():.3f}]")
print(f"   Overall date range      : {all_sentiment.index.min()} → {all_sentiment.index.max()}")

# Quick breakdown by source
print("\n   Source breakdown:")
print(all_sentiment['source_type'].value_counts().to_string(header=False))



# ════════════════════════════════════════════════════════════
# SECTION 5 — RESAMPLE SENTIMENT TO 15-MIN WINDOWS
# ════════════════════════════════════════════════════════════

print("─" * 60)
print("STEP 2 — Resample sentiment to 15-minute windows")
print("─" * 60)

# Resample: mean sentiment score + post count per 15-min window
sentiment_resampled = all_sentiment['sentiment_score'].resample(RESAMPLE_FREQ).agg(
    avg_sentiment  = 'mean',
    sentiment_count= 'count'
)

# Remove windows with no posts (count == 0 means no data fell in that bin)
sentiment_resampled = sentiment_resampled[sentiment_resampled['sentiment_count'] > 0]

print(f"\n  Windows with sentiment data : {len(sentiment_resampled):,}")
print(f"  Avg sentiment score range   : [{sentiment_resampled['avg_sentiment'].min():.4f}, "
      f"{sentiment_resampled['avg_sentiment'].max():.4f}]")
print(f"  Mean posts per window       : {sentiment_resampled['sentiment_count'].mean():.1f}")
print(f"  Max posts in single window  : {sentiment_resampled['sentiment_count'].max()}")

print("\n📋 Sample of resampled sentiment (first 5 rows):")
print(sentiment_resampled.head())



# ════════════════════════════════════════════════════════════
# SECTION 6 — LOAD & RESAMPLE PRICE DATA (YAHOO FINANCE)
# ════════════════════════════════════════════════════════════

print("─" * 60)
print("STEP 3 — Loading Yahoo Finance price data")
print("─" * 60)

price_df = safe_load_csv(PRICE_CSV, required_cols=['timestamp','ticker','open','high','low','close','volume'])

if price_df.empty:
    raise RuntimeError("❌ Price data is empty or missing. Cannot build time-series without prices.")

# ── Normalise column names ────────────────────────────────────────────────────
price_df.columns = price_df.columns.str.strip().str.lower()

# ── Parse timestamp ───────────────────────────────────────────────────────────
price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True, errors='coerce')
n_bad_ts = price_df['timestamp'].isna().sum()
if n_bad_ts > 0:
    print(f"  ⚠️  Dropped {n_bad_ts} rows with bad timestamps")
    price_df = price_df.dropna(subset=['timestamp'])

price_df = price_df.set_index('timestamp').sort_index()

# ── Cast OHLCV columns to numeric ─────────────────────────────────────────────
for col in ['open','high','low','close','volume']:
    if col in price_df.columns:
        price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

tickers = price_df['ticker'].unique()
print(f"\n  Tickers found     : {tickers.tolist()}")
print(f"  Total price rows  : {len(price_df):,}")
print(f"  Date range        : {price_df.index.min()} → {price_df.index.max()}")
print(f"  Close price range : [{price_df['close'].min():.2f}, {price_df['close'].max():.2f}]")

print("\n📋 Price data sample (first 5 rows):")
print(price_df.head())



# ════════════════════════════════════════════════════════════
# SECTION 7 — RESAMPLE PRICES TO 15-MIN OHLCV WINDOWS
# ════════════════════════════════════════════════════════════

print("─" * 60)
print("STEP 4 — Resample price data to 15-minute OHLCV windows")
print("─" * 60)

# ── If multiple tickers exist, process per ticker then stack ──────────────────
ticker_frames = []

for ticker in price_df['ticker'].unique():
    ticker_data = price_df[price_df['ticker'] == ticker].drop(columns='ticker')

    resampled_ticker = ticker_data.resample(RESAMPLE_FREQ).agg(
        open   = ('open',   'first'),
        high   = ('high',   'max'),
        low    = ('low',    'min'),
        close  = ('close',  'last'),
        volume = ('volume', 'sum')
    )

    # Drop windows where close price is missing (no trades in that window)
    resampled_ticker = resampled_ticker.dropna(subset=['close'])
    resampled_ticker['ticker'] = ticker
    ticker_frames.append(resampled_ticker)
    print(f"  [{ticker}] → {len(resampled_ticker):,} 15-min windows")

price_resampled = pd.concat(ticker_frames, axis=0).sort_index()

print(f"\n  Total price windows after resampling : {len(price_resampled):,}")
print(f"  Date range                           : {price_resampled.index.min()} → {price_resampled.index.max()}")

print("\n📋 Resampled OHLCV sample (first 5 rows):")
print(price_resampled.head())



# ════════════════════════════════════════════════════════════
# SECTION 8 — MERGE PRICE + SENTIMENT ON TIMESTAMP
# ════════════════════════════════════════════════════════════

print("─" * 60)
print("STEP 5 — Merge price and sentiment on timestamp")
print("─" * 60)

# ── Merge strategy: LEFT join on price windows ────────────────────────────────
#   Every price window is kept; sentiment is NaN if no posts in that window.
#   This avoids losing price data just because sentiment was sparse.

# For multi-ticker datasets, process each ticker separately so sentiment
# is correctly aligned with each ticker's price windows.

merged_frames = []

for ticker in price_resampled['ticker'].unique():
    ticker_prices = price_resampled[price_resampled['ticker'] == ticker].drop(columns='ticker')

    # LEFT join: keep all price windows, fill missing sentiment forward then with 0
    merged = ticker_prices.join(sentiment_resampled, how='left')

    # Forward-fill sentiment gaps (up to 3 periods = 45 min) then fill remaining with 0
    merged['avg_sentiment']   = merged['avg_sentiment'].fillna(method='ffill', limit=3).fillna(0.0)
    merged['sentiment_count'] = merged['sentiment_count'].fillna(0).astype(int)

    merged['ticker'] = ticker
    merged_frames.append(merged)
    print(f"  [{ticker}] merged → {len(merged):,} rows  (sentiment coverage: "
          f"{(merged['sentiment_count'] > 0).mean()*100:.1f}% of windows)")

df = pd.concat(merged_frames, axis=0).sort_index()

print(f"\n  ✅ Merged dataset shape : {df.shape}")
print(f"     Columns              : {df.columns.tolist()}")
print(f"     Date range           : {df.index.min()} → {df.index.max()}")

print("\n📋 Merged data sample (first 5 rows):")
print(df.head())



# ════════════════════════════════════════════════════════════
# SECTION 9 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════

print("─" * 60)
print("STEP 6 — Feature Engineering")
print("─" * 60)

# Work per-ticker so lag/rolling features don't bleed across tickers
feature_frames = []

for ticker in df['ticker'].unique():
    t = df[df['ticker'] == ticker].copy()
    print(f"\n  Building features for [{ticker}] — {len(t):,} rows")

    # ── 6a. Price percentage change ───────────────────────────────────────────
    #   How much did the close price move from the previous 15-min window?
    t['price_change_pct'] = t['close'].pct_change() * 100
    print(f"     ✅ price_change_pct          (range: {t['price_change_pct'].min():.2f}% to {t['price_change_pct'].max():.2f}%)")

    # ── 6b. Volume percentage change ─────────────────────────────────────────
    t['volume_change_pct'] = t['volume'].pct_change() * 100
    # Cap extreme volume spikes (e.g. first bar of day) at ±500%
    t['volume_change_pct'] = t['volume_change_pct'].clip(-500, 500)
    print(f"     ✅ volume_change_pct         (clipped to ±500%)")

    # ── 6c. Sentiment lag features ────────────────────────────────────────────
    #   Did sentiment from 15/30/45 minutes ago predict what prices do now?
    for lag in SENTIMENT_LAG_PERIODS:
        col = f'sentiment_lag_{lag}'
        t[col] = t['avg_sentiment'].shift(lag)
        print(f"     ✅ {col:<28}(lag {lag} × 15-min)")

    # ── 6d. Sentiment rolling mean ────────────────────────────────────────────
    #   Smooth out noisy sentiment with a 3-period (45-min) rolling average
    t['sentiment_ma_3'] = (
        t['avg_sentiment']
        .rolling(window=ROLLING_WINDOW, min_periods=1)
        .mean()
    )
    print(f"     ✅ sentiment_ma_3            ({ROLLING_WINDOW}-period rolling avg)")

    # ── 6e. Volume rolling mean ───────────────────────────────────────────────
    #   Baseline volume level — is current volume above or below trend?
    t['volume_ma_3'] = (
        t['volume']
        .rolling(window=ROLLING_WINDOW, min_periods=1)
        .mean()
    )
    print(f"     ✅ volume_ma_3               ({ROLLING_WINDOW}-period rolling avg)")

    # ── 6f. Price relative to moving average (bonus feature) ──────────────────
    t['price_ma_3'] = t['close'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    t['price_vs_ma'] = (t['close'] - t['price_ma_3']) / t['price_ma_3'] * 100
    print(f"     ✅ price_vs_ma               (close vs 3-period MA, in %)")

    # ── 6g. High-Low spread (intra-bar volatility) ────────────────────────────
    t['hl_spread_pct'] = (t['high'] - t['low']) / t['low'] * 100
    print(f"     ✅ hl_spread_pct             (intra-bar price range %)")

    feature_frames.append(t)

df_features = pd.concat(feature_frames, axis=0).sort_index()

print(f"\n✅ Feature engineering complete — shape: {df_features.shape}")



# ════════════════════════════════════════════════════════════
# SECTION 10 — CREATE TARGET VARIABLE (BINARY CLASSIFICATION)
# ════════════════════════════════════════════════════════════

print("─" * 60)
print("STEP 7 — Create target variable")
print("─" * 60)

target_frames = []

for ticker in df_features['ticker'].unique():
    t = df_features[df_features['ticker'] == ticker].copy()

    # future_price_change = (next window's close / current close) - 1
    # shift(-1) gives us the NEXT row's close price
    t['future_close']        = t['close'].shift(-1)
    t['future_price_change'] = (t['future_close'] / t['close']) - 1

    # Binary target: 1 = price goes UP, 0 = price stays flat or goes DOWN
    t['target'] = (t['future_price_change'] > TARGET_THRESHOLD).astype(int)

    target_frames.append(t)
    print(f"  [{ticker}] — UP: {(t['target']==1).sum():,}  |  DOWN: {(t['target']==0).sum():,}")

df_features = pd.concat(target_frames, axis=0).sort_index()

# Drop the helper column (not needed in final dataset)
df_features = df_features.drop(columns=['future_close'])

print(f"\n  ✅ Target variable created")
print(f"     1 (UP)   : {(df_features['target']==1).sum():,} samples")
print(f"     0 (DOWN) : {(df_features['target']==0).sum():,} samples")
print(f"     Class balance: {(df_features['target']==1).mean()*100:.1f}% UP")



# ════════════════════════════════════════════════════════════
# SECTION 11 — CLEAN DATASET & DEFINE FINAL COLUMNS
# ════════════════════════════════════════════════════════════

print("─" * 60)
print("STEP 8 — Final column selection & NaN removal")
print("─" * 60)

# ── Define the exact output columns (as specified in requirements) ────────────
PRICE_COLS      = ['open', 'high', 'low', 'close', 'volume']
SENTIMENT_COLS  = ['avg_sentiment', 'sentiment_count']
FEATURE_COLS    = [
    'price_change_pct',
    'volume_change_pct',
    'sentiment_lag_1',
    'sentiment_lag_2',
    'sentiment_lag_3',
    'sentiment_ma_3',
    'volume_ma_3',
    'price_ma_3',         # bonus
    'price_vs_ma',        # bonus
    'hl_spread_pct',      # bonus
]
TARGET_COL      = ['future_price_change', 'target']

ALL_COLS = PRICE_COLS + SENTIMENT_COLS + FEATURE_COLS + TARGET_COL

# Keep only columns that actually exist in df_features
existing_cols = [c for c in ALL_COLS if c in df_features.columns]
missing_cols  = [c for c in ALL_COLS if c not in df_features.columns]

if missing_cols:
    print(f"  ⚠️  Columns not found (will skip): {missing_cols}")

df_final = df_features[existing_cols].copy()

print(f"  Rows before NaN drop : {len(df_final):,}")
df_final = df_final.dropna()
print(f"  Rows after  NaN drop : {len(df_final):,}")
print(f"  Rows dropped         : {len(df_features) - len(df_final):,}")

# ── The FEATURE columns list (for ML teammate reference) ──────────────────────
ML_FEATURE_COLS = [c for c in FEATURE_COLS + SENTIMENT_COLS + PRICE_COLS
                   if c in df_final.columns and c != 'target']

print(f"\n  Final feature columns ({len(ML_FEATURE_COLS)}):")
for i, col in enumerate(ML_FEATURE_COLS, 1):
    print(f"    {i:>2}. {col}")



# ════════════════════════════════════════════════════════════
# SECTION 12 — SAVE OUTPUTS
# ════════════════════════════════════════════════════════════

print("─" * 60)
print("STEP 9 — Saving outputs")
print("─" * 60)

# ── Reset index so timestamp appears as a column in the CSV ───────────────────
df_save = df_final.reset_index()          # brings 'timestamp' back as column
df_save = df_save.rename(columns={'index': 'timestamp'}) if 'index' in df_save.columns else df_save

# ── 9a. Save main time-series dataset ─────────────────────────────────────────
try:
    df_save.to_csv(OUTPUT_CSV, index=False)
    size_kb = os.path.getsize(OUTPUT_CSV) / 1024
    print(f"  ✅ Saved: {OUTPUT_CSV}")
    print(f"            {len(df_save):,} rows × {df_save.shape[1]} columns  ({size_kb:.1f} KB)")
except Exception as e:
    print(f"  ❌ Could not save CSV: {e}")
    raise

# ── 9b. Save feature column list for ML teammate ──────────────────────────────
feature_meta = {
    "feature_columns": ML_FEATURE_COLS,
    "target_column": "target",
    "target_description": "1 = price UP in next 15-min window, 0 = price DOWN or flat",
    "timestamp_column": "timestamp",
    "resample_frequency": RESAMPLE_FREQ,
    "sentiment_sources": ["news", "reddit", "twitter"],
    "n_samples": int(len(df_save)),
    "date_range": {
        "start": str(df_final.index.min()),
        "end":   str(df_final.index.max())
    },
    "feature_descriptions": {
        "open":              "First price in 15-min window",
        "high":              "Highest price in 15-min window",
        "low":               "Lowest price in 15-min window",
        "close":             "Last price in 15-min window",
        "volume":            "Total volume in 15-min window",
        "avg_sentiment":     "Mean VADER compound score across all sources per window",
        "sentiment_count":   "Number of posts/articles in that window",
        "price_change_pct":  "% change in close price from previous window",
        "volume_change_pct": "% change in volume from previous window (clipped ±500%)",
        "sentiment_lag_1":   "avg_sentiment from 1 window ago (15 min)",
        "sentiment_lag_2":   "avg_sentiment from 2 windows ago (30 min)",
        "sentiment_lag_3":   "avg_sentiment from 3 windows ago (45 min)",
        "sentiment_ma_3":    "3-period rolling average of avg_sentiment (45-min smoothed)",
        "volume_ma_3":       "3-period rolling average of volume",
        "price_ma_3":        "3-period rolling average of close price",
        "price_vs_ma":       "% deviation of close from its 3-period moving average",
        "hl_spread_pct":     "(high - low) / low × 100 — intra-bar volatility"
    }
}

try:
    with open(FEATURE_JSON, 'w') as f:
        json.dump(feature_meta, f, indent=2)
    size_kb = os.path.getsize(FEATURE_JSON) / 1024
    print(f"\n  ✅ Saved: {FEATURE_JSON}")
    print(f"            Feature metadata for ML teammate  ({size_kb:.1f} KB)")
except Exception as e:
    print(f"  ❌ Could not save feature JSON: {e}")



# ════════════════════════════════════════════════════════════
# SECTION 13 — DATASET SUMMARY REPORT
# ════════════════════════════════════════════════════════════

print("=" * 65)
print("  TIMESERIES DATASET SUMMARY REPORT")
print("  Financial Market Movement Prediction — Areesha's Pipeline")
print("=" * 65)

# ── Total samples ─────────────────────────────────────────────────────────────
print(f"\n📊 DATASET OVERVIEW")
print(f"   Total samples       : {len(df_final):,}")
print(f"   Total columns       : {df_final.shape[1]}")
print(f"   Memory usage        : {df_final.memory_usage(deep=True).sum() / 1024:.1f} KB")

# ── Date range ────────────────────────────────────────────────────────────────
print(f"\n📅 DATE RANGE")
print(f"   Start               : {df_final.index.min()}")
print(f"   End                 : {df_final.index.max()}")
span = df_final.index.max() - df_final.index.min()
print(f"   Total span          : {span.days} days")

# ── Feature columns ───────────────────────────────────────────────────────────
print(f"\n🔧 FEATURE COLUMNS ({len(ML_FEATURE_COLS)} total)")
for i, col in enumerate(ML_FEATURE_COLS, 1):
    print(f"   {i:>2}. {col:<28}  mean={df_final[col].mean():>9.4f}  std={df_final[col].std():>9.4f}")

# ── Target distribution ───────────────────────────────────────────────────────
up_count   = (df_final['target'] == 1).sum()
down_count = (df_final['target'] == 0).sum()
print(f"\n🎯 TARGET DISTRIBUTION")
print(f"   UP   (target=1)     : {up_count:,}  ({up_count/len(df_final)*100:.1f}%)")
print(f"   DOWN (target=0)     : {down_count:,}  ({down_count/len(df_final)*100:.1f}%)")
imbalance = abs(up_count - down_count) / len(df_final) * 100
if imbalance > 10:
    print(f"   ⚠️  Class imbalance: {imbalance:.1f}% — consider SMOTE or class_weight in training")
else:
    print(f"   ✅ Classes are reasonably balanced ({imbalance:.1f}% difference)")

# ── Sentiment summary ─────────────────────────────────────────────────────────
print(f"\n💬 SENTIMENT SUMMARY")
print(f"   Avg sentiment score : {df_final['avg_sentiment'].mean():.4f}")
print(f"   Windows with posts  : {(df_final['sentiment_count'] > 0).sum():,} / {len(df_final):,}")
print(f"   Avg posts/window    : {df_final['sentiment_count'].mean():.1f}")

# ── NaN check ─────────────────────────────────────────────────────────────────
nan_counts = df_final.isna().sum()
print(f"\n🔍 DATA QUALITY")
if nan_counts.sum() == 0:
    print(f"   ✅ No NaN values — dataset is clean")
else:
    print(f"   ⚠️  NaN values found:")
    print(nan_counts[nan_counts > 0])

print(f"\n💾 OUTPUT FILES")
print(f"   Main dataset        : {OUTPUT_CSV}")
print(f"   Feature metadata    : {FEATURE_JSON}")
print()
print("=" * 65)
print("  ✅ Pipeline complete! Hand off timeseries_dataset.csv to ML teammate")
print("=" * 65)



# ════════════════════════════════════════════════════════════
# SECTION 14 — PREVIEW FIRST & LAST ROWS
# ════════════════════════════════════════════════════════════

print("FIRST 5 ROWS:")
print("=" * 80)
print(df_final.head())

print("\nLAST 5 ROWS:")
print("=" * 80)
print(df_final.tail())

print("\nDESCRIPTIVE STATISTICS (feature columns):")
print("=" * 80)
print(df_final[ML_FEATURE_COLS].describe().round(4))



# ════════════════════════════════════════════════════════════
# SECTION 15 — VISUALISATION (SANITY CHECKS)
# ════════════════════════════════════════════════════════════

if not PLOT_AVAILABLE:
    print("⚠️  matplotlib not installed — run: pip install matplotlib")
else:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Time-Series Dataset — Sanity Check Plots', fontsize=14, fontweight='bold')

    # Plot for the first ticker only
    first_ticker = df_features['ticker'].iloc[0] if 'ticker' in df_features.columns else None
    plot_df = df_final  # already filtered if single ticker

    # ── Plot 1: Close price ───────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(plot_df.index, plot_df['close'], linewidth=0.8, color='steelblue')
    ax.set_title('Close Price (15-min)')
    ax.set_ylabel('Price')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ── Plot 2: Sentiment over time ───────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(plot_df.index, plot_df['avg_sentiment'], linewidth=0.8,
            color='orange', alpha=0.7, label='avg_sentiment')
    ax.plot(plot_df.index, plot_df['sentiment_ma_3'], linewidth=1.2,
            color='red', label='sentiment_ma_3')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('Sentiment Score Over Time')
    ax.set_ylabel('Sentiment Score')
    ax.legend(fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ── Plot 3: Volume ────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.bar(plot_df.index, plot_df['volume'], width=0.01,
           color='teal', alpha=0.6, label='volume')
    ax.plot(plot_df.index, plot_df['volume_ma_3'],
            color='darkred', linewidth=1.2, label='volume_ma_3')
    ax.set_title('Volume (15-min windows)')
    ax.set_ylabel('Volume')
    ax.legend(fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ── Plot 4: Price change % ────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.hist(plot_df['price_change_pct'].dropna(), bins=50,
            color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_title('Distribution of price_change_pct')
    ax.set_xlabel('%')
    ax.set_ylabel('Count')

    # ── Plot 5: Target distribution ───────────────────────────────────────────
    ax = axes[2, 0]
    target_counts = plot_df['target'].value_counts().sort_index()
    bars = ax.bar(['DOWN (0)', 'UP (1)'], target_counts.values,
                  color=['salmon', 'mediumseagreen'], edgecolor='white')
    for bar, val in zip(bars, target_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:,}', ha='center', va='bottom', fontsize=10)
    ax.set_title('Target Class Distribution')
    ax.set_ylabel('Count')

    # ── Plot 6: Sentiment vs future price change ──────────────────────────────
    ax = axes[2, 1]
    scatter_df = plot_df[['avg_sentiment', 'future_price_change']].dropna().sample(
        min(500, len(plot_df)), random_state=42)
    ax.scatter(scatter_df['avg_sentiment'], scatter_df['future_price_change'],
               alpha=0.3, s=8, color='purple')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('Sentiment vs Future Price Change')
    ax.set_xlabel('avg_sentiment')
    ax.set_ylabel('future_price_change')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/timeseries_sanity_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Plots saved to {OUTPUT_DIR}/timeseries_sanity_plots.png")



# ════════════════════════════════════════════════════════════
# SECTION 16 — FASTAPI-READY REAL-TIME FUNCTION
# ════════════════════════════════════════════════════════════

def build_realtime_features(
    price_records:   list,   # list of dicts: {timestamp, ticker, open, high, low, close, volume}
    news_records:    list,   # list of dicts: {timestamp, vader_compound, sentiment}
    reddit_records:  list,   # list of dicts: {timestamp, vader_compound, sentiment}
    twitter_records: list,   # list of dicts: {timestamp, vader_compound, sentiment}
    resample_freq:   str  = RESAMPLE_FREQ,
    rolling_window:  int  = ROLLING_WINDOW,
    lag_periods:     list = SENTIMENT_LAG_PERIODS,
) -> pd.DataFrame:
    """
    Build a feature-engineered DataFrame from raw real-time records.

    Called by FastAPI at inference time with fresh live data.
    Applies the same transformations used during training so the model
    receives identically-shaped features.

    Parameters
    ----------
    price_records   : OHLCV rows (ideally last ~60 min so rolling/lag features are populated)
    news_records    : Scraped news sentiment rows
    reddit_records  : Reddit sentiment rows
    twitter_records : Twitter sentiment rows
    resample_freq   : Pandas offset alias (default '15min')
    rolling_window  : Periods for MA features (default 3)
    lag_periods     : Which lags to compute (default [1, 2, 3])

    Returns
    -------
    pd.DataFrame with all feature columns ready for model.predict()
    """

    # ── 1. Build sentiment DataFrame ─────────────────────────────────────────
    sent_frames = []
    for records in [news_records, reddit_records, twitter_records]:
        if not records:
            continue
        s = pd.DataFrame(records)
        if s.empty:
            continue
        s['timestamp'] = pd.to_datetime(s['timestamp'], utc=True, errors='coerce')
        s = s.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
        s['sentiment_score'] = map_sentiment_labels(s)
        sent_frames.append(s[['sentiment_score']])

    if sent_frames:
        all_sent = pd.concat(sent_frames).sort_index()
        sentiment_rt = all_sent['sentiment_score'].resample(resample_freq).agg(
            avg_sentiment='mean', sentiment_count='count'
        )
        sentiment_rt = sentiment_rt[sentiment_rt['sentiment_count'] > 0]
    else:
        # No sentiment available — create empty placeholder
        sentiment_rt = pd.DataFrame(columns=['avg_sentiment', 'sentiment_count'])

    # ── 2. Build price DataFrame ──────────────────────────────────────────────
    if not price_records:
        raise ValueError("price_records cannot be empty")

    p = pd.DataFrame(price_records)
    p['timestamp'] = pd.to_datetime(p['timestamp'], utc=True, errors='coerce')
    p = p.dropna(subset=['timestamp']).set_index('timestamp').sort_index()

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors='coerce')

    price_rt = p.resample(resample_freq).agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'),   close=('close', 'last'),
        volume=('volume', 'sum')
    ).dropna(subset=['close'])

    # ── 3. Merge ──────────────────────────────────────────────────────────────
    rt = price_rt.join(sentiment_rt, how='left')
    rt['avg_sentiment']   = rt['avg_sentiment'].fillna(method='ffill', limit=3).fillna(0.0)
    rt['sentiment_count'] = rt['sentiment_count'].fillna(0).astype(int)

    # ── 4. Features ───────────────────────────────────────────────────────────
    rt['price_change_pct']  = rt['close'].pct_change() * 100
    rt['volume_change_pct'] = rt['volume'].pct_change().clip(-5, 5) * 100

    for lag in lag_periods:
        rt[f'sentiment_lag_{lag}'] = rt['avg_sentiment'].shift(lag)

    rt['sentiment_ma_3'] = rt['avg_sentiment'].rolling(rolling_window, min_periods=1).mean()
    rt['volume_ma_3']    = rt['volume'].rolling(rolling_window, min_periods=1).mean()
    rt['price_ma_3']     = rt['close'].rolling(rolling_window, min_periods=1).mean()
    rt['price_vs_ma']    = (rt['close'] - rt['price_ma_3']) / rt['price_ma_3'] * 100
    rt['hl_spread_pct']  = (rt['high'] - rt['low']) / rt['low'] * 100

    # ── 5. Return cleaned, feature-ready rows ─────────────────────────────────
    rt = rt.dropna()
    return rt


# ── Quick smoke test ──────────────────────────────────────────────────────────
print("🧪 Smoke-testing build_realtime_features()...")

import random
from datetime import datetime, timedelta, timezone

base_ts  = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
base_price = 150.0

dummy_prices = []
for i in range(60):     # 60 one-minute ticks → covers 4 full 15-min windows
    base_price += random.uniform(-0.5, 0.5)
    dummy_prices.append({
        'timestamp': (base_ts + timedelta(minutes=i)).isoformat(),
        'ticker': 'AAPL',
        'open':   round(base_price - 0.1, 4),
        'high':   round(base_price + 0.3, 4),
        'low':    round(base_price - 0.3, 4),
        'close':  round(base_price, 4),
        'volume': random.randint(1000, 5000)
    })

dummy_news = [
    {'timestamp': (base_ts + timedelta(minutes=i*10)).isoformat(),
     'vader_compound': random.uniform(-1, 1), 'sentiment': 'positive'}
    for i in range(6)
]

try:
    rt_features = build_realtime_features(
        price_records=dummy_prices,
        news_records=dummy_news,
        reddit_records=[],
        twitter_records=[]
    )
    print(f"  ✅ Returned {len(rt_features)} rows × {rt_features.shape[1]} features")
    print(f"  ✅ Columns: {rt_features.columns.tolist()}")
    print(rt_features)
except Exception as e:
    print(f"  ❌ Smoke test failed: {e}")



# ════════════════════════════════════════════════════════════
# SECTION 17 — FASTAPI APP SKELETON (SAVE AS main.py)
# ════════════════════════════════════════════════════════════

fastapi_code = '''
"""
main.py — FastAPI app for real-time market movement prediction
Financial Market Movement Prediction Project
Areesha — Time-Series Construction & API Layer
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import json, joblib, os

# ── Import feature builder from this notebook (export as .py) ─────────────────
# from timeseries_construction import build_realtime_features, ML_FEATURE_COLS

app = FastAPI(
    title="Market Movement Prediction API",
    description="Predicts 15-min price direction from price + sentiment signals",
    version="1.0.0"
)

# ── Load trained model and feature list on startup ────────────────────────────
@app.on_event("startup")
async def load_model():
    global MODEL, FEATURE_COLS
    try:
        MODEL = joblib.load("model/trained_model.pkl")
        with open("dataset/processed/feature_columns.json") as f:
            meta = json.load(f)
        FEATURE_COLS = meta["feature_columns"]
        print(f"✅ Model loaded | {len(FEATURE_COLS)} features")
    except FileNotFoundError:
        print("⚠️  Model not found — /predict will return 503 until model is trained")
        MODEL = None
        FEATURE_COLS = []


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PriceRecord(BaseModel):
    timestamp: str
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class SentimentRecord(BaseModel):
    timestamp: str
    vader_compound: Optional[float] = 0.0
    sentiment: Optional[str] = "neutral"

class PredictionRequest(BaseModel):
    price_data:   List[PriceRecord]
    news_data:    Optional[List[SentimentRecord]] = []
    reddit_data:  Optional[List[SentimentRecord]] = []
    twitter_data: Optional[List[SentimentRecord]] = []

class PredictionResponse(BaseModel):
    prediction:        int          # 1 = UP, 0 = DOWN
    probability_up:    float
    probability_down:  float
    n_windows:         int
    latest_timestamp:  str


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


# ── Main prediction endpoint ──────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(503, detail="Model not loaded. Train the model first.")

    try:
        # Build feature matrix using the same pipeline as training
        features = build_realtime_features(
            price_records   = [r.dict() for r in request.price_data],
            news_records    = [r.dict() for r in request.news_data],
            reddit_records  = [r.dict() for r in request.reddit_data],
            twitter_records = [r.dict() for r in request.twitter_data],
        )

        if features.empty:
            raise HTTPException(422, detail="Could not build features — check input data")

        # Use the most recent complete window for prediction
        X = features[FEATURE_COLS].iloc[[-1]].values
        proba = MODEL.predict_proba(X)[0]          # [P(down), P(up)]
        pred  = int(MODEL.predict(X)[0])

        return PredictionResponse(
            prediction       = pred,
            probability_up   = round(float(proba[1]), 4),
            probability_down = round(float(proba[0]), 4),
            n_windows        = len(features),
            latest_timestamp = str(features.index[-1])
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction error: {str(e)}")


# ── Run locally ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

# Save main.py alongside this notebook
with open('main.py', 'w',encoding='utf-8') as f:
    f.write(fastapi_code.strip())

print("✅ main.py saved in current directory")
print()
print("To run the API:")
print("  pip install fastapi uvicorn joblib")
print("  uvicorn main:app --reload --port 8000")
print()
print("API Endpoints:")
print("  GET  http://localhost:8000/health")
print("  POST http://localhost:8000/predict")
print("  GET  http://localhost:8000/docs     ← Swagger UI")



# ════════════════════════════════════════════════════════════
# SECTION 18 — FINAL CHECKLIST
# ════════════════════════════════════════════════════════════

print("=" * 60)
print("  FINAL CHECKLIST — AREESHA'S DELIVERABLES")
print("=" * 60)

checks = [
    (OUTPUT_CSV,                       "Main timeseries dataset CSV"),
    (FEATURE_JSON,                     "Feature column metadata JSON"),
    ('main.py',                        "FastAPI app skeleton"),
    (f'{OUTPUT_DIR}/timeseries_sanity_plots.png', "Sanity check plots (optional)"),
]

all_ok = True
for path, description in checks:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✅ {description:<40} ({size:.1f} KB)  → {path}")
    else:
        print(f"  ❌ MISSING: {description:<38}  → {path}")
        all_ok = False

print()
if all_ok:
    print("🎉 All deliverables ready! Hand off to ML teammate (model training).")
else:
    print("⚠️  Some files are missing — re-run the cells above.")

print()
print("📌 Tell your ML teammate:")
print(f"   • Dataset    : {OUTPUT_CSV}")
print(f"   • Features   : load feature_columns.json → key 'feature_columns'")
print(f"   • Target col : 'target'  (1=UP, 0=DOWN)")
print(f"   • Window     : {RESAMPLE_FREQ} resampled")
print(f"   • Rows       : {len(df_final):,}")
print(f"   • Class bal  : {(df_final['target']==1).mean()*100:.1f}% UP / {(df_final['target']==0).mean()*100:.1f}% DOWN")


if __name__ == '__main__':
    print('\n✅ timeseries_construction.py finished.')
