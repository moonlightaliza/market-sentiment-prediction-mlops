"""
main.py — FastAPI Inference Server
====================================
Financial Market Movement Prediction Project
Role   : API Layer (Areesha)
Model  : GRU_best.pt  (PyTorch GRU — config from src/models/config.py)
Input  : Real-time OHLCV price + multi-source sentiment records
Output : Binary prediction  →  1 = price UP, 0 = price DOWN

Project structure:
    market-sentiment-prediction-mlops/
    ├── api/
    │   └── main.py                        ← THIS FILE
    ├── data/
    │   └── processed/
    │       └── timeseries_dataset.csv
    ├── src/
    │   ├── models/
    │   │   ├── config.py                  ← single source of truth
    │   │   └── saved_models/
    │   │       └── GRU_best.pt
    │   └── timeseries/
    │       └── timeseries_construction.py

Run:
    cd market-sentiment-prediction-mlops
    uvicorn api.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

# ════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# PATHS
# api/main.py  ->  parents[0]=api/  ->  parents[1]=project root
# ════════════════════════════════════════════════════════════════

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
CONFIG_PATH     = PROJECT_ROOT / "src" / "models" / "config.py"
TIMESERIES_PATH = PROJECT_ROOT / "src" / "timeseries" / "timeseries_construction.py"
FRONTEND_PATH   = PROJECT_ROOT / "frontend"

# ════════════════════════════════════════════════════════════════
# LOAD CONFIG  — src/models/config.py is the single source of truth
# ════════════════════════════════════════════════════════════════

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.py not found at {CONFIG_PATH}\n"
            f"Expected: src/models/config.py inside project root {PROJECT_ROOT}"
        )
    spec   = importlib.util.spec_from_file_location("model_config", str(CONFIG_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


try:
    CONFIG = _load_config()
    log.info(f"Loaded CONFIG from {CONFIG_PATH}")
except Exception as e:
    log.error(f"Could not load config.py: {e}")
    CONFIG = {}

# ════════════════════════════════════════════════════════════════
# SETTINGS — all derived from CONFIG, nothing hardcoded
# ════════════════════════════════════════════════════════════════

MODEL_PATH: Path = (
    PROJECT_ROOT / CONFIG.get("model_save_path", "src/models/saved_models/") / "GRU_best.pt"
)

FEATURE_COLS: List[str] = CONFIG.get("feature_columns", [])
SEQUENCE_LENGTH: int    = CONFIG.get("sequence_length", 5)

GRU_CONFIG = {
    "input_size"  : CONFIG.get("input_size",   len(FEATURE_COLS)),
    "hidden_size" : CONFIG.get("hidden_size",  32),
    "num_layers"  : CONFIG.get("num_layers",   1),
    "dropout"     : CONFIG.get("dropout",      0.3),
    "output_size" : CONFIG.get("output_size",  1),
}

log.info(f"Feature columns : {len(FEATURE_COLS)} -> {FEATURE_COLS}")
log.info(f"Sequence length : {SEQUENCE_LENGTH} x 15-min = {SEQUENCE_LENGTH * 15} min")
log.info(f"GRU config      : {GRU_CONFIG}")
log.info(f"Model path      : {MODEL_PATH}")


# ════════════════════════════════════════════════════════════════
# GRU MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════

class GRUModel(nn.Module):
    def __init__(
        self,
        input_size  : int,
        hidden_size : int   = 32,
        num_layers  : int   = 1,
        dropout     : float = 0.3,
        output_size : int   = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


# ════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ════════════════════════════════════════════════════════════════

class AppState:
    model         : Optional[GRUModel] = None
    device        : torch.device       = torch.device("cpu")
    model_loaded  : bool               = False
    startup_errors: List[str]          = []

state = AppState()


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _import_feature_builder():
    spec   = importlib.util.spec_from_file_location(
        "timeseries_construction", str(TIMESERIES_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)

    if not hasattr(module, "build_realtime_features"):
        raise AttributeError(
            "build_realtime_features not found in timeseries_construction.py."
        )
    return module.build_realtime_features


def prepare_sequence(
    features_df  : pd.DataFrame,
    feature_cols : List[str],
    seq_len      : int,
) -> torch.Tensor:
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    arr = features_df[feature_cols].values.astype(np.float32)

    if len(arr) < seq_len:
        raise ValueError(
            f"Not enough 15-min windows. Need {seq_len}, got {len(arr)}."
        )

    arr    = arr[-seq_len:]
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    return tensor


# ════════════════════════════════════════════════════════════════
# ── NEW: FRONTEND-FACING SENTIMENT HELPERS ───────────────────
# These power the /api/predict endpoint consumed by index.html
# ════════════════════════════════════════════════════════════════

def _vader_score(text: str) -> float:
    """Return VADER compound score in [-1, 1]. Falls back to 0 on import error."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    except ImportError:
        log.warning("vaderSentiment not installed — falling back to keyword scorer")
        return _keyword_score(text)


def _keyword_score(text: str) -> float:
    """Simple keyword-based fallback sentiment score in [-1, 1]."""
    t = text.lower()
    pos = ["beat", "record", "growth", "strong", "rally", "surge",
           "profit", "gain", "cuts", "bullish", "up", "rise", "higher",
           "positive", "optimistic", "recovery", "boom", "breakthrough"]
    neg = ["decline", "fall", "loss", "crisis", "default", "uncertainty",
           "risk", "recession", "bearish", "down", "drop", "lower",
           "negative", "pessimistic", "selloff", "crash", "concern"]
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    total = p + n
    if total == 0:
        return 0.0
    return (p - n) / total


def _score_to_sentiment(score: float) -> dict:
    """
    Convert a compound score → sentiment label + breakdown dict.
    Returns the shape expected by the frontend renderResult().
    """
    abs_s = abs(score)

    # ── label ─────────────────────────────────────────────────
    if score >= 0.05:
        label = "bullish"
    elif score <= -0.05:
        label = "bearish"
    else:
        label = "neutral"

    # ── confidence based on |score| ───────────────────────────
    confidence = 0.55 + abs_s * 0.40            # maps [0,1] → [0.55, 0.95]
    confidence = min(max(confidence, 0.50), 0.97)

    # ── top-class probability ──────────────────────────────────
    # Scale |score| into a [0.52, 0.97] range for the leading bar
    top_prob = 0.52 + abs_s * 0.45
    top_prob = min(max(top_prob, 0.52), 0.97)

    # ── three-way breakdown sums to 1.0 ───────────────────────
    remainder = 1.0 - top_prob
    if label == "bullish":
        breakdown = {
            "bullish": round(top_prob, 4),
            "neutral": round(remainder * 0.60, 4),
            "bearish": round(remainder * 0.40, 4),
        }
    elif label == "bearish":
        breakdown = {
            "bearish": round(top_prob, 4),
            "neutral": round(remainder * 0.55, 4),
            "bullish": round(remainder * 0.45, 4),
        }
    else:
        breakdown = {
            "neutral": round(top_prob, 4),
            "bullish": round(remainder * 0.52, 4),
            "bearish": round(remainder * 0.48, 4),
        }

    summaries = {
        "bullish": (
            "The model detects <span>positive market signals</span> in this input. "
            "Key indicators suggest optimism around near-term price movement."
        ),
        "bearish": (
            "The model identifies <span>negative market pressure</span>. "
            "Indicators point toward caution or downside risk in the short term."
        ),
        "neutral": (
            "The model finds <span>mixed or inconclusive signals</span>. "
            "Market direction appears uncertain based on available indicators."
        ),
    }

    return {
        "sentiment"  : label,
        "score"      : round(top_prob, 4),
        "confidence" : round(confidence, 4),
        "breakdown"  : breakdown,
        "summary"    : summaries[label],
    }


def _fetch_ticker_sentiment(ticker: str) -> dict:
    """
    Fetch recent OHLCV data via yfinance, score the built-in news
    headlines with VADER, and return the frontend sentiment shape.
    Falls back gracefully if yfinance is not installed.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="yfinance is not installed on this server. "
                   "Run: pip install yfinance"
        )

    tk = yf.Ticker(ticker.upper())

    # ── Price momentum signal ──────────────────────────────────
    hist = tk.history(period="5d", interval="15m")
    if hist.empty:
        raise HTTPException(
            status_code=422,
            detail=f"No price data found for ticker '{ticker}'. "
                   "Check the symbol and try again."
        )
    # Simple momentum: last close vs 5-period-ago close
    closes     = hist["Close"].dropna().values
    momentum   = (closes[-1] - closes[-min(5, len(closes))]) / closes[-min(5, len(closes))]
    price_score = float(np.tanh(momentum * 20))   # squash into (-1, 1)
    log.info(f"[{ticker}] price momentum score: {price_score:.3f}")

    # ── News / earnings headline sentiment ────────────────────
    text_scores: List[float] = []
    try:
        news = tk.news or []
        for item in news[:10]:                     # cap at 10 headlines
            title = item.get("title", "")
            if title:
                text_scores.append(_vader_score(title))
        log.info(f"[{ticker}] scored {len(text_scores)} headlines")
    except Exception as e:
        log.warning(f"[{ticker}] news fetch failed: {e}")

    # ── Combine price + news ───────────────────────────────────
    if text_scores:
        avg_news = float(np.mean(text_scores))
        # 40 % price momentum, 60 % news sentiment
        combined = 0.40 * price_score + 0.60 * avg_news
    else:
        combined = price_score

    log.info(f"[{ticker}] combined score: {combined:.3f}")
    return _score_to_sentiment(combined)


# ════════════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════════════

app = FastAPI(
    title    = "Market Movement Prediction API",
    description=(
        "Predicts 15-minute price direction using a GRU model.\n\n"
        "**Team:** Areesha · Mobeen · Aliza · Kulsoom\n\n"
        f"**Model:** `GRU_best.pt` | hidden={GRU_CONFIG['hidden_size']} "
        f"| layers={GRU_CONFIG['num_layers']} | seq={SEQUENCE_LENGTH} x 15-min\n\n"
        "**Target:** 1 = UP, 0 = DOWN"
    ),
    version  = "1.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    log.info("=" * 60)
    log.info("  Market Prediction API — starting up")
    log.info("=" * 60)

    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device          : {state.device}")

    if not FEATURE_COLS:
        msg = "CONFIG['feature_columns'] is empty — check src/models/config.py"
        log.error(f"  {msg}")
        state.startup_errors.append(msg)
    else:
        log.info(f"  Feature columns : {len(FEATURE_COLS)} (from config.py)")

    if not MODEL_PATH.exists():
        msg = f"GRU_best.pt not found at {MODEL_PATH}"
        log.error(f"  {msg}")
        state.startup_errors.append(msg)
    elif not FEATURE_COLS:
        state.startup_errors.append("Skipping model load — feature columns missing")
    else:
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=state.device)

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint
                else:
                    raise ValueError("Unrecognised checkpoint format in GRU_best.pt")

                model = GRUModel(
                    input_size  = GRU_CONFIG["input_size"],
                    hidden_size = GRU_CONFIG["hidden_size"],
                    num_layers  = GRU_CONFIG["num_layers"],
                    dropout     = GRU_CONFIG["dropout"],
                    output_size = GRU_CONFIG["output_size"],
                )
                model.load_state_dict(state_dict)
            else:
                model = checkpoint

            model.to(state.device)
            model.eval()
            state.model        = model
            state.model_loaded = True

            n_params = sum(p.numel() for p in model.parameters())
            log.info(
                f"  GRU loaded: input={GRU_CONFIG['input_size']}  "
                f"hidden={GRU_CONFIG['hidden_size']}  "
                f"layers={GRU_CONFIG['num_layers']}  "
                f"dropout={GRU_CONFIG['dropout']}  "
                f"params={n_params:,}"
            )

        except Exception as e:
            msg = f"Failed to load GRU model: {e}"
            log.error(f"  {msg}")
            state.startup_errors.append(msg)

    if TIMESERIES_PATH.exists():
        try:
            app.state.build_features = _import_feature_builder()
            log.info("  build_realtime_features() imported")
        except Exception as e:
            log.warning(f"  Feature builder import failed: {e} — using inline fallback")
            app.state.build_features = None
    else:
        log.warning("  timeseries_construction.py not found — using inline fallback")
        app.state.build_features = None

    log.info("=" * 60)
    status_str = "API ready" if state.model_loaded else "API started BUT model not loaded"
    log.info(f"  {status_str}  ->  http://localhost:8000/docs")
    log.info("=" * 60)

    # ── Serve the frontend from /frontend ─────────────────────
    if FRONTEND_PATH.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_PATH), html=True), name="frontend")
        log.info(f"  Serving frontend from {FRONTEND_PATH}  ->  http://localhost:8000/")
    else:
        log.warning(
            f"  Frontend folder not found at {FRONTEND_PATH}. "
            "Create a 'frontend/' directory next to 'api/' and put index.html inside."
        )


# ════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ════════════════════════════════════════════════════════════════

class PriceRecord(BaseModel):
    timestamp : str   = Field(..., example="2024-01-15T09:30:00+00:00")
    ticker    : str   = Field(..., example="AAPL")
    open      : float = Field(..., gt=0)
    high      : float = Field(..., gt=0)
    low       : float = Field(..., gt=0)
    close     : float = Field(..., gt=0)
    volume    : float = Field(..., ge=0)

    @validator("high")
    def high_gte_low(cls, v, values):
        if "low" in values and v < values["low"]:
            raise ValueError("high must be >= low")
        return v


class SentimentRecord(BaseModel):
    timestamp      : str            = Field(..., example="2024-01-15T09:25:00+00:00")
    vader_compound : Optional[float]= Field(None, ge=-1.0, le=1.0)
    sentiment      : Optional[str]  = Field(None, example="positive")


class PredictionRequest(BaseModel):
    price_data   : List[PriceRecord]               = Field(..., min_items=1)
    news_data    : Optional[List[SentimentRecord]] = Field(default=[])
    reddit_data  : Optional[List[SentimentRecord]] = Field(default=[])
    twitter_data : Optional[List[SentimentRecord]] = Field(default=[])

    class Config:
        schema_extra = {
            "example": {
                "price_data": [{
                    "timestamp": "2024-01-15T09:30:00+00:00",
                    "ticker": "AAPL",
                    "open": 185.20, "high": 186.10,
                    "low": 184.90,  "close": 185.80,
                    "volume": 12500
                }],
                "news_data": [{
                    "timestamp": "2024-01-15T09:25:00+00:00",
                    "vader_compound": 0.62,
                    "sentiment": "positive"
                }],
                "reddit_data": [], "twitter_data": [],
            }
        }


class PredictionResponse(BaseModel):
    prediction       : int   = Field(..., description="1 = UP, 0 = DOWN")
    probability_up   : float = Field(..., description="Confidence price will rise")
    probability_down : float = Field(..., description="Confidence price will fall")
    signal           : str   = Field(..., description="BUY / SELL / HOLD")
    confidence       : str   = Field(..., description="HIGH / MEDIUM / LOW")
    sequence_windows : int   = Field(..., description="15-min windows used")
    latest_timestamp : str   = Field(..., description="Most recent window timestamp")
    ticker           : str
    model            : str   = Field(default="GRU_best.pt")
    inferred_at      : str   = Field(..., description="UTC time of this inference")


# ── NEW: schema for the frontend /api/predict endpoint ────────

class FrontendPredictRequest(BaseModel):
    """
    Simple payload sent by index.html.
    Exactly one of `text` or `ticker` must be present.
    """
    text   : Optional[str] = Field(None, example="Fed signals rate cuts")
    ticker : Optional[str] = Field(None, example="AAPL")

    @validator("ticker", always=True)
    def at_least_one(cls, ticker, values):
        if not values.get("text") and not ticker:
            raise ValueError("Provide either 'text' or 'ticker'")
        return ticker


class FrontendPredictResponse(BaseModel):
    """Shape expected by renderResult() in main.js."""
    sentiment  : str
    score      : float
    confidence : float
    breakdown  : dict
    summary    : str


class BatchRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., min_items=1, max_items=20)


class HealthResponse(BaseModel):
    status          : str
    model_loaded    : bool
    model_file      : str
    device          : str
    feature_count   : int
    sequence_length : int
    gru_hidden_size : int
    gru_num_layers  : int
    startup_errors  : List[str]


# ════════════════════════════════════════════════════════════════
# CORE INFERENCE (GRU pipeline — unchanged)
# ════════════════════════════════════════════════════════════════

def _run_inference(request: PredictionRequest) -> PredictionResponse:
    if not state.model_loaded:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"Model not loaded. Errors: {state.startup_errors}"
        )

    build_features = getattr(app.state, "build_features", None)
    if build_features is not None:
        try:
            features_df = build_features(
                price_records   = [r.dict() for r in request.price_data],
                news_records    = [r.dict() for r in (request.news_data or [])],
                reddit_records  = [r.dict() for r in (request.reddit_data or [])],
                twitter_records = [r.dict() for r in (request.twitter_data or [])],
            )
        except Exception as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f"Feature engineering failed: {e}")
    else:
        log.warning("Using inline feature builder fallback")
        features_df = _inline_feature_builder(request)

    if features_df.empty:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Feature DataFrame is empty. Ensure price_data covers enough time "
                f"to produce at least {SEQUENCE_LENGTH} 15-minute windows."
            )
        )

    try:
        tensor = prepare_sequence(features_df, FEATURE_COLS, SEQUENCE_LENGTH).to(state.device)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    with torch.no_grad():
        logit     = state.model(tensor)
        prob_up   = float(torch.sigmoid(logit).item())
        prob_down = 1.0 - prob_up
        pred      = 1 if prob_up >= 0.5 else 0

    if   prob_up >= 0.70: signal, confidence = "BUY",  "HIGH"
    elif prob_up >= 0.55: signal, confidence = "BUY",  "MEDIUM"
    elif prob_up <= 0.30: signal, confidence = "SELL", "HIGH"
    elif prob_up <= 0.45: signal, confidence = "SELL", "MEDIUM"
    else:                 signal, confidence = "HOLD", "LOW"

    return PredictionResponse(
        prediction       = pred,
        probability_up   = round(prob_up,   4),
        probability_down = round(prob_down, 4),
        signal           = signal,
        confidence       = confidence,
        sequence_windows = len(features_df),
        latest_timestamp = str(features_df.index[-1]),
        ticker           = request.price_data[0].ticker,
        model            = "GRU_best.pt",
        inferred_at      = datetime.now(timezone.utc).isoformat(),
    )


def _inline_feature_builder(request: PredictionRequest) -> pd.DataFrame:
    RESAMPLE = "15min"
    SENT_MAP = {"positive": 1, "neutral": 0, "negative": -1}

    p = pd.DataFrame([r.dict() for r in request.price_data])
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
    p = p.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    for col in ["open","high","low","close","volume"]:
        p[col] = pd.to_numeric(p[col], errors="coerce")

    price_r = p.resample(RESAMPLE).agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),     close=("close","last"),
        volume=("volume","sum"),
    ).dropna(subset=["close"])

    all_sent = []
    for records in [request.news_data, request.reddit_data, request.twitter_data]:
        if not records:
            continue
        s = pd.DataFrame([r.dict() for r in records])
        if s.empty:
            continue
        s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True, errors="coerce")
        s = s.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        if "vader_compound" in s.columns:
            score = pd.to_numeric(s["vader_compound"], errors="coerce")
        elif "sentiment" in s.columns:
            score = s["sentiment"].str.strip().str.lower().map(SENT_MAP).fillna(0.0)
        else:
            continue
        all_sent.append(score.rename("sentiment_score"))

    if all_sent:
        sent_r = pd.concat(all_sent).sort_index().resample(RESAMPLE).agg(
            avg_sentiment="mean", sentiment_count="count"
        )
        sent_r = sent_r[sent_r["sentiment_count"] > 0]
    else:
        sent_r = pd.DataFrame(columns=["avg_sentiment","sentiment_count"])

    df = price_r.join(sent_r, how="left")
    df["avg_sentiment"]   = df["avg_sentiment"].fillna(method="ffill", limit=3).fillna(0.0)
    df["sentiment_count"] = df["sentiment_count"].fillna(0).astype(int)

    df["price_change_pct"]  = df["close"].pct_change() * 100
    df["volume_change_pct"] = df["volume"].pct_change().clip(-5, 5) * 100
    for lag in [1, 2, 3]:
        df[f"sentiment_lag_{lag}"] = df["avg_sentiment"].shift(lag)
    df["sentiment_ma_3"] = df["avg_sentiment"].rolling(3, min_periods=1).mean()
    df["volume_ma_3"]    = df["volume"].rolling(3, min_periods=1).mean()
    df["price_ma_3"]     = df["close"].rolling(3, min_periods=1).mean()
    df["price_vs_ma"]    = (df["close"] - df["price_ma_3"]) / df["price_ma_3"] * 100
    df["hl_spread_pct"]  = (df["high"] - df["low"]) / df["low"] * 100

    return df.dropna()


# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.get("/info", tags=["Info"])
async def root():
    return {
        "project"  : "Financial Market Movement Prediction",
        "api"      : "Market Prediction API v1.0",
        "team"     : {"data": "Aliza (DVC)", "sentiment": "Mobeen", "api": "Areesha"},
        "model"    : "GRU_best.pt",
        "endpoints": ["/health", "/config", "/features", "/model/info",
                      "/predict", "/predict/batch", "/api/predict"],
        "docs"     : "/docs",
    }


@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["Monitoring"])
async def health():
    return HealthResponse(
        status          = "ok" if state.model_loaded else "degraded",
        model_loaded    = state.model_loaded,
        model_file      = str(MODEL_PATH),
        device          = str(state.device),
        feature_count   = len(FEATURE_COLS),
        sequence_length = SEQUENCE_LENGTH,
        gru_hidden_size = GRU_CONFIG["hidden_size"],
        gru_num_layers  = GRU_CONFIG["num_layers"],
        startup_errors  = state.startup_errors,
    )


@app.get("/config", summary="View full CONFIG from config.py", tags=["Info"])
async def get_config():
    return {
        "config"          : CONFIG,
        "config_path"     : str(CONFIG_PATH),
        "feature_count"   : len(FEATURE_COLS),
        "sequence_length" : SEQUENCE_LENGTH,
    }


@app.get("/features", summary="List feature columns expected by the model", tags=["Info"])
async def get_features():
    return {
        "feature_columns"   : FEATURE_COLS,
        "feature_count"     : len(FEATURE_COLS),
        "sequence_length"   : SEQUENCE_LENGTH,
        "resample_frequency": "15min",
        "target_column"     : CONFIG.get("target_column", "target"),
        "source"            : "src/models/config.py",
    }


@app.get("/model/info", summary="GRU architecture details", tags=["Info"])
async def model_info():
    if not state.model_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model not loaded — check /health")
    total = sum(p.numel() for p in state.model.parameters())
    return {
        "model_file"      : MODEL_PATH.name,
        "architecture"    : "GRU (Gated Recurrent Unit)",
        "gru_config"      : GRU_CONFIG,
        "sequence_length" : SEQUENCE_LENGTH,
        "total_parameters": total,
        "device"          : str(state.device),
        "config_source"   : str(CONFIG_PATH),
    }


# ── NEW: Frontend-facing endpoint ─────────────────────────────

@app.post(
    "/api/predict",
    response_model = FrontendPredictResponse,
    summary        = "Sentiment endpoint for the MarketPulse UI",
    tags           = ["Frontend"],
)
async def api_predict(req: FrontendPredictRequest):
    """
    Lightweight endpoint consumed by **index.html / main.js**.

    - **text mode** : score the headline/excerpt with VADER
      (or keyword fallback) and return a sentiment breakdown.
    - **ticker mode**: fetch last-5-day 15-min OHLCV via yfinance,
      score the ticker's recent news headlines, combine price
      momentum + sentiment, and return the same breakdown shape.

    Response matches exactly what `renderResult()` in main.js expects:

    ```json
    {
      "sentiment":  "bullish" | "bearish" | "neutral",
      "score":      0.87,
      "confidence": 0.91,
      "breakdown":  { "bullish": 0.87, "neutral": 0.09, "bearish": 0.04 },
      "summary":    "HTML string"
    }
    ```
    """
    if req.ticker:
        log.info(f"/api/predict  ticker={req.ticker}")
        return _fetch_ticker_sentiment(req.ticker)

    # text mode
    log.info(f"/api/predict  text={req.text[:60]}...")
    score = _vader_score(req.text)
    return _score_to_sentiment(score)


# ── Original GRU pipeline endpoint (unchanged) ────────────────

@app.post("/predict", response_model=PredictionResponse,
          summary="Predict next 15-min price direction (GRU pipeline)", tags=["Prediction"])
async def predict(request: PredictionRequest):
    log.info(
        f"/predict ticker={request.price_data[0].ticker} "
        f"price_rows={len(request.price_data)} "
        f"sentiment_rows={len(request.news_data or []) + len(request.reddit_data or []) + len(request.twitter_data or [])}"
    )
    return _run_inference(request)


@app.post("/predict/batch", response_model=List[PredictionResponse],
          summary="Predict for up to 20 tickers (GRU pipeline)", tags=["Prediction"])
async def predict_batch(batch: BatchRequest):
    log.info(f"/predict/batch n={len(batch.requests)} tickers")
    results, errors = [], []
    for i, req in enumerate(batch.requests):
        try:
            results.append(_run_inference(req))
        except HTTPException as e:
            ticker = req.price_data[0].ticker if req.price_data else f"item_{i}"
            errors.append(f"[{ticker}] {e.detail}")
            log.warning(f"Batch item {i} failed: {e.detail}")

    if errors and not results:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "All batch items failed", "errors": errors},
        )
    return results


# ════════════════════════════════════════════════════════════════
# GLOBAL EXCEPTION HANDLER
# ════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_type": type(exc).__name__},
    )


# ════════════════════════════════════════════════════════════════
# RUN LOCALLY
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")