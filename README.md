# 📊 Market Sentiment Prediction — MLOps

### 🌐 [Live Demo → market-sentiment-prediction-mlops-five.vercel.app](https://market-sentiment-prediction-mlops-five.vercel.app/)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Pipeline Stages](#-pipeline-stages)
- [Data Sources](#-data-sources)
- [Sentiment Models](#-sentiment-models)
- [Project Structure](#-project-structure)
- [Setup & Installation](#️-setup--installation)
- [Usage](#-usage)
- [MLOps Pipeline](#-mlops-pipeline)
- [Docker & Deployment](#-docker--deployment)
- [CI/CD](#-cicd)
- [Team](#-team)

---

## 🔍 Overview

This project builds an end-to-end MLOps pipeline for financial market sentiment prediction. It covers the full ML lifecycle: multi-source data scraping, NLP-based sentiment labeling, model experiment tracking, REST API serving, and containerized deployment.

**Key components:**

| Component | Implementation |
|---|---|
| Data scraping | Yahoo Finance, Reddit, Twitter, News RSS |
| Sentiment analysis | VADER + Transformer-based models |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| API serving | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## 🌐 Live Demo

The frontend is deployed on Vercel and connected to the FastAPI backend:

**🔗 [market-sentiment-prediction-mlops-five.vercel.app](https://market-sentiment-prediction-mlops-five.vercel.app/)**

Enter any financial headline or text snippet and the model returns a sentiment prediction with a compound score and breakdown — all in real time.

---

## 🔄 Pipeline Stages

The pipeline runs in three sequential stages via `main.py`:

**Stage 1 — Scraping**
Collects raw financial text from Yahoo Finance, Reddit, Twitter, and news RSS feeds using dedicated scrapers in `src/scraping/`.

**Stage 2 — Sentiment Labeling**
Applies VADER sentiment analysis to score and label each text snippet as positive, negative, or neutral. Transformer-based inference via `transformers` + `torch` is available for higher-accuracy runs.

**Stage 3 — MLflow Tracking**
Logs all run artifacts, metrics, and parameters to MLflow for experiment tracking and model registry.

```bash
python main.py
```

```
=== STAGE 1: Scraping ===
  → Yahoo Finance scraper
  → Reddit scraper
  → Twitter scraper
  → News RSS scraper

=== STAGE 2: Sentiment Labeling ===
  → VADER sentiment scoring

=== STAGE 3: MLflow Tracking ===
  → Logging run to MLflow

=== Pipeline complete ===
```

---

## 📰 Data Sources

| Source | Module | Data Type |
|---|---|---|
| Yahoo Finance | `src/scraping/yahoo_scraper.py` | Stock news & headlines |
| Reddit | `src/scraping/reddit_scraper.py` | Community discussions |
| Twitter | `src/scraping/twitter_scraper.py` | Real-time tweets |
| News RSS | `src/scraping/news_scraper.py` | Financial news feeds |

All scraped data is tracked via DVC and stored in the configured remote.

---

## 🧠 Sentiment Models

### VADER

- Rule-based sentiment analysis optimized for social media and financial text
- Returns compound, positive, negative, and neutral scores
- Used as the primary labeling engine in `src/sentiment/vader_sentiment.py`

### Transformer (Optional)

- HuggingFace `transformers` with a pretrained financial sentiment model
- Higher accuracy for longer news articles and structured text
- Available via `torch` + `sentencepiece` backend

---

## 📁 Project Structure

```
market-sentiment-prediction-mlops/
│
├── .dvc/                          # DVC configuration
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline
│
├── api/                           # FastAPI model serving
│   └── main.py                    # /predict endpoint
│
├── docker/                        # Containerization
│   └── Dockerfile
│
├── frontend/                      # Web UI
│
├── mlops/
│   └── mlflow/
│       └── mlflow_tracker.py      # MLflow logging utilities
│
├── notebooks/                     # Exploratory analysis
│
├── src/
│   ├── scraping/
│   │   ├── yahoo_scraper.py       # Yahoo Finance scraper
│   │   ├── reddit_scraper.py      # Reddit scraper
│   │   ├── twitter_scraper.py     # Twitter scraper
│   │   └── news_scraper.py        # News RSS scraper
│   │
│   └── sentiment/
│       └── vader_sentiment.py     # VADER labeling pipeline
│
├── .dvcignore
├── .gitignore
├── data.dvc                       # DVC-tracked data pointer
├── main.py                        # Pipeline entry point
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- Git + DVC
- Docker (for deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/moonlightaliza/market-sentiment-prediction-mlops.git
cd market-sentiment-prediction-mlops
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Data with DVC

```bash
dvc pull
```

> Make sure you have access to the configured DVC remote. Ask the team for credentials.

---

## 🚀 Usage

### Run the Full Pipeline

```bash
python main.py
```

### Start the API

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Sample request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Markets rally as inflation data comes in lower than expected."}'
```

**Sample response:**

```json
{
  "sentiment": "positive",
  "compound_score": 0.6369,
  "scores": {
    "positive": 0.274,
    "negative": 0.0,
    "neutral": 0.726
  }
}
```

### View MLflow UI

```bash
mlflow ui --port 5000
```

Open `http://localhost:5000` to browse logged runs, metrics, and artifacts.

---

## 🔧 MLOps Pipeline

### DVC — Data Versioning

The scraped dataset is tracked with DVC:

```bash
dvc add data/              # Track data directory
dvc push                   # Push to remote
dvc pull                   # Pull on a new machine
```

`data.dvc` pins the exact dataset version used for each experiment run.

### MLflow — Experiment Tracking

Every pipeline run logs to MLflow automatically:

| Logged Item | Details |
|---|---|
| Parameters | Scraper config, sentiment thresholds |
| Metrics | Sentiment distribution, label counts |
| Artifacts | Labeled dataset, score distributions |
| Model | Serialized sentiment pipeline |

---

## 🐳 Docker & Deployment

### Build and Run Locally

```bash
docker build -f docker/Dockerfile -t market-sentiment-api .
docker run -p 8000:8000 market-sentiment-api
```

### EC2 Deployment

```bash
# On your EC2 instance
docker pull <your-registry>/market-sentiment-api:latest
docker run -d -p 80:8000 <your-registry>/market-sentiment-api:latest
```

---

## 🔄 CI/CD

GitHub Actions runs on every push and pull request to `main`:

```
.github/workflows/ci.yml
```

**Pipeline steps:**

1. **Lint** — `flake8` code style check
2. **Unit tests** — pytest on scrapers and sentiment modules
3. **DVC check** — validates `data.dvc` integrity
4. **Docker build** — builds the API image to catch Dockerfile issues early

---

## 👥 Team

| Name | GitHub |
|---|---|
| Umm e Kulsoom | [@KayTheCoder-101](https://github.com/KayTheCoder-101) |
| Aliza Zia | [@moonlightaliza](https://github.com/moonlightaliza) |
| Areesha Saqib | [@Areesha-008](https://github.com/Areesha-008) |
| Mobeen Rukhsar | [@mobeenrukhsar269-crypto](https://github.com/mobeenrukhsar269-crypto) |

---

## 📜 License

This project is for academic purposes only.

---

Made for the Artificial Neural Network + Machine Learning Operations Course · FAST-NUCES  
[🌐 Live Demo](https://market-sentiment-prediction-mlops-five.vercel.app/)
