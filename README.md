# Quantitative AI Research Platform

> A production-grade, institutional-style quantitative research environment for financial machine learning, alpha signal discovery, strategy simulation, portfolio optimization, and performance analytics.

---

## Table of Contents

- [Overview](#overview)
- [Why This Project Exists](#why-this-project-exists)
- [Core Design Philosophy](#core-design-philosophy)
- [System Scope](#system-scope)
- [High-Level Data Flow](#high-level-data-flow)
- [Architectural Layers](#architectural-layers)
- [Repository Structure](#repository-structure)
- [Module Responsibilities](#module-responsibilities)
- [Data Schemas](#data-schemas)
- [Database Tables](#database-tables)
- [Data Validation Rules](#data-validation-rules)
- [Technology Stack](#technology-stack)
- [Engineer Ownership Model](#engineer-ownership-model)
- [Module Dependency Rules](#module-dependency-rules)
- [Configuration Management](#configuration-management)
- [Infrastructure Architecture](#infrastructure-architecture)
- [Automation Architecture](#automation-architecture)
- [Experiment and Research Lifecycle](#experiment-and-research-lifecycle)
- [Financial Research Integrity Rules](#financial-research-integrity-rules)
- [Machine Learning Standards](#machine-learning-standards)
- [Backtesting Requirements](#backtesting-requirements)
- [Portfolio Optimization](#portfolio-optimization)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Scalability Considerations](#scalability-considerations)
- [Long-Term Vision](#long-term-vision)
- [Getting Started](#getting-started)
- [Contributing](#contributing)

---

## Overview

The **Quantitative AI Research Platform** is a modular, end-to-end research system designed to replicate the internal infrastructure used by quantitative hedge funds, systematic trading firms, and institutional financial research teams.

This is not a toy ML project. It is a complete research environment supporting the full lifecycle of quantitative strategy development — from raw data ingestion through portfolio-level performance evaluation.

The platform enables systematic experimentation on financial data across the following research stages:

- Automated multi-source financial data ingestion
- Schema validation and data cleaning
- Parquet-based data lake architecture
- Financial feature engineering
- Time-series and cross-sectional ML modeling
- MLflow experiment tracking and model versioning
- Predictive signal generation
- Realistic strategy backtesting
- Portfolio construction and optimization
- Risk-adjusted performance analytics
- Visualization dashboards

**Primary Focus:** Indian Financial Markets — NSE Equities and Indices  
**Architecture:** Extensible to multi-asset global research

---

## Why This Project Exists

Most beginner financial ML projects stop at:

- Predicting stock prices with simple models
- Running naive backtests without transaction costs
- Plotting accuracy metrics and calling it done

These projects fail to reflect how real quantitative research is actually performed.

Professional quantitative teams operate with:

- Structured, validated data pipelines
- Standardized, schema-enforced datasets
- Experiment tracking and model versioning
- Modular, independently testable signal research
- Realistic backtesting with slippage and fees
- Portfolio construction and capital allocation logic
- Risk analytics with benchmark comparison

This platform replicates that professional workflow in a compact, credible, institutional-style engineering system.

---

## Core Design Philosophy

### Modularity
Each subsystem has a single clearly defined responsibility. Modules must not absorb unrelated responsibilities or reimplement downstream logic.

### Reproducibility
All datasets, features, models, signals, and backtest results must be generated through deterministic, well-defined pipelines wherever possible.

### Interface Stability
Modules communicate through clearly defined datasets, schemas, configuration, and shared contracts. Upstream and downstream boundaries must remain stable.

### Scalability
The architecture supports future expansion from Indian equities to broader multi-asset global research.

### Research Flexibility
Engineers and researchers must be able to run many experiments without rewriting the core platform.

### Automation
The platform supports scheduled data refresh, feature recomputation, model retraining, and signal updates.

### Research Integrity
Financial research correctness is prioritized over engineering convenience. Any architecture decision that weakens research integrity is rejected.

---

## System Scope

### Current Production Scope
- NSE Equities
- Indian Indices

### Near-Term Expansion Scope
- ETFs
- Macroeconomic Indicators
- Mutual Fund NAV Datasets
- Commodities
- Forex
- Crypto

### Long-Term Expansion Scope
- Real-time market feeds
- Distributed training pipelines
- Higher-frequency data
- Cross-sectional ranking research
- Multi-strategy portfolio research
- Reinforcement learning portfolio allocation
- Cloud-native orchestration

---

## High-Level Data Flow

```
Multi-Asset Market Data Sources
(equities, indices, ETFs, macro, forex, commodities, crypto, news)
        ↓
Data Ingestion Pipeline
        ↓
Raw Data Storage (Parquet)
        ↓
Data Validation and Cleaning
        ↓
Processed Data Storage (Parquet)
        ↓
Feature Engineering
        ↓
Machine Learning Models
        ↓
Prediction Generation
        ↓
Signal Generation
        ↓
Backtesting Engine
        ↓
Portfolio Construction and Optimization
        ↓
Performance Analytics and Risk Evaluation
        ↓
Visualization Dashboard
```

Each stage produces outputs that become inputs to the next stage. No downstream stage should ever need to repair or reinterpret upstream outputs.

---

## Architectural Layers

### Layer 1 — Data Acquisition
Collects raw market, macro, and auxiliary datasets from external providers.

### Layer 2 — Data Storage and Validation
Stores data in efficient formats, validates schema quality, and cleans corrupted or inconsistent records.

### Layer 3 — Feature Engineering
Transforms processed market data into model-ready feature matrices.

### Layer 4 — Machine Learning and Signal Discovery
Trains predictive models, generates return forecasts, and converts them into actionable trading signals.

### Layer 5 — Strategy Simulation and Portfolio Research
Evaluates the profitability and robustness of model-generated signals under realistic conditions.

### Layer 6 — Analytics and Visualization
Summarizes model, strategy, and portfolio performance for researchers and engineers.

---

## Repository Structure

```
.
├── .env
├── .env.example
├── .gitattributes
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
├── README.md
│
├── config/
│   ├── constants.py
│   └── settings.py
│
├── data/
│   ├── features/
│   ├── predictions/
│   ├── processed/
│   │   ├── commodity/
│   │   ├── crypto/
│   │   ├── equity/
│   │   ├── etf/
│   │   ├── forex/
│   │   ├── index/
│   │   └── macro/
│   ├── raw/
│   │   ├── commodity/
│   │   ├── crypto/
│   │   ├── equity/
│   │   ├── etf/
│   │   ├── forex/
│   │   ├── index/
│   │   ├── macro/
│   │   └── news/
│   └── signals/
│
├── docker/
├── docs/
├── experiments/
├── models/
├── notebooks/
│
├── scripts/
│   ├── run_features.py
│   └── run_ingestion.py
│
├── src/
│   ├── analytics/
│   ├── backtesting/
│   ├── dashboard/
│   ├── database/
│   ├── data_pipeline/
│   ├── data_storage/
│   ├── feature_engineering/
│   ├── infra/
│   ├── models/
│   ├── portfolio/
│   ├── reinforcement_learning/
│   ├── signals/
│   └── utils/
│
└── tests/
    ├── analytics/
    ├── backtesting/
    ├── database/
    ├── data_pipeline/
    ├── data_storage/
    ├── feature_engineering/
    ├── models/
    ├── portfolio/
    ├── signals/
    └── utils/
```

---

## Module Responsibilities

### `src/data_pipeline/`
**Owner: Data Engineer**

Responsible for acquiring external financial data.

- Connect to market and macro data APIs
- Download historical data and perform incremental updates
- Normalize source-specific API responses
- Add source metadata
- Handle retries, rate limits, and timeouts

Supported data sources:
- `yfinance`
- `Alpha Vantage`
- `FRED`
- `NewsAPI`

---

### `src/data_storage/`
**Owner: Data Engineer**

Responsible for file-based dataset persistence and processed data management.

- Store raw and processed datasets in Parquet format
- Manage data paths and naming conventions
- Support partitioned storage by asset class

**Parquet file naming convention:**
```
<asset_class>_<symbol>_<timeframe>.parquet
```

Examples:
```
equity_RELIANCE_daily.parquet
index_NIFTY50_daily.parquet
crypto_BTCUSDT_hourly.parquet
forex_USDINR_daily.parquet
commodity_GOLD_daily.parquet
```

---

### `src/database/`
**Owner: Data Engineer (upstream schema); shared read access for all engineers**

Manages structured metadata and queryable research records.

- SQLAlchemy ORM models
- Database connections and session management
- Core metadata table definitions
- Index management
- Query utilities

Database: **PostgreSQL**

Core tables: `assets`, `prices`, `features`, `signals`, `trades`, `portfolio_metrics`

---

### `src/feature_engineering/`
**Owner: Machine Learning Engineer**

Transforms processed market data into predictive feature matrices.

Feature categories implemented:

| Category | Features |
|---|---|
| Trend | SMA_5, SMA_10, SMA_20, SMA_50, SMA_200, EMA_10, EMA_20, EMA_50 |
| Momentum | RSI_14, MACD, MACD_SIGNAL, ROC_5, ROC_10, MOMENTUM_5, MOMENTUM_10 |
| Volatility | VOLATILITY_10, VOLATILITY_20, VOLATILITY_50, ATR_14, BOLLINGER_WIDTH |
| Mean Reversion | ZSCORE_20, DISTANCE_FROM_SMA, DISTANCE_FROM_EMA |
| Volume | VOLUME_MA_20, VOLUME_RATIO, OBV |

**Target generation:**
```
future_return_n = close(t+n) / close(t) - 1
```

Supported targets: `future_return_1d`, `future_return_5d`, `future_return_10d`

---

### `src/models/`
**Owner: Machine Learning Engineer**

Responsible for predictive model training, evaluation, and forecast generation.

| Category | Models |
|---|---|
| Baseline | Linear Regression, Ridge Regression, Lasso Regression |
| Tree-Based | Random Forest, Gradient Boosting |
| Advanced | XGBoost, LightGBM |
| Deep Learning (Optional) | MLP, LSTM, Temporal CNN, Transformer |

Training pipeline steps:
1. Load feature dataset
2. Split train/validation/test **chronologically**
3. Train model
4. Evaluate performance
5. Save model artifacts
6. Log experiment to MLflow

Validation approach: **Walk-forward validation** (no random splits)

MLflow tracking URI: `http://localhost:5000`

Logged per experiment:
- Model type and hyperparameters
- Dataset version and feature set used
- Metrics: MSE, RMSE, MAE, R², Directional Accuracy, IC, Rank IC
- Trained model artifacts

---

### `src/signals/`
**Owner: Machine Learning Engineer**

Converts model predictions into research-ready trading signals.

Signal methods:
- Threshold-based signals
- Rank-based signals
- Top-N long/short selection

Signal encoding:
```
 1  =  BUY
 0  =  HOLD
-1  =  SELL
```

Output stored in: `data/signals/`

---

### `src/reinforcement_learning/`
**Owner: Machine Learning Engineer (future)**

Reserved for future portfolio RL systems. Placeholder architecture only.

Planned algorithms: PPO, DDPG, SAC

**Do NOT implement unless explicitly instructed.**

---

### `src/backtesting/`
**Owner: Quant Strategy Engineer**

Simulates realistic trading conditions using model-generated signals.

Required simulation features:
- Transaction costs
- Slippage
- Position sizing
- Portfolio constraints

Required performance metrics:
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Annualized Return
- Volatility
- Win Rate
- Calmar Ratio
- Profit Factor

---

### `src/portfolio/`
**Owner: Quant Strategy Engineer**

Constructs optimized portfolios from backtested strategies.

Supported methods:
- Mean Variance Optimization
- Risk Parity
- Maximum Diversification

Library: `PyPortfolioOpt`

---

### `src/analytics/`
**Owner: Quant Strategy Engineer**

Evaluates and summarizes strategy and portfolio performance.

- Compute risk-adjusted metrics
- Benchmark comparison (e.g., NIFTY 50)
- Trade analytics
- Performance attribution

---

### `src/dashboard/`
**Owner: Quant Strategy Engineer**

Visualizes research outputs for analysis and monitoring.

- Strategy performance charts
- Portfolio allocation visualization
- Research monitoring interfaces
- Benchmark comparison views

Technology: `Plotly`, `Streamlit` (prototype), `FastAPI` (backend)

---

## Data Schemas

### Raw Market Data Schema
**Location:** `data/raw/`

| Column | Type | Required | Description |
|---|---|---|---|
| symbol | string | Yes | Trading symbol / ticker |
| asset_class | string | Yes | equity / index / etf / forex / crypto / commodity / macro |
| exchange | string | Yes | NSE / BINANCE / FX / etc |
| currency | string | Yes | Trading currency |
| date | datetime | Yes | Observation timestamp |
| open | float | Yes | Opening price |
| high | float | Yes | Highest price |
| low | float | Yes | Lowest price |
| close | float | Yes | Closing price |
| adj_close | float | Optional | Adjusted close if available |
| volume | float | Optional | Trading volume |
| source | string | Yes | Data source provider |

**Primary index:** `(symbol, date)`

---

### Processed Market Data Schema
**Location:** `data/processed/`

| Column | Type | Required |
|---|---|---|
| symbol | string | Yes |
| asset_class | string | Yes |
| exchange | string | Yes |
| currency | string | Yes |
| date | datetime | Yes |
| open | float | Yes |
| high | float | Yes |
| low | float | Yes |
| close | float | Yes |
| adj_close | float | Optional |
| volume | float | Optional |
| returns | float | Yes |
| log_returns | float | Yes |

---

### Feature Dataset Schema
**Location:** `data/features/`

Base required columns: `symbol`, `date` + all feature columns + target columns

---

### Prediction Dataset Schema
**Location:** `data/predictions/`

| Column | Type |
|---|---|
| symbol | string |
| date | datetime |
| predicted_return | float |
| model_name | string |
| model_version | string |

---

### Signal Dataset Schema
**Location:** `data/signals/`

| Column | Type | Description |
|---|---|---|
| symbol | string | Asset symbol |
| date | datetime | Signal timestamp |
| predicted_return | float | Model forecast |
| signal | int | 1 buy / 0 hold / -1 sell |
| confidence | float | Optional model confidence |
| rank | int | Optional rank among universe |

---

### Trade Log Schema

| Column | Type |
|---|---|
| trade_id | int |
| symbol | string |
| entry_date | datetime |
| exit_date | datetime |
| entry_price | float |
| exit_price | float |
| position_size | float |
| fees | float |
| slippage_cost | float |
| pnl | float |
| return_pct | float |

---

### Portfolio History Schema

| Column | Type |
|---|---|
| date | datetime |
| cash | float |
| invested_capital | float |
| portfolio_value | float |
| daily_return | float |
| cumulative_return | float |
| drawdown | float |

---

### Portfolio Metrics Schema

| Column | Type |
|---|---|
| metric | string |
| value | float |

Example metrics: `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `annual_return`, `volatility`, `calmar_ratio`, `win_rate`, `profit_factor`

---

## Database Tables

### `assets`
| Column | Type |
|---|---|
| id | int (PK) |
| symbol | string |
| name | string |
| asset_class | string |
| exchange | string |
| currency | string |
| sector | string |
| is_active | bool |
| created_at | datetime |
| updated_at | datetime |

### `prices`
| Column | Type |
|---|---|
| id | int (PK) |
| symbol | string |
| date | datetime |
| open | float |
| high | float |
| low | float |
| close | float |
| adj_close | float |
| volume | float |
| source | string |
| created_at | datetime |

### `features`
| Column | Type |
|---|---|
| id | int (PK) |
| symbol | string |
| date | datetime |
| feature_set_version | string |
| created_at | datetime |

### `signals`
| Column | Type |
|---|---|
| id | int (PK) |
| symbol | string |
| date | datetime |
| signal | int |
| predicted_return | float |
| model_version | string |

### `trades`
| Column | Type |
|---|---|
| id | int (PK) |
| symbol | string |
| entry_date | datetime |
| exit_date | datetime |
| pnl | float |
| return_pct | float |

### `portfolio_metrics`
| Column | Type |
|---|---|
| id | int (PK) |
| strategy_name | string |
| metric | string |
| value | float |
| calculated_at | datetime |

**Required database indexes:**
```
(symbol, date)
(date)
(asset_class)
```

---

## Data Validation Rules

Before data enters processed datasets, all of the following checks must pass:

- No duplicate `(symbol, date)` rows
- No null required fields
- No negative volume
- `high >= low`
- `high >= open` and `high >= close`
- `low <= open` and `low <= close`
- Chronological sorting enforced

**If validation fails → STOP pipeline and report error. Do not proceed.**

Invalid rows must be rejected or quarantined, never silently dropped.

---

## Technology Stack

### Programming Language
- Python 3.12+

### Data Processing
- `pandas`
- `numpy`
- `pyarrow`

### Financial Data / Quant Libraries
- `yfinance`
- `pandas-ta`
- `quantstats`
- `PyPortfolioOpt`

### Machine Learning
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `pytorch`

### Experiment Tracking
- `MLflow`

### Database / Storage
- PostgreSQL
- Parquet Data Lake

### Backend / Dashboard
- `FastAPI`
- `Plotly`
- `Streamlit` (prototype)

### Infrastructure
- Docker
- Docker Compose

### Testing
- `pytest`

### ORM
- `SQLAlchemy`

---

## Engineer Ownership Model

The project is structured for three primary engineering roles working in parallel.

### Engineer 1 — Data & Infrastructure Engineer

**Owns:**
- `src/data_pipeline/`
- `src/data_storage/`
- `src/database/`
- `config/` (infrastructure configuration)

**Responsibilities:**
- Data ingestion pipelines
- Raw and processed data storage
- Database schema foundations
- Validation and cleaning pipelines
- Pipeline automation and scheduling
- Infrastructure foundations (Docker, PostgreSQL)

---

### Engineer 2 — Machine Learning & Signal Engineer

**Owns:**
- `src/feature_engineering/`
- `src/models/`
- `src/signals/`
- `src/reinforcement_learning/` (future)
- `config/` (ML-related configuration only)

**Responsibilities:**
- Feature engineering pipeline
- ML dataset construction
- Predictive model training and evaluation
- MLflow experiment tracking
- Return prediction pipeline
- Signal generation system

**Input:** `data/processed/`  
**Output:** `data/signals/`

---

### Engineer 3 — Quant Strategy & Portfolio Engineer

**Owns:**
- `src/backtesting/`
- `src/portfolio/`
- `src/analytics/`
- `src/dashboard/`

**Responsibilities:**
- Backtesting engine
- Portfolio construction and optimization
- Risk and performance analytics
- Benchmark evaluation
- Visualization and dashboard systems

**Input:** `data/signals/`  
**Output:** Strategy performance metrics and portfolio analytics

---

## Module Dependency Rules

The architecture enforces a strict one-directional dependency model.

```
data_pipeline
    → data_storage
        → feature_engineering
            → models
                → signals
                    → backtesting
                        → portfolio
                            → analytics
                                → dashboard
```

**Rules:**
- Reverse imports are strictly forbidden
- Downstream modules must not retrain or repair upstream outputs
- Shared utilities belong in `src/utils/`
- All configuration belongs in `config/`
- `backtesting` must NOT import from `models` directly
- No module may break interface contracts without updating `data_schema.md`

---

## Configuration Management

All runtime behavior must be configuration-driven.

**Allowed configuration sources:**
- `.env` — secrets and environment-specific values
- `config/settings.py` — platform-wide settings
- `config/constants.py` — static constants

**Never hardcode:**
- API keys
- Database URLs
- File paths
- Thresholds
- Hyperparameters
- Transaction costs
- Slippage values

**Examples of configuration values:**
- Database connection URLs
- API keys for yfinance, Alpha Vantage, FRED, NewsAPI
- Supported asset classes and symbols
- Historical data start dates
- Update frequencies
- ML model hyperparameters
- Signal thresholds
- Transaction cost and slippage parameters
- Portfolio constraints

No secrets or environment-specific values may appear in source files.

---

## Infrastructure Architecture

The platform runs using containerized services managed by Docker Compose.

### Core Services
- **PostgreSQL** — structured metadata and queryable records
- **MLflow Tracking Server** — experiment tracking and model registry

### Expected Future Service Additions
- Backend API service (FastAPI)
- Scheduler / orchestrator service
- Redis / cache layer
- Frontend UI service

### Containerization
```bash
docker-compose up --build
```

All services must run through Docker. This ensures reproducibility across teammates and environments.

---

## Automation Architecture

The system supports scheduled and repeatable pipeline execution.

### Current Automation Approach
- Cron-compatible workflows
- Script-driven orchestration via `scripts/`

**Entry point scripts:**
```
scripts/run_ingestion.py    # Trigger data ingestion pipeline
scripts/run_features.py     # Trigger feature engineering pipeline
```

### Scheduled Tasks
- Daily market data refresh
- Weekly model retraining
- Daily signal generation
- Recurring backtest refresh
- Benchmark analytics refresh

### Future Automation
- Apache Airflow or equivalent orchestration

---

## Experiment and Research Lifecycle

The full research lifecycle follows this sequence:

```
Acquire Data
    ↓
Validate and Clean
    ↓
Generate Features
    ↓
Train Models
    ↓
Generate Predictions
    ↓
Create Signals
    ↓
Backtest Strategies
    ↓
Construct Portfolios
    ↓
Evaluate Analytics
    ↓
Visualize and Compare
```

Every stage is modular and independently testable. The lifecycle must be reproducible and re-runnable over time.

---

## Financial Research Integrity Rules

All financial research code must avoid:

| Issue | Description |
|---|---|
| Lookahead Bias | Using future data to generate past-period signals |
| Data Leakage | Feature information crossing train/test boundaries |
| Survivorship Bias | Excluding delisted or failed assets from historical tests |
| Unrealistic Execution | Ignoring market impact or assuming perfect fills |
| Ignoring Costs | Omitting transaction fees and slippage from backtest simulation |

**Time Series Validation:**
- Chronological splits only — no random train/test splits
- Walk-forward validation is mandatory
- Rolling retraining must be supported

These rules take priority over engineering convenience.

---

## Machine Learning Standards

### Supported Prediction Tasks
- Time series prediction (future return forecasting)
- Cross-sectional prediction (relative ranking)
- Classification (direction prediction)
- Regression (magnitude prediction)

### Experiment Tracking (MLflow)
Every experiment must log:
- Model type
- Hyperparameters
- Dataset version
- Feature set used
- All evaluation metrics
- Trained model artifacts
- Model version

### Model Evaluation Metrics
| Metric | Description |
|---|---|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |
| Directional Accuracy | % of correct up/down predictions |
| IC | Information Coefficient (correlation of predictions to outcomes) |
| Rank IC | Spearman rank correlation of predictions |

### Model Retraining
- Default retraining schedule: every 7 days
- Retraining flow: load latest features → retrain → evaluate drift → log experiment → deploy

---

## Backtesting Requirements

Backtesting must simulate realistic trading conditions.

**Required simulation parameters:**
- Transaction costs
- Slippage
- Position sizing rules
- Portfolio constraints

**Required performance metrics:**

| Metric | Description |
|---|---|
| Sharpe Ratio | Risk-adjusted return relative to volatility |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Annualized Return | Annualized strategy return |
| Volatility | Annualized standard deviation of returns |
| Win Rate | Percentage of profitable trades |
| Calmar Ratio | Return relative to max drawdown |
| Profit Factor | Gross profit divided by gross loss |

---

## Portfolio Optimization

Supported optimization methods:

| Method | Description |
|---|---|
| Mean Variance Optimization | Maximize Sharpe / minimize variance |
| Risk Parity | Equal risk contribution across assets |
| Maximum Diversification | Maximize portfolio diversification ratio |

Library: `PyPortfolioOpt`

---

## Code Quality Standards

All code in this repository must conform to the following standards:

- **Python version:** 3.12+
- **Style:** PEP 8 formatting
- **Type hints:** Required on all functions and methods
- **Docstrings:** Required on all modules, classes, and public functions
- **Logging:** Required in all pipelines
- **Error handling:** Required in all data and model pipelines
- **Modularity:** No monolithic scripts; max recommended file size ~300 lines
- **No placeholders:** No TODO implementations, pseudo-code, or mock business logic
- **No hardcoded values:** All configuration via `.env` or `config/`

---

## Testing Requirements

Every module must include unit tests using `pytest`.

**Required test coverage:**
- Data transformations
- Feature generators
- Target builders
- Model training pipelines
- Prediction pipelines
- Signal generators
- Strategy evaluation logic

**Test locations:**
```
tests/feature_engineering/
tests/models/
tests/signals/
tests/data_pipeline/
tests/data_storage/
tests/database/
tests/backtesting/
tests/portfolio/
tests/analytics/
tests/utils/
```

Tests must be generated alongside new code unless explicitly instructed otherwise.

---

## Scalability Considerations

### Near-Term
- Support for more assets, symbols, and timeframes
- Support for more feature sets and strategy variants

### Long-Term
- Distributed data pipelines
- GPU-based model training
- Cloud storage integration
- Cloud-native orchestration
- Real-time streaming ingestion
- Larger experiment tracking infrastructure
- Full model registry
- Multi-strategy portfolio research

---

## Long-Term Vision

Future versions of this platform may support:

- Reinforcement learning portfolio agents (PPO, DDPG, SAC)
- Market regime detection and classification
- Concept drift detection and adaptive retraining
- Distributed training across GPU clusters
- Cloud deployment and auto-scaling
- Real-time streaming data ingestion
- Multi-strategy simultaneous portfolio research
- Autonomous AI research assistants
- Airflow-based pipeline orchestration

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/quant-ai-research-platform.git
cd quant-ai-research-platform
```

### 2. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your API keys and database settings
```

### 3. Start core infrastructure services
```bash
docker-compose up --build
```

This will start:
- PostgreSQL on port `5432`
- MLflow Tracking Server on port `5000`

### 4. Install Python dependencies
```bash
pip install -e .
# or
pip install -r requirements.txt
```

### 5. Run data ingestion
```bash
python scripts/run_ingestion.py
```

### 6. Run feature engineering
```bash
python scripts/run_features.py
```

### 7. Run tests
```bash
pytest tests/
```

---

## Contributing

This project follows a strict multi-engineer ownership model.

Before contributing:

1. Review your assigned engineer role file
2. Only modify modules within your ownership boundary
3. Never modify schemas without updating `docs/data_schema.md` and obtaining team agreement
4. All new code must include type hints, docstrings, logging, and tests
5. Never hardcode secrets, file paths, or hyperparameters
6. All experiments must be tracked in MLflow
7. Preserve interface stability — do not break existing public function signatures or data contracts

---

## Schema Authority

`docs/data_schema.md` is the **single source of truth** for all dataset and database schemas.

Any schema modification must be reviewed by:
- Data Engineer
- ML Engineer
- Quant Engineer

before implementation.

If generated code conflicts with the schema specification, **the code is wrong**.

---

*This platform is designed to function as a compact, credible, institutional-style quantitative research lab. Every architectural and implementation decision optimizes for long-term maintainability, research rigor, and professional engineering quality.*
