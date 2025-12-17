# Earnings Reaction ML (FMP + Postgres + Train + Predict)

End-to-end project for **modeling and predicting stock price reactions to earnings** using:
- **Daily OHLCV** stored in Postgres
- **Earnings** from Financial Modeling Prep (FMP)
- **Leak-safe** feature engineering + model selection
- Saved sklearn artifacts for fast inference

## What it predicts

Single target used across the project:

- **`target_gap_1`** = (next trading day **open** / **pre-earnings close**) − 1

Inference outputs:
- `prob_up` / `prob_down` (classifier)
- `expected_ret` (regressor)

---

## Key files

### API + DB utilities
- `fmp_client.py` — resilient FMP HTTP client (rate limiting + retries)
- `db.py` — psycopg2 helpers (`get_conn()`, `bulk_insert()`)
- `config.py` — project config (see `config_dummy.py` template)

### Data loaders (build your Postgres datasets)
- `load_symbols.py` — pulls symbol universe from FMP `/stable/stock-list` and upserts into `fmp.symbols`
- `load_prices.py` — full reload of historical daily prices into `fmp.daily_prices` for all symbols in `fmp.symbols`
- `load_earnings_calendar.py` — full reload of earnings reports into `fmp.earnings_calendar` for all symbols in `fmp.symbols`
- `csv_bulk_loads.py` — creates/loads *fundamentals tables* (e.g., DCF, ratings, scores, key_metrics_ttm, ratios_ttm) from CSVs

### Modeling
- Training pipeline (single-file OOP) — trains classifier + regressor for `target_gap_1`, saves artifacts to `artifacts/`
- `predict.py` — builds a single event row (leak-safe), attaches as-of fundamentals, loads artifacts, prints predictions

---

## Setup

### Install
```bash
pip install numpy pandas scikit-learn sqlalchemy requests joblib psycopg2-binary tqdm

```

### Configure

Copy `config_dummy.py` → `config.py` and set:

-   `PG_DSN`
    
-   `FMP_API_KEY`
    
-   `REQUESTS_PER_SECOND`
    

You can also set `PG_DSN` via environment variable.

----------

## Recommended workflow

<img width="1705" height="628" alt="files and table details" src="https://github.com/user-attachments/assets/ce49ac48-60b1-417d-810a-b1111e7fc0f2" />


### 1) Load fundamentals CSVs (optional, if you have the CSV exports)

This creates the `fmp` schema + fundamentals tables and bulk loads from a `Data/` folder:

```bash
python csv_bulk_loads.py

```

> Note: `csv_bulk_loads.py` assumes `Code/` and `Data/` are sibling directories (script in `Code/`, CSVs in `Data/`).

### 2) Load symbol universe

```bash
python load_symbols.py

```

### 3) Load daily prices (full reload)

```bash
python load_prices.py

```

### 4) Load earnings calendar (full reload)

```bash
python load_earnings_calendar.py

```

### 5) Train models (produces artifacts)

```bash
python train_pipeline.py

```

Artifacts:

```
artifacts/
  target_gap_1_cls.joblib
  target_gap_1_reg.joblib
  target_gap_1_meta.json

```
<img width="679" height="106" alt="Model_Training_Results" src="https://github.com/user-attachments/assets/59b1aa9a-736a-4f8e-979a-882fcd71feb9" />


### 6) Predict

```bash
python predict_earnings_all_features.py

```
<img width="1392" height="510" alt="Sample_Prediction" src="https://github.com/user-attachments/assets/7011fca0-cdee-41d7-8a10-a4456766b10b" />


----------

## Notes

-   The loaders `load_prices.py` / `load_earnings_calendar.py` are designed as **full reloads** (truncate table → reload all symbols).
    
-   The modeling code is **leak-safe by design** (pre-event feature shifting, as-of fundamentals, train-only winsorization).
    
-   Research code only — not financial advice.
