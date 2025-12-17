#!/usr/bin/env python3


import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import requests

from sqlalchemy import create_engine, text
from sklearn.base import BaseEstimator, TransformerMixin

from config import PG_DSN, FMP_API_KEY, REQUESTS_PER_SECOND


class InfToNan(BaseEstimator, TransformerMixin):
    def __init__(self, max_abs=1e12):
        self.max_abs = float(max_abs)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X[~np.isfinite(X)] = np.nan
        X[np.abs(X) > self.max_abs] = np.nan
        return X


class TrainOnlyWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, p: float = 0.01):
        self.p = float(p)
        self.lo_ = None
        self.hi_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.p, axis=0)
        self.hi_ = np.nanquantile(X, 1.0 - self.p, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_)



def safe_div(numer, denom, floor=1e-6):
    denom = denom.copy()
    denom = denom.where(np.abs(denom) >= floor, np.nan)
    return numer / denom

def signed_log1p(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def _pick_series(df: pd.DataFrame, *candidates, default=np.nan) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df), index=df.index)

def _throttle(rps: float):
    if not rps or rps <= 0:
        return
    time.sleep(1.0 / float(rps))

def load_earnings_from_fmp(symbol: str, start: str, end: str, apikey: str, rps: float = 20.0) -> pd.DataFrame:
   
    if not apikey or apikey.strip().lower() == "key":
        raise RuntimeError("FMP_API_KEY is not set (config.py has default 'key').")

    _throttle(rps)

    url = "https://financialmodelingprep.com/stable/earnings"
    resp = requests.get(url, params={"symbol": symbol, "apikey": apikey}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        raise ValueError(f"Unexpected FMP response (expected list), got: {type(data)}")

    raw = pd.DataFrame(data)
    if raw.empty:
        return pd.DataFrame(columns=["symbol_earnings", "date_earnings", "eps", "eps_est", "rev", "rev_est"])

    out = pd.DataFrame({
        "symbol_earnings": _pick_series(raw, "symbol", "Symbol"),
        "date_earnings": pd.to_datetime(_pick_series(raw, "date", "Date"), errors="coerce"),

        "eps":     pd.to_numeric(_pick_series(raw, "eps", "Eps", "epsActual", "eps_actual"), errors="coerce"),
        "eps_est": pd.to_numeric(_pick_series(raw, "epsEstimated", "EpsEstimated", "eps_estimated"), errors="coerce"),
        "rev":     pd.to_numeric(_pick_series(raw, "revenue", "Revenue", "revenueActual", "revenue_actual"), errors="coerce"),
        "rev_est": pd.to_numeric(_pick_series(raw, "revenueEstimated", "RevenueEstimated", "revenue_estimated"), errors="coerce"),
    })

    out["symbol_earnings"] = out["symbol_earnings"].fillna(symbol)
    out = out.dropna(subset=["date_earnings"])

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    out = out[(out["date_earnings"] >= start_dt) & (out["date_earnings"] <= end_dt)].copy()

    return out.sort_values(["symbol_earnings", "date_earnings"]).reset_index(drop=True)



def load_metrics_asof(pg_dsn: str, symbol: str, asof_date) -> pd.DataFrame:
    
    engine = create_engine(pg_dsn)
    asof_date = pd.to_datetime(asof_date).date()

    with engine.connect() as conn:
        dcf = pd.read_sql(text("""
            SELECT "date" AS dcf_date,
                   dcf    AS dcf_value,
                   stock_price AS dcf_stock_price
            FROM fmp.dcf
            WHERE symbol = :sym AND "date" <= :asof
            ORDER BY "date" DESC
            LIMIT 1
        """), conn, params={"sym": symbol, "asof": asof_date})

        ratings = pd.read_sql(text("""
            SELECT "date" AS ratings_date,
                   discounted_cash_flow_score,
                   return_on_equity_score,
                   return_on_assets_score,
                   debt_to_equity_score,
                   price_to_earnings_score,
                   price_to_book_score
            FROM fmp.ratings
            WHERE symbol = :sym AND "date" <= :asof
            ORDER BY "date" DESC
            LIMIT 1
        """), conn, params={"sym": symbol, "asof": asof_date})

        scores = pd.read_sql(text("""
            SELECT
              altman_z_score,
              piotroski_score,
              working_capital,
              total_assets,
              retained_earnings,
              ebit,
              market_cap,
              total_liabilities
            FROM fmp.scores
            WHERE symbol = :sym
            LIMIT 1
        """), conn, params={"sym": symbol})

        km = pd.read_sql(text("""
            SELECT
              enterprise_value_ttm,
              net_debt_to_ebitda_ttm,
              current_ratio_ttm,
              return_on_assets_ttm,
              return_on_equity_ttm,
              earnings_yield_ttm,
              free_cash_flow_yield_ttm,
              capex_to_revenue_ttm,
              working_capital_ttm
            FROM fmp.key_metrics_ttm
            WHERE symbol = :sym
            LIMIT 1
        """), conn, params={"sym": symbol})

        rt = pd.read_sql(text("""
            SELECT
              gross_profit_margin_ttm,
              ebit_margin_ttm,
              ebitda_margin_ttm,
              operating_profit_margin_ttm,
              net_profit_margin_ttm,
              debt_to_equity_ratio_ttm,
              price_to_earnings_ratio_ttm,
              price_to_book_ratio_ttm,
              price_to_sales_ratio_ttm,
              dividend_yield_ttm,
              enterprise_value_multiple_ttm,
              effective_tax_rate_ttm
            FROM fmp.ratios_ttm
            WHERE symbol = :sym
            LIMIT 1
        """), conn, params={"sym": symbol})

    parts = [dcf, ratings, scores, km, rt]
    out = pd.concat([p.reset_index(drop=True) for p in parts if p is not None and not p.empty], axis=1)

    if out.empty:
        return pd.DataFrame([{}])

    return out.iloc[:1].copy()


def build_event_row(df_daily_earn, symbol, event_date,
                    surprise_pct_clip=2.0, vol_chg_clip=3.0, inter_clip=6.0,
                    date_mode="on_or_before", max_days_tolerance=10):
   
    df = df_daily_earn.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    event_date = pd.to_datetime(event_date)

    s = df[df["symbol"] == symbol].copy()
    if s.empty:
        raise ValueError(f"No rows for symbol={symbol} in loaded window.")

    available = np.array(sorted(s["date"].dropna().unique()))
    if len(available) == 0:
        raise ValueError(f"No valid dates for symbol={symbol} in loaded window.")

    if date_mode == "exact":
        eff_date = event_date
        if eff_date not in set(available):
            raise ValueError(f"Exact date {event_date.date()} not found for {symbol}.")
    elif date_mode == "on_or_before":
        le = available[available <= event_date]
        eff_date = le[-1] if len(le) else available[0]
    elif date_mode == "nearest":
        idx = np.argmin(np.abs(available - event_date))
        eff_date = available[idx]
    else:
        raise ValueError("date_mode must be one of: exact, on_or_before, nearest")

    gap_days = int(abs((pd.Timestamp(eff_date) - pd.Timestamp(event_date)).days))
    if gap_days > max_days_tolerance:
        raise ValueError(
            f"Requested date {event_date.date()} but nearest usable trading date is {pd.Timestamp(eff_date).date()} "
            f"({gap_days} days away). Increase lookback/tolerance or fix your dates."
        )

    g = df.groupby("symbol", sort=False)

    df["pre_close"] = g["close"].shift(1)

    df["ret_1d_pre"]  = g["close"].pct_change(1).shift(1)
    df["ret_5d_pre"]  = g["close"].pct_change(5).shift(1)
    df["ret_21d_pre"] = g["close"].pct_change(21).shift(1)
    df["ret_63d_pre"] = g["close"].pct_change(63).shift(1)

    ret_1d = g["close"].pct_change(1)
    df["vol_21d_pre"] = ret_1d.rolling(21, min_periods=10).std().shift(1).reset_index(level=0, drop=True)
    df["vol_63d_pre"] = ret_1d.rolling(63, min_periods=20).std().shift(1).reset_index(level=0, drop=True)

    df["dollar_vol"] = df["close"] * df["volume"]
    df["log_dollar_vol_21d"] = (
        g["dollar_vol"].rolling(21, min_periods=10).mean().shift(1).pipe(np.log1p).reset_index(level=0, drop=True)
    )

    vol_chg_1d = g["volume"].pct_change(1)
    df["vol_chg_1d_pre"] = vol_chg_1d.shift(1).clip(-vol_chg_clip, vol_chg_clip)

    vol_mean = g["volume"].rolling(21, min_periods=10).mean().shift(1).reset_index(level=0, drop=True)
    vol_std  = g["volume"].rolling(21, min_periods=10).std().shift(1).reset_index(level=0, drop=True)
    vol_std = vol_std.where(vol_std > 0, np.nan)
    df["vol_z_21d_pre"] = (df["volume"].shift(1) - vol_mean) / vol_std

    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14_pre"] = tr.groupby(df["symbol"]).rolling(14, min_periods=10).mean().shift(1).reset_index(level=0, drop=True)
    df["atr_14_pct_pre"] = safe_div(df["atr_14_pre"], df["pre_close"])

    df.loc[df["eps_est"].abs() < 1e-6, "eps_est"] = np.nan
    df.loc[df["rev_est"].abs() < 1e-6, "rev_est"] = np.nan

    df["eps_surprise"] = df["eps"] - df["eps_est"]
    df["rev_surprise"] = df["rev"] - df["rev_est"]

    df["eps_surprise_pct"] = safe_div(df["eps_surprise"], df["eps_est"].abs()).clip(-surprise_pct_clip, surprise_pct_clip)
    df["rev_surprise_pct"] = safe_div(df["rev_surprise"], df["rev_est"].abs()).clip(-surprise_pct_clip, surprise_pct_clip)

    df["eps_surprise_pct_slog"] = signed_log1p(df["eps_surprise_pct"])
    df["rev_surprise_pct_slog"] = signed_log1p(df["rev_surprise_pct"])

    df["eps_mag"] = np.abs(df["eps_surprise_pct"])
    df["rev_mag"] = np.abs(df["rev_surprise_pct"])
    df["eps_pos"] = (df["eps_surprise_pct"] > 0).astype("int8")
    df["rev_pos"] = (df["rev_surprise_pct"] > 0).astype("int8")

    df["eps_x_volchg"] = (df["eps_surprise_pct"] * df["vol_chg_1d_pre"]).clip(-inter_clip, inter_clip)
    df["rev_x_volchg"] = (df["rev_surprise_pct"] * df["vol_chg_1d_pre"]).clip(-inter_clip, inter_clip)

    df["is_earnings_day"] = df["eps"].notna()

    earn_days = (
        df.loc[df["is_earnings_day"], ["symbol", "date"]]
          .drop_duplicates()
          .sort_values(["symbol", "date"])
          .copy()
    )
    earn_days["prev_earn_date"] = earn_days.groupby("symbol")["date"].shift(1)
    earn_days["days_since_earn"] = (earn_days["date"] - earn_days["prev_earn_date"]).dt.days

    df = df.merge(
        earn_days[["symbol", "date", "days_since_earn"]],
        on=["symbol", "date"],
        how="left"
    )

    earn_events = (
        df.loc[df["is_earnings_day"], ["symbol", "date", "eps", "rev"]]
          .drop_duplicates(["symbol", "date"], keep="last")
          .sort_values(["symbol", "date"])
          .copy()
    )

    earn_events["eps_ttm"] = (
        earn_events.groupby("symbol")["eps"]
                   .rolling(4, min_periods=4).sum()
                   .shift(1)
                   .reset_index(level=0, drop=True)
    )

    earn_events["eps_prev_q"] = earn_events.groupby("symbol")["eps"].shift(1)
    earn_events["rev_prev_q"] = earn_events.groupby("symbol")["rev"].shift(1)
    earn_events["eps_prev_y"] = earn_events.groupby("symbol")["eps"].shift(4)
    earn_events["rev_prev_y"] = earn_events.groupby("symbol")["rev"].shift(4)

    earn_events["eps_qoq"] = safe_div(
        earn_events["eps"] - earn_events["eps_prev_q"],
        earn_events["eps_prev_q"].abs()
    ).clip(-surprise_pct_clip, surprise_pct_clip)

    earn_events["rev_qoq"] = safe_div(
        earn_events["rev"] - earn_events["rev_prev_q"],
        earn_events["rev_prev_q"].abs()
    ).clip(-surprise_pct_clip, surprise_pct_clip)

    earn_events["eps_yoy"] = safe_div(
        earn_events["eps"] - earn_events["eps_prev_y"],
        earn_events["eps_prev_y"].abs()
    ).clip(-surprise_pct_clip, surprise_pct_clip)

    earn_events["rev_yoy"] = safe_div(
        earn_events["rev"] - earn_events["rev_prev_y"],
        earn_events["rev_prev_y"].abs()
    ).clip(-surprise_pct_clip, surprise_pct_clip)

    df = df.merge(
        earn_events[["symbol", "date", "eps_ttm", "eps_qoq", "rev_qoq", "eps_yoy", "rev_yoy"]],
        on=["symbol", "date"],
        how="left"
    )

    df["eps_ttm_pos"] = (df["eps_ttm"] > 0).astype("int8")

    df["pe_pre"] = np.where(
        (df["eps_ttm"] > 0) & (df["pre_close"] > 0),
        df["pre_close"] / df["eps_ttm"],
        np.nan
    )
    df["log_pe_pre"] = np.log(df["pe_pre"])
    df["earnings_yield_pre"] = np.where(
        df["pre_close"] > 0,
        df["eps_ttm"] / df["pre_close"],
        np.nan
    )

    row = df[(df["symbol"] == symbol) & (df["date"] == pd.to_datetime(eff_date))].copy()
    if row.empty:
        raise ValueError(f"Could not build row for {symbol} at effective date {pd.Timestamp(eff_date).date()}")

    row["requested_date"] = event_date
    row["effective_date"] = pd.to_datetime(eff_date)
    row["date_gap_days"] = gap_days

    row["dcf_to_preclose"] = np.nan

    return row


def load_data_from_postgres(pg_dsn, symbol, event_date, lookback_days=400,
                           fmp_apikey=None, rps=20.0):
    engine = create_engine(pg_dsn)
    event_date = pd.to_datetime(event_date)

    start = (event_date - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end   = (event_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    daily_sql = text("""
        SELECT symbol, date, open, high, low, close, volume
        FROM fmp.daily_prices_analysis
        WHERE symbol = :sym AND date BETWEEN :start AND :end
        ORDER BY symbol, date
    """)

    params = {"sym": symbol, "start": start, "end": end}

    with engine.connect() as conn:
        daily_df = pd.read_sql(daily_sql, conn, params=params)

    daily_df["date"] = pd.to_datetime(daily_df["date"])

    earn_df = load_earnings_from_fmp(symbol, start, end, fmp_apikey, rps=rps)

    df = daily_df.merge(
        earn_df,
        left_on=["symbol", "date"],
        right_on=["symbol_earnings", "date_earnings"],
        how="left"
    )

    return df


def predict_one(symbol, date, target_key,
                artifacts_dir, pg_dsn,
                fmp_apikey, rps=20.0,
                eps_actual=None, eps_est=None, rev_actual=None, rev_est=None):

    target_col = f"target_{target_key}"
    meta_path = os.path.join(artifacts_dir, f"{target_col}_meta.json")
    cls_path  = os.path.join(artifacts_dir, f"{target_col}_cls.joblib")
    reg_path  = os.path.join(artifacts_dir, f"{target_col}_reg.joblib")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    cls_model = joblib.load(cls_path)
    reg_model = joblib.load(reg_path)

    df = load_data_from_postgres(pg_dsn, symbol, date, lookback_days=450, fmp_apikey=fmp_apikey, rps=rps)

    req = pd.to_datetime(date)
    s = df[df["symbol"] == symbol].copy()
    s["date"] = pd.to_datetime(s["date"])
    avail = np.array(sorted(s["date"].dropna().unique()))
    if len(avail) == 0:
        raise ValueError(f"No daily dates available for {symbol} in pulled window.")

    le = avail[avail <= req]
    eff = le[-1] if len(le) else avail[0]

    mask = (df["symbol"] == symbol) & (df["date"] == pd.to_datetime(eff))
    if eps_actual is not None: df.loc[mask, "eps"] = float(eps_actual)
    if eps_est    is not None: df.loc[mask, "eps_est"] = float(eps_est)
    if rev_actual is not None: df.loc[mask, "rev"] = float(rev_actual)
    if rev_est    is not None: df.loc[mask, "rev_est"] = float(rev_est)

    row = build_event_row(
        df, symbol, date,
        surprise_pct_clip=2.0, vol_chg_clip=3.0, inter_clip=6.0,
        date_mode="on_or_before", max_days_tolerance=10
    )

    print(
        f"[INFO] {symbol}: requested_date={pd.to_datetime(date).date()} "
        f"effective_date={pd.to_datetime(row['effective_date'].iloc[0]).date()} "
        f"gap_days={int(row['date_gap_days'].iloc[0])}"
    )

    asof_date = pd.to_datetime(row["effective_date"].iloc[0]) - pd.Timedelta(days=1)
    m = load_metrics_asof(pg_dsn, symbol, asof_date)

    for c in m.columns:
        row[c] = m.iloc[0][c]

    if "dcf_value" in row.columns and "pre_close" in row.columns:
        row["dcf_to_preclose"] = (
            pd.to_numeric(row["dcf_value"], errors="coerce") / pd.to_numeric(row["pre_close"], errors="coerce")
        ) - 1.0

    for c in feature_cols:
        if c not in row.columns:
            row[c] = np.nan

    X = row[feature_cols].astype(float).values

    proba = cls_model.predict_proba(X)[0]
    prob_down = float(proba[0])
    prob_up   = float(proba[1])
    exp_ret   = float(reg_model.predict(X)[0])

    print("\nPREDICTION")
    print("----------")
    print(f"symbol       : {symbol}")
    print(f"event_date   : {date}")
    print(f"target       : {target_col}")
    print(f"prob_up      : {prob_up:.4f}")
    print(f"prob_down    : {prob_down:.4f}")
    print(f"expected_ret : {exp_ret:.4%}")

    cols = [
        "eps_est", "eps", "rev_est", "rev",
        "eps_surprise_pct", "rev_surprise_pct",
        "eps_surprise_pct_slog", "rev_surprise_pct_slog",
        "eps_qoq", "rev_qoq", "eps_yoy", "rev_yoy",
        "dcf_value", "dcf_to_preclose",
        "piotroski_score", "altman_z_score"
    ]
    cols = [c for c in cols if c in row.columns]
    if cols:
        print("\nINPUT SNAPSHOT")
        print("--------------")
        print(row[cols].tail(1).to_string(index=False))


if __name__ == "__main__":
    ARTIFACTS_DIR = "artifacts"
    EVENT_DATE = "2025-12-08"
    TARGET_KEY = "gap_1"   

    injections = {}

    injections["AVGO"] = dict(
        eps_actual=1.95,
        eps_est=1.87,
        rev_actual=18259111720.0,
        rev_est=17459111720.0,
    )
    
    injections["COST"] = dict(
        eps_actual=4.5,
        eps_est=4.26,
        rev_actual=67085401740.0,
        rev_est=67115401740.0,
    )
    
    injections["LULU"] = dict(
        eps_actual=2.59,
        eps_est=2.22,
        rev_actual=2608637550.0,
        rev_est=2478637550.0,
    )
    
    injections["NTSK"] = dict(
        eps_actual=-0.1,
        eps_est=-0.25158,
        rev_actual=184049180.0,
        rev_est=176049180.0,
    )
    
    injections["RH"] = dict(
        eps_actual=None,
        eps_est=2.13,
        rev_actual=None,
        rev_est=883654010.0,
    )
    
    injections["KMTS"] = dict(
        eps_actual=-0.64,
        eps_est=-0.56,
        rev_actual=22663250.0,
        rev_est=20963250.0,
    )
    
    injections["NX"] = dict(
        eps_actual=0.83,
        eps_est=0.50,
        rev_actual=489844250.0,
        rev_est=470744250.0,
    )
    
    injections["MITK"] = dict(
        eps_actual=0.24,
        eps_est=0.18,
        rev_actual=44855200,
        rev_est=40855200.0,
    )

    injections["FEIM"] = dict(
        eps_actual=None,
        eps_est=0.295,
        rev_actual=None,
        rev_est=16871500.0,
    )
    for sym, inj in injections.items():
        predict_one(
            symbol=sym,
            date=EVENT_DATE,
            target_key=TARGET_KEY,
            artifacts_dir=ARTIFACTS_DIR,
            pg_dsn=PG_DSN,
            fmp_apikey=FMP_API_KEY,
            rps=REQUESTS_PER_SECOND,
            eps_actual=inj["eps_actual"],
            eps_est=inj["eps_est"],
            rev_actual=inj["rev_actual"],
            rev_est=inj["rev_est"],
        )
