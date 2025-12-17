#!/usr/bin/env python3

import os
import time
import warnings
from dataclasses import dataclass
from contextlib import contextmanager
import joblib
import json
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, roc_auc_score, f1_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


@contextmanager
def section_timer(name: str):
    print(f"\n[START] {name}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[END] {name} — {time.perf_counter() - t0:.2f} sec")


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


@dataclass
class PipelineConfig:
    pg_dsn: str

    winsor_p: float = 0.01  
    surprise_pct_clip: float = 2.0    
    vol_chg_clip: float = 3.0
    inter_clip: float = 6.0
    target_clip: float = 0.50         

    target_key: str = "gap_1"

    min_hist_days: int = 260  

    train_frac: float = 0.60
    bt_frac: float = 0.20  

    n_jobs: int = -1



class PostgresDataLoader:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.engine = create_engine(cfg.pg_dsn)

    def load(self):
        daily_sql = """
            SELECT symbol, date, open, high, low, close, volume
            FROM fmp.daily_prices_analysis
            ORDER BY symbol, date
        """

        earn_sql = """
            SELECT symbol, report_date, eps_actual, eps_estimated,
                   revenue_actual, revenue_estimated
            FROM fmp.earnings_calendar_analysis
            ORDER BY symbol, report_date
        """

        dcf_sql = """
            SELECT symbol, "date", dcf, stock_price, loaded_at
            FROM fmp.dcf
            ORDER BY symbol, "date"
        """

        ratings_sql = """
            SELECT symbol, "date",
                   rating,
                   discounted_cash_flow_score,
                   return_on_equity_score,
                   return_on_assets_score,
                   debt_to_equity_score,
                   price_to_earnings_score,
                   price_to_book_score,
                   loaded_at
            FROM fmp.ratings
            ORDER BY symbol, "date"
        """

        scores_sql = """
            SELECT symbol,
                   altman_z_score,
                   piotroski_score,
                   working_capital,
                   total_assets,
                   retained_earnings,
                   ebit,
                   market_cap,
                   total_liabilities,
                   revenue,
                   loaded_at
            FROM fmp.scores
        """

        key_metrics_sql = """
            SELECT symbol,
                   market_cap,
                   enterprise_value_ttm,
                   net_debt_to_ebitda_ttm,
                   current_ratio_ttm,
                   return_on_assets_ttm,
                   return_on_equity_ttm,
                   earnings_yield_ttm,
                   free_cash_flow_yield_ttm,
                   capex_to_revenue_ttm,
                   working_capital_ttm,
                   loaded_at
            FROM fmp.key_metrics_ttm
        """

        ratios_sql = """
            SELECT symbol,
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
                   effective_tax_rate_ttm,
                   loaded_at
            FROM fmp.ratios_ttm
        """

        daily_df = pd.read_sql(daily_sql, self.engine)
        earn_df  = pd.read_sql(earn_sql,  self.engine)

        dcf_df     = pd.read_sql(dcf_sql, self.engine)
        ratings_df = pd.read_sql(ratings_sql, self.engine)
        scores_df  = pd.read_sql(scores_sql, self.engine)
        km_df      = pd.read_sql(key_metrics_sql, self.engine)
        rt_df      = pd.read_sql(ratios_sql, self.engine)

        daily_df["date"] = pd.to_datetime(daily_df["date"])
        earn_df["report_date"] = pd.to_datetime(earn_df["report_date"])
        dcf_df["date"] = pd.to_datetime(dcf_df["date"])
        ratings_df["date"] = pd.to_datetime(ratings_df["date"])

        earn_df = earn_df.rename(columns={
            "symbol": "symbol_earnings",
            "report_date": "date_earnings",
            "eps_actual": "eps",
            "eps_estimated": "eps_est",
            "revenue_actual": "rev",
            "revenue_estimated": "rev_est",
        })

        full = daily_df.merge(
            earn_df,
            left_on=["symbol", "date"],
            right_on=["symbol_earnings", "date_earnings"],
            how="left",
        )

        metrics = {
            "dcf": dcf_df,
            "ratings": ratings_df,
            "scores": scores_df,
            "key_metrics_ttm": km_df,
            "ratios_ttm": rt_df,
        }
        return full, metrics



class EarningsFeatureEngineer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    @staticmethod
    def _safe_div(numer, denom, floor=1e-6):
        denom = denom.copy()
        denom = denom.where(np.abs(denom) >= floor, np.nan)
        return numer / denom

    @staticmethod
    def _signed_log1p(x: pd.Series) -> pd.Series:
        return np.sign(x) * np.log1p(np.abs(x))

    def build(self, df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        g = df.groupby("symbol", sort=False)

        df["pre_close"] = g["close"].shift(1)
        df["pre_open"]  = g["open"].shift(1)

        df["ret_1d_pre"]  = g["close"].pct_change(1).shift(1)
        df["ret_5d_pre"]  = g["close"].pct_change(5).shift(1)
        df["ret_21d_pre"] = g["close"].pct_change(21).shift(1)
        df["ret_63d_pre"] = g["close"].pct_change(63).shift(1)

        ret_1d = g["close"].pct_change(1)
        df["vol_21d_pre"] = (
            ret_1d.rolling(21, min_periods=10).std().shift(1).reset_index(level=0, drop=True)
        )
        df["vol_63d_pre"] = (
            ret_1d.rolling(63, min_periods=20).std().shift(1).reset_index(level=0, drop=True)
        )

        df["dollar_vol"] = df["close"] * df["volume"]
        df["log_dollar_vol_21d"] = (
            g["dollar_vol"]
            .rolling(21, min_periods=10).mean()
            .shift(1)
            .pipe(np.log1p)
            .reset_index(level=0, drop=True)
        )

        vol_chg_1d = g["volume"].pct_change(1)
        df["vol_chg_1d_pre"] = vol_chg_1d.shift(1)

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

        df["atr_14_pre"] = (
            tr.groupby(df["symbol"])
              .rolling(14, min_periods=10).mean()
              .shift(1)
              .reset_index(level=0, drop=True)
        )
        df["atr_14_pct_pre"] = self._safe_div(df["atr_14_pre"], df["pre_close"])

        df.loc[df["eps_est"].abs() < 1e-6, "eps_est"] = np.nan
        df.loc[df["rev_est"].abs() < 1e-6, "rev_est"] = np.nan

        df["eps_surprise"] = df["eps"] - df["eps_est"]
        df["rev_surprise"] = df["rev"] - df["rev_est"]

        df["eps_surprise_pct"] = self._safe_div(df["eps_surprise"], df["eps_est"].abs())
        df["rev_surprise_pct"] = self._safe_div(df["rev_surprise"], df["rev_est"].abs())

        df["eps_surprise_pct"] = df["eps_surprise_pct"].clip(-self.cfg.surprise_pct_clip, self.cfg.surprise_pct_clip)
        df["rev_surprise_pct"] = df["rev_surprise_pct"].clip(-self.cfg.surprise_pct_clip, self.cfg.surprise_pct_clip)
        df["vol_chg_1d_pre"] = df["vol_chg_1d_pre"].clip(-self.cfg.vol_chg_clip, self.cfg.vol_chg_clip)

        df["eps_surprise_pct_slog"] = self._signed_log1p(df["eps_surprise_pct"])
        df["rev_surprise_pct_slog"] = self._signed_log1p(df["rev_surprise_pct"])

        df["eps_mag"] = np.abs(df["eps_surprise_pct"])
        df["rev_mag"] = np.abs(df["rev_surprise_pct"])
        df["eps_pos"] = (df["eps_surprise_pct"] > 0).astype("int8")
        df["rev_pos"] = (df["rev_surprise_pct"] > 0).astype("int8")

        df["eps_x_volchg"] = (df["eps_surprise_pct"] * df["vol_chg_1d_pre"]).clip(-self.cfg.inter_clip, self.cfg.inter_clip)
        df["rev_x_volchg"] = (df["rev_surprise_pct"] * df["vol_chg_1d_pre"]).clip(-self.cfg.inter_clip, self.cfg.inter_clip)

        df["is_earnings_day"] = df["eps"].notna()

        earn_days = df.loc[df["is_earnings_day"], ["symbol", "date"]].drop_duplicates().sort_values(["symbol", "date"])
        earn_days["prev_earn_date"] = earn_days.groupby("symbol")["date"].shift(1)
        earn_days["days_since_earn"] = (earn_days["date"] - earn_days["prev_earn_date"]).dt.days
        df = df.merge(earn_days[["symbol", "date", "days_since_earn"]], on=["symbol", "date"], how="left")

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

        earn_events["eps_qoq"] = self._safe_div(
            earn_events["eps"] - earn_events["eps_prev_q"],
            earn_events["eps_prev_q"].abs()
        ).clip(-self.cfg.surprise_pct_clip, self.cfg.surprise_pct_clip)

        earn_events["rev_qoq"] = self._safe_div(
            earn_events["rev"] - earn_events["rev_prev_q"],
            earn_events["rev_prev_q"].abs()
        ).clip(-self.cfg.surprise_pct_clip, self.cfg.surprise_pct_clip)

        earn_events["eps_yoy"] = self._safe_div(
            earn_events["eps"] - earn_events["eps_prev_y"],
            earn_events["eps_prev_y"].abs()
        ).clip(-self.cfg.surprise_pct_clip, self.cfg.surprise_pct_clip)

        earn_events["rev_yoy"] = self._safe_div(
            earn_events["rev"] - earn_events["rev_prev_y"],
            earn_events["rev_prev_y"].abs()
        ).clip(-self.cfg.surprise_pct_clip, self.cfg.surprise_pct_clip)

        df = df.merge(
            earn_events[["symbol", "date", "eps_ttm", "eps_qoq", "rev_qoq", "eps_yoy", "rev_yoy"]],
            on=["symbol", "date"],
            how="left"
        )

        df["pe_pre"] = np.where(
            (df["eps_ttm"] > 0) & (df["pre_close"] > 0),
            df["pre_close"] / df["eps_ttm"],
            np.nan
        )
        df["log_pe_pre"] = np.log(df["pe_pre"])
        df["earnings_yield_pre"] = np.where(df["pre_close"] > 0, df["eps_ttm"] / df["pre_close"], np.nan)
        df["eps_ttm_pos"] = (df["eps_ttm"] > 0).astype("int8")

        events = df.loc[df["is_earnings_day"]].copy()
        events = events.dropna(subset=["pre_close"]).copy()

        events["asof_date"] = events["date"] - pd.Timedelta(days=1)
        events = events.sort_values(["asof_date", "symbol"]).reset_index(drop=True)

        scores_df = metrics["scores"].copy()
        km_df     = metrics["key_metrics_ttm"].copy()
        rt_df     = metrics["ratios_ttm"].copy()

        events = events.merge(scores_df, on="symbol", how="left", suffixes=("", "_scores"))
        events = events.merge(km_df,     on="symbol", how="left", suffixes=("", "_km"))
        events = events.merge(rt_df,     on="symbol", how="left", suffixes=("", "_rt"))

        dcf_df = metrics["dcf"].copy()
        dcf_df["date"] = pd.to_datetime(dcf_df["date"])
        dcf_df = dcf_df.rename(columns={
            "date": "dcf_date",
            "dcf": "dcf_value",
            "stock_price": "dcf_stock_price",
            "loaded_at": "dcf_loaded_at",
        })

        if "dcf_loaded_at" in dcf_df.columns:
            dcf_df = (dcf_df.sort_values(["symbol", "dcf_date", "dcf_loaded_at"])
                            .drop_duplicates(["symbol", "dcf_date"], keep="last"))

        dcf_df = dcf_df.sort_values(["dcf_date", "symbol"]).reset_index(drop=True)

        ratings_df = metrics["ratings"].copy()
        ratings_df["date"] = pd.to_datetime(ratings_df["date"])

        ratings_df = ratings_df.rename(columns={
            "date": "ratings_date",
            "loaded_at": "ratings_loaded_at",
        })

        if "ratings_loaded_at" in ratings_df.columns:
            ratings_df = (ratings_df.sort_values(["symbol", "ratings_date", "ratings_loaded_at"])
                                .drop_duplicates(["symbol", "ratings_date"], keep="last"))

        ratings_df = ratings_df.sort_values(["ratings_date", "symbol"]).reset_index(drop=True)

        events = pd.merge_asof(
            events, dcf_df,
            left_on="asof_date", right_on="dcf_date",
            by="symbol", direction="backward"
        )

        events = pd.merge_asof(
            events, ratings_df,
            left_on="asof_date", right_on="ratings_date",
            by="symbol", direction="backward"
        )

        events["dcf_to_preclose"] = (events["dcf_value"] / events["pre_close"]) - 1.0

        feature_cols = [
            "eps_surprise_pct", "rev_surprise_pct",
            "eps_surprise_pct_slog", "rev_surprise_pct_slog",
            "eps_mag", "rev_mag",
            "eps_pos", "rev_pos",

            "vol_chg_1d_pre", "vol_z_21d_pre",
            "log_dollar_vol_21d",
            "vol_21d_pre", "vol_63d_pre",
            "atr_14_pct_pre",

            "ret_1d_pre", "ret_5d_pre", "ret_21d_pre", "ret_63d_pre",

            "eps_ttm", "eps_ttm_pos",
            "pe_pre", "log_pe_pre", "earnings_yield_pre",

            "eps_qoq", "rev_qoq",
            "eps_yoy", "rev_yoy",

            "days_since_earn",

            "eps_x_volchg", "rev_x_volchg",

            "dcf_to_preclose",

            "discounted_cash_flow_score",
            "return_on_equity_score",
            "return_on_assets_score",
            "debt_to_equity_score",
            "price_to_earnings_score",
            "price_to_book_score",

            "altman_z_score",
            "piotroski_score",
            "market_cap",
            "total_liabilities",
            "working_capital",
            "total_assets",
            "retained_earnings",
            "ebit",

            "enterprise_value_ttm",
            "net_debt_to_ebitda_ttm",
            "current_ratio_ttm",
            "return_on_assets_ttm",
            "return_on_equity_ttm",
            "earnings_yield_ttm",
            "free_cash_flow_yield_ttm",
            "capex_to_revenue_ttm",
            "working_capital_ttm",

            "gross_profit_margin_ttm",
            "ebit_margin_ttm",
            "ebitda_margin_ttm",
            "operating_profit_margin_ttm",
            "net_profit_margin_ttm",
            "debt_to_equity_ratio_ttm",
            "price_to_earnings_ratio_ttm",
            "price_to_book_ratio_ttm",
            "price_to_sales_ratio_ttm",
            "dividend_yield_ttm",
            "enterprise_value_multiple_ttm",
            "effective_tax_rate_ttm",
        ]

        for c in feature_cols:
            events[c] = pd.to_numeric(events[c], errors="coerce")

        keep = ["symbol", "date", "pre_close", "open", "close"] + feature_cols
        events = events[keep].copy()

        return events, feature_cols


class TargetBuilder:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def add_targets(self, events: pd.DataFrame, full_daily: pd.DataFrame) -> pd.DataFrame:
      
        df = full_daily.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
        g = df.groupby("symbol", sort=False)

        df["open_fwd_1"] = g["open"].shift(-1)

        map_df = df[["symbol", "date", "open_fwd_1"]].copy()

        out = events.merge(map_df, on=["symbol", "date"], how="left")

        out["target_gap_1"] = (out["open_fwd_1"] / out["pre_close"]) - 1.0
        out["target_gap_1"] = out["target_gap_1"].clip(-self.cfg.target_clip, self.cfg.target_clip)

        return out


class TimeSplitter:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def split_masks(self, df: pd.DataFrame):
        dates = np.sort(df["date"].dropna().unique())
        if len(dates) < 50:
            raise ValueError(f"Not enough unique dates to split: {len(dates)}")

        train_end = dates[int(self.cfg.train_frac * len(dates))]
        bt_end = dates[int((self.cfg.train_frac + self.cfg.bt_frac) * len(dates))]

        train_m = df["date"] <= train_end
        bt_m = (df["date"] > train_end) & (df["date"] <= bt_end)
        fw_m = df["date"] > bt_end

        return train_m, bt_m, fw_m, train_end, bt_end



class ModelZoo:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def make_classifiers(self):
        cls = {
            "LogReg": LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=self.cfg.n_jobs
            ),
            "HGB": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=300,
                random_state=RANDOM_STATE,
                early_stopping=True
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
                n_jobs=self.cfg.n_jobs,
                class_weight="balanced_subsample"
            ),
        }
        return cls

    def make_regressors(self):
        reg = {
            "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
            "HGB": HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=6,
                max_iter=500,
                random_state=RANDOM_STATE,
                early_stopping=True
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=800,
                max_depth=None,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
                n_jobs=self.cfg.n_jobs
            ),
        }
        return reg


class TrainerEvaluator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    @staticmethod
    def _eval_cls(model, X, y):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            "acc": accuracy_score(y, pred),
            "f1": f1_score(y, pred),
            "auc": roc_auc_score(y, proba),
        }

    @staticmethod
    def _eval_reg(model, X, y):
        pred = model.predict(X)
        return {
            "mae": mean_absolute_error(y, pred),
            "r2": r2_score(y, pred),
            "corr": np.corrcoef(y, pred)[0, 1] if np.std(y) > 0 and np.std(pred) > 0 else np.nan
        }

    def _make_preprocess(self):
        return Pipeline([
            ("inf_to_nan", InfToNan(max_abs=1e12)),
            ("winsor", TrainOnlyWinsorizer(p=self.cfg.winsor_p)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ])

    def fit_and_score(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str,
        split_masks,
    ):
        train_m, bt_m, fw_m, train_end, bt_end = split_masks

        d = df.dropna(subset=[target_col, "date"]).copy()
        X = d[feature_cols].astype(float).values
        y = d[target_col].astype(float).values

        y_cls = (y > 0).astype(int)

        X_tr, y_tr, y_tr_cls = X[train_m.loc[d.index]], y[train_m.loc[d.index]], y_cls[train_m.loc[d.index]]
        X_bt, y_bt, y_bt_cls = X[bt_m.loc[d.index]], y[bt_m.loc[d.index]], y_cls[bt_m.loc[d.index]]
        X_fw, y_fw, y_fw_cls = X[fw_m.loc[d.index]], y[fw_m.loc[d.index]], y_cls[fw_m.loc[d.index]]

        zoo = ModelZoo(self.cfg)
        preprocess = self._make_preprocess()

        cls_models = {}
        cls_scores = []
        for name, base in zoo.make_classifiers().items():
            pipe = Pipeline([("prep", preprocess), ("model", base)])
            pipe.fit(X_tr, y_tr_cls)
            s_tr = self._eval_cls(pipe, X_tr, y_tr_cls)
            s_bt = self._eval_cls(pipe, X_bt, y_bt_cls) if len(y_bt_cls) else {"acc": np.nan, "f1": np.nan, "auc": np.nan}
            s_fw = self._eval_cls(pipe, X_fw, y_fw_cls) if len(y_fw_cls) else {"acc": np.nan, "f1": np.nan, "auc": np.nan}
            cls_models[name] = pipe
            cls_scores.append((name, s_tr, s_bt, s_fw))

        reg_models = {}
        reg_scores = []
        for name, base in zoo.make_regressors().items():
            pipe = Pipeline([("prep", preprocess), ("model", base)])
            pipe.fit(X_tr, y_tr)
            s_tr = self._eval_reg(pipe, X_tr, y_tr)
            s_bt = self._eval_reg(pipe, X_bt, y_bt) if len(y_bt) else {"mae": np.nan, "r2": np.nan, "corr": np.nan}
            s_fw = self._eval_reg(pipe, X_fw, y_fw) if len(y_fw) else {"mae": np.nan, "r2": np.nan, "corr": np.nan}
            reg_models[name] = pipe
            reg_scores.append((name, s_tr, s_bt, s_fw))

        best_cls = max(cls_scores, key=lambda t: (t[2]["auc"] if not np.isnan(t[2]["auc"]) else -1e9))[0]
        best_reg = min(reg_scores, key=lambda t: (t[2]["mae"] if not np.isnan(t[2]["mae"]) else 1e9))[0]

        summary = {
            "target": target_col,
            "train_end": pd.Timestamp(train_end).date(),
            "bt_end": pd.Timestamp(bt_end).date(),
            "n_train": int(len(X_tr)),
            "n_bt": int(len(X_bt)),
            "n_fw": int(len(X_fw)),
            "best_cls": best_cls,
            "best_reg": best_reg,
            "cls_scores": cls_scores,
            "reg_scores": reg_scores,
        }
        return summary, cls_models, reg_models, (d, X, y, y_cls)


class SimpleBacktester:
    def __init__(self, txn_cost_bps: int = 10):
        self.txn_cost_bps = txn_cost_bps

    def run_threshold_sweep(self, df_slice, X_slice, model, target_col, thresholds=(0.45, 0.5, 0.55, 0.6)):
     
        proba = model.predict_proba(X_slice)[:, 1]
        best = None

        for th in thresholds:
            tmp = df_slice.copy()
            tmp["p"] = proba
            tmp["pos"] = (tmp["p"] >= th).astype(int)
            tmp["gross"] = tmp[target_col] * tmp["pos"]

            tmp["prev_pos"] = tmp.groupby("symbol")["pos"].shift(1).fillna(0)
            traded = (tmp["pos"] != tmp["prev_pos"]).astype(int)
            cost = traded * (self.txn_cost_bps / 10000.0)
            tmp["net"] = tmp["gross"] - cost

            daily = tmp.groupby("date")["net"].mean().sort_index()
            eq = (1 + daily).cumprod()
            sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252)

            score = float(sharpe)
            if best is None or score > best["sharpe"]:
                best = {
                    "th": th,
                    "sharpe": score,
                    "tot_ret": float(eq.iloc[-1] - 1),
                    "daily_mean": float(daily.mean()),
                    "daily_vol": float(daily.std()),
                }
        return best


class EarningsReactionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.loader = PostgresDataLoader(cfg)
        self.fe = EarningsFeatureEngineer(cfg)
        self.tb = TargetBuilder(cfg)
        self.splitter = TimeSplitter(cfg)
        self.trainer = TrainerEvaluator(cfg)
        self.backtester = SimpleBacktester(txn_cost_bps=10)

    def _leakage_check_notes(self):
     
        return True

    def _compute_and_print_correlations(self, df, feature_cols, target_col, train_mask):
       
        print("\n==============================")
        print("FEATURE–TARGET CORRELATIONS")
        print("(Train-only, Pearson vs", target_col, ")")
        print("==============================")
        train_df = df.loc[train_mask].copy()
        cols = feature_cols + [target_col]
        corr_df = train_df[cols].astype(float).corr()
        corrs = corr_df[target_col].drop(labels=[target_col])
        corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)

        for feat, val in corrs.items():
            print(f"{feat:35s} {val:+.4f}")

    def run(self, do_backtest: bool = True):
        with section_timer("1) Load data"):
            full, metrics = self.loader.load()

        with section_timer("2) Build features (events only)"):
            events, feature_cols = self.fe.build(full, metrics)

        with section_timer("3) Add target (gap_1 only)"):
            events_t = self.tb.add_targets(events, full_daily=full[["symbol", "date", "open", "close"]].copy())

        core_needed = ["eps_surprise_pct", "rev_surprise_pct", "pre_close", "vol_21d_pre", "target_gap_1"]
        events_t = events_t.dropna(subset=core_needed).copy()

        split_masks = self.splitter.split_masks(events_t)
        train_m, bt_m, fw_m, train_end, bt_end = split_masks

        target_col = f"target_{self.cfg.target_key}"

        self._compute_and_print_correlations(
            df=events_t,
            feature_cols=feature_cols,
            target_col=target_col,
            train_mask=train_m,
        )

        all_summaries = []
        models_by_target = {}

        with section_timer(f"4) Train/Eval for {target_col}"):
            summary, cls_models, reg_models, packed = self.trainer.fit_and_score(
                df=events_t,
                feature_cols=feature_cols,
                target_col=target_col,
                split_masks=split_masks,
            )
            all_summaries.append(summary)
            models_by_target[target_col] = (summary, cls_models, reg_models, packed)

        if do_backtest:
            d, X_all, y_all, y_cls_all = packed
            train_m2, bt_m2, fw_m2, train_end2, bt_end2 = split_masks  

            bt_idx = bt_m2.loc[d.index].values
            fw_idx = fw_m2.loc[d.index].values

            best_cls_name = summary["best_cls"]
            best_cls_model = cls_models[best_cls_name]

            bt_df = d.loc[bt_m2.loc[d.index]].copy()
            fw_df = d.loc[fw_m2.loc[d.index]].copy()

            bt_df[target_col] = d.loc[bt_m2.loc[d.index], target_col].values
            fw_df[target_col] = d.loc[fw_m2.loc[d.index], target_col].values

            bt_res = self.backtester.run_threshold_sweep(
                bt_df, X_all[bt_idx], best_cls_model, target_col,
                thresholds=(0.45, 0.5, 0.55, 0.6)
            )
            print(f"Backtest threshold sweep best for {target_col}: {bt_res}")

        print("\n====================")
        print("FINAL MODEL SUMMARY")
        print("====================")
        for s in all_summaries:
            print(
                f"{s['target']}: "
                f"rows tr/bt/fw={s['n_train']}/{s['n_bt']}/{s['n_fw']} | "
                f"best_cls={s['best_cls']} best_reg={s['best_reg']}"
            )
            best_cls = next(x for x in s["cls_scores"] if x[0] == s["best_cls"])
            best_reg = next(x for x in s["reg_scores"] if x[0] == s["best_reg"])
            print(f"  CLS (BT)  acc={best_cls[2]['acc']:.3f} f1={best_cls[2]['f1']:.3f} auc={best_cls[2]['auc']:.3f}")
            print(f"  REG (BT)  mae={best_reg[2]['mae']:.5f} r2={best_reg[2]['r2']:.3f} corr={best_reg[2]['corr']:.3f}")

        return all_summaries, models_by_target, feature_cols, events_t


def save_best_artifacts(models_by_target, feature_cols, out_dir="artifacts", target_key="gap_1"):
    os.makedirs(out_dir, exist_ok=True)
    target_col = f"target_{target_key}"

    summary, cls_models, reg_models, packed = models_by_target[target_col]
    best_cls_name = summary["best_cls"]
    best_reg_name = summary["best_reg"]

    joblib.dump(cls_models[best_cls_name], os.path.join(out_dir, f"{target_col}_cls.joblib"))
    joblib.dump(reg_models[best_reg_name], os.path.join(out_dir, f"{target_col}_reg.joblib"))

    meta = {
        "target_key": target_key,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "best_cls_name": best_cls_name,
        "best_reg_name": best_reg_name,
    }
    with open(os.path.join(out_dir, f"{target_col}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)



def main():
    pg_dsn = os.getenv("PG_DSN", "").strip()
    if not pg_dsn:
        try:
            from config import PG_DSN as _PG_DSN
            pg_dsn = _PG_DSN
        except Exception as e:
            raise RuntimeError(
                "PG_DSN not set. Set env var PG_DSN or provide config.py with PG_DSN."
            ) from e

    cfg = PipelineConfig(pg_dsn=pg_dsn)

    pipe = EarningsReactionPipeline(cfg)
    summaries, models_by_target, feature_cols, events_t = pipe.run(do_backtest=True)

    save_best_artifacts(models_by_target, feature_cols, out_dir="artifacts", target_key=cfg.target_key)


if __name__ == "__main__":
    main()
