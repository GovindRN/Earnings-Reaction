from __future__ import annotations

from pathlib import Path

import psycopg2

from config import PG_DSN


DDL = """
CREATE SCHEMA IF NOT EXISTS fmp;

CREATE TABLE IF NOT EXISTS fmp.dcf (
  symbol       text NOT NULL,
  "date"       date NOT NULL,
  dcf          double precision,
  stock_price  double precision,
  loaded_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, "date")
);

CREATE TABLE IF NOT EXISTS fmp.scores (
  symbol            text PRIMARY KEY,
  reported_currency text,
  altman_z_score    double precision,
  piotroski_score   integer,
  working_capital   numeric,
  total_assets      numeric,
  retained_earnings numeric,
  ebit              numeric,
  market_cap        numeric,
  total_liabilities numeric,
  revenue           numeric,
  loaded_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fmp.ratings (
  symbol                      text NOT NULL,
  "date"                      date NOT NULL,
  rating                      text,
  discounted_cash_flow_score  integer,
  return_on_equity_score      integer,
  return_on_assets_score      integer,
  debt_to_equity_score        integer,
  price_to_earnings_score     integer,
  price_to_book_score         integer,
  loaded_at                   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, "date")
);

CREATE TABLE IF NOT EXISTS fmp.key_metrics_ttm (
  symbol                                      text PRIMARY KEY,
  market_cap                                  numeric,
  enterprise_value_ttm                        numeric,
  ev_to_sales_ttm                             double precision,
  ev_to_operating_cash_flow_ttm               double precision,
  ev_to_free_cash_flow_ttm                    double precision,
  ev_to_ebitda_ttm                            double precision,
  net_debt_to_ebitda_ttm                      double precision,
  current_ratio_ttm                           double precision,
  income_quality_ttm                          double precision,
  graham_number_ttm                           double precision,
  graham_net_net_ttm                          double precision,
  tax_burden_ttm                              double precision,
  interest_burden_ttm                         double precision,
  working_capital_ttm                         numeric,
  invested_capital_ttm                        numeric,
  return_on_assets_ttm                        double precision,
  operating_return_on_assets_ttm              double precision,
  return_on_tangible_assets_ttm               double precision,
  return_on_equity_ttm                        double precision,
  return_on_invested_capital_ttm              double precision,
  return_on_capital_employed_ttm              double precision,
  earnings_yield_ttm                          double precision,
  free_cash_flow_yield_ttm                    double precision,
  capex_to_operating_cash_flow_ttm            double precision,
  capex_to_depreciation_ttm                   double precision,
  capex_to_revenue_ttm                        double precision,
  sales_general_and_administrative_to_revenue_ttm double precision,
  research_and_developement_to_revenue_ttm    double precision,
  stock_based_compensation_to_revenue_ttm     double precision,
  intangibles_to_total_assets_ttm             double precision,
  average_receivables_ttm                     numeric,
  average_payables_ttm                        numeric,
  average_inventory_ttm                       numeric,
  days_of_sales_outstanding_ttm               double precision,
  days_of_payables_outstanding_ttm            double precision,
  days_of_inventory_outstanding_ttm           double precision,
  operating_cycle_ttm                         double precision,
  cash_conversion_cycle_ttm                   double precision,
  free_cash_flow_to_equity_ttm                numeric,
  free_cash_flow_to_firm_ttm                  numeric,
  tangible_asset_value_ttm                    numeric,
  net_current_asset_value_ttm                 numeric,
  loaded_at                                   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fmp.ratios_ttm (
  symbol                                     text PRIMARY KEY,
  gross_profit_margin_ttm                    double precision,
  ebit_margin_ttm                            double precision,
  ebitda_margin_ttm                          double precision,
  operating_profit_margin_ttm                double precision,
  pretax_profit_margin_ttm                   double precision,
  continuous_operations_profit_margin_ttm    double precision,
  net_profit_margin_ttm                      double precision,
  bottom_line_profit_margin_ttm              double precision,
  receivables_turnover_ttm                   double precision,
  payables_turnover_ttm                      double precision,
  inventory_turnover_ttm                     double precision,
  fixed_asset_turnover_ttm                   double precision,
  asset_turnover_ttm                         double precision,
  current_ratio_ttm                          double precision,
  quick_ratio_ttm                            double precision,
  solvency_ratio_ttm                         double precision,
  cash_ratio_ttm                             double precision,
  price_to_earnings_ratio_ttm                double precision,
  price_to_earnings_growth_ratio_ttm         double precision,
  forward_price_to_earnings_growth_ratio_ttm double precision,
  price_to_book_ratio_ttm                    double precision,
  price_to_sales_ratio_ttm                   double precision,
  price_to_free_cash_flow_ratio_ttm          double precision,
  price_to_operating_cash_flow_ratio_ttm     double precision,
  debt_to_assets_ratio_ttm                   double precision,
  debt_to_equity_ratio_ttm                   double precision,
  debt_to_capital_ratio_ttm                  double precision,
  long_term_debt_to_capital_ratio_ttm        double precision,
  financial_leverage_ratio_ttm               double precision,
  working_capital_turnover_ratio_ttm         double precision,
  operating_cash_flow_ratio_ttm              double precision,
  operating_cash_flow_sales_ratio_ttm        double precision,
  free_cash_flow_operating_cash_flow_ratio_ttm double precision,
  debt_service_coverage_ratio_ttm            double precision,
  interest_coverage_ratio_ttm                double precision,
  short_term_operating_cash_flow_coverage_ratio_ttm double precision,
  operating_cash_flow_coverage_ratio_ttm     double precision,
  capital_expenditure_coverage_ratio_ttm     double precision,
  dividend_paid_and_capex_coverage_ratio_ttm double precision,
  dividend_payout_ratio_ttm                  double precision,
  dividend_yield_ttm                         double precision,
  enterprise_value_ttm                       numeric,
  revenue_per_share_ttm                      double precision,
  net_income_per_share_ttm                   double precision,
  interest_debt_per_share_ttm                double precision,
  cash_per_share_ttm                         double precision,
  book_value_per_share_ttm                   double precision,
  tangible_book_value_per_share_ttm          double precision,
  shareholders_equity_per_share_ttm          double precision,
  operating_cash_flow_per_share_ttm          double precision,
  capex_per_share_ttm                        double precision,
  free_cash_flow_per_share_ttm               double precision,
  net_income_per_ebt_ttm                     double precision,
  ebt_per_ebit_ttm                           double precision,
  price_to_fair_value_ttm                    double precision,
  debt_to_market_cap_ttm                     double precision,
  effective_tax_rate_ttm                     double precision,
  enterprise_value_multiple_ttm              double precision,
  dividend_per_share_ttm                     double precision,
  loaded_at                                  timestamptz NOT NULL DEFAULT now()
);
"""


FILES = [
    ("dcf.csv", "fmp.dcf", ["symbol", "date", "dcf", "stock_price"]),
    ("scores.csv", "fmp.scores", [
        "symbol", "reported_currency", "altman_z_score", "piotroski_score",
        "working_capital", "total_assets", "retained_earnings", "ebit",
        "market_cap", "total_liabilities", "revenue"
    ]),
    ("ratings.csv", "fmp.ratings", [
        "symbol", "date", "rating", "discounted_cash_flow_score",
        "return_on_equity_score", "return_on_assets_score", "debt_to_equity_score",
        "price_to_earnings_score", "price_to_book_score"
    ]),
    ("key_metrics_ttm.csv", "fmp.key_metrics_ttm", [
        "symbol", "market_cap", "enterprise_value_ttm", "ev_to_sales_ttm",
        "ev_to_operating_cash_flow_ttm", "ev_to_free_cash_flow_ttm", "ev_to_ebitda_ttm",
        "net_debt_to_ebitda_ttm", "current_ratio_ttm", "income_quality_ttm",
        "graham_number_ttm", "graham_net_net_ttm", "tax_burden_ttm", "interest_burden_ttm",
        "working_capital_ttm", "invested_capital_ttm", "return_on_assets_ttm",
        "operating_return_on_assets_ttm", "return_on_tangible_assets_ttm",
        "return_on_equity_ttm", "return_on_invested_capital_ttm",
        "return_on_capital_employed_ttm", "earnings_yield_ttm", "free_cash_flow_yield_ttm",
        "capex_to_operating_cash_flow_ttm", "capex_to_depreciation_ttm", "capex_to_revenue_ttm",
        "sales_general_and_administrative_to_revenue_ttm", "research_and_developement_to_revenue_ttm",
        "stock_based_compensation_to_revenue_ttm", "intangibles_to_total_assets_ttm",
        "average_receivables_ttm", "average_payables_ttm", "average_inventory_ttm",
        "days_of_sales_outstanding_ttm", "days_of_payables_outstanding_ttm",
        "days_of_inventory_outstanding_ttm", "operating_cycle_ttm", "cash_conversion_cycle_ttm",
        "free_cash_flow_to_equity_ttm", "free_cash_flow_to_firm_ttm",
        "tangible_asset_value_ttm", "net_current_asset_value_ttm"
    ]),
    ("ratios_ttm.csv", "fmp.ratios_ttm", [
        "symbol",
        "gross_profit_margin_ttm", "ebit_margin_ttm", "ebitda_margin_ttm", "operating_profit_margin_ttm",
        "pretax_profit_margin_ttm", "continuous_operations_profit_margin_ttm", "net_profit_margin_ttm",
        "bottom_line_profit_margin_ttm", "receivables_turnover_ttm", "payables_turnover_ttm",
        "inventory_turnover_ttm", "fixed_asset_turnover_ttm", "asset_turnover_ttm", "current_ratio_ttm",
        "quick_ratio_ttm", "solvency_ratio_ttm", "cash_ratio_ttm", "price_to_earnings_ratio_ttm",
        "price_to_earnings_growth_ratio_ttm", "forward_price_to_earnings_growth_ratio_ttm",
        "price_to_book_ratio_ttm", "price_to_sales_ratio_ttm", "price_to_free_cash_flow_ratio_ttm",
        "price_to_operating_cash_flow_ratio_ttm", "debt_to_assets_ratio_ttm", "debt_to_equity_ratio_ttm",
        "debt_to_capital_ratio_ttm", "long_term_debt_to_capital_ratio_ttm", "financial_leverage_ratio_ttm",
        "working_capital_turnover_ratio_ttm", "operating_cash_flow_ratio_ttm", "operating_cash_flow_sales_ratio_ttm",
        "free_cash_flow_operating_cash_flow_ratio_ttm", "debt_service_coverage_ratio_ttm",
        "interest_coverage_ratio_ttm", "short_term_operating_cash_flow_coverage_ratio_ttm",
        "operating_cash_flow_coverage_ratio_ttm", "capital_expenditure_coverage_ratio_ttm",
        "dividend_paid_and_capex_coverage_ratio_ttm", "dividend_payout_ratio_ttm", "dividend_yield_ttm",
        "enterprise_value_ttm", "revenue_per_share_ttm", "net_income_per_share_ttm",
        "interest_debt_per_share_ttm", "cash_per_share_ttm", "book_value_per_share_ttm",
        "tangible_book_value_per_share_ttm", "shareholders_equity_per_share_ttm",
        "operating_cash_flow_per_share_ttm", "capex_per_share_ttm", "free_cash_flow_per_share_ttm",
        "net_income_per_ebt_ttm", "ebt_per_ebit_ttm", "price_to_fair_value_ttm", "debt_to_market_cap_ttm",
        "effective_tax_rate_ttm", "enterprise_value_multiple_ttm", "dividend_per_share_ttm"
    ]),
]


def copy_csv(cur, table: str, columns: list[str], csv_path: Path) -> None:
    cols = ", ".join([('"date"' if c == "date" else c) for c in columns])
    sql = f"""
        COPY {table} ({cols})
        FROM STDIN
        WITH (
            FORMAT csv,
            HEADER true,
            NULL '',
            QUOTE '"',
            ESCAPE '"'
        )
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        cur.copy_expert(sql, f)


def main():
    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parent
    data_dir = project_root / "Data"

    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)

            cur.execute("""
                TRUNCATE TABLE
                  fmp.dcf,
                  fmp.scores,
                  fmp.ratings,
                  fmp.key_metrics_ttm,
                  fmp.ratios_ttm
                CASCADE
            """)

            for filename, table, columns in FILES:
                path = data_dir / filename
                if not path.exists():
                    raise FileNotFoundError(f"Missing CSV: {path}")
                print(f"Loading {path.name} -> {table}")
                copy_csv(cur, table, columns, path)

        conn.commit()

    print("✅ Done. Truncated + loaded all CSVs into fmp schema.")


if __name__ == "__main__":
    main()
