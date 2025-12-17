from __future__ import annotations

import datetime
from typing import Dict, Tuple

from tqdm import tqdm

from db import bulk_insert, get_conn
from fmp_client import FMPClient


def get_symbols_to_load(limit: int | None = None) -> list[str]:
   
    sql = """
        SELECT s.symbol
        FROM fmp.symbols s
        ORDER BY s.symbol
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]


def truncate_earnings_calendar_table() -> None:
  
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE fmp.earnings_calendar")
        conn.commit()


def reset_earnings_loaded_flags() -> None:
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE fmp.symbols
                SET earnings_loaded = FALSE,
                    updated_at = NOW()
                """
            )
        conn.commit()


def load_earnings_for_symbol(client: FMPClient, symbol: str) -> str:
    
    try:
        data = client.get_earnings_report(symbol, limit=1000) or []
    except Exception as e:
        print(f"[ERROR] API error for {symbol}: {e}")
        return "error"

   
    dedup: Dict[Tuple[str, str], tuple] = {}

    for item in data:
        report_date = item.get("date")  
        if not report_date:
            continue

        last_updated = item.get("lastUpdated")
        if isinstance(last_updated, str) and len(last_updated) > 10:
            last_updated_date = last_updated[:10]
        else:
            last_updated_date = last_updated

        row = (
            symbol,                       
            report_date,                  
            item.get("epsActual"),        
            item.get("epsEstimated"),     
            None,                         
            item.get("revenueActual"),    
            item.get("revenueEstimated"), 
            last_updated_date,            
            None,                         
            datetime.datetime.utcnow(),   
        )

        dedup[(symbol, report_date)] = row  

    rows = list(dedup.values())

    insert_sql = """
    INSERT INTO fmp.earnings_calendar (
        symbol, report_date, eps_actual, eps_estimated,
        time, revenue_actual, revenue_estimated,
        updated_from_date, fiscal_date_ending, loaded_at
    )
    VALUES %s
    ON CONFLICT (symbol, report_date) DO UPDATE SET
        eps_actual = EXCLUDED.eps_actual,
        eps_estimated = EXCLUDED.eps_estimated,
        time = EXCLUDED.time,
        revenue_actual = EXCLUDED.revenue_actual,
        revenue_estimated = EXCLUDED.revenue_estimated,
        updated_from_date = EXCLUDED.updated_from_date,
        fiscal_date_ending = EXCLUDED.fiscal_date_ending,
        loaded_at = EXCLUDED.loaded_at
    """

    with get_conn() as conn:
        if rows:
            bulk_insert(conn, insert_sql, rows)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE fmp.symbols
                SET earnings_loaded = TRUE, updated_at = NOW()
                WHERE symbol = %s
                """,
                (symbol,),
            )
        conn.commit()

    return "ok" if rows else "no_data"


def main(limit: int | None = None):

    print("Truncating fmp.earnings_calendar and resetting earnings_loaded flags...")
    truncate_earnings_calendar_table()
    reset_earnings_loaded_flags()

    client = FMPClient()
    symbols = get_symbols_to_load(limit=limit)
    print(f"Loading earnings reports for {len(symbols)} symbols (from scratch)")

    ok = no_data = errors = 0

    for sym in tqdm(symbols):
        try:
            status = load_earnings_for_symbol(client, sym)
        except Exception as e:
            print(f"[ERROR] Unexpected error for {sym}: {e}")
            status = "error"

        if status == "ok":
            ok += 1
        elif status == "no_data":
            no_data += 1
        else:
            errors += 1

    print(f"Done. ok={ok}, no_data={no_data}, errors={errors}")


if __name__ == "__main__":
    main()
