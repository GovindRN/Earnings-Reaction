from __future__ import annotations

import datetime
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from db import bulk_insert, get_conn
from fmp_client import FMPClient

import math

BIGINT_MIN = -(2**63)
BIGINT_MAX =  (2**63 - 1)

def safe_bigint(x):
    if x is None:
        return None
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return None
        v = int(xf)
    except Exception:
        return None
    if v < BIGINT_MIN or v > BIGINT_MAX:
        return None
    return v


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


def truncate_prices_table() -> None:
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE fmp.daily_prices")
        conn.commit()


def reset_prices_loaded_flags() -> None:
 
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE fmp.symbols
                SET prices_loaded = FALSE,
                    updated_at = NOW()
                """
            )
        conn.commit()


def build_rows_for_symbol(symbol: str, historical: list[dict]) -> list[tuple]:
    
    by_date: Dict[str, tuple] = {}
    for h in historical or []:
        d = h.get("date")
        if isinstance(d, str):
            d = d.strip()
        if not d:
            continue
        by_date[d] = (
            symbol,
            d,
            h.get("open"),
            h.get("high"),
            h.get("low"),
            h.get("close"),
            h.get("adjClose"),
            safe_bigint(h.get("volume")),                
            safe_bigint(h.get("unadjustedVolume")),      
            h.get("change"),
            h.get("changePercent"),
            h.get("vwap"),
            h.get("label"),
            h.get("changeOverTime"),
            datetime.datetime.utcnow(),  
        )

    return [by_date[d] for d in sorted(by_date.keys())]


def load_prices_for_symbol(client: FMPClient, symbol: str) -> str:
 
    try:
        data = client.get_historical_prices_full_chunked(symbol)
    except Exception as e:
        print(f"[ERROR] API error for {symbol}: {e}")
        return "error"

    historical = data or []
    if not historical:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE fmp.symbols
                    SET prices_loaded = TRUE, updated_at = NOW()
                    WHERE symbol = %s
                    """,
                    (symbol,),
                )
            conn.commit()
        return "no_data"

    rows = build_rows_for_symbol(symbol, historical)
    if not rows:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE fmp.symbols
                    SET prices_loaded = TRUE, updated_at = NOW()
                    WHERE symbol = %s
                    """,
                    (symbol,),
                )
            conn.commit()
        return "no_data"

    insert_sql = """
        INSERT INTO fmp.daily_prices (
            symbol, date, open, high, low, close,
            adj_close, volume, unadj_volume,
            change, change_percent, vwap, label,
            change_over_time, loaded_at
        )
        VALUES %s
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume,
            unadj_volume = EXCLUDED.unadj_volume,
            change = EXCLUDED.change,
            change_percent = EXCLUDED.change_percent,
            vwap = EXCLUDED.vwap,
            label = EXCLUDED.label,
            change_over_time = EXCLUDED.change_over_time,
            loaded_at = EXCLUDED.loaded_at
    """


    with get_conn() as conn:
        bulk_insert(conn, insert_sql, rows)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE fmp.symbols
                SET prices_loaded = TRUE, updated_at = NOW()
                WHERE symbol = %s
                """,
                (symbol,),
            )
        conn.commit()

    return "ok"


def main(limit: int | None = None):
    
    print("Truncating fmp.daily_prices and resetting prices_loaded flags...")
    truncate_prices_table()
    reset_prices_loaded_flags()

    client = FMPClient()
    symbols = get_symbols_to_load(limit=limit)
    print(f"Loading prices for {len(symbols)} symbols (from scratch)")

    ok = no_data = errors = 0

    for sym in tqdm(symbols):
        status = load_prices_for_symbol(client, sym)
        if status == "ok":
            ok += 1
        elif status == "no_data":
            no_data += 1
        elif status == "error":
            errors += 1

    print(f"Done. ok={ok}, no_data={no_data}, errors={errors}")


if __name__ == "__main__":
    main()
