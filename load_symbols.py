import datetime
from db import get_conn, bulk_insert
from fmp_client import FMPClient


def upsert_symbols(client: FMPClient):
  
    data = client.get_symbol_list()

    rows = []
    for item in data:
        rows.append((
            item.get("symbol"),
            item.get("companyName") or item.get("name"),
            datetime.datetime.utcnow(),     
        ))

    sql = """
    INSERT INTO fmp.symbols (
        symbol, name, updated_at
    )
    VALUES %s
    ON CONFLICT (symbol) DO UPDATE SET
        name       = EXCLUDED.name,
        updated_at = NOW();
    """

    with get_conn() as conn:
        bulk_insert(conn, sql, rows)


if __name__ == "__main__":
    client = FMPClient()

    print("Loading symbols from /stable/stock-list ...")
    upsert_symbols(client)
    print("Symbols loaded / updated.")
