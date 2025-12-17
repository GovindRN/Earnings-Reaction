import time
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import execute_values

from config import PG_DSN

@contextmanager
def get_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()


def bulk_insert(conn, sql, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
