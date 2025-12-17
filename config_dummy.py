import os

FMP_API_KEY = os.getenv("key") or "key"

PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://postgres:password@localhost:5432/trading"
)

REQUESTS_PER_SECOND = 20