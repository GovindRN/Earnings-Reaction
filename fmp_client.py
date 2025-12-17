import time
import datetime
from http.client import RemoteDisconnected

import requests

from config import FMP_API_KEY, REQUESTS_PER_SECOND

BASE_V3 = "https://financialmodelingprep.com/api/v3"
BASE_STABLE = "https://financialmodelingprep.com/stable"


class FMPClient:
    def __init__(self, api_key: str = FMP_API_KEY, rps: float = REQUESTS_PER_SECOND):
        self.api_key = api_key
        self.session = requests.Session()
        self.min_interval = 1.0 / rps
        self._last_call = 0.0

    def _throttle(self) -> None:
        now = time.time()
        dt = now - self._last_call
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last_call = time.time()

    def _get(self, url: str, params: dict | None = None):
      
        if params is None:
            params = {}
        params.setdefault("apikey", self.api_key)

        for attempt in range(5):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code != 200:
                    print(
                        f"HTTP {resp.status_code} from FMP on {url}\n"
                        f"Response (truncated): {resp.text[:300]}"
                    )

                    if resp.status_code in (429, 500, 502, 503, 504):
                        time.sleep(10 * (attempt + 1))
                        continue

                    raise RuntimeError(
                        f"FMP returned {resp.status_code} for {url}: "
                        f"{resp.text[:200]}"
                    )

                return resp.json()

            except (requests.exceptions.ConnectionError, RemoteDisconnected) as e:
                print(
                    f"Connection error on {url}: {e} "
                    f"(attempt {attempt + 1}/5), backing off..."
                )
                time.sleep(10 * (attempt + 1))
                continue

            except Exception as e:
                print(
                    f"Unexpected error {e} on {url}, "
                    f"attempt {attempt + 1}/5; backing off..."
                )
                time.sleep(5 * (attempt + 1))

        raise RuntimeError(f"Failed to fetch {url} after retries")


    def get_hourly_1h_prices(
        self,
        ticker: str,
        from_date: str | None = None,
        to_date: str | None = None,
        nonadjusted: bool = False,
    ) -> list[dict]:
      
        url = f"{BASE_STABLE}/historical-chart/1hour"

        params: dict[str, str] = {
            "symbol": ticker,
            "nonadjusted": "true" if nonadjusted else "false",
        }
        if from_date:
            params["from"] = from_date      
        if to_date:
            params["to"] = to_date          

        data = self._get(url, params)

        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("historical", []) or []
        return []
    
    def get_earnings_report(self, symbol: str, limit: int = 1000):
        
        if limit > 1000:
            limit = 1000
    
        url = f"{BASE_STABLE}/earnings"
        params = {
            "symbol": symbol,
            "limit": limit,
        }
        return self._get(url, params)

    def search_symbol(self, query: str, limit: int = 1, exchange: str | None = None):
        
        url = f"{BASE_STABLE}/search-symbol"
        params: dict[str, object] = {"query": query, "limit": limit}
        if exchange:
            params["exchange"] = exchange
        return self._get(url, params)

    def get_exchange_variants(self, symbol: str):
     
        url = f"{BASE_STABLE}/search-exchange-variants"
        params = {"symbol": symbol}
        return self._get(url, params)

    def get_symbol_list(self):
  
        url = f"{BASE_STABLE}/stock-list"
        return self._get(url)


    def get_stock_list(
        self,
        exchange: str | None = None,
        exchange_short_name: str | None = None,
        actively_trading: bool | None = None,
        security_type: str | None = None,
    ):
        
        data = self.get_symbol_list() or []
        if not isinstance(data, list):
            return data

        def _match(rec: dict) -> bool:
            if exchange_short_name:
                if (rec.get("exchangeShortName") or "").upper() != exchange_short_name.upper():
                    return False
            if exchange:
                if (rec.get("exchange") or "").upper() != exchange.upper():
                    return False
            if actively_trading is not None:
                if bool(rec.get("isActivelyTrading")) != bool(actively_trading):
                    return False
            if security_type:
                if (rec.get("type") or "").upper() != security_type.upper():
                    return False
            return True

        return [r for r in data if isinstance(r, dict) and _match(r)]
    def get_historical_prices_full(self, symbol: str, from_date: str | None = None, to_date: str | None = None):

        url = f"{BASE_STABLE}/historical-price-eod/full"
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get(url, params)


    def get_historical_prices_eod(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ):
      
        url = f"{BASE_STABLE}/historical-price-eod/full"
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get(url, params)

    def get_historical_prices_full_chunked(
        self,
        symbol: str,
        start_year: int = 1980,
        years_per_chunk: int = 10,
    ):
       
        today = datetime.date.today()
        end_year = today.year

        all_rows: list[dict] = []

        for start in range(start_year, end_year + 1, years_per_chunk):
            from_date = f"{start}-01-01"
            chunk_end_year = min(start + years_per_chunk - 1, end_year)
            to_date = f"{chunk_end_year}-12-31"

            try:
                chunk = self.get_historical_prices_eod(
                    symbol, from_date=from_date, to_date=to_date
                )
            except Exception as e:
                print(
                    f"[ERROR] Price chunk {from_date}–{to_date} for {symbol}: {e}"
                )
                continue

            if isinstance(chunk, list):
                all_rows.extend(chunk)
            elif isinstance(chunk, dict):
                hist = chunk.get("historical") or []
                if isinstance(hist, list):
                    all_rows.extend(hist)

        all_rows.sort(key=lambda x: x.get("date") or "")

        return all_rows


    def get_historical_earning_calendar(self, symbol: str):
        
        url = f"{BASE_V3}/historical/earning_calendar/{symbol}"
        return self._get(url)
