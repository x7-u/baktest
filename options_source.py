"""
Options Data Source — fetches US equity options data from philippdubach/options-data CDN.

Data is pre-processed Parquet files hosted on Cloudflare R2:
  - underlying.parquet  (~5MB)  — daily OHLCV for the equity
  - options.parquet     (~50-600MB) — full options chain with greeks

104 US equities/ETFs, date range 2008-01-02 to 2025-12-16.
"""

import os
import time
import urllib.request
import shutil

import pandas as pd

# ── Available tickers (104 US equities + ETFs) ──────────────────────────────
AVAILABLE_TICKERS = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
    'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK.B', 'C',
    'CAT', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX',
    'DE', 'DHR', 'DIS', 'DUK', 'EMR', 'FDX', 'GD', 'GE', 'GILD', 'GM',
    'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTU', 'ISRG', 'IWM', 'JNJ',
    'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ', 'MDT',
    'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX', 'NKE',
    'NOW', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PLTR', 'PM', 'PYPL', 'QCOM',
    'QQQ', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'SPY', 'T', 'TGT', 'TMO',
    'TMUS', 'TSLA', 'TXN', 'UBER', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VIX',
    'VZ', 'WFC', 'WMT', 'XOM',
]

CDN_BASE = 'https://static.philippdubach.com/data/options'


class OptionsDataSource:
    """Download, cache, and serve options data from the CDN."""

    def __init__(self, cache_dir=None):
        if cache_dir is None:
            base = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(base, 'uploads', 'options_cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    # ── Ticker list ──────────────────────────────────────────────────────
    @staticmethod
    def get_tickers():
        return list(AVAILABLE_TICKERS)

    # ── Cache paths ──────────────────────────────────────────────────────
    def _ticker_dir(self, ticker):
        """Cache directory for a ticker. CDN uses lowercase paths."""
        return os.path.join(self.cache_dir, ticker.upper())

    def _underlying_path(self, ticker):
        return os.path.join(self._ticker_dir(ticker), 'underlying.parquet')

    def _options_path(self, ticker):
        return os.path.join(self._ticker_dir(ticker), 'options.parquet')

    def is_cached(self, ticker):
        return {
            'underlying': os.path.isfile(self._underlying_path(ticker)),
            'options': os.path.isfile(self._options_path(ticker)),
        }

    # ── Download ─────────────────────────────────────────────────────────
    def _cdn_url(self, ticker, filename):
        """CDN uses lowercase ticker in the URL path."""
        return f'{CDN_BASE}/{ticker.lower()}/{filename}'

    def download_file(self, ticker, filename, progress_callback=None):
        """
        Download a single file from the CDN to the local cache.
        Uses atomic write: downloads to .tmp then renames on success.
        Returns the local file path.
        """
        ticker_dir = self._ticker_dir(ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        dest = os.path.join(ticker_dir, filename)
        tmp = dest + '.tmp'
        url = self._cdn_url(ticker, filename)

        # Skip if already cached
        if os.path.isfile(dest):
            return dest

        # Skip if another process is downloading (marker file)
        lock = dest + '.lock'
        if os.path.isfile(lock):
            lock_age = time.time() - os.path.getmtime(lock)
            if lock_age < 3600:  # less than 1 hour old — wait for it
                waited = 0
                while os.path.isfile(lock) and waited < 300:
                    time.sleep(2)
                    waited += 2
                if os.path.isfile(dest):
                    return dest
                raise RuntimeError(f'{filename} for {ticker} is currently being downloaded by another process.')
            # Stale lock (>1 hour) — remove and proceed
            os.remove(lock)

        try:
            # Write lock marker
            with open(lock, 'w') as f:
                f.write('downloading')

            # Download with User-Agent header (Cloudflare blocks bare urllib)
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Baktest/1.0'
            })
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                block_size = 64 * 1024  # 64KB chunks
                downloaded = 0
                with open(tmp, 'wb') as out_f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_size, filename)

            # Atomic rename
            shutil.move(tmp, dest)
            return dest

        except Exception as e:
            # Clean up partial download
            if os.path.isfile(tmp):
                os.remove(tmp)
            raise RuntimeError(f'Failed to download {filename} for {ticker}: {e}')

        finally:
            # Remove lock
            if os.path.isfile(lock):
                os.remove(lock)

    def ensure_underlying(self, ticker, progress_callback=None):
        """Download underlying.parquet if not cached. Returns local path."""
        return self.download_file(ticker, 'underlying.parquet', progress_callback)

    def ensure_options(self, ticker, progress_callback=None):
        """Download options.parquet if not cached. Returns local path."""
        return self.download_file(ticker, 'options.parquet', progress_callback)

    # ── Load data ────────────────────────────────────────────────────────
    def load_underlying(self, ticker, start_date=None, end_date=None):
        """
        Load underlying OHLCV data and convert to standard Baktest DataFrame.

        Output columns: [datetime, open, high, low, close, volume]
        datetime format: 'YYYY.MM.DD' (MetaTrader convention, matches parse_metatrader_csv)
        """
        path = self._underlying_path(ticker)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Underlying data not cached for {ticker}. Call ensure_underlying() first.')

        df = pd.read_parquet(path)

        # Normalize column names (the CDN may use various casings)
        df.columns = [c.lower() for c in df.columns]

        # Validate required columns
        required = ['date', 'open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'Missing required columns in underlying data for {ticker}: {missing}')

        # Convert date to datetime string in MetaTrader format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Filter by date range
        if start_date:
            df = df[df['date'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['date'] <= pd.Timestamp(end_date)]

        # Format datetime as 'YYYY.MM.DD' — consistent with MetaTrader CSV
        df['datetime'] = df['date'].dt.strftime('%Y.%m.%d')

        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Volume handling — use 'volume' if available, else 0
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        else:
            df['volume'] = 0

        # Select final columns
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)

        return df

    def load_options_chain(self, ticker, start_date=None, end_date=None):
        """
        Load options chain data with date filtering via PyArrow predicate pushdown.
        This avoids loading the entire 600MB file into memory.

        Returns a pandas DataFrame with all 21 options columns.
        """
        path = self._options_path(ticker)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Options data not cached for {ticker}. Call ensure_options() first.')

        try:
            import pyarrow.parquet as pq

            # Detect date column type to build correct filter values
            schema = pq.read_schema(path)
            try:
                date_field = schema.field('date')
                date_type = str(date_field.type)
            except KeyError:
                # No 'date' column — load without filters, let caller handle
                return pq.read_table(path).to_pandas()

            filters = []
            if start_date:
                if 'string' in date_type or 'utf8' in date_type:
                    filters.append(('date', '>=', str(pd.Timestamp(start_date).date())))
                elif 'timestamp' in date_type:
                    filters.append(('date', '>=', pd.Timestamp(start_date)))
                else:  # date32, date64
                    filters.append(('date', '>=', pd.Timestamp(start_date).date()))
            if end_date:
                if 'string' in date_type or 'utf8' in date_type:
                    filters.append(('date', '<=', str(pd.Timestamp(end_date).date())))
                elif 'timestamp' in date_type:
                    filters.append(('date', '<=', pd.Timestamp(end_date)))
                else:
                    filters.append(('date', '<=', pd.Timestamp(end_date).date()))

            if filters:
                table = pq.read_table(path, filters=filters)
            else:
                table = pq.read_table(path)

            return table.to_pandas()

        except ImportError:
            # Fallback: load with pandas (no predicate pushdown — uses more memory)
            df = pd.read_parquet(path)
            df['date'] = pd.to_datetime(df['date'])
            if start_date:
                df = df[df['date'] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df['date'] <= pd.Timestamp(end_date)]
            return df

    def _has_underlying_data(self, ticker):
        """Check if the underlying.parquet has actual rows (some ETFs have empty files)."""
        path = self._underlying_path(ticker)
        if not os.path.isfile(path):
            return False
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(path)
            return meta.num_rows > 0
        except ImportError:
            pass
        except Exception:
            # Corrupt parquet — treat as no data
            return False
        try:
            df = pd.read_parquet(path)
            return len(df) > 0
        except Exception:
            return False

    def load_underlying_from_options(self, ticker, start_date=None, end_date=None):
        """
        Fallback: derive daily OHLCV from the options chain for tickers where
        underlying.parquet is empty (ETFs like SPY, QQQ, IWM, VIX).

        Uses ATM call marks to approximate the underlying price.
        """
        options_df = self.load_options_chain(ticker, start_date=start_date, end_date=end_date)
        if options_df is None or len(options_df) == 0:
            raise ValueError(f'No options data available for {ticker} to derive underlying prices.')

        # Work on a copy so we don't mutate the caller's DataFrame
        options_df = options_df.copy()
        options_df['date'] = pd.to_datetime(options_df['date'])
        calls = options_df[options_df['type'] == 'call'].copy()
        if len(calls) == 0:
            raise ValueError(f'No call options found for {ticker} to derive underlying.')

        rows = []
        for date_val, group in calls.groupby('date'):
            # Find nearest ATM option: closest delta to 0.50
            group = group.copy()
            # Filter to rows with valid delta; if none, fall back to mid-strike
            has_delta = group['delta'].notna()
            if has_delta.any():
                valid = group[has_delta].copy()
                valid['_atm_dist'] = (valid['delta'] - 0.50).abs()
                atm = valid.nsmallest(1, '_atm_dist')
            else:
                # No delta data — pick strike closest to median strike as ATM proxy
                median_strike = group['strike'].median()
                group['_atm_dist'] = (group['strike'] - median_strike).abs()
                atm = group.nsmallest(1, '_atm_dist')
            if len(atm) == 0:
                continue
            strike = float(atm.iloc[0]['strike'])
            # For ATM calls (delta ≈ 0.50), underlying ≈ strike.
            # Refine with put-call parity if both sides available:
            #   S = Strike + Call_mark - Put_mark
            # Fallback: just use strike (ATM by definition means S ≈ K)
            underlying_price = strike
            atm_row = atm.iloc[0]
            # Use mark if available (not NaN), otherwise fall back to last
            _mark = atm_row.get('mark')
            _last = atm_row.get('last')
            call_mark = float(_mark) if pd.notna(_mark) else (float(_last) if pd.notna(_last) else 0.0)
            # Try to find matching put for better estimate via put-call parity
            date_puts = options_df[(options_df['date'] == date_val) &
                                   (options_df['type'] == 'put') &
                                   (options_df['strike'] == strike)]
            if len(date_puts) > 0:
                _pm = date_puts.iloc[0].get('mark')
                _pl = date_puts.iloc[0].get('last')
                put_mark = float(_pm) if pd.notna(_pm) else (float(_pl) if pd.notna(_pl) else 0.0)
                # Put-call parity: S = K + C - P
                underlying_price = strike + call_mark - put_mark

            total_vol = int(group['volume'].sum()) if 'volume' in group.columns else 0
            rows.append({
                'datetime': date_val.strftime('%Y.%m.%d'),
                'open': underlying_price,
                'high': underlying_price * 1.001,   # tiny spread for charting
                'low': underlying_price * 0.999,
                'close': underlying_price,
                'volume': total_vol,
            })

        if not rows:
            raise ValueError(f'Could not derive underlying prices for {ticker}.')

        df = pd.DataFrame(rows).sort_values('datetime').reset_index(drop=True)
        return df

    def get_date_range(self, ticker):
        """
        Get earliest and latest dates available for a ticker.
        First checks underlying.parquet, then falls back to options.parquet.
        Returns (start_date_str, end_date_str) in 'YYYY-MM-DD' format.
        """
        def _extract_range(path):
            """Read date column and return (min, max) as strings, or None if empty."""
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(path, columns=['date'])
                dates = table.to_pandas()['date']
            except ImportError:
                dates = pd.read_parquet(path, columns=['date'])['date']
            dates = pd.to_datetime(dates, errors='coerce').dropna()
            if len(dates) == 0:
                return None
            return dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d')

        # Try underlying first
        path = self._underlying_path(ticker)
        if os.path.isfile(path) and self._has_underlying_data(ticker):
            result = _extract_range(path)
            if result:
                return result

        # Fallback: use options.parquet date range
        opt_path = self._options_path(ticker)
        if os.path.isfile(opt_path):
            result = _extract_range(opt_path)
            if result:
                return result

        raise FileNotFoundError(f'No data cached for {ticker}.')

    # ── Cache management ─────────────────────────────────────────────────
    def cache_size(self, ticker=None):
        """Get cache size in bytes. If ticker is None, returns total cache size."""
        total = 0
        try:
            if ticker:
                tdir = self._ticker_dir(ticker)
                if os.path.isdir(tdir):
                    for f in os.listdir(tdir):
                        fp = os.path.join(tdir, f)
                        try:
                            total += os.path.getsize(fp)
                        except OSError:
                            pass
            else:
                for root, dirs, files in os.walk(self.cache_dir):
                    for f in files:
                        try:
                            total += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
        except OSError:
            pass
        return total

    def clear_cache(self, ticker=None):
        """Clear cached data. If ticker is None, clears all cached data."""
        if ticker:
            tdir = self._ticker_dir(ticker)
            if os.path.isdir(tdir):
                shutil.rmtree(tdir)
        else:
            if os.path.isdir(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
