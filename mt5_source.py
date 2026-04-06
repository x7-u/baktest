"""
MetaTrader 5 Data Source — fetches historical OHLCV data directly from the MT5 terminal.

Requires:
  - pip install MetaTrader5
  - MetaTrader 5 terminal running and logged in

Returns pandas DataFrames in the same format as parse_metatrader_csv(),
so the rest of the backtester works identically whether data comes from CSV or MT5.
"""

import pandas as pd
from datetime import datetime


# Timeframe string → MT5 constant mapping (deferred import)
_TF_MAP = None


def _get_tf_map():
    global _TF_MAP
    if _TF_MAP is None:
        import MetaTrader5 as mt5
        _TF_MAP = {
            'M1': mt5.TIMEFRAME_M1, 'M2': mt5.TIMEFRAME_M2, 'M3': mt5.TIMEFRAME_M3,
            'M4': mt5.TIMEFRAME_M4, 'M5': mt5.TIMEFRAME_M5, 'M6': mt5.TIMEFRAME_M6,
            'M10': mt5.TIMEFRAME_M10, 'M12': mt5.TIMEFRAME_M12, 'M15': mt5.TIMEFRAME_M15,
            'M20': mt5.TIMEFRAME_M20, 'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1, 'H2': mt5.TIMEFRAME_H2, 'H3': mt5.TIMEFRAME_H3,
            'H4': mt5.TIMEFRAME_H4, 'H6': mt5.TIMEFRAME_H6, 'H8': mt5.TIMEFRAME_H8,
            'H12': mt5.TIMEFRAME_H12,
            'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1,
        }
    return _TF_MAP


class MT5DataSource:
    """Fetches OHLCV data from a running MetaTrader 5 terminal."""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Connect to the MT5 terminal. Returns True on success."""
        import MetaTrader5 as mt5
        if mt5.initialize():
            self._initialized = True
            return True
        return False

    def shutdown(self):
        """Disconnect from the MT5 terminal."""
        if self._initialized:
            import MetaTrader5 as mt5
            mt5.shutdown()
            self._initialized = False

    def is_connected(self):
        """Check if MT5 is connected."""
        return self._initialized

    def last_error(self):
        """Get last MT5 error."""
        import MetaTrader5 as mt5
        return mt5.last_error()

    def fetch_bars(self, symbol, tf_string, date_from, date_to):
        """
        Fetch OHLCV bars from MT5.

        Args:
            symbol: Trading pair (e.g. 'EURUSD', 'NZDJPY')
            tf_string: Timeframe string ('M1','M5','H1','H4','D1', etc.)
            date_from: Start datetime
            date_to: End datetime

        Returns:
            pandas DataFrame with columns: datetime, open, high, low, close, volume
            Same format as parse_metatrader_csv() output.

        Raises:
            RuntimeError: MT5 terminal not running
            ValueError: Symbol not found or no data
        """
        import MetaTrader5 as mt5

        if not self._initialized:
            raise RuntimeError(
                'MT5 terminal not connected. Make sure MetaTrader 5 is running and logged in.'
            )

        # Resolve timeframe
        tf_map = _get_tf_map()
        tf_const = tf_map.get(tf_string)
        if tf_const is None:
            raise ValueError(f'Unknown timeframe: {tf_string}')

        # Fetch data
        rates = mt5.copy_rates_range(symbol, tf_const, date_from, date_to)

        if rates is None:
            error = mt5.last_error()
            raise ValueError(
                f'Failed to fetch {symbol} data. MT5 error: {error}. '
                f'Make sure the symbol exists on your broker.'
            )

        if len(rates) == 0:
            raise ValueError(
                f'No data returned for {symbol} ({tf_string}) '
                f'from {date_from.strftime("%Y-%m-%d")} to {date_to.strftime("%Y-%m-%d")}.'
            )

        # Convert to DataFrame
        df = pd.DataFrame(rates)

        # Convert Unix timestamp to datetime string (matching CSV format)
        df['datetime'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y.%m.%d %H:%M:%S')

        # Rename columns to match CSV parser output
        df = df.rename(columns={'tick_volume': 'volume'})

        # Keep only the columns the backtester expects
        cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        if 'spread' in df.columns:
            cols.append('spread')
        df = df[cols]

        return df.reset_index(drop=True)

    def list_symbols(self, filter_text=''):
        """
        Get available symbols from the broker.

        Args:
            filter_text: Optional filter string (e.g. 'EUR', 'JPY')

        Returns:
            List of symbol name strings
        """
        import MetaTrader5 as mt5

        if not self._initialized:
            return []

        if filter_text:
            symbols = mt5.symbols_get(filter_text)
        else:
            symbols = mt5.symbols_get()

        if symbols is None:
            return []

        return sorted([s.name for s in symbols])

    def get_data_range(self, symbol, tf_string):
        """Get the earliest and latest available bar dates for a symbol/timeframe."""
        import MetaTrader5 as mt5

        if not self._initialized:
            return None

        tf_map = _get_tf_map()
        tf_const = tf_map.get(tf_string)
        if tf_const is None:
            return None

        # Fetch first bar (oldest available)
        first_bars = mt5.copy_rates_from_pos(symbol, tf_const, 0, 1)
        # Fetch last bar count to find the oldest
        total = mt5.copy_rates_from_pos(symbol, tf_const, 0, 1)

        # Get the very first available bar by requesting from epoch
        from datetime import datetime
        oldest = mt5.copy_rates_range(symbol, tf_const, datetime(2000, 1, 1), datetime(2000, 2, 1))
        if oldest is None or len(oldest) == 0:
            # Try getting total bar count and fetch the last one
            bars = mt5.copy_rates_from_pos(symbol, tf_const, 99999, 1)
            if bars is not None and len(bars) > 0:
                earliest = pd.Timestamp(bars[0]['time'], unit='s')
            else:
                earliest = None
        else:
            earliest = pd.Timestamp(oldest[0]['time'], unit='s')

        # Get most recent bar
        latest_bars = mt5.copy_rates_from_pos(symbol, tf_const, 0, 1)
        if latest_bars is not None and len(latest_bars) > 0:
            latest = pd.Timestamp(latest_bars[0]['time'], unit='s')
        else:
            latest = None

        # Better approach: count total bars available
        # Fetch 1 bar from very far back position
        for pos in [500000, 200000, 100000, 50000, 20000, 10000]:
            far = mt5.copy_rates_from_pos(symbol, tf_const, pos, 1)
            if far is not None and len(far) > 0:
                earliest = pd.Timestamp(far[0]['time'], unit='s')
                break

        if earliest is None or latest is None:
            return None

        return {
            'earliest': earliest.strftime('%Y-%m-%d'),
            'latest': latest.strftime('%Y-%m-%d'),
        }

    def get_symbol_info(self, symbol):
        """Get basic info about a symbol (digits, point, contract size)."""
        import MetaTrader5 as mt5

        if not self._initialized:
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return {
            'name': info.name,
            'digits': info.digits,
            'point': info.point,
            'spread': info.spread,
            'trade_contract_size': info.trade_contract_size,
            'currency_profit': info.currency_profit,
            'currency_base': info.currency_base,
        }
