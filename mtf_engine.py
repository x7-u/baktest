"""
Multi-Timeframe (MTF) aggregation engine.

Takes base timeframe OHLCV data and builds higher-timeframe candles
by aggregating bars. Provides per-bar access to any HTF via shift.

Supported timeframes:
  M1, M3, M5, M10, M15, M30, H1, H4, D1, W1, MN1

Usage:
  mtf = MTFEngine(base_tf='M5')
  mtf.build(df)  # build all HTF candles from base data
  mtf.get('H4', 'high', shift=0)  # current H4 high
  mtf.get('D1', 'close', shift=1)  # previous daily close
"""

import pandas as pd
import numpy as np

# Timeframe minutes
TF_MINUTES = {
    'M1': 1, 'M3': 3, 'M5': 5, 'M10': 10, 'M15': 15, 'M30': 30,
    'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200,
}

# MQL5 enum values → timeframe string
ENUM_TO_TF = {
    0: None,  # PERIOD_CURRENT = use base
    1: 'M1', 3: 'M3', 5: 'M5', 10: 'M10', 15: 'M15', 30: 'M30',
    60: 'H1', 240: 'H4', 1440: 'D1', 10080: 'W1', 43200: 'MN1',
}

# Pine timeframe strings
PINE_TF_MAP = {
    '1': 'M1', '3': 'M3', '5': 'M5', '10': 'M10', '15': 'M15', '30': 'M30',
    '60': 'H1', '240': 'H4', 'D': 'D1', '1D': 'D1', 'W': 'W1', '1W': 'W1', 'M': 'MN1', '1M': 'MN1',
}


class MTFEngine:
    def __init__(self, base_tf='M5'):
        self.base_tf = base_tf
        self.base_minutes = TF_MINUTES.get(base_tf, 5)
        self._htf_data = {}   # {tf_str: DataFrame with open/high/low/close/volume}
        self._htf_map = {}    # {tf_str: list mapping base_bar_index → htf_bar_index}
        self._bar_count = 0

    def build(self, df):
        """Build all higher timeframes from base OHLCV DataFrame.
        DataFrame must have columns: open, high, low, close, volume
        and optionally a datetime column."""
        self._bar_count = len(df)

        # Find datetime column
        dt_col = None
        for col in ('datetime', 'date', 'Date', 'Datetime'):
            if col in df.columns:
                dt_col = col
                break

        # Parse datetimes if available
        if dt_col:
            try:
                timestamps = pd.to_datetime(df[dt_col])
            except Exception:
                timestamps = None
        else:
            timestamps = None

        # Build each HTF that's higher than base
        for tf_str, tf_min in TF_MINUTES.items():
            if tf_min <= self.base_minutes:
                continue  # skip same or lower timeframes

            ratio = tf_min // self.base_minutes
            if ratio < 2:
                continue

            if timestamps is not None:
                # Use real timestamps for grouping
                htf_data, bar_map = self._aggregate_by_time(df, timestamps, tf_min, ratio)
            else:
                # Use fixed-ratio grouping
                htf_data, bar_map = self._aggregate_by_ratio(df, ratio)

            self._htf_data[tf_str] = htf_data
            self._htf_map[tf_str] = bar_map

    def _aggregate_by_time(self, df, timestamps, tf_minutes, ratio):
        """Group bars by real time boundaries."""
        tf_seconds = tf_minutes * 60
        ts_unix = timestamps.astype(np.int64) // 10**9  # to Unix seconds
        # Round down to HTF boundary
        htf_keys = (ts_unix // tf_seconds).values

        # Build HTF candles
        htf_rows = []
        bar_map = np.zeros(len(df), dtype=np.int32)

        unique_keys = []
        prev_key = None
        htf_idx = -1

        for i in range(len(df)):
            key = htf_keys[i]
            if key != prev_key:
                htf_idx += 1
                unique_keys.append(key)
                htf_rows.append({
                    'open': df.iloc[i]['open'],
                    'high': df.iloc[i]['high'],
                    'low': df.iloc[i]['low'],
                    'close': df.iloc[i]['close'],
                    'volume': df.iloc[i]['volume'] if 'volume' in df.columns else 0,
                })
                prev_key = key
            else:
                row = htf_rows[htf_idx]
                row['high'] = max(row['high'], df.iloc[i]['high'])
                row['low'] = min(row['low'], df.iloc[i]['low'])
                row['close'] = df.iloc[i]['close']
                row['volume'] = row['volume'] + (df.iloc[i]['volume'] if 'volume' in df.columns else 0)
            bar_map[i] = htf_idx

        htf_df = pd.DataFrame(htf_rows)
        return htf_df, bar_map.tolist()

    def _aggregate_by_ratio(self, df, ratio):
        """Group bars by fixed ratio (no timestamp)."""
        htf_rows = []
        bar_map = []

        for i in range(len(df)):
            htf_idx = i // ratio
            if htf_idx >= len(htf_rows):
                htf_rows.append({
                    'open': df.iloc[i]['open'],
                    'high': df.iloc[i]['high'],
                    'low': df.iloc[i]['low'],
                    'close': df.iloc[i]['close'],
                    'volume': df.iloc[i]['volume'] if 'volume' in df.columns else 0,
                })
            else:
                row = htf_rows[htf_idx]
                row['high'] = max(row['high'], df.iloc[i]['high'])
                row['low'] = min(row['low'], df.iloc[i]['low'])
                row['close'] = df.iloc[i]['close']
                row['volume'] = row['volume'] + (df.iloc[i]['volume'] if 'volume' in df.columns else 0)
            bar_map.append(htf_idx)

        htf_df = pd.DataFrame(htf_rows)
        return htf_df, bar_map

    def get(self, tf_str, field, base_bar_index, shift=0):
        """Get a HTF value at a given base bar index with optional shift.

        Args:
            tf_str: Timeframe string ('H1', 'H4', 'D1', etc.)
            field: 'open', 'high', 'low', 'close', 'volume'
            base_bar_index: Current base timeframe bar index
            shift: HTF bar offset (0=current, 1=previous, etc.)

        Returns:
            float value or None if out of range
        """
        if tf_str not in self._htf_data:
            return None

        htf_df = self._htf_data[tf_str]
        bar_map = self._htf_map[tf_str]

        if base_bar_index < 0 or base_bar_index >= len(bar_map):
            return None

        # Current HTF bar index for this base bar
        current_htf_idx = bar_map[base_bar_index]

        # Apply shift (look back)
        target_idx = current_htf_idx - shift

        if target_idx < 0 or target_idx >= len(htf_df):
            return None

        return float(htf_df.iloc[target_idx][field])

    def get_htf_bar_index(self, tf_str, base_bar_index):
        """Get the HTF bar index for a given base bar index."""
        if tf_str not in self._htf_map:
            return -1
        bar_map = self._htf_map[tf_str]
        if base_bar_index < 0 or base_bar_index >= len(bar_map):
            return -1
        return bar_map[base_bar_index]

    def get_htf_time(self, tf_str, base_bar_index):
        """Get a unique timestamp-like value for the current HTF bar.
        Used by MQL5's iTime(symbol, tf, 0) for new-bar detection."""
        idx = self.get_htf_bar_index(tf_str, base_bar_index)
        if idx < 0:
            return 0
        # Return a unique value per HTF bar
        tf_min = TF_MINUTES.get(tf_str, 5)
        return 1700000000 + idx * tf_min * 60  # synthetic but unique per HTF bar

    def resolve_tf(self, tf_value):
        """Convert MQL5 enum or Pine string to timeframe string.
        Returns None if it matches the base timeframe."""
        if isinstance(tf_value, (int, float)):
            tf_int = int(tf_value)
            if tf_int == 0:
                return None  # PERIOD_CURRENT
            return ENUM_TO_TF.get(tf_int)
        if isinstance(tf_value, str):
            return PINE_TF_MAP.get(tf_value, tf_value if tf_value in TF_MINUTES else None)
        return None

    def has_tf(self, tf_str):
        """Check if a timeframe has been built."""
        return tf_str in self._htf_data

    def available_timeframes(self):
        """Return list of available HTF strings."""
        return list(self._htf_data.keys())
