"""
Platform-level SMT (Smart Money Technique) Divergence Engine.

Computes swing highs/lows on both main and secondary (correlated) symbol data,
then detects divergence. Results are injected as variables into any interpreter
so ANY strategy can use SMT divergence without needing SMT-specific code.

Variables injected per bar:
  _smt_bull       - True if bullish SMT divergence detected this bar
  _smt_bear       - True if bearish SMT divergence detected this bar
  _smt_bull_bar   - Bar index of last bullish SMT divergence
  _smt_bear_bar   - Bar index of last bearish SMT divergence
  _smt_available  - True if SMT data is loaded
"""


class SMTEngine:
    """Detects SMT divergence between main and secondary symbol data."""

    def __init__(self, swing_len=5, lookback=2):
        self.swing_len = swing_len
        self.lookback = lookback  # how many recent swings to compare

        # Main symbol swing tracking
        self._main_highs = []  # list of (bar, price)
        self._main_lows = []

        # SMT symbol swing tracking
        self._smt_highs = []
        self._smt_lows = []

        # History buffers for pivot detection
        self._main_high_buf = []
        self._main_low_buf = []
        self._smt_high_buf = []
        self._smt_low_buf = []

        # Last divergence state
        self.last_bull_bar = 0
        self.last_bear_bar = 0

    def update(self, bar_index, main_high, main_low, smt_high, smt_low):
        """Call each bar with OHLC highs/lows for both symbols.
        Returns (bull_divergence, bear_divergence) booleans."""

        self._main_high_buf.append(main_high)
        self._main_low_buf.append(main_low)
        self._smt_high_buf.append(smt_high)
        self._smt_low_buf.append(smt_low)

        bull = False
        bear = False

        n = self.swing_len

        # Need at least 2*n+1 bars for pivot detection
        if len(self._main_high_buf) < 2 * n + 1:
            return False, False

        # Detect pivot high on main symbol (center = n bars ago)
        main_ph = self._detect_pivot_high(self._main_high_buf, n)
        smt_ph = self._detect_pivot_high(self._smt_high_buf, n)

        # Detect pivot low on main symbol
        main_pl = self._detect_pivot_low(self._main_low_buf, n)
        smt_pl = self._detect_pivot_low(self._smt_low_buf, n)

        # Track swing highs
        if main_ph is not None:
            self._main_highs.append((bar_index - n, main_ph))
            if len(self._main_highs) > 10:
                self._main_highs.pop(0)

        if smt_ph is not None:
            self._smt_highs.append((bar_index - n, smt_ph))
            if len(self._smt_highs) > 10:
                self._smt_highs.pop(0)

        # Track swing lows
        if main_pl is not None:
            self._main_lows.append((bar_index - n, main_pl))
            if len(self._main_lows) > 10:
                self._main_lows.pop(0)

        if smt_pl is not None:
            self._smt_lows.append((bar_index - n, smt_pl))
            if len(self._smt_lows) > 10:
                self._smt_lows.pop(0)

        # Check for bearish SMT divergence (swing highs)
        # Main makes higher high, SMT makes lower high (or vice versa)
        if len(self._main_highs) >= 2 and len(self._smt_highs) >= 2:
            mh1 = self._main_highs[-2][1]
            mh2 = self._main_highs[-1][1]
            sh1 = self._smt_highs[-2][1]
            sh2 = self._smt_highs[-1][1]

            # Main HH + SMT LH = bearish divergence
            if mh2 > mh1 and sh2 < sh1:
                bear = True
                self.last_bear_bar = bar_index
            # Main LH + SMT HH = also bearish divergence (disagreement at highs)
            elif mh2 < mh1 and sh2 > sh1:
                bear = True
                self.last_bear_bar = bar_index

        # Check for bullish SMT divergence (swing lows)
        # Main makes lower low, SMT makes higher low (or vice versa)
        if len(self._main_lows) >= 2 and len(self._smt_lows) >= 2:
            ml1 = self._main_lows[-2][1]
            ml2 = self._main_lows[-1][1]
            sl1 = self._smt_lows[-2][1]
            sl2 = self._smt_lows[-1][1]

            # Main LL + SMT HL = bullish divergence
            if ml2 < ml1 and sl2 > sl1:
                bull = True
                self.last_bull_bar = bar_index
            # Main HL + SMT LL = also bullish divergence (disagreement at lows)
            elif ml2 > ml1 and sl2 < sl1:
                bull = True
                self.last_bull_bar = bar_index

        return bull, bear

    def _detect_pivot_high(self, buf, n):
        """Check if buf[-n-1] is a pivot high (strictly higher than n bars on each side)."""
        if len(buf) < 2 * n + 1:
            return None
        center = buf[-n - 1]
        for i in range(1, n + 1):
            if buf[-n - 1 - i] >= center:
                return None
            if buf[-n - 1 + i] >= center:
                return None
        return center

    def _detect_pivot_low(self, buf, n):
        """Check if buf[-n-1] is a pivot low (strictly lower than n bars on each side)."""
        if len(buf) < 2 * n + 1:
            return None
        center = buf[-n - 1]
        for i in range(1, n + 1):
            if buf[-n - 1 - i] <= center:
                return None
            if buf[-n - 1 + i] <= center:
                return None
        return center

    def get_variables(self, bar_index):
        """Return dict of variables to inject into the interpreter."""
        return {
            '_smt_available': True,
            '_smt_bull': self.last_bull_bar == bar_index,
            '_smt_bear': self.last_bear_bar == bar_index,
            '_smt_bull_bar': self.last_bull_bar,
            '_smt_bear_bar': self.last_bear_bar,
            '_smt_bull_active': (bar_index - self.last_bull_bar) <= 10 if self.last_bull_bar > 0 else False,
            '_smt_bear_active': (bar_index - self.last_bear_bar) <= 10 if self.last_bear_bar > 0 else False,
        }
