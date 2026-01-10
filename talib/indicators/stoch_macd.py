import pandas as pd
import numpy as np
from typing import Union
from talib.base import OHLCIndicator, register_indicator

@register_indicator
class STOCH_MACD(OHLCIndicator):
    """
    Stochastic MACD Oscillator - Combines stochastic oscillator and MACD
    
    Parameters:
        period: Lookback period for stochastic high/low (default 45)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns DataFrame with:
        STMACD: Stochastic MACD line
        STMACD_SIGNAL: Signal line
        
    Reference:
    https://traders.com/Documentation/FEEDbk_docs/2019/11/TradersTips.html
    """
    
    def __init__(self, source, period: int = 45, fast_period: int = 12,
                 slow_period: int = 26, signal: int = 9, **kwargs):
        super().__init__(source, **kwargs)
        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal = signal
        
    def compute(self) -> pd.DataFrame:
        # Calculate highest high and lowest low
        highest_high = self.data['high'].rolling(window=self.period).max()
        lowest_low = self.data['low'].rolling(window=self.period).min()
        
        # Calculate EMAs
        ema_fast = self.data['close'].ewm(
            span=self.fast_period
        ).mean()
        
        ema_slow = self.data['close'].ewm(
            span=self.slow_period
        ).mean()
        
        # Calculate stochastic components
        stoch_fast = (ema_fast - lowest_low) / (highest_high - lowest_low)
        stoch_slow = (ema_slow - lowest_low) / (highest_high - lowest_low)
        
        # Calculate STMACD and signal line
        stmacd = (stoch_fast - stoch_slow) * 100
        signal = stmacd.ewm(span=self.signal).mean()
        
        return pd.DataFrame({
            'STMACD': stmacd,
            'STMACD_SIGNAL': signal
        }, index=self.data.index)