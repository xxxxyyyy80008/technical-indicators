import pandas as pd
import numpy as np
from typing import Optional
from talib.base import SeriesIndicator, register_indicator
from .er import ER

@register_indicator
class KAMA(SeriesIndicator):
    """
    Kaufman's Adaptive Moving Average (KAMA)
    Adaptive moving average that accounts for market noise and volatility
    
    Parameters:
        er_period: Efficiency Ratio period (default 10)
        ema_fast: Fast EMA period (default 2)
        ema_slow: Slow EMA period (default 30)
        period: Initial SMA period (default 20)
        column: Price column to use (default "close")
    
    Returns:
        Series with KAMA values
    """
    
    def __init__(self, source, er_period: int = 10, ema_fast: int = 2, 
                 ema_slow: int = 30, period: int = 20, column: str = "close", **kwargs):
        super().__init__(source, column=column, **kwargs)
        self.er_period = er_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.period = period
        
    def compute(self) -> pd.Series:
        # Calculate components
        er = ER(self.data, period=self.er_period, column=self.column).compute()
        fast_alpha = 2 / (self.ema_fast + 1)
        slow_alpha = 2 / (self.ema_slow + 1)
        
        # Calculate smoothing constant
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        # Initialize with SMA
        sma = self.series.rolling(window=self.period).mean()
        kama = sma.copy()
        
        # Calculate KAMA recursively
        for i in range(1, len(kama)):
            if pd.notna(kama.iloc[i-1]) and pd.notna(sc.iloc[i]):
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (self.series.iloc[i] - kama.iloc[i-1])
        
        return kama.rename("KAMA")