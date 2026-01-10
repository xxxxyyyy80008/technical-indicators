import pandas as pd
import numpy as np
from typing import Union
from talib.base import OHLCIndicator, register_indicator

@register_indicator
class AMA(OHLCIndicator):
    """
    Adaptive Moving Average (TASC April 2018)
    
    Parameters:
        period: Lookback period for price range (default 10)
        fast_period: Fast EMA period (default 2)
        slow_period: Slow EMA period (default 30)
    
    Reference:
    https://traders.com/documentation/feedbk_docs/2018/04/traderstips.html
    Created on 2022-09-10
    """
    
    def __init__(self, source, period: int = 10, fast_period: int = 2, 
                 slow_period: int = 30, **kwargs):
        super().__init__(source, **kwargs)
        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def compute(self) -> pd.Series:
        close = self.data["close"]
        highest_high = self.data["high"].rolling(self.period).max()
        lowest_low = self.data["low"].rolling(self.period).min()
        
        # Calculate smoothing constants
        fast_sc = 2 / (self.fast_period + 1)
        slow_sc = 2 / (self.slow_period + 1)
        
        # Calculate efficiency ratio
        price_range = (highest_high - lowest_low).replace(0, np.nan)
        mltp = np.abs((close - lowest_low) - (highest_high - close)) / price_range
        ssc = mltp * (fast_sc - slow_sc) + slow_sc
        cst = ssc * ssc
        
        # Initialize AMA
        ama = np.zeros(len(close))
        ama[0] = close[0]
        
        # Calculate AMA recursively
        for i in range(1, len(close)):
            if i < self.period:
                ama[i] = close[i-1] + cst[i] * (close[i] - close[i-1])
            else:
                ama[i] = ama[i-1] + cst[i] * (close[i] - ama[i-1])
        
        return pd.Series(ama, index=self.data.index, name=f"AMA{self.period}")