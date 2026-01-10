import pandas as pd
import numpy as np
from typing import Union
from talib.base import SeriesIndicator, register_indicator

@register_indicator
class MABW(SeriesIndicator):
    """
    Moving Average Bands Width (MABW) Indicator
    
    Parameters:
        fast_period: Short-term EMA period (default 10)
        slow_period: Long-term EMA period (default 50)
        multiplier: Band width multiplier (default 1.0)
        column: Price column to use (default "close")
    """
    
    def __init__(self, source, fast_period: int = 10, slow_period: int = 50,
                 multiplier: float = 1.0, column: str = "close", **kwargs):
        super().__init__(source, column=column, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.multiplier = multiplier
        
    def compute(self) -> pd.DataFrame:
        ma_slow = self.series.ewm(span=self.slow_period).mean()
        ma_fast = self.series.ewm(span=self.fast_period).mean()
        
        dst = ma_slow - ma_fast
        dv = dst.pow(2).rolling(self.fast_period).mean().apply(np.sqrt)
        dev = dv * self.multiplier
        
        upper = ma_slow + dev
        lower = ma_slow - dev
        width = (upper - lower) / ma_slow * 100
        llv = width.rolling(self.slow_period).min()
        
        return pd.DataFrame({
            "MAB_UPPER": upper,
            "MAB_MIDDLE": ma_fast,
            "MAB_LOWER": lower,
            "MAB_WIDTH": width,
            "MAB_LLV": llv
        }, index=self.data.index)