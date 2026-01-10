import pandas as pd
import numpy as np
from talib.base import OHLCVIndicator, register_indicator
from .tp import TP
from .atr import ATR

@register_indicator
class VPN(OHLCVIndicator):
    """
    Volume Positive Negative (VPN) Indicator
    Compares volume on up days vs down days, normalized between -100 and 100
    
    Parameters:
        period: Lookback period for volume comparison (default 30)
        ema_period: Smoothing period for VPN line (default 3)
        mav_period: Moving average period for signal line (default 30)
        adjust: Whether to adjust EMA calculation (default True)
    
    Returns DataFrame with:
        VPN: Main indicator line
        MA_VPN: Smoothed signal line
        
    Reference:
    https://traders.com/Documentation/FEEDbk_docs/2021/04/TradersTips.html
    """
    
    def __init__(self, source, period: int = 30, ema_period: int = 3, 
                 mav_period: int = 30, **kwargs):
        super().__init__(source, **kwargs)
        self.period = period
        self.ema_period = ema_period
        self.mav_period = mav_period
        
    def compute(self) -> pd.DataFrame:
        v = self.data['volume']
        mav_ = v.rolling(self.period).mean()

        tp_ = TP(self.data).compute()  #typical price: (high + low + close)/3
        atr_ =  ATR(self.data, period = self.period).compute() #Average True Range is moving average of True Range
        mf_ = tp_.diff(1) #momentum of typical price
        mc_ = atr_*0.1
        vol_up = (mf_ > mc_).astype(int)*v
        vol_down = (mf_ < (-1*mc_)).astype(int)*v
        vp_ = vol_up.rolling(self.period).sum()
        vn_ = vol_down.rolling(self.period).sum()

        mav_[mav_<=0] = 1
        vpn_ = ((vp_ - vn_)/mav_/self.period*100).ewm(span=self.ema_period).mean()
        ma_vpn_ = vpn_.rolling(self.mav_period).mean()   

        return pd.DataFrame(data = {'VPN': vpn_.values, 
                                    'MA_VPN': ma_vpn_.values, 
                                   },
                            index=self.data.index, )   
