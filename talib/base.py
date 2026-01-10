import pandas as pd
from typing import Union, Optional, Dict, Any, Callable, List, Type
from inspect import isclass
import sys
import importlib
from pathlib import Path
from .utils import standardize_ohlcv_columns

# Global registry for concrete indicators
CONCRETE_INDICATORS: Dict[str, Type] = {}

class Indicator:
    """Base class for all technical indicators"""
    def __init__(self, 
                 source: Union[pd.DataFrame, pd.Series],
                 column: str = 'close',
                 **kwargs):
        """
        Initialize with data source
        :param source: OHLCV DataFrame or price Series
        :param column: Relevant column if DataFrame provided
        :param kwargs: Additional indicator-specific parameters
        """
        self.column = column
        self.params = kwargs
        
        # Standardize column names if DataFrame provided
        if isinstance(source, pd.DataFrame):
            self.data = standardize_ohlcv_columns(source)
            self.series = self.data[column]
            self._validate_columns()
        elif isinstance(source, pd.Series):
            self.data = pd.DataFrame({'close': source})
            self.series = source
        else:
            raise TypeError("Input must be DataFrame or Series")
            
    def _validate_columns(self):
        """Ensure required columns exist - to be overridden by subclasses"""
        pass
        
    def compute(self) -> Union[pd.Series, pd.DataFrame]:
        """Main calculation method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement compute() method")

class SeriesIndicator(Indicator):
    """Base for indicators requiring only price series"""
    def _validate_columns(self):
        """Ensure series exists"""
        if self.column not in self.data.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")

class OHLCIndicator(Indicator):
    """Base for indicators requiring full OHLC data"""
    def _validate_columns(self):
        """Ensure OHLC columns exist"""
        required = {'open', 'high', 'low', 'close'}
        if not required.issubset(self.data.columns):
            missing = required - set(self.data.columns)
            raise ValueError(f"Missing required columns: {missing}")

class OHLCVIndicator(OHLCIndicator):
    """Base for indicators requiring OHLC + Volume data"""
    def _validate_columns(self):
        """Ensure OHLCV columns exist"""
        super()._validate_columns()  
        if 'volume' not in self.data.columns:
            raise ValueError("Missing required column: volume")

def register_indicator(cls: Type) -> Type:
    """Decorator to register concrete indicators"""
    if (isclass(cls) and 
        issubclass(cls, Indicator) and 
        cls != Indicator and 
        hasattr(cls, 'compute') and 
        cls.compute != Indicator.compute):
        CONCRETE_INDICATORS[cls.__name__] = cls
    return cls

class TA:
    """Unified interface for all indicators"""
    def __init__(self, 
                 ohlc: Optional[pd.DataFrame] = None,
                 series: Optional[pd.Series] = None):
        """
        Initialize with either:
        - OHLCV DataFrame for full indicators
        - Price Series for series-only indicators
        
        Args:
            ohlc: DataFrame containing Open, High, Low, Close, Volume columns
            series: Price series (typically close prices)
            
        Raises:
            ValueError: If neither ohlc nor series is provided
            TypeError: If inputs are of wrong type
        """
        # Automatically import indicators from the indicators subpackage
        self._import_indicators()
        
        if ohlc is not None:
            if not isinstance(ohlc, pd.DataFrame):
                raise TypeError("ohlc must be a pandas DataFrame")
            self.source = standardize_ohlcv_columns(ohlc)
            self.source_type = 'ohlc'
        elif series is not None:
            if not isinstance(series, pd.Series):
                raise TypeError("series must be a pandas Series")
            self.source = series
            self.source_type = 'series'
        else:
            raise ValueError("Must provide either OHLCV DataFrame or price Series")
    
    def _import_indicators(self):
        """Dynamically import all indicator modules"""
        # Only run once
        if hasattr(TA, '_indicators_imported'):
            return
            
        try:
            # Get the indicators package path
            package = sys.modules[__name__].__package__
            indicators_path = Path(sys.modules[package].__file__).parent / 'indicators'
            
            # Import all .py files in the indicators directory
            for file in indicators_path.glob('*.py'):
                if file.name.startswith('_') or file.name == '__init__.py':
                    continue
                    
                module_name = f"{package}.indicators.{file.stem}"
                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    print(f"Warning: Could not import indicator module {module_name}: {e}")
                    
        except Exception as e:
            print(f"Warning: Could not import indicators: {e}")
            
        TA._indicators_imported = True
    
    def __getattr__(self, name: str) -> Callable:
        """Dynamically handle indicator calls"""
        # Get indicator class from global registry
        indicator_cls = CONCRETE_INDICATORS.get(name)
        
        if indicator_cls is None:
            raise AttributeError(f"Indicator '{name}' not found or not valid")
        
        # Handle different indicator types
        if self.source_type == 'ohlc':
            def ohlc_wrapper(*args, **kwargs):
                # For OHLC data, all indicators are available
                column = kwargs.pop('column', 'close')
                return indicator_cls(self.source, column=column, **kwargs).compute(*args)
            return ohlc_wrapper
        else:
            # For Series data, only allow SeriesIndicator types
            if issubclass(indicator_cls, (OHLCIndicator, OHLCVIndicator)):
                raise ValueError(
                    f"Indicator '{name}' requires OHLCV data but series data was provided"
                )
            def series_wrapper(*args, **kwargs):
                return indicator_cls(self.source, **kwargs).compute(*args)
            return series_wrapper
    
    @property
    def available_indicators(self) -> List[str]:
        """Return list of available concrete indicators for current data type"""
        available = []
        
        for name, cls in CONCRETE_INDICATORS.items():
            # Check compatibility with current data type
            if self.source_type == 'ohlc':
                available.append(name)
            else:
                # For series data, only allow SeriesIndicator types
                if not issubclass(cls, (OHLCIndicator, OHLCVIndicator)):
                    available.append(name)
        
        return sorted(available)