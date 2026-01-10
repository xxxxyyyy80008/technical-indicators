from .base import TA, Indicator, SeriesIndicator, OHLCIndicator, OHLCVIndicator
from .utils import standardize_ohlcv_columns, validate_ohlcv

import importlib
import sys
from pathlib import Path
from typing import Dict, Type, Any

INDICATORS: Dict[str, Type[Any]] = {}

def _discover_indicators():
    """Discover and register all indicator classes from the indicators subpackage."""
    try:
        from . import indicators
        indicators_dir = Path(indicators.__path__[0])
        
        for module_file in indicators_dir.glob("*.py"):
            if module_file.name.startswith('_') or module_file.name == "__init__.py":
                continue
                
            module_name = module_file.stem
            try:
                module = importlib.import_module(f".indicators.{module_name}", package=__name__)
                
                for name, obj in vars(module).items():
                    if isinstance(obj, type) and issubclass(obj, (Indicator, SeriesIndicator, OHLCIndicator, OHLCVIndicator)):
                        if obj.__module__ == module.__name__: 
                            INDICATORS[name] = obj
                            
                            # Add to globals for direct import
                            if name not in globals():
                                globals()[name] = obj
                            
            except Exception as e:
                print(f"Warning: Could not process indicator module {module_name}: {str(e)}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not initialize indicators subpackage: {str(e)}", file=sys.stderr)

_discover_indicators()

__all__ = [
    'TA',
    'Indicator',
    'SeriesIndicator',
    'OHLCIndicator',
    'OHLCVIndicator',
    'standardize_ohlcv_columns',
    'validate_ohlcv',
    'INDICATORS',
] + list(INDICATORS.keys())

# Clean up namespace
del importlib, sys, Path, Dict, Type, Any
