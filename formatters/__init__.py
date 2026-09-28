# formatters/__init__.py

'''
import os
import sys
# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
'''

from formatters.narrative import NarrativeFormatter 
from formatters.structured import StructuredFormatter 
from formatters.json_formatter import JSONFormatter
from formatters.fpi import FPIFormatter


def get_formatter(name: str, **kwargs):
    if name == "narrative":
        return NarrativeFormatter()
    elif name == "structured":
        return StructuredFormatter()
    elif name == "json":
        return JSONFormatter()
    elif name == "fpi_structured":
        return FPIFormatter(style="structured", **kwargs)
    elif name == "fpi_narrative":
        return FPIFormatter(style="narrative", **kwargs) 
    elif name == "fpi_json":
        return FPIFormatter(style="json", **kwargs)
    elif name == "fpi":  # default
        return FPIFormatter(style="structured", **kwargs)
    else:
        raise ValueError(f"Unknown formatter: {name}")
    