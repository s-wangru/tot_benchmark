from dataclasses import dataclass 
from typing import Any

@dataclass
class Trace:
    begin_time: float 
    end_time: float
    latency: float
    messages: Any
    response: str
    
traces: list[Trace] = []

VERBOSE = True