"""Norm dataclass and types"""

from dataclasses import dataclass
from typing import List, Optional, Callable


@dataclass
class Norm:
    """Norm definition"""

    name: str
    description: str
    check: str
    severity: str  # critical, high, medium, low
    auto_fix: bool
    suggested_fix: str
    hooks: Optional[List[str]] = None
    check_fn: Optional[Callable] = None
