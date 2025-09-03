from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy


def _json_default(o: Any):
    if isinstance(o, datetime | date):
        return o.isoformat()
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, numpy.ndarray):
        return o.tolist()
    return str(o)
