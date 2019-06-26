from collections import defaultdict
from typing import Dict

from lahja import BaseEvent, AsyncioEndpoint


_bus = None


_counts: Dict[str, int] = defaultdict(int)


class NoBusError(Exception):
    pass


def set_bus(event_bus: AsyncioEndpoint) -> None:
    global _bus
    _bus = event_bus


def _check_bus() -> None:
    if _bus == None:
        raise NoBusError()


class Metric(BaseEvent):
    def __init__(self, metric: str, value: int) -> None:
        self.metric = metric
        self.value = value


def gauge(metric: str, value: int) -> None:
    _check_bus()
    _bus.broadcast_nowait(Metric(metric, value))


def count(metric: str, increment: int = 1) -> None:
    _check_bus()

    _counts[metric] += increment
    _bus.broadcast_nowait(Metric(metric, _counts[metric]))
