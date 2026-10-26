import threading
import time
from collections import deque


class RequestLimiter:
    def __init__(self, limit: int = 20, window_seconds: float = 60) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit and window_seconds must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client_id: str, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        with self._lock:
            requests = self._requests.setdefault(client_id, deque())
            while requests and current_time - requests[0] >= self.window_seconds:
                requests.popleft()
            if len(requests) >= self.limit:
                return False
            requests.append(current_time)
            return True
