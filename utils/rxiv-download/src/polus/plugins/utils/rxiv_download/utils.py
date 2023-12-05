import logging
import time
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class RateLimiter:
    calls: float = 0
    """The number of calls that count toward the rate limit."""
    rps: int
    """The backoff threshold."""
    lock: threading.Lock
    """Lock to impose rate limit restrictions."""
    last: float
    """Seconds since the last call."""

    def __init__(self, rps: int):
        """Wrapper for rate limited functions.

        For some functions that do things like call APIs, a common task is to rate limit
        calls. This class tracks call rates and will delay execution when the rate limit
        is reached.

        Args:
            rps: Permitted requests per second.
        """

        self.rps = rps
        self.lock = threading.Lock()
        self.call_count = 0
        self.last = 0

    def _limiter(self):
        """Rate limiting logic."""

        with self.lock:
            self.calls += 1

            if self.calls >= self.rps:
                logger.warning(
                    f"Current rate ({self.calls:0.1f}/s) is over the limit "
                    + f"({self.rps}/s). Pausing for 1 second."
                )

                time.sleep(1)

        now = time.time()
        self.calls = max(0, self.calls - (now - self.last) * self.rps)
        self.last = now

    def limit(self, func: Callable):
        """Wrap with rate limit.

        This can probably be removed. It might have utility in the future for
        customizing imposing rate limits on a function.

        """

        def wrapper(*args, **kwargs):
            self._limiter()
            return func(*args, **kwargs)

        return wrapper