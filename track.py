from dataclasses import dataclass
import time
from typing import Optional

@dataclass
class TargetQualityTracker:
    target: float
    start_time: float
    reached: bool = False
    ittq: Optional[int] = None
    tttq: Optional[float] = None

    def check(self, iteration: int, incumbent_cost: float) -> None:
        """Record first hit only (ITTQ/TTTQ)."""
        if (not self.reached) and (incumbent_cost <= self.target):
            self.reached = True
            self.ittq = iteration
            self.tttq = time.perf_counter() - self.start_time

    def finalize(self, max_iterations: int) -> None:
        """Ensure fields are filled even if target is never reached."""
        if not self.reached:
            self.ittq = max_iterations
            self.tttq = time.perf_counter() - self.start_time


