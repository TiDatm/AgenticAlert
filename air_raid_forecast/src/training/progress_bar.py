"""
Lightweight, Zero-Dependency Progress Bar & Training Status Indicator.
Provides smooth visual progress bars, rate tracking, and dynamic metric updates
both in interactive TTY consoles and headless environments.
"""

import sys
import time
from typing import Iterable, Any, Optional

class ProgressBar:
    """
    Zero-dependency visual progress bar with ETA, speed (it/s), and dynamic status descriptors.
    """
    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        total: Optional[int] = None,
        desc: str = "",
        bar_len: int = 30,
        leave: bool = True
    ):
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else 0)
        self.desc = desc
        self.bar_len = bar_len
        self.leave = leave
        self.current = 0
        self.start_time = time.time()
        self.last_print_time = 0.0
        self.postfix_str = ""

    def __iter__(self):
        self.start_time = time.time()
        self.current = 0
        self._display()
        for item in self.iterable:
            yield item
            self.current += 1
            now = time.time()
            # Throttle screen updates to max 10 times per second to prevent stdout clutter
            if now - self.last_print_time > 0.08 or self.current == self.total:
                self._display()
        if self.leave:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def set_description(self, desc: str):
        self.desc = desc

    def set_postfix(self, **kwargs):
        parts = []
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        self.postfix_str = " | " + " ".join(parts) if parts else ""

    def _display(self):
        elapsed = time.time() - self.start_time
        rate = (self.current / elapsed) if elapsed > 0 else 0.0
        pct = (self.current / self.total) if self.total > 0 else 0.0
        filled = int(self.bar_len * pct)
        bar = "█" * filled + "░" * (self.bar_len - filled)

        # Estimate time remaining (ETA)
        if rate > 0 and self.total > 0:
            remaining = (self.total - self.current) / rate
            eta_str = f"{int(remaining // 60):02d}:{int(remaining % 60):02d}"
        else:
            eta_str = "--:--"

        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        out = f"\r{self.desc} [{bar}] {self.current}/{self.total} ({pct*100:5.1f}%) [{elapsed_str}<{eta_str}, {rate:5.1f}it/s]{self.postfix_str}"
        sys.stdout.write(out)
        sys.stdout.flush()
        self.last_print_time = time.time()


def track_progress(iterable: Iterable, desc: str = "", total: Optional[int] = None, **kwargs) -> ProgressBar:
    """
    Convenience wrapper function matching the tqdm API.
    """
    try:
        import tqdm
        return tqdm.tqdm(iterable, desc=desc, total=total, **kwargs)
    except ImportError:
        return ProgressBar(iterable=iterable, total=total, desc=desc, **kwargs)
