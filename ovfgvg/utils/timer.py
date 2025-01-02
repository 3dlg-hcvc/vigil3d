import time
from typing import Self


class TimerNotStartedError(Exception):
    pass


class Timer:
    def __init__(self):
        self._cumulative = 0
        self._num_trials = 0
        self._start_time = None

    @property
    def cumulative(self) -> float:
        return self._cumulative

    def average(self, pretty: bool = False) -> float | str:
        if self._num_trials == 0:
            return None
        if pretty:
            return self.format_time(self._cumulative / self._num_trials)
        return self._cumulative / self._num_trials

    def is_running(self) -> bool:
        return self._start_time is not None

    def start(self) -> Self:
        self._start_time = time.time()
        return self

    def elapsed(self, pretty: bool = False) -> float | str:
        if not self.is_running:
            raise TimerNotStartedError("Tried to lapse a timer without starting it.")

        diff = time.time() - self._start_time
        if pretty:
            return self.format_time(diff)
        return diff

    def end(self, pretty: bool = False, batch: int = 1) -> float | str:
        diff = self.elapsed()
        self._cumulative += diff
        self._num_trials += batch
        self._start_time = None

        if pretty:
            return self.format_time(diff)
        return diff

    def reset(self) -> Self:
        self._cumulative = 0
        self._num_trials = 0
        self._start_time = None
        return self

    def print(self) -> str:
        if self._num_trials == 0:
            return "Average time: N/A (no trials)"
        return f"Average time: {self.average(pretty=True)} (num_trials={self._num_trials})"

    def __repr__(self) -> str:
        return f"Timer(cumulative={self._cumulative}, num_trials={self._num_trials}, start_time={self._start_time})"

    @staticmethod
    def format_time(time_s: float):
        if time_s < 1e-3:  # order of microseconds
            return f"{time_s * 1e6:.2f}µs"
        if time_s < 1:  # order of ms
            return f"{time_s * 1e3:.2f}ms"
        if time_s < 60:  # order of seconds
            return f"{time_s:.2f}s"
        elif time_s < 3600:  # order of minutes
            time_min = int(time_s // 60)
            time_s = time_s - 60 * time_min
            return f"{time_min:02d}m{time_s:04.1f}s"
        elif time_s < 60 * 60 * 24:  # order of hours
            time_h = int(time_s // 3600)
            time_s = time_s - time_h * 3600
            time_min = int(time_s // 60)
            time_s = int(time_s - time_min * 60)
            return f"{time_h}h{time_min:02d}m{time_s:02d}s"
        else:
            time_d = int(time_s // (60 * 60 * 24))
            time_s = time_s - time_d * 60 * 60 * 24
            time_h = int(time_s // 3600)
            time_s = time_s - time_h * 3600
            time_min = int(time_s // 60)
            time_s = int(time_s - time_min * 60)
            return f"{time_d}d{time_h:02d}h{time_min:02d}m{time_s:02d}s"
