import pathlib
import time
import os
import pathlib

from .stdlib.stopwatch import Stopwatch

class StemConstants:
    @property
    def STEM_LOG_DIR(self):
        return pathlib.Path('stem_log')

STEM_CONSTANTS = StemConstants()

class StateChangeListener:
    def __init__(self):
        self._last_update_time = None
        self._current_state = None
        self._last_state = None
        self._is_active = False
        self._has_changed = False

    def fetch(self, timeout=float('inf')):
        is_active = self._is_active
        has_changed = self._has_changed
        state = self._current_state

        if not is_active:
            return is_active, has_changed, state

        if (time.time() - self._last_update_time > timeout
            or self._has_changed):
            self._has_changed = False
            has_changed = True
            state = self._last_state
        else:
            state = self._current_state

        return is_active, has_changed, state
        
    def update(self, state):
        if (self._is_active
            and self._current_state != state):
            self._last_state = self._current_state
            self._has_changed = True

        self._is_active = True
        self._current_state = state
        self._last_update_time = time.time()


class SamplingRateWatchdog:
    def __init__(self, sampling_rate_min):
        self._sampling_rate_min = sampling_rate_min
        self._sw = Stopwatch()
        self._sampling_rate_average = 0

    @property
    def sampling_rate(self):
        return self._sampling_rate_average

    def start(self):
        self._sw.start()
    
    def lap(self):
        rate = 1 / (self._sw.lap() + 1e-8)
        self._sampling_rate_average = 0.5 * (rate + self._sampling_rate_average)
        return self._sampling_rate_average >= self._sampling_rate_min

