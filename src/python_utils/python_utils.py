import os
import sys
import time

class Timer:
    """A simple timer class for performance profiling Taken from https://github.com/flat
    ironinstitute/online_psp/blob/master/online_psp/online_psp_simulations.py.

    Usage:
    with Timer() as t:
        DO SOMETHING HERE
    print('Above (DO SOMETHING HERE) took %f sec.' % (t.interval))
    """

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start