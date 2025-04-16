from __future__ import annotations
import copy
import threading
from collections import OrderedDict
from contextlib import contextmanager

from dspy.dsp.utils.utils import dotdict

DEFAULT_CONFIG = dotdict(
    lm=None,
    adapter=None,
    rm=None,
    branch_idx=0,
    trace=[],
    bypass_assert=False,
    bypass_suggest=False,
    assert_failures=0,
    suggest_failures=0,
    experimental=False,
    backoff_time=10,
    callbacks=[],
    async_max_workers=8,
    send_stream=None,
    disable_history=False,
    track_usage=False,
    usage_tracker=None,
    caller_predict=None,
    stream_listeners=[],
    provide_traceback=False, # Whether to include traceback information in error logs.
    num_threads=8, # Number of threads to use for parallel processing.
)

class Settings(dotdict):
    def copy(self) -> Settings:
        return Settings({**settings, **self})

    @property
    def config(self):
        return self.copy()

    def configure(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    @contextmanager
    def context(self, **kwargs):
        """
        Context manager for temporary configuration changes at the thread level.
        Does not affect global configuration.
        """

        try:
            new_settings = self.copy()
            new_settings.configure(**kwargs)
            yield new_settings
        finally:
            pass

settings = Settings(**copy.deepcopy(DEFAULT_CONFIG))
