import asyncio
import threading

from typing import Tuple, List, Any

from dspy.primitives.example import Example
from dspy.utils.parallelizer import ParallelExecutor


class Parallel:
    def __init__(
        self,
        num_threads: int = 32,
        max_errors: int = 10,
        return_failed_examples: bool = False,
        provide_traceback: bool = False,
        disable_progress_bar: bool = False,
    ):
        super().__init__()
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.return_failed_examples = return_failed_examples
        self.provide_traceback = provide_traceback
        self.disable_progress_bar = disable_progress_bar

        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()
        self.failed_examples = []
        self.exceptions = []


    async def forward(self, settings, exec_pairs: List[Tuple[Any, Example]], num_threads: int = None) -> List[Any]:

        async def process_pair(pair):
            result = None
            module, example = pair

            if isinstance(example, Example):
                result = await module(settings, example)
            elif isinstance(example, dict):
                result = await module(settings, **example)
            elif isinstance(example, list) and module.__class__.__name__ == "Parallel":
                result = await module(settings, example)
            elif isinstance(example, tuple):
                result = await module(settings, *example)
            else:
                raise ValueError(f"Invalid example type: {type(example)}, only supported types are Example, dict, list and tuple")
            return result

        # Execute the processing function over the execution pairs
        results = await asyncio.gather(*[process_pair(pair) for pair in exec_pairs])

        if self.return_failed_examples:
            return results, self.failed_examples, self.exceptions
        else:
            return results


    async def __call__(self, settings, *args: Any, **kwargs: Any) -> Any:
        return await self.forward(settings, *args, **kwargs)
