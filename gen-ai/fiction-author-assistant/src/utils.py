#!/usr/bin/env python

import time
import random
from typing import Callable, Any

def retry(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    initial_delay: float = 0.5,
    multiplier: float = 1.5,
    jitter: float | None = 0.5,
    max_delay: float = 32.0,
    max_retries: int = 10,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """
    Retry a function with exponential backoff. Defaults:
    - 0.5s initial delay between retries
    - 1.5x multiplier for exponential backoff
    - 50% jitter (so actual delay ranges from 50% to 150% of calculated delay)
    - 32s max delay
    - 10 max retries
    - retry on any exception
    """
    for attempt in range(max_retries):
        try:
            return func(*args)
        except exceptions as e:
            if attempt == max_retries - 1:
                raise
            print(e)

            delay = initial_delay * (multiplier ** attempt)

            if jitter is not None:
                # with jitter = 0.5, delay can be 50% higher or lower, so jitter_range = 1.0
                jitter_range = 2 * jitter
                # select a random value from the range
                # subtract the jitter to center it around 0
                # add 1 to center it around 1 to get final jitter multiplier
                jitter_factor = 1 + (random.random() * jitter_range - jitter)
                # adjust delay by jitter_factor
                delay = delay * jitter_factor

            delay = min(delay, max_delay)

            print(f'{func.__name__} - attempt {attempt + 1} failed, retrying in {delay:.2f} seconds...')
            time.sleep(delay)
