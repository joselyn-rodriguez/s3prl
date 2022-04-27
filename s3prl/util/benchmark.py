import logging
from contextlib import ContextDecorator
from time import time
from typing import Any

logger = logging.getLogger(__name__)


class benchmark(ContextDecorator):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __enter__(self):
        logger.info(f"{self.name}: start")
        self.start = time()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        logger.info(f"{self.name}: finish with {time() - self.start} seconds")
