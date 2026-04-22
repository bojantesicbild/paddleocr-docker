"""RQ worker entrypoint.

Runs inside the same container as the API, managed by supervisord.
Pre-warms the OCR pipeline before polling so the first job doesn't pay
the ~60-120s model-load cost.
"""
from __future__ import annotations

import logging
import sys

from rq import Worker

from . import jobs, ocr_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("worker")


def main() -> None:
    log.info("warming OCR pipeline (this may take 1-2 min)...")
    ocr_service.get_pipeline()
    log.info("pipeline warm, starting RQ worker on queue '%s'", jobs.QUEUE_NAME)

    queue = jobs.get_queue()
    worker = Worker([queue], connection=jobs.get_redis())
    worker.work(logging_level="INFO")


if __name__ == "__main__":
    main()
