"""RQ worker entrypoint.

Runs inside the same container as the API, managed by supervisord.
Pre-warms the OCR pipeline before polling so the first job doesn't pay
the ~60-120s model-load cost.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

from rq import SimpleWorker

from . import jobs, ocr_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("worker")


def _publish_worker_diagnostics() -> None:
    """Write the worker's paddle/CUDA state to Redis so /debug/worker can
    surface it. This is the source of truth for what device inference is
    running on — the API process has a *separate* paddle import."""
    info: dict = {"published_at": time.time(), "pid": os.getpid()}
    try:
        import paddle
        info["paddle_version"] = paddle.__version__
        info["compiled_with_cuda"] = paddle.is_compiled_with_cuda()
        info["current_device"] = paddle.device.get_device()
        gpu_count = paddle.device.cuda.device_count() if info["compiled_with_cuda"] else 0
        info["gpu_count"] = gpu_count
        if gpu_count:
            info["gpu_name"] = paddle.device.cuda.get_device_name(0)
            try:
                info["memory_allocated_mib"] = paddle.device.cuda.memory_allocated() // (1024 * 1024)
                info["memory_reserved_mib"] = paddle.device.cuda.memory_reserved() // (1024 * 1024)
            except Exception as e:
                info["memory_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        info["paddle_error"] = f"{type(e).__name__}: {e}"
    try:
        jobs.get_redis().set("worker:diagnostics", json.dumps(info), ex=jobs.RESULT_TTL)
    except Exception as e:
        log.warning("failed to publish worker diagnostics: %s", e)


def main() -> None:
    log.info("warming OCR pipeline (this may take 1-2 min)...")
    ocr_service.get_pipeline()
    log.info("pipeline warm, starting RQ worker on queue '%s'", jobs.QUEUE_NAME)

    _publish_worker_diagnostics()

    queue = jobs.get_queue()
    # SimpleWorker runs jobs in the same process (no fork). This is required
    # for GPU work because CUDA contexts do not survive fork() — the default
    # forking Worker raises cudaErrorInitializationError in the child.
    worker = SimpleWorker([queue], connection=jobs.get_redis())
    worker.work(logging_level="INFO")


if __name__ == "__main__":
    main()
