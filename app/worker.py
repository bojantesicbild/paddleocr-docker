"""RQ worker entrypoint.

Runs inside the same container as the API, managed by supervisord. Pre-warms
the Docling pipeline before polling so the first job doesn't pay the
~30-60s model-load cost.
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
    """Publish docling + torch state to Redis so /debug/worker can surface
    where inference is actually running."""
    info: dict = {"published_at": time.time(), "pid": os.getpid(), "backend": "docling"}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                info["memory_allocated_mib"] = torch.cuda.memory_allocated() // (1024 * 1024)
                info["memory_reserved_mib"] = torch.cuda.memory_reserved() // (1024 * 1024)
            except Exception as e:
                info["memory_error"] = f"{type(e).__name__}: {e}"
            info["current_device"] = f"gpu:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        else:
            info["current_device"] = "cpu"
    except Exception as e:
        info["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        import docling
        info["docling_version"] = getattr(docling, "__version__", "unknown")
    except Exception as e:
        info["docling_error"] = f"{type(e).__name__}: {e}"
    try:
        jobs.get_redis().set("worker:diagnostics", json.dumps(info), ex=jobs.RESULT_TTL)
    except Exception as e:
        log.warning("failed to publish worker diagnostics: %s", e)


def main() -> None:
    log.info("warming Docling pipeline (this may take 30-60s)...")
    ocr_service.get_converter()
    log.info("pipeline warm, starting RQ worker on queue '%s'", jobs.QUEUE_NAME)

    _publish_worker_diagnostics()

    queue = jobs.get_queue()
    # SimpleWorker runs jobs in the same process (no fork). Required for GPU
    # work — CUDA contexts don't survive fork() and the default forking
    # Worker raises cudaErrorInitializationError in the child.
    worker = SimpleWorker([queue], connection=jobs.get_redis())
    worker.work(logging_level="INFO")


if __name__ == "__main__":
    main()
