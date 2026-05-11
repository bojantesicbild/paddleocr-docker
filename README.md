# paddleocr-docker

Dockerized **PaddleOCR-VL-1.5** HTTP service for the HillMetrics.AI pipeline.
Accepts images or PDFs and returns Markdown with `> Page N` blockquote page
markers, GitHub-flavored tables, `<image label="chart">[region_id]</image>`
placeholders, plus base64-encoded crops for downstream vision-LLM description.

Ships as a single container running supervisord → redis + FastAPI + RQ worker.

## Stack

- FastAPI + Uvicorn (sync + async API, HTML testing UI at `/`)
- **PaddleOCR-VL-1.5** (0.9B-parameter VLM — handles layout, OCR, tables,
  formulas end-to-end; much better structured-table output than PP-StructureV3)
- Redis + RQ for background job queueing
- supervisord as PID 1 to run all three processes in one image
- PyMuPDF for PDF → page images
- BeautifulSoup for HTML table → GFM conversion

## Endpoints

| Method | Path              | Body                                                              | Returns |
|--------|-------------------|-------------------------------------------------------------------|---------|
| GET    | `/`               | —                                                                 | HTML testing UI |
| GET    | `/health`         | —                                                                 | `{"status":"ready"}` once Redis is reachable |
| GET    | `/config`         | —                                                                 | `{"auth_required": true\|false}` |
| POST   | `/ocr/image`      | multipart: `file`, `page_number` (default 1), `settings` (JSON)   | `OCRResponse` (sync) or `{"job_id": ...}` (with `?async=1`) |
| POST   | `/ocr/pdf`        | multipart: `file`, `dpi` (default 200), `settings` (JSON)         | `OCRResponse` (sync) or `{"job_id": ...}` (with `?async=1`) |
| GET    | `/jobs/{job_id}`  | _(auth-gated)_                                                    | `{"status": "queued\|started\|finished\|failed", "result": ..., "error": ...}` |
| GET    | `/debug/env`      | —                                                                 | API-process paddle/CUDA wiring + env vars (no secrets). Note: API is a separate process from the worker. |
| GET    | `/debug/worker`   | _(auth-gated)_                                                    | **Worker** process's paddle state — `current_device`, GPU memory. This is where inference runs; check this to confirm GPU binding. |
| GET    | `/debug/gpu`      | _(auth-gated)_                                                    | Live `nvidia-smi` snapshot: utilization, memory, temp, power per GPU |
| GET    | `/docs`           | —                                                                 | Swagger UI |

`POST /ocr/*` endpoints always enqueue the job on the internal Redis. With
`?async=1` the response is 202 with the job id — poll `/jobs/{id}`. Without
`?async=1` the API blocks until the worker finishes (up to `OCR_JOB_TIMEOUT`)
and returns the full `OCRResponse`.

### `OCRResponse` shape

```json
{
  "markdown": "> Page 1\n\n# Title\n\nBody…\n\n<image label=\"chart\">[region_3]</image>\n\n| a | b |\n| --- | --- |\n| 1 | 2 |",
  "images": [
    {
      "region_id": "region_3",
      "page": 1,
      "label": "chart",
      "bbox": [x0, y0, x1, y1],
      "png_base64": "iVBORw0KGgoAAAA…"
    }
  ],
  "metadata": {
    "library": "paddleocr",
    "model": "PaddleOCR-VL-1.5",
    "version": "3.4.0",
    "language": "ch",
    "detected_language": { "code": "fr", "confidence": 0.9997 },
    "page_count": 1,
    "duration_ms": 1234,
    "settings": { "include_header": false, ... }
  }
}
```

`language` is the configured recognition model (env `OCR_LANGUAGE`).
`detected_language` is a post-OCR `langdetect` guess on the final markdown:
`{code, confidence}` where `code` is a 2-letter ISO 639-1 tag. It is `null`
when the page has too little text to classify.

For multi-page PDFs, `region_id` values are prefixed with `p{N}_` to stay
unique across pages.

## Configuration

Copy `.env.example` to `.env` and edit. Both `docker-compose.yml` and the
overlays read variables from there.

| Variable            | Default  | Purpose |
|---------------------|----------|---------|
| `OCR_LANGUAGE`      | `ch`     | PaddleOCR language code. `ch` is multilingual (CJK + Latin). |
| `OCR_JOB_TIMEOUT`   | `1800`   | Per-job timeout in the RQ worker (seconds). |
| `OCR_RESULT_TTL`    | `86400`  | How long finished job results stay in Redis (seconds). |
| `OCR_API_KEY`       | _(unset)_| If set, `/ocr/*` and `/jobs/*` require `X-API-Key: <value>`. |
| `VL_REMOTE_URL`     | _(unset)_| If set, the VLM step is offloaded to a remote vllm/sglang/fastdeploy server. Lets you run the API on a slow GPU (e.g. V100) and route generation to a fast one. |
| `VL_REMOTE_BACKEND` | `vllm-server` | Used only when `VL_REMOTE_URL` is set. One of `vllm-server`, `sglang-server`, `fastdeploy-server`, `mlx-vlm-server`. |
| `VL_REMOTE_API_KEY` | _(unset)_| Auth header for the remote VL server, if it requires one. |
| `HOST_PORT`         | `8090`   | Host port mapped to the container's 8080. |
| `DOMAIN`            | _(unset)_| TLS overlay only: public hostname for Caddy. |
| `CADDY_EMAIL`       | _(unset)_| TLS overlay only: Let's Encrypt contact email. |

`ocr_settings.json` holds the defaults sent to `PaddleOCRVL.predict()` — per
request these are overridable via the `settings` JSON form field (the UI's
checkboxes build this for you).

### Speed knobs (per-request, V100 users especially)

V100/V100S can't run vLLM/FlashAttention 2 (CC < 8.0) and inference is
~100 s/page. To trade some quality for speed:

| Setting | Effect | Notes |
|---|---|---|
| `max_pixels` (in JSON `settings`) | Caps the VLM's input image size. 1280×960 ≈ 1228800 ≈ half of default → roughly half the tokens → roughly half the time. | Small text may be missed. |
| `max_new_tokens` (in JSON `settings`) | Caps generation length per block. Default ~1024. | Set lower to bound worst-case latency; risk truncating long blocks. |
| `dpi` (form field, PDFs only) | Lowers the rasterized page size before layout/VL. Default 200. | 150 gives noticeable speedup with mild quality loss. |
| Include-block toggles | Each enabled label is an extra VLM call. | Default OFF for header/footer/page-number is intentional. |

The HTML UI exposes `max_pixels`, `max_new_tokens`, and `dpi` directly so you
can A/B them.

## Build & run (Docker)

### CPU (local dev)

```bash
cp .env.example .env
docker compose up -d --build        # first build: ~10 min (wheels + image)
open http://localhost:8090/         # HTML testing UI
```

First OCR call downloads the VL model (~4 GB) into the `paddle-models` named
volume; subsequent starts reuse it.

### GPU (any NVIDIA host, including OVH)

Requires NVIDIA driver + NVIDIA Container Toolkit on the host. GPU wheels are
`linux/amd64` only.

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

GPU build pre-caches the PP-DocLayoutV3 + PaddleOCR-VL-1.5 weights into the
image (~5 GB total), so first request pays no download cost.

## Deploy to OVH (GPU instance)

One-time host setup — on the OVH VM (Ubuntu 22.04/24.04):

```bash
# 1. Verify NVIDIA driver
nvidia-smi

# 2. Install Docker Engine (if not present)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# 3. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Sanity check
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 nvidia-smi
```

### Deploy

Two routes — pulling a pre-built image from GHCR is much faster than
building on the OVH VM (skips ~30 min of wheel + model bake):

#### Option 1 — pull from GHCR (recommended)

CI builds `linux/amd64` GPU images on every push to `main`. CI runners
have no NVIDIA driver, so the image is built with `PRECACHE_MODELS=0`
— the worker downloads ~5 GB of weights into the `paddle-models` named
volume on first boot (5–10 min). Every subsequent start reuses the
volume. If you want a fully pre-baked image, build on the GPU host
(Option 2).

```bash
git clone <this repo>
cd paddle-ocr-docker
cp .env.example .env
# Set OCR_API_KEY, DOMAIN, CADDY_EMAIL in .env

# Pull the GHCR image and tag it as the local name compose expects.
docker pull ghcr.io/<your-gh-org>/paddleocr-docker:gpu-latest
docker tag  ghcr.io/<your-gh-org>/paddleocr-docker:gpu-latest paddle-ocr-viewer:gpu

# A) GPU + prod, direct HTTP (behind your own LB / VPN)
docker compose -f docker-compose.yml \
               -f docker-compose.gpu.yml \
               -f docker-compose.prod.yml \
               up -d   # no --build; uses the pulled image

# B) GPU + prod + automatic HTTPS (Let's Encrypt via Caddy)
docker compose -f docker-compose.yml \
               -f docker-compose.gpu.yml \
               -f docker-compose.prod.yml \
               -f docker-compose.tls.yml \
               up -d
```

#### Option 2 — build on the VM

Same commands as Option 1 but with `--build`. First build is ~30 min
(installs CUDA wheels + bakes models). Subsequent builds reuse layer
cache.

For (B) the DNS `A` record for `$DOMAIN` must already point at the VM and
ports 80 + 443 must be open on the OVH firewall before you bring it up —
Caddy's ACME challenge needs both.

#### First-boot expectations

- Pulled image (Option 1): container becomes healthy in ~30–60 s; first
  inference ~5–15 s on a V100S.
- Built locally without `PRECACHE_MODELS=1`: first boot downloads ~5 GB
  of weights from HuggingFace into the `paddle-models` named volume —
  expect 5–10 min before the worker is warm. Subsequent starts reuse the
  volume and are fast.

### Verifying

```bash
curl https://$DOMAIN/health        # {"status":"ready"}
curl https://$DOMAIN/config        # {"auth_required":true|false}
# End-to-end sanity check with the CLI helper:
OCR_API_KEY=$OCR_API_KEY HOST=https://$DOMAIN ./extract.sh sample.jpg
```

### Operations

- **Process logs (host)**: `docker compose logs -f paddle-ocr` — full
  supervisord/uvicorn/redis/worker output. There is no `/logs` API
  endpoint by design: supervisord forwards subprocess stdout to the
  Docker daemon, so the host's `docker logs` is the source of truth.
- **GPU monitoring (API)**: `GET /debug/gpu` — JSON `nvidia-smi` snapshot
  per call. Pair it with your monitoring stack to alert on memory or
  temperature thresholds without granting shell access.
- **Restart**: `docker compose restart paddle-ocr`
- **Update**: `git pull && docker compose up -d --build` (models stay cached in the volume)
- **Nuke and re-download models**: `docker volume rm paddle-ocr-docker_paddle-models`

## CLI testing

`extract.sh` wraps the API for one-shot CLI use:

```bash
./extract.sh ./document.pdf                           # → ./document.md
SAVE_IMAGES=1 ./extract.sh ./scan.png                 # + ./scan.images/*.png
HOST=https://ocr.example.com OCR_API_KEY=xyz ./extract.sh ./doc.pdf
```

## Layout

```
app/
├── main.py              # FastAPI: routes, auth dep, UI, job polling
├── jobs.py              # RQ queue + run_ocr_* job functions
├── worker.py            # RQ worker entrypoint (warms the pipeline, then polls)
├── ocr_service.py       # PaddleOCRVL singleton + block/crop extraction
├── markdown_format.py   # HTML-table → GFM, region-id prefixing, page markers
├── pdf_split.py         # PyMuPDF PDF → PNG pages
├── schemas.py           # Pydantic models (OCRResponse, ImageCrop, OCRMetadata)
└── templates/index.html # Testing UI
Dockerfile               # CPU build
Dockerfile.gpu           # CUDA 12.6 runtime, paddlepaddle-gpu, linux/amd64
docker-compose.yml       # Base (CPU)
docker-compose.gpu.yml   # Overlay: GPU target, pre-cache models
docker-compose.prod.yml  # Overlay: restart:always, log rotation
docker-compose.tls.yml   # Overlay: Caddy reverse proxy with ACME
supervisord.conf         # Runs redis + api + worker in one container
Caddyfile                # Caddy config (used by docker-compose.tls.yml)
ocr_settings.json        # Default VL flags, block inclusion toggles
.env.example             # Template for per-host env vars
extract.sh               # CLI helper
```
