#!/usr/bin/env bash
# Post a file to the running paddle-ocr service and save the markdown response.
# Usage:
#   ./extract.sh <file> [output.md]
#   ./extract.sh doc.pdf
#   ./extract.sh scan.png result.md
# Env:
#   HOST           (default http://localhost:8090)
#   DPI            (default 200, PDFs only)
#   OCR_API_KEY    sent as X-API-Key header if set (server-side auth)
#   SAVE_IMAGES=1  also write the base64 crops to <output>.images/region_*.png

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <file> [output.md]" >&2
  exit 1
fi

for bin in curl jq; do
  command -v "$bin" >/dev/null || { echo "missing dependency: $bin" >&2; exit 1; }
done

input="$1"
[[ -f "$input" ]] || { echo "not found: $input" >&2; exit 1; }

host="${HOST:-http://localhost:8090}"
output="${2:-${input%.*}.md}"
ext="${input##*.}"
ext="${ext,,}"

case "$ext" in
  pdf)  endpoint="$host/ocr/pdf";  extra=(-F "dpi=${DPI:-200}") ;;
  png|jpg|jpeg|webp|bmp|tif|tiff) endpoint="$host/ocr/image"; extra=(-F "page_number=1") ;;
  *) echo "unsupported extension: $ext" >&2; exit 1 ;;
esac

echo "POST $endpoint ($input)" >&2
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

auth_args=()
if [[ -n "${OCR_API_KEY:-}" ]]; then
  auth_args=(-H "X-API-Key: $OCR_API_KEY")
fi

http=$(curl -sS -o "$tmp" -w '%{http_code}' "${auth_args[@]}" -F "file=@${input}" "${extra[@]}" "$endpoint")
if [[ "$http" != "200" ]]; then
  echo "API error ($http):" >&2
  jq . "$tmp" 2>/dev/null || cat "$tmp" >&2
  exit 1
fi

jq -r '.markdown' "$tmp" > "$output"
echo "wrote $output ($(wc -c < "$output") bytes)" >&2

if [[ "${SAVE_IMAGES:-0}" == "1" ]]; then
  img_dir="${output%.md}.images"
  mkdir -p "$img_dir"
  count=$(jq -r '.images | length' "$tmp")
  for ((i=0; i<count; i++)); do
    rid=$(jq -r ".images[$i].region_id" "$tmp")
    jq -r ".images[$i].png_base64" "$tmp" | base64 -D > "$img_dir/$rid.png"
  done
  echo "wrote $count crops to $img_dir/" >&2
fi

jq -r '.metadata | "pages=\(.page_count) duration=\(.duration_ms)ms"' "$tmp" >&2
