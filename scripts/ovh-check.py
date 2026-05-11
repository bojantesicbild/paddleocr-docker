#!/usr/bin/env python3
"""Local OVH credentials + rights smoke test.

Reads OVH_* env vars (loaded from .env if present) and tries each call the
redeploy workflow needs. Reports PASS/FAIL per right, never actually
stops/starts your app.

Usage:
    pip install ovh python-dotenv
    cp .env.example .env   # add OVH_* values
    python scripts/ovh-check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Tiny dotenv reader (avoids the python-dotenv dependency)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        os.environ.setdefault(key, val)


def main() -> int:
    load_env_file(Path(__file__).resolve().parent.parent / ".env")

    required = ["OVH_APPLICATION_KEY", "OVH_APPLICATION_SECRET", "OVH_CONSUMER_KEY",
                "OVH_PROJECT_ID", "OVH_APP_ID"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"❌ missing env vars: {missing}")
        print("   Put them in .env or export them before running.")
        return 2

    try:
        import ovh
    except ImportError:
        print("❌ python-ovh not installed. run: pip install ovh")
        return 2

    client = ovh.Client(
        endpoint="ovh-eu",
        application_key=os.environ["OVH_APPLICATION_KEY"],
        application_secret=os.environ["OVH_APPLICATION_SECRET"],
        consumer_key=os.environ["OVH_CONSUMER_KEY"],
    )
    project = os.environ["OVH_PROJECT_ID"]
    app = os.environ["OVH_APP_ID"]

    checks: list[tuple[str, str, callable]] = [
        ("GET /cloud/project",
         "list projects (needed by discover workflow)",
         lambda: client.get("/cloud/project")),
        (f"GET /cloud/project/{project[:6]}…",
         "read project metadata",
         lambda: client.get(f"/cloud/project/{project}")),
        (f"GET /cloud/project/{project[:6]}…/ai/app",
         "list AI apps in project",
         lambda: client.get(f"/cloud/project/{project}/ai/app")),
        (f"GET /cloud/project/{project[:6]}…/ai/app/{app[:6]}…",
         "read your specific app",
         lambda: client.get(f"/cloud/project/{project}/ai/app/{app}")),
    ]

    print(f"Testing OVH credentials against project {project[:6]}…, app {app[:6]}…\n")
    state_payload = None
    all_pass = True
    for label, why, fn in checks:
        try:
            payload = fn()
            print(f"✅ {label:55s}  {why}")
            if "ai/app/" in label:
                state_payload = payload
        except Exception as e:
            print(f"❌ {label:55s}  {why}")
            print(f"   {type(e).__name__}: {e}")
            all_pass = False

    if state_payload:
        spec = state_payload.get("spec") or {}
        status = state_payload.get("status") or {}
        print(f"\nApp summary:")
        print(f"  name      : {spec.get('name', '?')}")
        print(f"  state     : {status.get('state', '?')}")
        print(f"  image     : {spec.get('image', '?')}")
        print(f"  url       : {status.get('url', '?')}")

    # Show the current token's actual rules — answers "did the regenerated
    # token really get PUT /start and /stop?" without needing to try them.
    try:
        cred = client.get("/auth/currentCredential")
        print(f"\nToken rules (id={cred.get('credentialId')}):")
        for rule in cred.get("rules", []):
            print(f"  {rule.get('method', '?'):6s} {rule.get('path', '?')}")
    except Exception as e:
        print(f"\n(can't list rules: {type(e).__name__}: {e})")

    if not all_pass:
        print("\nReads failed → token rights are wrong (or wrong project/app ID).")
        return 1

    # Don't actually stop/start in a smoke test. Just confirm OPTIONS allows it.
    # OVH doesn't expose HTTP OPTIONS for ACLs, so we can't fully test PUT
    # without invoking — but if reads work and you set the same scope rules
    # for PUTs, the workflow will succeed.
    print("\n✅ All reads pass. PUT /start and PUT /stop still need to be tested")
    print("   by running the redeploy workflow (would stop your app).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
