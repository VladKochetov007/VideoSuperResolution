"""Thin fal.ai helpers shared by the fal-backed adapters.

Retry policy is cost-aware: upload and download are free, so we retry them on transient
network errors. `subscribe` is BILLABLE, so it is NOT auto-retried — a silent retry could
double-charge if the first call actually generated but the response was lost. On a subscribe
failure we surface the error; the caller records it and a manual re-run skips completed jobs.
"""

import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from dotenv import find_dotenv, load_dotenv

T = TypeVar("T")
_RETRIES = 3
_BACKOFF = 2.0
_MIN_BYTES = 1024


def ensure_fal_key() -> None:
    """fal_client reads FAL_KEY; map it from the project's FALAI_TOKEN if needed."""
    load_dotenv(find_dotenv(usecwd=True))
    if "FAL_KEY" not in os.environ and os.environ.get("FALAI_TOKEN"):
        os.environ["FAL_KEY"] = os.environ["FALAI_TOKEN"]
    if "FAL_KEY" not in os.environ:
        raise RuntimeError("no FAL_KEY / FALAI_TOKEN in environment")


def _retry(fn: Callable[[], T], what: str, retries: int = _RETRIES) -> T:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - transient net/fal errors, retry
            last = exc
            if attempt < retries:
                wait = _BACKOFF ** attempt
                print(f"  {what} failed (attempt {attempt}/{retries}): {exc} — retry in {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"{what} failed after {retries} attempts: {last}")


def upload(path: Path) -> str:
    import fal_client

    return _retry(lambda: fal_client.upload_file(str(path)), "upload")  # free → safe to retry


def subscribe(endpoint: str, arguments: dict) -> dict:
    import fal_client

    # BILLABLE: single attempt only (see module docstring).
    return fal_client.subscribe(endpoint, arguments=arguments, with_logs=False)


def download(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Stream to disk with a timeout, then verify it is not a truncated/empty file. Retried (free)."""

    def _dl() -> None:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        size = dest.stat().st_size
        if size < _MIN_BYTES:
            raise RuntimeError(f"downloaded file suspiciously small ({size} B)")

    _retry(_dl, "download")
