#!/usr/bin/env python3
"""One-shot verification pipeline for the stripe reconstruction service.

Stages, in order:

1. unit tests      -- ``python -m unittest discover -s tests``
2. package build   -- ``python scripts/package_build.py`` (wheel + sdist)
3. API smoke test  -- drives a running service through reconstruction,
                      over-capacity (422) and contradiction (409) cases

The process exits 0 only when every stage passes. It is intended to run
as the ``verify`` service from docker-compose.yml, but also works
against a locally started server:

    python verify.py                          # http://localhost:8000
    APP_URL=http://127.0.0.1:9000 python verify.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
APP_URL = os.environ.get("APP_URL", "http://localhost:8000").rstrip("/")
HEALTH_TIMEOUT_SECONDS = float(os.environ.get("VERIFY_HEALTH_TIMEOUT", "60"))


# --------------------------------------------------------------------------
# stage 1: unit tests
# --------------------------------------------------------------------------

def run_unit_tests() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError("unit tests exited with code %d" % proc.returncode)


# --------------------------------------------------------------------------
# stage 2: publishable package build
# --------------------------------------------------------------------------

def run_package_build() -> None:
    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", "package_build.py")],
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError("package build exited with code %d" % proc.returncode)


# --------------------------------------------------------------------------
# stage 3: API smoke test
# --------------------------------------------------------------------------

def http_json(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        APP_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def wait_for_health() -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while True:
        try:
            status, body = http_json("GET", "/health")
            if status == 200 and body.get("status") == "ok":
                return
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("service at %s did not become healthy" % APP_URL)
        time.sleep(0.5)


def make_stripe(data_count: int, size: int, seed: int) -> tuple[list[bytes], list[str], list[str]]:
    sys.path.insert(0, ROOT)
    from app import reed_solomon as rs

    rng = random.Random(seed)
    data = [rng.randbytes(size) for _ in range(data_count)]
    p, q = rs.compute_parity(data)
    stripe = data + [p, q]
    encoded = [base64.b64encode(s).decode("ascii") for s in stripe]
    digests = [hashlib.sha256(s).hexdigest() for s in stripe]
    return stripe, encoded, digests


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_smoke_test() -> None:
    wait_for_health()

    data_count, size = 6, 257
    stripe, encoded, digests = make_stripe(data_count, size, seed=20261005)

    def request(shards, digest_list=digests):
        return {
            "dataShards": data_count,
            "shardSize": size,
            "shards": shards,
            "digests": digest_list,
        }

    # Case 1: two missing data shards are reconstructed exactly.
    shards = list(encoded)
    shards[1] = None
    shards[4] = None
    status, body = http_json("POST", "/api/stripes/reconstruct", request(shards))
    expect(status == 200, "case 1: expected 200, got %d (%s)" % (status, body))
    expect(body["recoveredIndices"] == [1, 4], "case 1: wrong recoveredIndices")
    expect(body["shards"] == encoded, "case 1: reconstructed shards differ")
    expect(
        body["digests"] == [hashlib.sha256(s).hexdigest() for s in stripe],
        "case 1: digest list mismatch",
    )

    # Case 2: missing P and Q parity shards are recomputed.
    shards = list(encoded)
    shards[data_count] = None
    shards[data_count + 1] = None
    status, body = http_json("POST", "/api/stripes/reconstruct", request(shards))
    expect(status == 200, "case 2: expected 200, got %d (%s)" % (status, body))
    expect(body["recoveredIndices"] == [data_count, data_count + 1], "case 2: wrong indices")
    expect(body["shards"] == encoded, "case 2: parity shards differ")

    # Case 3: three missing shards exceed the recovery capability -> 422.
    shards = list(encoded)
    for i in (0, 2, data_count):
        shards[i] = None
    status, body = http_json("POST", "/api/stripes/reconstruct", request(shards))
    expect(status == 422, "case 3: expected 422, got %d (%s)" % (status, body))
    expect(body["error"]["code"] == "TOO_MANY_MISSING_SHARDS", "case 3: wrong error code")
    expect("shards" not in body, "case 3: failure leaked shard data")

    # Case 4: a silently corrupted surviving shard -> 409, never recovered.
    shards = list(encoded)
    corrupted = bytes([stripe[3][0] ^ 0xFF]) + stripe[3][1:]
    shards[3] = base64.b64encode(corrupted).decode("ascii")
    status, body = http_json("POST", "/api/stripes/reconstruct", request(shards))
    expect(status == 409, "case 4: expected 409, got %d (%s)" % (status, body))
    expect(body["error"]["code"] == "SHARD_DIGEST_MISMATCH", "case 4: wrong error code")
    expect(body["error"]["shardIndex"] == 3, "case 4: wrong shardIndex")
    expect("shards" not in body, "case 4: failure leaked shard data")

    # Case 5: parity contradiction with self-consistent digests -> 409.
    shards = list(encoded)
    bad_p = bytes(b ^ 0x5A for b in stripe[data_count])
    shards[data_count] = base64.b64encode(bad_p).decode("ascii")
    tampered_digests = list(digests)
    tampered_digests[data_count] = hashlib.sha256(bad_p).hexdigest()
    status, body = http_json(
        "POST", "/api/stripes/reconstruct", request(shards, tampered_digests)
    )
    expect(status == 409, "case 5: expected 409, got %d (%s)" % (status, body))
    expect(body["error"]["code"] == "PARITY_RELATION_MISMATCH", "case 5: wrong error code")
    expect("shards" not in body, "case 5: failure leaked shard data")


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def main() -> int:
    stages = [
        ("unit-tests", run_unit_tests),
        ("package-build", run_package_build),
        ("api-smoke", run_smoke_test),
    ]
    failures = []
    for name, stage in stages:
        print("[verify] --- stage: %s ---" % name, flush=True)
        try:
            stage()
        except Exception as err:  # noqa: BLE001 - report and continue
            print("[verify] %s: FAIL (%s)" % (name, err), flush=True)
            failures.append(name)
        else:
            print("[verify] %s: OK" % name, flush=True)

    if failures:
        print("[verify] FAILED stages: %s" % ", ".join(failures), flush=True)
        return 1
    print("[verify] all stages passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
