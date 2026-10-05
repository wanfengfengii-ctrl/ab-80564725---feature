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


def scatter_physical(
    data_count: int, logical: list[bytes], p_index: int, q_index: int
) -> list[bytes]:
    """Lay out [D0..Dn-1, P, Q] onto physical slots of a rotated stripe."""
    total = data_count + 2
    physical = [b""] * total
    data_slots = [i for i in range(total) if i != p_index and i != q_index]
    for logical_index, slot in enumerate(data_slots):
        physical[slot] = logical[logical_index]
    physical[p_index] = logical[data_count]
    physical[q_index] = logical[data_count + 1]
    return physical


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

    # ------------------------------------------------------------------
    # Rotated physical layouts: P/Q are not in the final two slots.
    # ------------------------------------------------------------------
    rot_n, rot_size = 5, 201
    rot_logical, _, _ = make_stripe(rot_n, rot_size, seed=20261006)
    rot_total = rot_n + 2

    def rotated_request(p_index, q_index, shards, digest_list):
        return {
            "dataShards": rot_n,
            "shardSize": rot_size,
            "parityIndices": [p_index, q_index],
            "shards": shards,
            "digests": digest_list,
        }

    # Case 6: layout P,Q,D0..D4 (parity at physical slots 0 and 1) with
    # two data shards missing; reconstruction must stay in physical order.
    p_index, q_index = 0, 1
    physical = scatter_physical(rot_n, rot_logical, p_index, q_index)
    physical_enc = [base64.b64encode(s).decode("ascii") for s in physical]
    physical_digests = [hashlib.sha256(s).hexdigest() for s in physical]
    shards = list(physical_enc)
    shards[3] = None  # D1
    shards[5] = None  # D3
    status, body = http_json(
        "POST",
        "/api/stripes/reconstruct",
        rotated_request(p_index, q_index, shards, physical_digests),
    )
    expect(status == 200, "case 6: expected 200, got %d (%s)" % (status, body))
    expect(body["parityIndices"] == [0, 1], "case 6: parityIndices not echoed")
    expect(body["recoveredIndices"] == [3, 5], "case 6: wrong recoveredIndices")
    expect(body["shards"] == physical_enc, "case 6: reconstructed shards differ")
    expect(body["digests"] == physical_digests, "case 6: digest order drifted")

    # Case 7: one data slot plus the Q physical slot missing; the Q slot
    # is slot 1 here, so recoveredIndices must name physical slot 1.
    shards = list(physical_enc)
    shards[2] = None  # D0
    shards[q_index] = None
    status, body = http_json(
        "POST",
        "/api/stripes/reconstruct",
        rotated_request(p_index, q_index, shards, physical_digests),
    )
    expect(status == 200, "case 7: expected 200, got %d (%s)" % (status, body))
    expect(body["recoveredIndices"] == [1, 2], "case 7: wrong recoveredIndices")
    expect(body["shards"] == physical_enc, "case 7: reconstructed shards differ")

    # Case 8: illegal parity index (out of range) -> locatable 422.
    shards = list(physical_enc)
    status, body = http_json(
        "POST",
        "/api/stripes/reconstruct",
        rotated_request(rot_total, 1, shards, physical_digests),
    )
    expect(status == 422, "case 8: expected 422, got %d (%s)" % (status, body))
    expect(body["error"]["code"] == "INVALID_PARITY_INDICES", "case 8: wrong code")
    expect(body["error"]["shardIndex"] == rot_total, "case 8: wrong physical index")
    expect("shards" not in body, "case 8: failure leaked shard data")

    # Case 9: repeated parity index (P == Q slot) -> locatable 422.
    status, body = http_json(
        "POST",
        "/api/stripes/reconstruct",
        rotated_request(3, 3, physical_enc, physical_digests),
    )
    expect(status == 422, "case 9: expected 422, got %d (%s)" % (status, body))
    expect(body["error"]["code"] == "DUPLICATE_PARITY_INDEX", "case 9: wrong code")
    expect(body["error"]["shardIndex"] == 3, "case 9: wrong physical index")

    # Case 10: valid layout but a surviving shard contradicts its digest;
    # the reported index must be the physical disk slot, not a logical one.
    shards = list(physical_enc)
    corrupted = bytes([physical[4][0] ^ 0x7E]) + physical[4][1:]
    shards[4] = base64.b64encode(corrupted).decode("ascii")
    status, body = http_json(
        "POST",
        "/api/stripes/reconstruct",
        rotated_request(p_index, q_index, shards, physical_digests),
    )
    expect(status == 409, "case 10: expected 409, got %d (%s)" % (status, body))
    expect(body["error"]["code"] == "SHARD_DIGEST_MISMATCH", "case 10: wrong code")
    expect(body["error"]["shardIndex"] == 4, "case 10: wrong physical slot")
    expect("shards" not in body, "case 10: failure leaked shard data")

    # Case 11: valid layout, digests self-consistent, but the P relation
    # does not hold at the named physical P slot -> 409 with physical slots.
    shards = list(physical_enc)
    bad_p = bytes(b ^ 0x3C for b in physical[p_index])
    shards[p_index] = base64.b64encode(bad_p).decode("ascii")
    bad_digests = list(physical_digests)
    bad_digests[p_index] = hashlib.sha256(bad_p).hexdigest()
    status, body = http_json(
        "POST",
        "/api/stripes/reconstruct",
        rotated_request(p_index, q_index, shards, bad_digests),
    )
    expect(status == 409, "case 11: expected 409, got %d (%s)" % (status, body))
    expect(body["error"]["code"] == "PARITY_RELATION_MISMATCH", "case 11: wrong code")
    expect(body["error"]["pIndex"] == 0 and body["error"]["qIndex"] == 1,
           "case 11: wrong physical parity indices")
    expect("P" in body["error"]["parity"], "case 11: P defect not reported")
    expect("shards" not in body, "case 11: failure leaked shard data")


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
