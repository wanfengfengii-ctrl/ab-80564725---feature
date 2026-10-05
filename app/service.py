"""Request validation and orchestration for the reconstruct endpoint.

Error taxonomy
--------------
* 422 -- the request itself is invalid or asks for more than the code
  can recover (e.g. more than two missing shards).
* 409 -- the request is well formed but the supplied material is
  contradictory: a surviving shard does not match its expected digest,
  a reconstructed shard does not match its expected digest, or the
  P/Q parity relations are inconsistent.

No error response ever contains shard data; only a fully verified
stripe is returned.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from typing import Any

from . import reed_solomon as rs

MIN_DATA_SHARDS = 2
MAX_DATA_SHARDS = 16
MIN_SHARD_SIZE = 1
MAX_SHARD_SIZE = 4096
MAX_RECOVERABLE_MISSING = 2

_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ApiError(Exception):
    """An error that maps directly onto an HTTP response."""

    def __init__(self, status: int, code: str, message: str, **details: Any):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                **self.details,
            }
        }


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    # bool is a subclass of int; reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(
            422,
            "INVALID_FIELD",
            "field '%s' must be an integer" % field,
            field=field,
        )
    if not (minimum <= value <= maximum):
        raise ApiError(
            422,
            "INVALID_FIELD",
            "field '%s' must be between %d and %d" % (field, minimum, maximum),
            field=field,
        )
    return value


def _decode_shard(value: Any, index: int, shard_size: int) -> bytes:
    if not isinstance(value, str):
        raise ApiError(
            422,
            "INVALID_SHARD_ENCODING",
            "shard %d must be a Base64 string or null" % index,
            shardIndex=index,
        )
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(
            422,
            "INVALID_SHARD_ENCODING",
            "shard %d is not valid canonical Base64" % index,
            shardIndex=index,
        ) from None
    if len(raw) != shard_size:
        raise ApiError(
            422,
            "SHARD_SIZE_MISMATCH",
            "shard %d decodes to %d bytes, expected %d"
            % (index, len(raw), shard_size),
            shardIndex=index,
        )
    return raw


def _parse_request(body: Any) -> tuple[int, int, list[bytes | None], list[str]]:
    if not isinstance(body, dict):
        raise ApiError(422, "INVALID_BODY", "request body must be a JSON object")

    data_shards = _require_int(
        body.get("dataShards"), "dataShards", MIN_DATA_SHARDS, MAX_DATA_SHARDS
    )
    shard_size = _require_int(
        body.get("shardSize"), "shardSize", MIN_SHARD_SIZE, MAX_SHARD_SIZE
    )

    shard_count = data_shards + 2
    raw_shards = body.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != shard_count:
        raise ApiError(
            422,
            "INVALID_SHARD_COUNT",
            "field 'shards' must be an array of %d entries "
            "(%d data shards + P + Q)" % (shard_count, data_shards),
            field="shards",
        )
    raw_digests = body.get("digests")
    if not isinstance(raw_digests, list) or len(raw_digests) != shard_count:
        raise ApiError(
            422,
            "INVALID_DIGEST_COUNT",
            "field 'digests' must be an array of %d SHA-256 hex strings"
            % shard_count,
            field="digests",
        )

    shards: list[bytes | None] = []
    for i, item in enumerate(raw_shards):
        shards.append(None if item is None else _decode_shard(item, i, shard_size))

    digests: list[str] = []
    for i, item in enumerate(raw_digests):
        if not isinstance(item, str) or not _DIGEST_RE.match(item):
            raise ApiError(
                422,
                "INVALID_DIGEST_FORMAT",
                "digest %d must be a 64-character hex SHA-256 string" % i,
                shardIndex=i,
            )
        digests.append(item.lower())

    return data_shards, shard_size, shards, digests


def reconstruct_stripe_request(body: Any) -> dict[str, Any]:
    """Validate, verify, reconstruct and re-verify a stripe.

    Returns the success response payload; raises :class:`ApiError`
    otherwise. A failure response never carries shard material.
    """
    data_shards, shard_size, shards, digests = _parse_request(body)
    shard_count = data_shards + 2

    missing = [i for i, s in enumerate(shards) if s is None]
    if len(missing) > MAX_RECOVERABLE_MISSING:
        raise ApiError(
            422,
            "TOO_MANY_MISSING_SHARDS",
            "%d shards are missing; at most %d can be reconstructed"
            % (len(missing), MAX_RECOVERABLE_MISSING),
            missingIndices=missing,
        )

    # Surviving shards must match their expected digests. A mismatch is a
    # contradiction (possible silent corruption), never a licence to treat
    # the shard as missing.
    for i, shard in enumerate(shards):
        if shard is not None and _sha256_hex(shard) != digests[i]:
            raise ApiError(
                409,
                "SHARD_DIGEST_MISMATCH",
                "surviving shard %d does not match its expected SHA-256; "
                "refusing to treat it as missing" % i,
                shardIndex=i,
            )

    full, recovered = rs.reconstruct_stripe(shards, data_shards)

    # Reconstructed shards must match their expected digests, otherwise the
    # surviving material contradicts the archival metadata.
    for i in recovered:
        if _sha256_hex(full[i]) != digests[i]:
            raise ApiError(
                409,
                "RECONSTRUCTED_DIGEST_MISMATCH",
                "reconstructed shard %d does not match its expected SHA-256; "
                "surviving shards and digests are contradictory" % i,
                shardIndex=i,
            )

    # The complete stripe must satisfy both parity relations.
    defects = rs.parity_defects(
        full[:data_shards], full[data_shards], full[data_shards + 1]
    )
    if defects:
        raise ApiError(
            409,
            "PARITY_RELATION_MISMATCH",
            "parity relation(s) %s do not hold for the assembled stripe"
            % ", ".join(defects),
            parity=defects,
            pIndex=data_shards,
            qIndex=data_shards + 1,
        )

    return {
        "dataShards": data_shards,
        "shardSize": shard_size,
        "shardCount": shard_count,
        "shards": [base64.b64encode(s).decode("ascii") for s in full],
        "recoveredIndices": recovered,
        "digests": [_sha256_hex(s) for s in full],
    }
