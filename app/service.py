"""Request validation and orchestration for the reconstruct endpoint.

Error taxonomy
--------------
* 422 -- the request itself is invalid or asks for more than the code
  can recover (e.g. more than two missing shards).
* 409 -- the request is well formed but the supplied material is
  contradictory: a surviving shard does not match its expected digest,
  a reconstructed shard does not match its expected digest, or the
  P/Q parity relations are inconsistent.

Shards are always addressed by their *physical* slot indices. The
optional ``parityIndices`` field names the physical slots holding P and
Q (the remaining slots are D0..Dn-1 in ascending physical order); all
error locations therefore point at real disk slots.

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


def _parse_parity_indices(
    body: dict[str, Any], shard_count: int, data_shards: int
) -> tuple[int, int]:
    """Extract and validate the optional ``parityIndices`` [p, q].

    Absent or null means the canonical layout (P, Q in the last two
    slots). Malformed, out-of-range or duplicated indices are 422 with
    the offending physical index reported so archivists can locate it.
    """
    if "parityIndices" not in body or body["parityIndices"] is None:
        return data_shards, data_shards + 1

    raw = body["parityIndices"]
    if not isinstance(raw, list) or len(raw) != 2:
        raise ApiError(
            422,
            "INVALID_PARITY_INDICES",
            "field 'parityIndices' must be a [pIndex, qIndex] pair of "
            "distinct integers, or be omitted",
            field="parityIndices",
        )

    indices: list[int] = []
    for position, value in enumerate(raw):
        label = "p" if position == 0 else "q"
        if isinstance(value, bool) or not isinstance(value, int):
            raise ApiError(
                422,
                "INVALID_PARITY_INDEX",
                "parityIndices.%s must be an integer physical slot index" % label,
                field="parityIndices",
                parity=label,
            )
        if not (0 <= value < shard_count):
            raise ApiError(
                422,
                "PARITY_INDEX_OUT_OF_RANGE",
                "parityIndices.%s = %d is outside the valid physical slot "
                "range [0, %d)" % (label, value, shard_count),
                field="parityIndices",
                parity=label,
                parityIndex=value,
            )
        indices.append(value)

    if indices[0] == indices[1]:
        raise ApiError(
            422,
            "DUPLICATE_PARITY_INDEX",
            "P and Q must occupy distinct physical slots; both name slot %d"
            % indices[0],
            field="parityIndices",
            parityIndex=indices[0],
        )
    return indices[0], indices[1]


def _parse_request(
    body: Any,
) -> tuple[int, int, list[bytes | None], list[str], int, int]:
    if not isinstance(body, dict):
        raise ApiError(422, "INVALID_BODY", "request body must be a JSON object")

    data_shards = _require_int(
        body.get("dataShards"), "dataShards", MIN_DATA_SHARDS, MAX_DATA_SHARDS
    )
    shard_size = _require_int(
        body.get("shardSize"), "shardSize", MIN_SHARD_SIZE, MAX_SHARD_SIZE
    )

    shard_count = data_shards + 2
    p_index, q_index = _parse_parity_indices(body, shard_count, data_shards)

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

    return data_shards, shard_size, shards, digests, p_index, q_index


def reconstruct_stripe_request(body: Any) -> dict[str, Any]:
    """Validate, verify, reconstruct and re-verify a stripe.

    Returns the success response payload; raises :class:`ApiError`
    otherwise. A failure response never carries shard material.
    """
    data_shards, shard_size, shards, digests, p_index, q_index = _parse_request(body)
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
    # the shard as missing. All indices are physical slot positions.
    for i, shard in enumerate(shards):
        if shard is not None and _sha256_hex(shard) != digests[i]:
            raise ApiError(
                409,
                "SHARD_DIGEST_MISMATCH",
                "surviving shard at physical slot %d does not match its "
                "expected SHA-256; refusing to treat it as missing" % i,
                shardIndex=i,
            )

    full, recovered = rs.reconstruct_stripe(
        shards, data_shards, p_index=p_index, q_index=q_index
    )

    # Reconstructed shards must match their expected digests, otherwise the
    # surviving material contradicts the archival metadata.
    for i in recovered:
        if _sha256_hex(full[i]) != digests[i]:
            raise ApiError(
                409,
                "RECONSTRUCTED_DIGEST_MISMATCH",
                "reconstructed shard at physical slot %d does not match its "
                "expected SHA-256; surviving shards and digests are "
                "contradictory" % i,
                shardIndex=i,
            )

    # The complete stripe must satisfy both parity relations. Data shards
    # are the physical slots other than P/Q in ascending order; their list
    # position is the logical index whose 2**i coefficient feeds Q.
    data_slots = [i for i in range(shard_count) if i != p_index and i != q_index]
    data_shard_values = [full[i] for i in data_slots]
    defects = rs.parity_defects(data_shard_values, full[p_index], full[q_index])
    if defects:
        raise ApiError(
            409,
            "PARITY_RELATION_MISMATCH",
            "parity relation(s) %s do not hold for the assembled stripe "
            "(P at physical slot %d, Q at physical slot %d)"
            % (", ".join(defects), p_index, q_index),
            parity=defects,
            pIndex=p_index,
            qIndex=q_index,
        )

    return {
        "dataShards": data_shards,
        "shardSize": shard_size,
        "shardCount": shard_count,
        "parityIndices": [p_index, q_index],
        "shards": [base64.b64encode(s).decode("ascii") for s in full],
        "recoveredIndices": recovered,
        "digests": [_sha256_hex(s) for s in full],
    }
