"""P/Q parity computation and stripe reconstruction over GF(2^8).

A stripe consists of ``data_count`` data shards followed by two parity
shards:

* P (index ``data_count``)     -- bytewise XOR of all data shards.
* Q (index ``data_count + 1``) -- sum over GF(2^8) of ``2**i * D_i``
  where ``i`` is the zero-based data shard index and 2 is the field
  generator (primitive polynomial 0x11d).

Any one or two missing shards can be reconstructed from the survivors.
"""

from __future__ import annotations

from . import gf256 as gf


class TooManyMissingShards(Exception):
    """Raised when more than two shards of a stripe are absent."""

    def __init__(self, missing_indices: list[int]):
        self.missing_indices = list(missing_indices)
        super().__init__(
            "cannot reconstruct stripe: %d shards missing (at most 2 recoverable)"
            % len(self.missing_indices)
        )


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """Bytewise XOR of two equal-length byte strings."""
    if len(a) != len(b):
        raise ValueError("xor_bytes requires equal lengths")
    return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).to_bytes(len(a), "big")


def compute_parity(data_shards: list[bytes]) -> tuple[bytes, bytes]:
    """Return the (P, Q) parity shards for the given data shards."""
    if not data_shards:
        raise ValueError("at least one data shard is required")
    size = len(data_shards[0])
    if any(len(d) != size for d in data_shards):
        raise ValueError("all data shards must have equal length")
    p = bytes(size)
    q = bytes(size)
    for i, shard in enumerate(data_shards):
        p = xor_bytes(p, shard)
        q = xor_bytes(q, gf.mul_bytes(gf.pow2(i), shard))
    return p, q


def parity_defects(data_shards: list[bytes], p: bytes, q: bytes) -> list[str]:
    """Return ["P"]/["Q"]/["P", "Q"] for parity relations that do not hold."""
    expected_p, expected_q = compute_parity(data_shards)
    defects = []
    if p != expected_p:
        defects.append("P")
    if q != expected_q:
        defects.append("Q")
    return defects


def reconstruct_stripe(
    shards: list[bytes | None], data_count: int
) -> tuple[list[bytes], list[int]]:
    """Reconstruct a full stripe from ``shards`` (None marks a missing shard).

    ``shards`` must have ``data_count + 2`` entries; the last two are the
    P and Q parity shards. Returns ``(full_shards, recovered_indices)``.
    Raises :class:`TooManyMissingShards` when more than two are missing.
    """
    if len(shards) != data_count + 2:
        raise ValueError("expected %d shards, got %d" % (data_count + 2, len(shards)))
    missing = [i for i, s in enumerate(shards) if s is None]
    if len(missing) > 2:
        raise TooManyMissingShards(missing)

    out: list[bytes | None] = list(shards)
    p_index = data_count
    q_index = data_count + 1
    missing_data = [i for i in missing if i < data_count]

    if missing_data:
        present_data = [i for i in range(data_count) if out[i] is not None]

        # Remainders equal the contribution of the missing data shards to
        # each parity equation: P - sum(present) and Q - sum(2^i * present).
        p_rem: bytes | None = None
        if out[p_index] is not None:
            p_rem = out[p_index]
            for i in present_data:
                p_rem = xor_bytes(p_rem, out[i])  # type: ignore[arg-type]

        q_rem: bytes | None = None
        if out[q_index] is not None:
            q_rem = out[q_index]
            for i in present_data:
                q_rem = xor_bytes(q_rem, gf.mul_bytes(gf.pow2(i), out[i]))  # type: ignore[arg-type]

        if len(missing_data) == 1:
            k = missing_data[0]
            if p_rem is not None:
                # P equation alone determines the single missing shard.
                out[k] = p_rem
            elif q_rem is not None:
                # Q equation: 2^k * D_k = q_rem  =>  D_k = q_rem / 2^k.
                out[k] = gf.mul_bytes(gf.inv(gf.pow2(k)), q_rem)
            else:  # pragma: no cover - unreachable with <= 2 missing shards
                raise TooManyMissingShards(missing)
        else:
            # Two missing data shards; both parity shards must be present.
            if p_rem is None or q_rem is None:  # pragma: no cover - defensive
                raise TooManyMissingShards(missing)
            a, b = missing_data
            coef_a = gf.pow2(a)
            coef_b = gf.pow2(b)
            # D_a + D_b = p_rem ; coef_a*D_a + coef_b*D_b = q_rem
            # => D_a = (q_rem + coef_b * p_rem) / (coef_a + coef_b)
            inv_denom = gf.inv(coef_a ^ coef_b)
            out[a] = gf.mul_bytes(inv_denom, xor_bytes(q_rem, gf.mul_bytes(coef_b, p_rem)))
            out[b] = xor_bytes(p_rem, out[a])  # type: ignore[arg-type]

    # Recompute parity shards themselves if they are the missing ones.
    if out[p_index] is None or out[q_index] is None:
        data = [d for d in out[:data_count]]
        if any(d is None for d in data):  # pragma: no cover - defensive
            raise TooManyMissingShards(missing)
        new_p, new_q = compute_parity(data)  # type: ignore[arg-type]
        if out[p_index] is None:
            out[p_index] = new_p
        if out[q_index] is None:
            out[q_index] = new_q

    full: list[bytes] = []
    for s in out:
        if s is None:  # pragma: no cover - defensive, unreachable
            raise TooManyMissingShards(missing)
        full.append(s)
    return full, missing
