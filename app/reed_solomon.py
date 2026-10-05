"""P/Q parity computation and stripe reconstruction over GF(2^8).

A stripe logically consists of ``data_count`` data shards plus two
parity shards:

* P -- bytewise XOR of all data shards.
* Q -- sum over GF(2^8) of ``2**i * D_i`` where ``i`` is the zero-based
  *logical* data shard index and 2 is the field generator (primitive
  polynomial 0x11d).

Newer array controllers may place P and Q in any two distinct physical
slots to rotate write load. :func:`reconstruct_stripe` therefore takes
explicit ``p_index``/``q_index`` physical positions; the remaining
physical slots, in ascending order, are D0..Dn-1. When the indices are
omitted the canonical layout (P and Q in the final two slots) is used.

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


def logical_layout(
    data_count: int,
    p_index: int | None = None,
    q_index: int | None = None,
) -> tuple[list[int], int, int]:
    """Return ``(data_slots, p_index, q_index)`` physical slot indices.

    The data slots are the physical positions other than the P/Q slots,
    in ascending physical order; their list position is the logical data
    shard index (D0, D1, ...). When ``p_index``/``q_index`` are omitted
    the canonical layout applies: P then Q in the last two slots.
    """
    total = data_count + 2
    if p_index is None:
        p_index = data_count
    if q_index is None:
        q_index = data_count + 1
    if not (0 <= p_index < total and 0 <= q_index < total):
        raise ValueError("parity indices must be within [0, %d)" % total)
    if p_index == q_index:
        raise ValueError("P and Q must occupy distinct physical slots")
    data_slots = [i for i in range(total) if i != p_index and i != q_index]
    return data_slots, p_index, q_index


def reconstruct_stripe(
    shards: list[bytes | None],
    data_count: int,
    p_index: int | None = None,
    q_index: int | None = None,
) -> tuple[list[bytes], list[int]]:
    """Reconstruct a full stripe from ``shards`` (None marks a missing shard).

    ``shards`` must have ``data_count + 2`` entries kept in *physical*
    slot order. ``p_index``/``q_index`` name the physical slots holding
    P and Q; every other physical slot, in ascending order, holds
    D0..D(n-1). The Q coefficient of a data shard is ``2**i`` for its
    logical index ``i`` regardless of where the controller placed it.
    When the parity indices are omitted the canonical layout (P, Q in
    the last two slots) is assumed.

    Returns ``(full_shards, recovered_indices)`` in physical slot order.
    Raises :class:`TooManyMissingShards` when more than two are missing.
    """
    if len(shards) != data_count + 2:
        raise ValueError("expected %d shards, got %d" % (data_count + 2, len(shards)))
    data_slots, p_index, q_index = logical_layout(data_count, p_index, q_index)

    # Remap the physical stripe into the canonical logical order
    # [D0, ..., Dn-1, P, Q] so the erasure math below stays layout-free.
    logical = [shards[slot] for slot in data_slots]
    logical.append(shards[p_index])
    logical.append(shards[q_index])

    full_logical, recovered_logical = _reconstruct_canonical(logical, data_count)

    full: list[bytes | None] = [None] * (data_count + 2)
    for logical_i, physical_i in enumerate(data_slots):
        full[physical_i] = full_logical[logical_i]
    full[p_index] = full_logical[data_count]
    full[q_index] = full_logical[data_count + 1]

    physical_by_logical = {logical: physical
                           for logical, physical in enumerate(data_slots)}
    physical_by_logical[data_count] = p_index
    physical_by_logical[data_count + 1] = q_index
    recovered = sorted(physical_by_logical[i] for i in recovered_logical)

    return [s for s in full], recovered  # type: ignore[list-item]


def _reconstruct_canonical(
    shards: list[bytes | None], data_count: int
) -> tuple[list[bytes], list[int]]:
    """Reconstruct a stripe laid out as [D0, ..., Dn-1, P, Q]."""
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
