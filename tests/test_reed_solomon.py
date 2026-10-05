import itertools
import random
import unittest

from app import reed_solomon as rs


def make_stripe(data_count, size, rng):
    data = [rng.randbytes(size) for _ in range(data_count)]
    p, q = rs.compute_parity(data)
    return data + [p, q]


class TestParity(unittest.TestCase):
    def test_p_is_bytewise_xor(self):
        rng = random.Random(7)
        data = [rng.randbytes(64) for _ in range(5)]
        p, _ = rs.compute_parity(data)
        expect = bytearray(64)
        for shard in data:
            for i, b in enumerate(shard):
                expect[i] ^= b
        self.assertEqual(p, bytes(expect))

    def test_q_uses_generator_powers(self):
        # Single data shard: Q must equal 2^0 * D0 == D0.
        shard = bytes(range(256))
        _, q = rs.compute_parity([shard])
        self.assertEqual(q, shard)
        # Two shards: Q = D0 + 2*D1.
        d0 = bytes([0x11] * 32)
        d1 = bytes([0x80] * 32)
        _, q = rs.compute_parity([d0, d1])
        self.assertEqual(q, bytes([0x11 ^ 0x1D] * 32))  # 2 * 0x80 = 0x1d

    def test_parity_defects_detect_tampering(self):
        rng = random.Random(11)
        stripe = make_stripe(4, 128, rng)
        data, p, q = stripe[:4], stripe[4], stripe[5]
        self.assertEqual(rs.parity_defects(data, p, q), [])
        bad_p = bytes([p[0] ^ 1]) + p[1:]
        self.assertEqual(rs.parity_defects(data, bad_p, q), ["P"])
        bad_q = q[:-1] + bytes([q[-1] ^ 1])
        self.assertEqual(rs.parity_defects(data, bad_p, bad_q), ["P", "Q"])


class TestReconstruction(unittest.TestCase):
    def test_every_single_and_double_erasure_is_recovered(self):
        rng = random.Random(2024)
        for data_count in range(2, 17):
            stripe = make_stripe(data_count, 96, rng)
            total = data_count + 2
            combos = [(i,) for i in range(total)]
            combos += list(itertools.combinations(range(total), 2))
            for missing in combos:
                damaged = [None if i in missing else s for i, s in enumerate(stripe)]
                full, recovered = rs.reconstruct_stripe(damaged, data_count)
                self.assertEqual(sorted(recovered), sorted(missing))
                self.assertEqual(
                    full,
                    stripe,
                    "data_count=%d missing=%s" % (data_count, missing),
                )

    def test_no_missing_shards_returns_stripe_unchanged(self):
        rng = random.Random(3)
        stripe = make_stripe(6, 33, rng)
        full, recovered = rs.reconstruct_stripe(list(stripe), 6)
        self.assertEqual(full, stripe)
        self.assertEqual(recovered, [])

    def test_three_missing_shards_are_refused(self):
        rng = random.Random(5)
        stripe = make_stripe(4, 16, rng)
        damaged = [None, None, None] + stripe[3:]
        with self.assertRaises(rs.TooManyMissingShards) as ctx:
            rs.reconstruct_stripe(damaged, 4)
        self.assertEqual(ctx.exception.missing_indices, [0, 1, 2])

    def test_max_size_stripe_roundtrip(self):
        rng = random.Random(99)
        stripe = make_stripe(16, 4096, rng)
        damaged = [None if i in (0, 17) else s for i, s in enumerate(stripe)]
        full, recovered = rs.reconstruct_stripe(damaged, 16)
        self.assertEqual(full, stripe)
        self.assertEqual(recovered, [0, 17])


def make_rotated_stripe(data, p, q, p_index, q_index):
    """Place [D0..Dn-1, P, Q] into physical slots with P/Q rotated out."""
    total = len(data) + 2
    data_slots = [i for i in range(total) if i != p_index and i != q_index]
    physical = [None] * total
    for logical_i, physical_i in enumerate(data_slots):
        physical[physical_i] = data[logical_i]
    physical[p_index] = p
    physical[q_index] = q
    return physical


class TestRotatedLayout(unittest.TestCase):
    """New controllers rotate P/Q across physical slots to balance writes."""

    LAYOUTS = [(0, 1), (1, 0), (0, 5), (5, 0), (2, 4), (4, 2), (3, 1), (5, 3)]

    def test_logical_layout_maps_remaining_slots_in_ascending_order(self):
        data_slots, p_index, q_index = rs.logical_layout(4, 1, 4)
        self.assertEqual(p_index, 1)
        self.assertEqual(q_index, 4)
        self.assertEqual(data_slots, [0, 2, 3, 5])

    def test_logical_layout_defaults_to_last_two_slots(self):
        data_slots, p_index, q_index = rs.logical_layout(4)
        self.assertEqual((p_index, q_index), (4, 5))
        self.assertEqual(data_slots, [0, 1, 2, 3])

    def test_logical_layout_rejects_invalid_indices(self):
        with self.assertRaises(ValueError):
            rs.logical_layout(4, 0, 6)
        with self.assertRaises(ValueError):
            rs.logical_layout(4, 2, 2)

    def test_every_erasure_combo_recovered_for_every_rotated_layout(self):
        rng = random.Random(2026)
        for p_index, q_index in self.LAYOUTS:
            data_count = 4
            data = [rng.randbytes(96) for _ in range(data_count)]
            p, q = rs.compute_parity(data)
            total = data_count + 2
            stripe = make_rotated_stripe(data, p, q, p_index, q_index)

            combos = [(i,) for i in range(total)]
            combos += list(itertools.combinations(range(total), 2))
            for missing in combos:
                damaged = [None if i in missing else s for i, s in enumerate(stripe)]
                full, recovered = rs.reconstruct_stripe(
                    damaged, data_count, p_index=p_index, q_index=q_index
                )
                self.assertEqual(
                    full,
                    stripe,
                    "layout=(P:%d,Q:%d) missing=%s" % (p_index, q_index, missing),
                )
                self.assertEqual(sorted(recovered), sorted(missing))
                # recovered indices are reported in physical slot order
                self.assertEqual(recovered, sorted(missing))

    def test_rotated_recovery_uses_logical_q_coefficients(self):
        # Hand-checked case: P at slot 0, Q at slot 1; physical slots
        # 2..5 are D0..D3. Erasing slot 0 (P) and slot 3 (= D1) must
        # rebuild D1 using coefficient 2**1, i.e. the Q equation.
        rng = random.Random(77)
        data = [rng.randbytes(48) for _ in range(4)]
        p, q = rs.compute_parity(data)
        p_index, q_index = 0, 1
        stripe = make_rotated_stripe(data, p, q, p_index, q_index)
        damaged = [None if i in (0, 3) else s for i, s in enumerate(stripe)]
        full, recovered = rs.reconstruct_stripe(
            damaged, 4, p_index=p_index, q_index=q_index
        )
        self.assertEqual(recovered, [0, 3])
        self.assertEqual(full, stripe)

    def test_full_rotated_stripe_round_trips_unchanged(self):
        rng = random.Random(5)
        data = [rng.randbytes(33) for _ in range(5)]
        p, q = rs.compute_parity(data)
        stripe = make_rotated_stripe(data, p, q, 2, 6)
        full, recovered = rs.reconstruct_stripe(stripe, 5, p_index=2, q_index=6)
        self.assertEqual(full, stripe)
        self.assertEqual(recovered, [])


if __name__ == "__main__":
    unittest.main()
