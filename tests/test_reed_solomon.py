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


def rotate_layout(logical_stripe, data_count, p_index, q_index):
    """Scatter [D0..Dn-1, P, Q] into physical slots for the given layout."""
    total = data_count + 2
    positions = rs.data_positions(data_count, p_index, q_index)
    physical = [b""] * total
    for logical_index, shard in enumerate(logical_stripe[:data_count]):
        physical[positions[logical_index]] = shard
    physical[p_index] = logical_stripe[data_count]
    physical[q_index] = logical_stripe[data_count + 1]
    return physical


class TestRotatedPhysicalLayout(unittest.TestCase):
    def test_data_positions_skip_parity_slots(self):
        # P/Q at slots 1 and 3 of a 6-slot stripe -> data slots 0,2,4,5.
        self.assertEqual(rs.data_positions(4, 1, 3), [0, 2, 4, 5])
        # Default layout: parity at the end, data fills the front.
        self.assertEqual(rs.data_positions(4, 4, 5), [0, 1, 2, 3])

    def test_every_layout_and_erasure_pair_roundtrips(self):
        rng = random.Random(5150)
        for data_count in (2, 5, 16):
            logical = make_stripe(data_count, 80, rng)
            total = data_count + 2
            for p_index in range(total):
                for q_index in range(total):
                    if p_index == q_index:
                        continue
                    physical = rotate_layout(
                        logical, data_count, p_index, q_index
                    )
                    raw_cases = [
                        (0,),
                        (p_index,),
                        (q_index,),
                        tuple(sorted((0, p_index))),
                        tuple(sorted((p_index, q_index))),
                    ]
                    cases = {
                        missing
                        for missing in raw_cases
                        if len(set(missing)) == len(missing)
                    }
                    for missing in cases:
                        damaged = [
                            None if i in missing else s
                            for i, s in enumerate(physical)
                        ]
                        full, recovered = rs.reconstruct_physical_stripe(
                            damaged, data_count, p_index, q_index
                        )
                        self.assertEqual(
                            full,
                            physical,
                            "n=%d p=%d q=%d missing=%s"
                            % (data_count, p_index, q_index, missing),
                        )
                        self.assertEqual(sorted(recovered), list(missing))

    def test_q_coefficient_follows_logical_data_numbering(self):
        # Layout P,Q,D0,D1: the data shards occupy late physical slots but
        # Q must still be D0 + 2*D1, not weighted by physical position.
        # Erase both data slots (2, 3); the Q equation only solves correctly
        # when coefficients follow the logical numbering.
        rng = random.Random(808)
        logical = make_stripe(2, 48, rng)
        physical = rotate_layout(logical, 2, p_index=0, q_index=1)
        damaged = [None if i in (2, 3) else s for i, s in enumerate(physical)]
        full, recovered = rs.reconstruct_physical_stripe(damaged, 2, 0, 1)
        self.assertEqual(full, physical)
        self.assertEqual(recovered, [2, 3])

    def test_invalid_physical_parity_indices_rejected(self):
        with self.assertRaises(ValueError):
            rs.data_positions(4, 2, 2)
        with self.assertRaises(ValueError):
            rs.data_positions(4, -1, 3)
        with self.assertRaises(ValueError):
            rs.data_positions(4, 0, 6)


if __name__ == "__main__":
    unittest.main()
