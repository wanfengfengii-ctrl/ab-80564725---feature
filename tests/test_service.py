import base64
import hashlib
import random
import unittest

from app import reed_solomon as rs
from app.service import ApiError, reconstruct_stripe_request


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_request(data_count=4, size=64, seed=42, parity_indices=None):
    rng = random.Random(seed)
    data = [rng.randbytes(size) for _ in range(data_count)]
    p, q = rs.compute_parity(data)
    if parity_indices is None:
        p_index, q_index = data_count, data_count + 1
        stripe = data + [p, q]
        request_parity = None
    else:
        p_index, q_index = parity_indices
        total = data_count + 2
        data_slots = [i for i in range(total) if i != p_index and i != q_index]
        stripe = [b""] * total
        for logical_i, physical_i in enumerate(data_slots):
            stripe[physical_i] = data[logical_i]
        stripe[p_index] = p
        stripe[q_index] = q
        request_parity = [p_index, q_index]
    body = {
        "dataShards": data_count,
        "shardSize": size,
        "shards": [b64(s) for s in stripe],
        "digests": [sha(s) for s in stripe],
    }
    if request_parity is not None:
        body["parityIndices"] = request_parity
    return body, stripe


class TestSuccess(unittest.TestCase):
    def test_full_stripe_returns_canonical_response(self):
        body, stripe = make_request()
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["dataShards"], 4)
        self.assertEqual(result["shardSize"], 64)
        self.assertEqual(result["shardCount"], 6)
        self.assertEqual(result["shards"], [b64(s) for s in stripe])
        self.assertEqual(result["recoveredIndices"], [])
        self.assertEqual(result["digests"], [sha(s) for s in stripe])

    def test_two_missing_data_shards_recovered(self):
        body, stripe = make_request(data_count=6, size=100)
        body["shards"][1] = None
        body["shards"][4] = None
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["recoveredIndices"], [1, 4])
        self.assertEqual(result["shards"], [b64(s) for s in stripe])

    def test_missing_parity_shards_recovered(self):
        body, stripe = make_request(data_count=3, size=17)
        body["shards"][3] = None  # P
        body["shards"][4] = None  # Q
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["recoveredIndices"], [3, 4])
        self.assertEqual(result["shards"], [b64(s) for s in stripe])

    def test_data_and_q_missing_recovered(self):
        body, stripe = make_request(data_count=5, size=33)
        body["shards"][2] = None  # data
        body["shards"][6] = None  # Q
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["recoveredIndices"], [2, 6])
        self.assertEqual(result["shards"], [b64(s) for s in stripe])


class TestRotatedLayout(unittest.TestCase):
    ROTATED_LAYOUTS = [(0, 1), (1, 0), (0, 5), (5, 0), (2, 4), (4, 1)]

    def test_full_rotated_stripes_returned_in_physical_order(self):
        for layout in self.ROTATED_LAYOUTS:
            body, stripe = make_request(parity_indices=layout)
            result = reconstruct_stripe_request(body)
            self.assertEqual(result["parityIndices"], list(layout))
            self.assertEqual(
                result["shards"], [b64(s) for s in stripe], "layout=%s" % (layout,)
            )
            self.assertEqual(result["recoveredIndices"], [])
            self.assertEqual(
                result["digests"], [sha(s) for s in stripe], "layout=%s" % (layout,)
            )

    def test_default_response_also_echoes_canonical_parity_indices(self):
        body, stripe = make_request()
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["parityIndices"], [4, 5])

    def test_two_missing_shards_recovered_in_rotated_layout(self):
        # P at physical slot 1, Q at slot 4; erase a data slot and Q.
        body, stripe = make_request(
            data_count=4, size=70, seed=9, parity_indices=(1, 4)
        )
        body["shards"][0] = None  # D0
        body["shards"][4] = None  # Q
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["recoveredIndices"], [0, 4])
        self.assertEqual(result["shards"], [b64(s) for s in stripe])

    def test_missing_rotated_parity_shards_recomputed(self):
        body, stripe = make_request(
            data_count=3, size=21, seed=13, parity_indices=(0, 1)
        )
        body["shards"][0] = None  # P
        body["shards"][1] = None  # Q
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["recoveredIndices"], [0, 1])
        self.assertEqual(result["shards"], [b64(s) for s in stripe])

    def test_all_erasure_pairs_recovered_for_rotated_layouts(self):
        import itertools

        for layout in ((0, 3), (2, 5)):
            body, stripe = make_request(
                data_count=4, size=50, seed=21, parity_indices=layout
            )
            total = 6
            for missing in itertools.combinations(range(total), 2):
                damaged = dict(body)
                damaged["shards"] = list(body["shards"])
                for i in missing:
                    damaged["shards"][i] = None
                result = reconstruct_stripe_request(damaged)
                self.assertEqual(
                    result["shards"],
                    [b64(s) for s in stripe],
                    "layout=%s missing=%s" % (layout, missing),
                )
                self.assertEqual(result["recoveredIndices"], list(missing))


class TestParityIndexValidation(unittest.TestCase):
    def assert_422(self, body, code):
        with self.assertRaises(ApiError) as ctx:
            reconstruct_stripe_request(body)
        self.assertEqual(ctx.exception.status, 422)
        self.assertEqual(ctx.exception.code, code)
        self.assertNotIn("shards", ctx.exception.body())
        return ctx.exception

    def test_rejects_non_array_parity_indices(self):
        body, _ = make_request()
        body["parityIndices"] = {"p": 4}
        self.assert_422(body, "INVALID_PARITY_INDICES")
        body["parityIndices"] = [4]
        self.assert_422(body, "INVALID_PARITY_INDICES")
        body["parityIndices"] = [4, 5, 0]
        self.assert_422(body, "INVALID_PARITY_INDICES")

    def test_rejects_non_integer_entries(self):
        body, _ = make_request()
        body["parityIndices"] = [4, "5"]
        err = self.assert_422(body, "INVALID_PARITY_INDEX")
        self.assertEqual(err.body()["error"]["parity"], "q")
        body["parityIndices"] = [True, 5]
        err = self.assert_422(body, "INVALID_PARITY_INDEX")
        self.assertEqual(err.body()["error"]["parity"], "p")
        body["parityIndices"] = [None, 5]
        self.assert_422(body, "INVALID_PARITY_INDEX")

    def test_rejects_out_of_range_index_with_physical_value(self):
        body, _ = make_request(data_count=4)
        body["parityIndices"] = [4, 6]
        err = self.assert_422(body, "PARITY_INDEX_OUT_OF_RANGE")
        error = err.body()["error"]
        self.assertEqual(error["parity"], "q")
        self.assertEqual(error["parityIndex"], 6)
        body["parityIndices"] = [-1, 5]
        err = self.assert_422(body, "PARITY_INDEX_OUT_OF_RANGE")
        self.assertEqual(err.body()["error"]["parityIndex"], -1)

    def test_rejects_duplicate_parity_slots(self):
        body, _ = make_request()
        body["parityIndices"] = [3, 3]
        err = self.assert_422(body, "DUPLICATE_PARITY_INDEX")
        self.assertEqual(err.body()["error"]["parityIndex"], 3)

    def test_null_parity_indices_means_canonical_layout(self):
        body, stripe = make_request()
        body["parityIndices"] = None
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["parityIndices"], [4, 5])
        self.assertEqual(result["shards"], [b64(s) for s in stripe])


class TestRotatedConflict(unittest.TestCase):
    def assert_409(self, body, code):
        with self.assertRaises(ApiError) as ctx:
            reconstruct_stripe_request(body)
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(ctx.exception.code, code)
        self.assertNotIn("shards", ctx.exception.body())
        return ctx.exception

    def test_corrupt_survivor_reports_physical_slot(self):
        body, stripe = make_request(parity_indices=(0, 1))
        # Physical slot 2 is D0; corrupt it in place.
        corrupted = bytes([stripe[2][0] ^ 0x0F]) + stripe[2][1:]
        body["shards"][2] = b64(corrupted)
        err = self.assert_409(body, "SHARD_DIGEST_MISMATCH")
        self.assertEqual(err.body()["error"]["shardIndex"], 2)

    def test_reconstructed_digest_mismatch_reports_physical_slot(self):
        body, _ = make_request(parity_indices=(1, 4))
        body["shards"][3] = None  # a data slot
        body["digests"][3] = sha(b"not the real reconstructed shard")
        err = self.assert_409(body, "RECONSTRUCTED_DIGEST_MISMATCH")
        self.assertEqual(err.body()["error"]["shardIndex"], 3)

    def test_parity_contradiction_reports_rotated_slots(self):
        body, stripe = make_request(parity_indices=(2, 0))  # Q at slot 0
        bad_q = bytes(b ^ 0xC3 for b in stripe[0])
        body["shards"][0] = b64(bad_q)
        body["digests"][0] = sha(bad_q)
        err = self.assert_409(body, "PARITY_RELATION_MISMATCH")
        error = err.body()["error"]
        self.assertEqual(error["parity"], ["Q"])
        self.assertEqual(error["pIndex"], 2)
        self.assertEqual(error["qIndex"], 0)

    def test_rotated_layout_detects_q_relation_that_canonical_ordering_misses(self):
        # Build a rotated stripe where P is fine but Q was computed with
        # the *physical* (wrong) coefficient order instead of the logical
        # data order. Digests stay self-consistent, so only the Q relation
        # must trip -- and only when Q coefficients follow logical order.
        body, stripe = make_request(
            data_count=4, size=40, seed=31, parity_indices=(0, 5)
        )
        # Slots 1..4 are D0..D3. Forge Q using physical-order coefficients
        # 2^1..2^4 instead of logical 2^0..2^3.
        from app import gf256 as gf

        forged_q = bytes(40)
        for logical_i, physical_i in enumerate((1, 2, 3, 4)):
            coef = gf.pow2(physical_i)  # wrong: physical, not logical
            forged_q = bytes(a ^ b for a, b in zip(forged_q, gf.mul_bytes(coef, stripe[physical_i])))
        body["shards"][5] = b64(forged_q)
        body["digests"][5] = sha(forged_q)
        err = self.assert_409(body, "PARITY_RELATION_MISMATCH")
        self.assertEqual(err.body()["error"]["parity"], ["Q"])
        self.assertEqual(err.body()["error"]["qIndex"], 5)


class TestUnprocessable(unittest.TestCase):
    def assert_422(self, body, code):
        with self.assertRaises(ApiError) as ctx:
            reconstruct_stripe_request(body)
        self.assertEqual(ctx.exception.status, 422)
        self.assertEqual(ctx.exception.code, code)
        self.assertNotIn("shards", ctx.exception.body())

    def test_rejects_non_object_body(self):
        self.assert_422([1, 2, 3], "INVALID_BODY")

    def test_rejects_out_of_range_data_shards(self):
        body, _ = make_request()
        body["dataShards"] = 1
        self.assert_422(body, "INVALID_FIELD")
        body["dataShards"] = 17
        self.assert_422(body, "INVALID_FIELD")
        body["dataShards"] = True
        self.assert_422(body, "INVALID_FIELD")

    def test_rejects_out_of_range_shard_size(self):
        body, _ = make_request()
        body["shardSize"] = 0
        self.assert_422(body, "INVALID_FIELD")
        body["shardSize"] = 4097
        self.assert_422(body, "INVALID_FIELD")

    def test_rejects_wrong_array_lengths(self):
        body, _ = make_request()
        body["shards"] = body["shards"][:-1]
        self.assert_422(body, "INVALID_SHARD_COUNT")
        body, _ = make_request()
        body["digests"] = body["digests"] + [body["digests"][0]]
        self.assert_422(body, "INVALID_DIGEST_COUNT")

    def test_rejects_bad_base64_and_wrong_size(self):
        body, _ = make_request()
        body["shards"][0] = "!!!not-base64!!!"
        self.assert_422(body, "INVALID_SHARD_ENCODING")
        body, _ = make_request()
        body["shards"][0] = b64(b"too-short")
        self.assert_422(body, "SHARD_SIZE_MISMATCH")

    def test_rejects_malformed_digest(self):
        body, _ = make_request()
        body["digests"][2] = "zz" * 32
        self.assert_422(body, "INVALID_DIGEST_FORMAT")
        body["digests"][2] = "ab" * 16
        self.assert_422(body, "INVALID_DIGEST_FORMAT")

    def test_three_missing_shards_is_422(self):
        body, _ = make_request()
        body["shards"][0] = None
        body["shards"][2] = None
        body["shards"][4] = None
        with self.assertRaises(ApiError) as ctx:
            reconstruct_stripe_request(body)
        self.assertEqual(ctx.exception.status, 422)
        self.assertEqual(ctx.exception.code, "TOO_MANY_MISSING_SHARDS")
        self.assertEqual(ctx.exception.body()["error"]["missingIndices"], [0, 2, 4])


class TestConflict(unittest.TestCase):
    def assert_409(self, body, code):
        with self.assertRaises(ApiError) as ctx:
            reconstruct_stripe_request(body)
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(ctx.exception.code, code)
        # A failure must never emit anything resembling a complete stripe.
        self.assertNotIn("shards", ctx.exception.body())
        return ctx.exception

    def test_surviving_shard_with_wrong_digest_is_not_treated_as_missing(self):
        body, stripe = make_request()
        # Silently corrupt a surviving shard in place.
        corrupted = bytes([stripe[1][0] ^ 0xFF]) + stripe[1][1:]
        body["shards"][1] = b64(corrupted)
        err = self.assert_409(body, "SHARD_DIGEST_MISMATCH")
        self.assertEqual(err.body()["error"]["shardIndex"], 1)

    def test_corrupted_survivor_is_refused_even_when_recovery_is_possible(self):
        body, stripe = make_request()
        # One shard missing (recoverable) plus one corrupted survivor:
        # the corruption must win and produce 409, not a reconstruction.
        body["shards"][0] = None
        corrupted = bytes([stripe[2][0] ^ 0x01]) + stripe[2][1:]
        body["shards"][2] = b64(corrupted)
        err = self.assert_409(body, "SHARD_DIGEST_MISMATCH")
        self.assertEqual(err.body()["error"]["shardIndex"], 2)

    def test_reconstructed_shard_digest_mismatch_is_409(self):
        body, stripe = make_request()
        body["shards"][3] = None
        body["digests"][3] = sha(b"something-else entirely")
        err = self.assert_409(body, "RECONSTRUCTED_DIGEST_MISMATCH")
        self.assertEqual(err.body()["error"]["shardIndex"], 3)

    def test_parity_contradiction_with_consistent_digests_is_409(self):
        body, stripe = make_request()
        # Replace P with garbage but update its expected digest to match,
        # so every digest check passes and only the parity relation fails.
        bad_p = bytes(b ^ 0x5A for b in stripe[4])
        body["shards"][4] = b64(bad_p)
        body["digests"][4] = sha(bad_p)
        err = self.assert_409(body, "PARITY_RELATION_MISMATCH")
        self.assertIn("P", err.body()["error"]["parity"])

    def test_q_contradiction_is_located(self):
        body, stripe = make_request()
        bad_q = bytes(b ^ 0xA5 for b in stripe[5])
        body["shards"][5] = b64(bad_q)
        body["digests"][5] = sha(bad_q)
        err = self.assert_409(body, "PARITY_RELATION_MISMATCH")
        error = err.body()["error"]
        self.assertEqual(error["parity"], ["Q"])
        self.assertEqual(error["qIndex"], 5)


if __name__ == "__main__":
    unittest.main()
