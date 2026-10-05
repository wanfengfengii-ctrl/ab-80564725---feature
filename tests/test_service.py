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


def make_request(data_count=4, size=64, seed=42):
    rng = random.Random(seed)
    data = [rng.randbytes(size) for _ in range(data_count)]
    p, q = rs.compute_parity(data)
    stripe = data + [p, q]
    return {
        "dataShards": data_count,
        "shardSize": size,
        "shards": [b64(s) for s in stripe],
        "digests": [sha(s) for s in stripe],
    }, stripe


def scatter(data_count, logical, p_index, q_index):
    """Place [D0..Dn-1, P, Q] into physical slots of a rotated layout."""
    total = data_count + 2
    physical = [b""] * total
    data_slots = [i for i in range(total) if i != p_index and i != q_index]
    for logical_index, slot in enumerate(data_slots):
        physical[slot] = logical[logical_index]
    physical[p_index] = logical[data_count]
    physical[q_index] = logical[data_count + 1]
    return physical


def make_rotated_request(p_index, q_index, data_count=4, size=64, seed=42):
    rng = random.Random(seed)
    data = [rng.randbytes(size) for _ in range(data_count)]
    p, q = rs.compute_parity(data)
    logical = data + [p, q]
    physical = scatter(data_count, logical, p_index, q_index)
    body = {
        "dataShards": data_count,
        "shardSize": size,
        "parityIndices": [p_index, q_index],
        "shards": [b64(s) for s in physical],
        "digests": [sha(s) for s in physical],
    }
    return body, physical


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

    def test_default_response_echoes_last_two_slots_as_parity(self):
        body, stripe = make_request()
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["parityIndices"], [4, 5])


class TestRotatedLayoutSuccess(unittest.TestCase):
    def test_full_stripe_with_pq_at_front(self):
        # Physical layout: P, Q, D0, D1, D2, D3.
        body, physical = make_rotated_request(0, 1)
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["parityIndices"], [0, 1])
        self.assertEqual(result["shards"], [b64(s) for s in physical])
        self.assertEqual(result["recoveredIndices"], [])
        self.assertEqual(result["digests"], [sha(s) for s in physical])

    def test_rotated_layout_two_data_shards_missing(self):
        body, physical = make_rotated_request(2, 5, data_count=6, size=100)
        body["shards"][0] = None  # D0
        body["shards"][4] = None  # D3
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["recoveredIndices"], [0, 4])
        self.assertEqual(result["shards"], [b64(s) for s in physical])

    def test_rotated_layout_both_parity_shards_missing(self):
        body, physical = make_rotated_request(1, 3, data_count=3, size=17)
        body["shards"][1] = None  # P
        body["shards"][3] = None  # Q
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["recoveredIndices"], [1, 3])
        self.assertEqual(result["shards"], [b64(s) for s in physical])

    def test_rotated_layout_data_and_q_missing(self):
        body, physical = make_rotated_request(0, 4, data_count=5, size=33)
        body["shards"][3] = None  # some data slot
        body["shards"][4] = None  # Q
        result = reconstruct_stripe_request(body)
        self.assertEqual(result["recoveredIndices"], [3, 4])
        self.assertEqual(result["shards"], [b64(s) for s in physical])

    def test_explicit_last_two_indices_equals_omitted_layout(self):
        default_body, stripe = make_request(data_count=4, size=50, seed=7)
        explicit_body = dict(default_body, parityIndices=[4, 5])
        self.assertEqual(
            reconstruct_stripe_request(explicit_body)["shards"],
            reconstruct_stripe_request(default_body)["shards"],
        )
        self.assertEqual([b64(s) for s in stripe][:4],
                         reconstruct_stripe_request(default_body)["shards"][:4])


class TestRotatedLayoutUnprocessable(unittest.TestCase):
    def assert_422(self, body, code):
        with self.assertRaises(ApiError) as ctx:
            reconstruct_stripe_request(body)
        self.assertEqual(ctx.exception.status, 422)
        self.assertEqual(ctx.exception.code, code)
        self.assertNotIn("shards", ctx.exception.body())
        return ctx.exception

    def test_parity_indices_not_a_pair(self):
        body, _ = make_rotated_request(0, 1)
        body["parityIndices"] = [0]
        self.assert_422(body, "INVALID_PARITY_INDICES")
        body["parityIndices"] = [0, 1, 2]
        self.assert_422(body, "INVALID_PARITY_INDICES")
        body["parityIndices"] = "0,1"
        self.assert_422(body, "INVALID_PARITY_INDICES")

    def test_parity_indices_wrong_type(self):
        body, _ = make_rotated_request(0, 1)
        body["parityIndices"] = [0, "1"]
        self.assert_422(body, "INVALID_PARITY_INDICES")
        body["parityIndices"] = [True, 1]
        self.assert_422(body, "INVALID_PARITY_INDICES")

    def test_parity_index_out_of_range_is_located(self):
        body, _ = make_rotated_request(0, 1)
        body["parityIndices"] = [0, 6]
        err = self.assert_422(body, "INVALID_PARITY_INDICES")
        self.assertEqual(err.body()["error"]["shardIndex"], 6)
        body["parityIndices"] = [-1, 1]
        err = self.assert_422(body, "INVALID_PARITY_INDICES")
        self.assertEqual(err.body()["error"]["shardIndex"], -1)

    def test_duplicate_parity_index_is_located(self):
        body, _ = make_rotated_request(0, 1)
        body["parityIndices"] = [3, 3]
        err = self.assert_422(body, "DUPLICATE_PARITY_INDEX")
        self.assertEqual(err.body()["error"]["shardIndex"], 3)

    def test_too_many_missing_reports_physical_slots(self):
        body, _ = make_rotated_request(1, 4)
        body["shards"][0] = None
        body["shards"][2] = None
        body["shards"][4] = None
        err = self.assert_422(body, "TOO_MANY_MISSING_SHARDS")
        self.assertEqual(err.body()["error"]["missingIndices"], [0, 2, 4])


class TestRotatedLayoutConflict(unittest.TestCase):
    def assert_409(self, body, code):
        with self.assertRaises(ApiError) as ctx:
            reconstruct_stripe_request(body)
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(ctx.exception.code, code)
        self.assertNotIn("shards", ctx.exception.body())
        return ctx.exception

    def test_surviving_shard_mismatch_reports_physical_slot(self):
        body, physical = make_rotated_request(0, 1)
        corrupted = bytes([physical[4][0] ^ 0xFF]) + physical[4][1:]
        body["shards"][4] = b64(corrupted)
        err = self.assert_409(body, "SHARD_DIGEST_MISMATCH")
        self.assertEqual(err.body()["error"]["shardIndex"], 4)

    def test_reconstructed_data_shard_mismatch_reports_physical_slot(self):
        # Physical slot 2 is D0 under layout (0, 1); its wrong digest must
        # be reported at slot 2, not at logical index 0.
        body, _ = make_rotated_request(0, 1)
        body["shards"][2] = None
        body["digests"][2] = sha(b"bogus archival metadata")
        err = self.assert_409(body, "RECONSTRUCTED_DIGEST_MISMATCH")
        self.assertEqual(err.body()["error"]["shardIndex"], 2)

    def test_parity_contradiction_reports_physical_pq_slots(self):
        body, physical = make_rotated_request(2, 5, data_count=6)
        bad_p = bytes(b ^ 0x5A for b in physical[2])
        body["shards"][2] = b64(bad_p)
        body["digests"][2] = sha(bad_p)
        err = self.assert_409(body, "PARITY_RELATION_MISMATCH")
        error = err.body()["error"]
        self.assertEqual(error["parity"], ["P"])
        self.assertEqual(error["pIndex"], 2)
        self.assertEqual(error["qIndex"], 5)

    def test_q_contradiction_at_physical_slot_zero(self):
        body, physical = make_rotated_request(1, 0, data_count=4)
        bad_q = bytes(b ^ 0xA5 for b in physical[0])
        body["shards"][0] = b64(bad_q)
        body["digests"][0] = sha(bad_q)
        err = self.assert_409(body, "PARITY_RELATION_MISMATCH")
        error = err.body()["error"]
        self.assertEqual(error["parity"], ["Q"])
        self.assertEqual(error["pIndex"], 1)
        self.assertEqual(error["qIndex"], 0)


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
