import unittest

from app import gf256 as gf


class TestTables(unittest.TestCase):
    def test_exp_log_are_inverse(self):
        for x in range(1, 256):
            self.assertEqual(gf.EXP[gf.LOG[x]], x)

    def test_generator_has_order_255(self):
        self.assertEqual(gf.EXP[0], 1)
        self.assertEqual(gf.EXP[255], 1)
        self.assertEqual(len(set(gf.EXP[:255])), 255)

    def test_reduction_uses_polynomial_0x11d(self):
        # 2^7 = 0x80, so 2^8 reduces modulo 0x11d to 0x1d.
        self.assertEqual(gf.EXP[7], 0x80)
        self.assertEqual(gf.EXP[8], 0x1D)
        self.assertEqual(gf.mul(0x80, 0x02), 0x1D)


class TestArithmetic(unittest.TestCase):
    def test_add_is_xor(self):
        self.assertEqual(gf.add(0x53, 0xCA), 0x99)
        self.assertEqual(gf.sub(0x53, 0xCA), 0x99)

    def test_mul_zero_and_identity(self):
        for x in range(256):
            self.assertEqual(gf.mul(x, 0), 0)
            self.assertEqual(gf.mul(0, x), 0)
            self.assertEqual(gf.mul(x, 1), x)
            self.assertEqual(gf.mul(1, x), x)

    def test_mul_known_values(self):
        self.assertEqual(gf.mul(2, 2), 4)
        self.assertEqual(gf.mul(0xFF, 0xFF), gf.EXP[(gf.LOG[0xFF] * 2) % 255])

    def test_mul_div_inverse_roundtrip(self):
        for a in (1, 2, 3, 0x80, 0xFE, 0xFF):
            for b in (1, 2, 7, 0x35, 0xFF):
                self.assertEqual(gf.div(gf.mul(a, b), b), a)
                self.assertEqual(gf.mul(a, gf.inv(a)), 1)

    def test_div_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            gf.div(1, 0)
        with self.assertRaises(ZeroDivisionError):
            gf.inv(0)

    def test_pow2_cycles_with_period_255(self):
        self.assertEqual(gf.pow2(0), 1)
        self.assertEqual(gf.pow2(1), 2)
        self.assertEqual(gf.pow2(255), 1)
        self.assertEqual(gf.pow2(256), 2)

    def test_mul_bytes(self):
        data = bytes(range(256))
        self.assertEqual(gf.mul_bytes(1, data), data)
        self.assertEqual(gf.mul_bytes(0, data), bytes(256))
        doubled = gf.mul_bytes(2, data)
        self.assertEqual(doubled[0x80], 0x1D)
        self.assertEqual(len(doubled), 256)


if __name__ == "__main__":
    unittest.main()
