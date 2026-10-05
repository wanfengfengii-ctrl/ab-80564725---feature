"""GF(2^8) arithmetic modulo the primitive polynomial 0x11d.

The field uses 2 as its generator (0x11d is primitive for x, so the
exponential table below cycles through all 255 non-zero elements).
Addition and subtraction in GF(2^8) are both bitwise XOR.
"""

POLYNOMIAL = 0x11D
GENERATOR = 0x02
FIELD_ORDER = 256
FIELD_CHAR_MINUS_ONE = 255  # multiplicative group order

# EXP[i] = GENERATOR ** i for i in [0, 255); the table is duplicated to
# length 512 so that EXP[LOG[a] + LOG[b]] never needs a modulo.
EXP = [0] * 512
LOG = [0] * FIELD_ORDER

_x = 1
for _i in range(FIELD_CHAR_MINUS_ONE):
    EXP[_i] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & FIELD_ORDER:
        _x ^= POLYNOMIAL
for _i in range(FIELD_CHAR_MINUS_ONE, 512):
    EXP[_i] = EXP[_i - FIELD_CHAR_MINUS_ONE]

del _x, _i


def add(a: int, b: int) -> int:
    """Field addition (XOR)."""
    return a ^ b


def sub(a: int, b: int) -> int:
    """Field subtraction (identical to addition in characteristic 2)."""
    return a ^ b


def mul(a: int, b: int) -> int:
    """Field multiplication."""
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


def div(a: int, b: int) -> int:
    """Field division; raises ZeroDivisionError for a zero divisor."""
    if b == 0:
        raise ZeroDivisionError("division by zero in GF(2^8)")
    if a == 0:
        return 0
    return EXP[(LOG[a] - LOG[b]) % FIELD_CHAR_MINUS_ONE]


def inv(a: int) -> int:
    """Multiplicative inverse of a non-zero element."""
    if a == 0:
        raise ZeroDivisionError("inverse of zero in GF(2^8)")
    return EXP[FIELD_CHAR_MINUS_ONE - LOG[a]]


def pow2(i: int) -> int:
    """GENERATOR ** i, i.e. the element 2^i used as a Q coefficient."""
    return EXP[i % FIELD_CHAR_MINUS_ONE]


_MUL_TABLES: dict[int, bytes] = {0: bytes(FIELD_ORDER), 1: bytes(range(FIELD_ORDER))}


def mul_table(c: int) -> bytes:
    """A 256-byte translation table mapping b -> c * b, cached."""
    table = _MUL_TABLES.get(c)
    if table is None:
        table = bytes(mul(c, b) for b in range(FIELD_ORDER))
        _MUL_TABLES[c] = table
    return table


def mul_bytes(c: int, data: bytes) -> bytes:
    """Multiply every byte of ``data`` by the field element ``c``."""
    if c == 1:
        return data
    return data.translate(mul_table(c))
