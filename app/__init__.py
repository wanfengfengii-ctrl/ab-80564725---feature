"""RAID-6 style stripe reconstruction service.

Reconstructs missing shards of a storage stripe protected by dual
parity shards (P and Q) computed over GF(2^8) with the primitive
polynomial 0x11d and generator 2.
"""

__version__ = "1.0.0"
