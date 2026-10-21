import struct

M = 0xffffffff
rot = lambda x, k: ((x << k) | (x >> (32 - k))) & M


def _mix(a, b, c):
    a = (a - c) & M; a ^= rot(c, 4);  c = (c + b) & M
    b = (b - a) & M; b ^= rot(a, 6);  a = (a + c) & M
    c = (c - b) & M; c ^= rot(b, 8);  b = (b + a) & M
    a = (a - c) & M; a ^= rot(c, 16); c = (c + b) & M
    b = (b - a) & M; b ^= rot(a, 19); a = (a + c) & M
    c = (c - b) & M; c ^= rot(b, 4);  b = (b + a) & M
    return a, b, c


def _final(a, b, c):
    c ^= b; c = (c - rot(b, 14)) & M
    a ^= c; a = (a - rot(c, 11)) & M
    b ^= a; b = (b - rot(a, 25)) & M
    c ^= b; c = (c - rot(b, 16)) & M
    a ^= c; a = (a - rot(c, 4)) & M
    b ^= a; b = (b - rot(a, 14)) & M
    c ^= b; c = (c - rot(b, 24)) & M
    return c


def lookup3(k, init=0):
    """Bob Jenkins hashlittle, as in H5_checksum_lookup3."""
    n = len(k); a = b = c = (0xdeadbeef + n + init) & M; i = 0
    while n > 12:
        a = (a + int.from_bytes(k[i:i+4], 'little')) & M
        b = (b + int.from_bytes(k[i+4:i+8], 'little')) & M
        c = (c + int.from_bytes(k[i+8:i+12], 'little')) & M
        a, b, c = _mix(a, b, c); n -= 12; i += 12
    if n == 0:
        return c
    t = k[i:] + bytes(12 - n)
    a = (a + int.from_bytes(t[0:4], 'little')) & M
    b = (b + int.from_bytes(t[4:8], 'little')) & M
    c = (c + int.from_bytes(t[8:12], 'little')) & M
    return _final(a, b, c)


def clear_status_flags(path):
    """Clear the HDF5 superblock (v2/v3) file-consistency flags, like `h5clear -s`.

    A writer killed mid-run (e.g. at a Slurm time limit) leaves the file flagged as open for write,
    and h5py then refuses to open it with mode 'a'.
    """
    with open(path, 'r+b') as f:
        sb = bytearray(f.read(48))
        if sb[11] == 0:
            return
        assert sb[:8] == b'\x89HDF\r\n\x1a\n' and sb[8] in (2, 3) and sb[9] == 8
        if lookup3(bytes(sb[:44])) != struct.unpack('<I', sb[44:48])[0]:
            raise RuntimeError('superblock checksum does not reproduce, not touching ' + path)
        sb[11] = 0
        sb[44:48] = struct.pack('<I', lookup3(bytes(sb[:44])))
        f.seek(0)
        f.write(sb)
