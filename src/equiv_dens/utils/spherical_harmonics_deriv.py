import torch
import math
import numpy as np
from scipy.special import binom

"""
This returns a list of all spherical harmonics (up to order L)
derived from the input unit vectors u. For up to L=5, this
function is optimized to reduce the number of floating point
operations. When L>5 is requested, a general formula is used,
which is not as efficient.
NOTE: All spherical harmonics lack the constant 1/sqrt(4*Pi)
for efficiency and work only for unit vectors (to
remove unnecessary terms involving radius r)
The m values are stored from -L (index 0), -L+1 (index 1), ..., L
Condon-Shortley phase is included!

input:
    L: integer that specifies order (0:s, 1:p, 2:d, 3:f, 4:g, 5:h, ...)
    u: Cartesian unit vectors of shape [...,3] (last dimension must be 3, the rest of the shape is arbitrary)
output:
    Y: list of length L+1 containing the spherical harmonics of shape [...,2*L+1]
"""


def spherical_harmonics_deriv(L, u):
    Y = []
    if L >= 0:
        Y.append(Y0_deriv(u))
    if L >= 1:
        shape = (*u.shape[:-1], 1)
        x = torch.gather(u, -1, u.new_full(shape, 0, dtype=torch.long))
        y = torch.gather(u, -1, u.new_full(shape, 1, dtype=torch.long))
        z = torch.gather(u, -1, u.new_full(shape, 2, dtype=torch.long))
        dx = torch.cat([torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
        dy = torch.cat([torch.zeros_like(y), torch.ones_like(y), torch.zeros_like(y)], dim=-1)
        dz = torch.cat([torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z)], dim=-1)
        Y.append(Y1_deriv(dx, dy, dz))
    if L >= 2:
        x2 = x * x
        y2 = y * y
        z2 = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        _x2my2 = x2 - y2
        _3z2m1 = 3 * z2 - 1
        dx2 = 2 * x * dx 
        dy2 = 2 * y * dy 
        dz2 = 2 * z * dz 
        dxy = y * dx + x * dy
        dyz = y * dz + z * dy
        dxz = z * dx + x * dz
        _dx2my2 = dx2 - dy2
        _d3z2m1 = 3 * dz2
        Y.append(Y2_deriv(dxy, dyz, dxz, _d3z2m1, _dx2my2))
    if L >= 3:
        xyz = xy * z
        _3x2my2 = 3 * x2 - y2
        _x2m3y2 = x2 - 3 * y2
        dxyz = dxy * z + xy * dz 
        _d3x2my2 = 3 * dx2 - dy2
        _dx2m3y2 = dx2 - 3 * dy2
        Y.append(Y3_deriv(x, y, z, z2, _x2my2, _3x2my2, _x2m3y2,
                          dx, dy, dz, dz2, dxyz, _dx2my2, _d3x2my2, _dx2m3y2))
    if L >= 4:
        x4 = x2 * x2
        y4 = y2 * y2
        x2y2 = x2 * y2
        dx4 = 4*x**3 * dx
        dy4 = 4*y**3 * dy 
        dx2y2 = dx2 * y2 + x2 * dy2
        Y.append(Y4_deriv(x2, y2, z2, xy, yz, xz, _x2my2, _x2m3y2,
                    dx2, dy2, dz2, dxy, dyz, dxz, dx4, dy4, dx2y2, _dx2my2, _dx2m3y2))
    if L >= 5:
        Y.append(Y5_deriv(x, y, z, z2, xyz, x4, y4, x2y2,
                          _x2my2, _3z2m1, _3x2my2, _x2m3y2,
                          dx, dy, dz, dz2, dxyz, dx4, dy4, dx2y2,
                          _dx2my2, _d3z2m1, _d3x2my2, _dx2m3y2))
    if L >= 6:
        raise NotImplementedError("Derivatives for order 6 and above are not yet implemented.")
    #     for order in range(6, L + 1):
    #         Y.append(Yl(order, x, y, z))
    return Y


# spherical harmonics of order 0
def Y0_deriv(u):
    return u.new_zeros((*u.shape[:-1], 1, u.shape[-1]))

# spherical harmonics of order 1
sqrt3 = np.sqrt(3)
def Y1_deriv(dx, dy, dz):
    # return (sqrt3 / 3) * (dx + dy + dz).unsqueeze(-1).expand(-1, -1, -1, 3)
    return sqrt3 * torch.stack((dy, dz,dx), dim=-2)
# # spherical harmonics of order 2
sqrt15 = np.sqrt(15)
sqrt5over2 = np.sqrt(5) / 2
sqrt15over2 = sqrt15 / 2
#
#
def Y2_deriv(dxy, dyz, dxz, _d3z2m1, _dx2my2):
    return torch.stack(
        (
            sqrt15 * dxy,
            sqrt15 * dyz,
            sqrt5over2 * _d3z2m1,
            sqrt15 * dxz,
            sqrt15over2 * _dx2my2,
        ), dim=-2)


# spherical harmonics of order 3
sqrt70over4 = np.sqrt(70) / 4
sqrt105 = np.sqrt(105)
sqrt42over4 = np.sqrt(42) / 4
sqrt7over2 = np.sqrt(7) / 2
sqrt105over2 = sqrt105 / 2

def Y3_deriv(x, y, z, z2, _x2my2, _3x2my2, _x2m3y2,
             dx, dy, dz, dz2, dxyz, _dx2my2, _d3x2my2, _dx2m3y2):
    _5z2 = 5 * z2
    _d5z2 = 5 * dz2
    _5z2m1 = _5z2 - 1
    return torch.stack(
        (
            sqrt70over4 * (y * _d3x2my2 + dy * _3x2my2),
            sqrt105 * dxyz,
            sqrt42over4 * (y * _d5z2 + dy * _5z2m1),
            sqrt7over2 * (dz * (_5z2 - 3) + z * _d5z2),
            sqrt42over4 * (dx * _5z2m1 + x * _d5z2),
            sqrt105over2 * (z * _dx2my2 + dz * _x2my2),
            sqrt70over4 * (dx * _x2m3y2 + x * _dx2m3y2),
        ),
        dim=-2,
    )
#
#
# spherical harmonics of order 4
sqrt35_3over2 = np.sqrt(35) * 3 / 2
sqrt70_9over4 = np.sqrt(70) * 9 / 4
sqrt45over2 = np.sqrt(45) / 2
sqrt10_3over4 = np.sqrt(10) * 3 / 4
oneover8 = 1 / 8
sqrt45over4 = sqrt45over2 / 2
sqrt70_3over4 = sqrt70_9over4 / 3
sqrt35_3over8 = sqrt35_3over2 / 4


def Y4_deriv(x2, y2, z2, xy, yz, xz, _x2my2, _x2m3y2,
       dx2, dy2, dz2, dxy, dyz, dxz, dx4, dy4, dx2y2, _dx2my2, _dx2m3y2,):
    _7z2 = 7 * z2
    _d7z2 = 7 * dz2
    _7z2m1 = _7z2 - 1
    _7z2m3 = _7z2 - 3
    return torch.stack(
        (
            sqrt35_3over2 * (xy * _dx2my2 + dxy * _x2my2),
            sqrt70_9over4 * (dyz * (x2 - y2 / 3) + yz * (dx2 - dy2 / 3)),
            sqrt45over2 * (xy * _d7z2 + dxy * _7z2m1),
            sqrt10_3over4 * (yz * _d7z2 + dyz * _7z2m3),
            oneover8 * (dz2 * (105 * z2 - 90) + z2 * 105 * dz2),
            sqrt10_3over4 * (xz * _d7z2 + dxz * _7z2m3),
            sqrt45over4 * (_7z2m1 * _dx2my2 + _d7z2 * _x2my2),
            sqrt70_3over4 * (dxz * _x2m3y2 + xz * _dx2m3y2),
            sqrt35_3over8 * (dx4 - 6 * dx2y2 + dy4),
        ),
        dim=-2,
    )

# spherical harmonics of order 5
sqrt154_3over16 = np.sqrt(154) * 3 / 16
sqrt385_3over2 = np.sqrt(385) * 3 / 2
sqrt770over16 = np.sqrt(770) / 16
sqrt1155over2 = np.sqrt(1155) / 2
sqrt165over8 = np.sqrt(165) / 8
sqrt11over8 = np.sqrt(11) / 8
sqrt1155over4 = sqrt1155over2 / 2
sqrt385_3over8 = sqrt385_3over2 / 4


def Y5_deriv(x, y, z, z2, xyz, x4, y4, x2y2,
             _x2my2, _3z2m1, _3x2my2, _x2m3y2,
             dx, dy, dz, dz2, dxyz, dx4, dy4, dx2y2,
             _dx2my2, _d3z2m1, _d3x2my2, _dx2m3y2):
    z4 = z2 * z2
    dz4 = 4 * z**3 * dz
    _9z2m1 = 9 * z2 - 1
    _d9z2m1 = 9 * dz2
    _21z4m14z2p1 = 21 * z4 - 14 * z2 + 1
    _d21z4m14z2p1 = 21 * dz4 - 14 * dz2
    return torch.stack(
        (
            sqrt154_3over16 * (dy * (5 * x4 - 10 * x2y2 + y4) +
                               y * (5 * dx4 - 10 * dx2y2 + dy4)),
            sqrt385_3over2 * (dxyz * _x2my2 + xyz * _dx2my2),
            sqrt770over16 * (dy * (_3x2my2 * _9z2m1) +
                             y * (_d3x2my2 * _9z2m1 + _3x2my2 * _d9z2m1)),
            sqrt1155over2 * (dxyz * _3z2m1 + xyz * _d3z2m1),
            sqrt165over8 * (dy * _21z4m14z2p1 + y * _d21z4m14z2p1),
            sqrt11over8 * (dz * (63 * z4 - 70 * z2 + 15) + z * (63 * dz4 - 70 * dz2)),
            sqrt165over8 * (dx * _21z4m14z2p1 + x * _d21z4m14z2p1),
            sqrt1155over4 * (dz * (_3z2m1 * _x2my2) +
                             z * (_d3z2m1 * _x2my2 + _3z2m1 * _dx2my2)),
            sqrt770over16 * (dx * (_x2m3y2 * _9z2m1) +
                             x * (_dx2m3y2 * _9z2m1 + _x2m3y2 * _d9z2m1)),
            sqrt385_3over8 * (dz * (x4 - 6 * x2y2 + y4) + z * (dx4 - 6 * dx2y2 + dy4)),
            sqrt154_3over16 * (dx * (x4 - 10 * x2y2 + 5 * y4) +
                               x * (dx4 - 10 * dx2y2 + 5 * dy4)),
        ),
        dim=-2,
    )
#
#
# # utility functions to generate higher order spherical harmonics
# def _A(m, x, y):
#     A = torch.zeros_like(x)
#     for p in range(m + 1):
#         A += binom(m, p) * x ** p * y ** (m - p) * math.cos((m - p) * math.pi / 2)
#     return A
#
#
# def _B(m, x, y):
#     B = torch.zeros_like(x)
#     for p in range(m + 1):
#         B += binom(m, p) * x ** p * y ** (m - p) * math.sin((m - p) * math.pi / 2)
#     return B
#
#
# def _Pi(L, m, z):
#     Pi = torch.zeros_like(z)
#     for k in range((L - m) // 2 + 1):
#         Pi += (
#             (-1) ** k
#             * 2 ** (-L)
#             * binom(L, k)
#             * binom(2 * L - 2 * k, L)
#             * math.factorial(L - 2 * k)
#             / math.factorial(L - 2 * k - m)
#             * z ** (L - 2 * k - m)
#         )
#     return math.sqrt(math.factorial(L - m) / math.factorial(L + m)) * Pi
#
#
# # Herglotz generating function for Y(l,m)
# def _Y(L, m, x, y, z):
#     if m > 0:
#         return math.sqrt(4 * L + 2) * _Pi(L, m, z) * _A(m, x, y)
#     elif m < 0:
#         return math.sqrt(4 * L + 2) * _Pi(L, -m, z) * _B(-m, x, y)
#     else:
#         return math.sqrt(2 * L + 1) * _Pi(L, m, z)
#
#
# # spherical harmonics of order l (works for any order)
# def Yl(L, x, y, z):
#     Yl = []
#     for m in range(-L, L + 1):
#         Yl.append(_Y(L, m, x, y, z))
#     return torch.cat(Yl, dim=-1)
