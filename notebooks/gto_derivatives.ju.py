# %%
import torch
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
import equiv_dens.utils.spherical_harmonics_deriv as sph_deriv
# %load_ext autoreload
# %autoreload 2

# %%
coords = torch.randn(1, 3, 3)
coords.requires_grad = True
print('coords', coords)
max_L = 5
d, u = utils.calculate_distances_and_directions(coords, center=torch.zeros(1, 1, 3))
s = spherical_harmonics(max_L, u)
print('s requires grad', [ss.requires_grad for ss in s])
for L in range(len(s)):
    zeros = torch.zeros_like(s[L])
    s[L] = torch.where(torch.isnan(s[L]), zeros, s[L])  # making sure there are no nans to avoid NaNs
print('dist', d)
print('unit', u)
scales = [torch.randn(1, 1, 1, 1) for L in range(max_L + 1)]
widths = [torch.randn(1, 1, 1, 1)**2 for L in range(max_L + 1)]
coeffs = [torch.randn(1, 1, 2 * L + 1, 1) for L in range(max_L + 1)]
print([ss.shape for ss in s])
print('scales', scales)
print('widths', widths)
print('coeffs', coeffs)

# %%
gto = 0

for L in range(max_L + 1):
    scale = scales[L]
    width = widths[L]
    coeff = coeffs[L]
    sph = s[L].unsqueeze(-1) * coeff
    print('sph shape', sph.shape)
    # if L == 1:
        # print('sL shape', s[L].shape)
        # print('sph grad', torch.autograd.grad(sph[0, 0, 0, 0], coords))
    rbf = orbitals.gaussian_rbf(d.unsqueeze(-1), width, scale, L)
    gto += torch.sum(rbf * sph, dim=(-2, -1))
print(gto)

# %%
grad = torch.autograd.grad(gto[0,1], coords, retain_graph=True)
print(grad)

# %%
print('coords requires grad', coords.requires_grad)
print('u requires grad', u.requires_grad)
print('s requres grad', s[0].requires_grad)
gto = 0
s_deriv = sph_deriv.spherical_harmonics_deriv(max_L, u)
for L in range(max_L + 1):
    scale = scales[L]
    width = widths[L]
    coeff = coeffs[L]
    sph = s[L].unsqueeze(-1) * coeff
    print('L', L)
    print('coeff shape', coeff.shape)
    print('s[L] shape', s[L].shape)
    print('sph shape', sph.shape)
    print('s[l] deriv shape', s_deriv[L].shape)
    sph_autograd = 0
    if L != 0:
        for i in range(sph.squeeze().shape[0]):
            for j in range(sph.squeeze().shape[1]):
                sph_autograd += torch.autograd.grad(sph.squeeze()[i, j], u, retain_graph=True)[0]
    print('sph autograd', sph_autograd)
    if L != 1:
        sph_grad = (s_deriv[L].unsqueeze(-1) * coeff.unsqueeze(0)).sum((-1, -2))
    else:
        sph_grad = (s_deriv[L].unsqueeze(-1) * coeff[..., [2, 0, 1], :]).sum(-1)
    print('sph deriv', sph_grad) 
    rbf = orbitals.gaussian_rbf(d.unsqueeze(-1), width, scale, L)
    rbf_autograd = 0
    for i in range(3):
        rbf_autograd += torch.autograd.grad(rbf.squeeze()[i], coords, retain_graph=True)[0]
    print('rbf autograd', rbf_autograd)
    rbf_deriv = orbitals.gaussian_rbf_deriv(d.unsqueeze(-1), width, scale, L).squeeze(-1) * coords / d
    print('rbf_deriv', rbf_deriv)
    gto += torch.sum(rbf * sph, dim=(-2, -1))

# %%


