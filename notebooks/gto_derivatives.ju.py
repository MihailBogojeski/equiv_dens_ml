# %%
import torch
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
import equiv_dens.utils.spherical_harmonics_deriv as sph_deriv
import numpy as np
from torchviz import make_dot
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
print('coords', coords)
print('u', u)
gto = 0
gto_grad = 0
s_deriv = sph_deriv.spherical_harmonics_deriv(max_L, u)
dcoords = torch.eye(3).unsqueeze(0).unsqueeze(0)
print(dcoords.shape)
for L in range(max_L + 1):
    scale = scales[L]
    width = widths[L]
    coeff = coeffs[L]
    sph = s[L].unsqueeze(-1) * coeff
    print('L', L)
    print('coeff shape', coeff.shape)
    print('s[L] shape', s[L].shape)
    print('s[l] deriv shape', s_deriv[L].shape)
    print('sph', sph)
    sph_autograd = 0
    du = 0
    if L != 0:
        for i in range(sph.squeeze().shape[0]):
            for j in range(sph.squeeze().shape[1]):
                sph_autograd += torch.autograd.grad(sph.squeeze()[i, j], coords, retain_graph=True)[0]
                du += torch.autograd.grad(sph.squeeze()[i, j], u, retain_graph=True)[0]
                # print(sph_autograd)
    # print('sph du autograd', du)
    if L != 1:
        sph_grad = (s_deriv[L].unsqueeze(-1) * coeff.unsqueeze(0)).sum((-1, -2))
    else:
        sph_grad = (s_deriv[L].unsqueeze(-1) * coeff[..., [2, 0, 1], :]).sum(-1)
    # print('sph du deriv', sph_grad)
    # print('sph autograd', sph_autograd)
    sph_grad_c = sph_grad.unsqueeze(-2) * -(dcoords/d.unsqueeze(-1) -
                 (coords.unsqueeze(-2) * coords.unsqueeze(-1)/d.unsqueeze(-1)**3))
    sph_grad_c = sph_grad_c.sum(-1)
    # print('sph u and c ratio', sph_autograd / sph_grad)
    rbf = orbitals.gaussian_rbf(d.unsqueeze(-1), width, scale, L)
    rbf_autograd = 0
    for i in range(3):
        rbf_autograd += torch.autograd.grad(rbf.squeeze()[i], coords, retain_graph=True)[0]
    # print('rbf autograd', rbf_autograd)
    rbf_grad = orbitals.gaussian_rbf_deriv(d.unsqueeze(-1), width, scale, L).squeeze(-1) * coords / d
    # print('rbf_deriv', rbf_grad)
    print('coords.shape', coords.shape)
    gto_l = torch.sum(rbf * sph, dim=(-2, -1)) 
    gto_autograd = 0
    for i in range(3):
        gto_autograd += torch.autograd.grad(gto_l.squeeze()[i], coords, retain_graph=True)[0]
    print('rbf shape', rbf.shape)
    print('rbf grad shape', rbf_grad.shape)
    print('rbf grad unsqueeze shape', rbf_grad.unsqueeze(-2).shape)
    print('sph shape', sph.shape)
    print('sph deriv unsqueeze coords', sph_grad_c.unsqueeze(-1).shape) 
    print('rbf_grad unsqueeze * sph', (rbf_grad.unsqueeze(-2) * sph).shape)
    print('rbf* sph_grad unsqueeze ', (rbf * sph_grad_c.unsqueeze(-1)).shape)
    gto_deriv = torch.sum(rbf_grad.unsqueeze(-2) * sph, -2) + rbf.squeeze(-1) * sph_grad_c
    print('gto autograd', gto_autograd)
    print('gto deriv', gto_deriv)
    gto += gto_l 

# gto_autodiff = 0
# %%
du = 0.0
for i in range(u.squeeze().shape[0]):
    for j in range(u.squeeze().shape[1]):
        du += torch.autograd.grad(u.squeeze()[i, j], coords, retain_graph=True)[0]

dcoords = torch.eye(3).unsqueeze(0).unsqueeze(-1)
dcoords = dcoords.expand((-1 ,-1 ,-1, 3))

u_deriv = -(1 / d - (coords * torch.sum(coords, dim=(-1), keepdim=True) / d**3))
print('du', du)
print('u_deriv', u_deriv)
# %%
sph_grad = (s_deriv[1].unsqueeze(-1) * coeffs[1][..., [2, 0, 1], :]).sum(-1)
# sph_grad = (s_deriv[2].unsqueeze(-1) * coeffs[2]).sum((-1, -2))
print('sph grad shape', sph_grad.shape)
dcoords = torch.eye(3).unsqueeze(0).unsqueeze(0)
# dcoords = dcoords.expand((-1 , 3 ,-1, -1))
u_deriv = -(dcoords/d.unsqueeze(-1) - (coords.unsqueeze(-2) * coords.unsqueeze(-1)/d.unsqueeze(-1)**3))
print('u_deriv shape', u_deriv.shape)
print('u_deriv', (sph_grad.unsqueeze(-2) * u_deriv).sum(-1))
# %%
coords2 = torch.tensor(coords)
coords2.requires_grad = True
coeffs1 = torch.tensor(coeffs[1])
class Direction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones((1, 3, 3)))

    def forward(self, Ri):
        Ri = Ri * self.dummy
        Rj = torch.zeros_like(Ri)
        rij = Rj - Ri  # displacement vectors
        dij = torch.norm(rij, dim=-1, keepdim=True)  # distances
        uij = rij / dij  # unit displacement vectors
        return uij

class L1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones((1, 3, 3)))

    def forward(self, u):
        u = u * self.dummy
        print('u', u)
        return np.sqrt(3) * u[..., [1, 2, 0]] 

class SphCoeffs(torch.nn.Module):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = torch.nn.Parameter(coeffs)

    def forward(self, s):
        return s.unsqueeze(-1) * self.coeffs

l1 = L1()
dir = Direction()
sphcoeffs = SphCoeffs(coeffs1)
model = torch.nn.Sequential(dir, l1, sphcoeffs)

print('coords', coords2)

sph_l1 = model(coords2)
# make_dot(sph_l1.mean(), params=dict(model.named_parameters()))
print("l1", sph_l1)
l = sph_l1.sum()
l.backward()
print('coords_grad', coords2.grad)
print('l1 dummy', l1.dummy.grad)
print('dir dummy', dir.dummy.grad)
# %% 
coords2 = torch.tensor(coords)
coords2.requires_grad = True
coeffs1 = torch.tensor(coeffs[1])
d = torch.norm(coords2, dim=-1, keepdim=True)  # distances
d.retain_grad()
u = coords2 / d  # unit displacement vectors
u.retain_grad()
s = np.sqrt(3) * u[..., [1, 2, 0]]
s.retain_grad()
sph = s.unsqueeze(-1) * coeffs1
sph.retain_grad()

l = sph.sum()
l.backward(retain_graph=True)
print('coords grad', coords2.grad)
print('sph grad', sph.grad)
print('u grad', u.grad)
print('d grad', d.grad)
# %% 
coords2 = coords.clone().detach()
coords2.requires_grad = True
coeffs1 = coeffs[2].clone().detach()
c_pow2 = coords2**2
c_pow2.retain_grad()
c_pow2_sum = c_pow2.sum(-1, keepdim=True)
c_pow2_sum.retain_grad()
d = c_pow2_sum**(0.5)
d.retain_grad()
u2 = coords2 * d**(-1)  # unit displacement vectors
u2.retain_grad()
l = (u2 * sph_grad).sum()
# l = u2.sum()
l.backward(retain_graph=True)
print('u grad', u2.grad)
print('d grad', d.grad)
print('c_pow2_sum grad', c_pow2_sum.grad)
print('c_pow2 grad', c_pow2.grad)
print('one side?', 2 * coords2 * c_pow2.grad)
print('coords grad', coords2.grad)
print('coords', coords)
print('final grad?', 2 * coords2 * c_pow2.grad + sph_grad/d)
make_dot(l)
# %%

s_deriv = sph_deriv.spherical_harmonics_deriv(2, u)

sph_grad = (s_deriv[2].unsqueeze(-1) * coeffs[2]).sum((-1,-2))
print('sph_grad.shape', sph_grad.shape)

dudr = -(1 / d - (coords * torch.sum(coords, dim=(-1), keepdim=True) / d**3)) 
dudd = (coords / -d**2)
dddp = 1/(2 *  torch.sqrt(torch.sum(coords**2, -1, keepdim=True)))
dpds = 2*coords
sph_grad_c = sph_grad * dudr
print('s grad', s_deriv[1])
print('sph_grad', sph_grad)
print('u grad r', dudr)
print('u grad d', (sph_grad*dudd).sum(-1, keepdim=True))
print('d grad sqrt', (sph_grad*dudd*dddp).sum(-1, keepdim=True))
print('sqrt grad sum', dpds*(sph_grad*dddp*dudd).sum(-1, keepdim=True))
print('sqrt grad sum', dpds*(sph_grad*dddp*dudd).sum(-1, keepdim=True))
print('sum up', sph_grad*1/d + dpds*(sph_grad*dddp*dudd).sum(-1, keepdim=True))
# print('second part', (coords * torch.sum(coords, dim=(-1), keepdim=True) / d**3))
# print('sph_grad_c', sph_grad_c)
