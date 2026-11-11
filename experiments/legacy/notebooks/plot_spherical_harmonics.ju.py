# %%
import numpy as np
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from equiv_dens.utils import base as utils
import torch
plt.rc('text', usetex=True)

# %%
# Grids of polar and azimuthal angles
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
# Create a 2-D meshgrid of (theta, phi) angles.
theta, phi = np.meshgrid(theta, phi)
# Calculate the Cartesian coordinates of each point in the mesh.
xyz = np.array([np.sin(theta) * np.sin(phi),
                np.sin(theta) * np.cos(phi),
                np.cos(theta)])
print('xyz shape', xyz.shape)
rot_mat = utils.random_rotation_matrix()
print('rot mat shape', rot_mat.shape)
xyz_rot = np.einsum('ij,jkl->ikl', rot_mat, xyz)
# xyz = rot_mat @ xyz

# xyz = np.einsum('ij,klj->kli', rot_mat, xyz)

# %%
def gauss_3d(x, y, z, a):
    return np.exp(-a * (x**2 + y**2 + z**2))

def compute_Y(el, m, phi, theta, equiv=False):
    Y = sph_harm(abs(m), el, phi, theta)
    #print('Y', Y)
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real

        Y = np.sqrt(2) * (-1)**m * Y.real
    # print('sph.numpy() shape', sph.numpy().shape)
    # print('xyz shape', xyz.shape)
    # print('Y.real shape', Y.real.shape)
    # Yx, Yy, Yz = np.abs(Y) * sph.numpy()
    #print('Y', Y)
    if equiv:
        order = m + el
        sph = 0.25 * spherical_harmonics(el, torch.tensor(xyz).permute(1, 2, 0))[-1][:, :, order]
    else:
        sph = Y.real
    return sph #  * gauss_3d(*xyz, a=1)

def plot_Y(ax, el, m, equiv=False):
    """Plot the spherical harmonic of degree el and order m on Axes ax."""
    # NB In SciPy's sph_harm function the azimuthal coordinate, theta,
    # comes before the polar coordinate, phi.

    sph = compute_Y(el, m, phi, theta, equiv=equiv)
    Yx, Yy, Yz = np.abs(sph) * xyz_rot
    # print("Yx shape", Yx.shape)
    # print('xyz norms', np.sqrt(np.sum(xyz**2, axis=0)))
    # print('Y real 10, 10', Y.real[20, 20])
    # print('sph [10, 10]', sph[20, 20])
    #print('sph', sph)

    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('bwr'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(sph),
                    rstride=2, cstride=2)

    # Draw a set of x, y, z axes for reference.
    ax_lim = 0.5
    ax.plot([-ax_lim, ax_lim], [0, 0], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [-ax_lim, ax_lim], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [0, 0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    ax.set_title(r'$Y_{{{}}}$'.format(el))
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('off')

# %%
fig = plt.figure(figsize=plt.figaspect(1.))
ax1 = fig.add_subplot(121, projection='3d')
l, m = 2, 0
plot_Y(ax1, l, m, equiv=False)
ax2 = fig.add_subplot(122, projection='3d')
l, m = 3, 2
plot_Y(ax2, l, m, equiv=True)
#plt.savefig('Y{}_{}.png'.format(l, m))
plt.show()

# %%
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
cg_mat = ClebschGordanMatrix(order_max=3)

def tensor_product_contraction(a, b, el1, m1, el2, m2, el3, m3):
    # print('e1', el1, 'm1', m1, 'el2', el2, 'm2', m2, 'el3', el3, 'm3', m3)
    #print('cg mat', cg_mat(el1, el2, el3)[0].numpy()[el1**2 + el1 + m1,
    #                                                 el2**2 + el2 + m2,
    #                                                 el3**2:(el3 + 1)**2])
    cg = cg_mat(el1, el2, el3)[0].numpy()[el1**2 + el1 + m1,
                                          el2**2 + el2 + m2,
                                          el3**2 + el3 + m3]
    # print('cg mat shape', cg_mat(el1, el2, el3)[0].numpy()[m1 + el1, m2 + el2].shape)
    #print('cg coeff', cg)
    tp_out = a * b * cg

    return tp_out * (2 * el3 + 1)

# %%
def plot_tp_Y(ax, el1, m1, el2, m2, el3, m3, equiv=False):
    """Plot the spherical harmonic of degree el and order m on Axes ax."""
    # NB In SciPy's sph_harm function the azimuthal coordinate, theta,
    # comes before the polar coordinate, phi.

    sph1 = compute_Y(el1, m1, phi, theta, equiv=equiv)
    sph2 = compute_Y(el2, m2, phi, theta, equiv=equiv)

    #print('norm sph1', np.sqrt(np.sum(sph1**2)))
    #print('norm sph2', np.sqrt(np.sum(sph2**2)))
    sph_tp = tensor_product_contraction(sph1, sph2, el1, m1, el2, m2, el3, m3)

    #print('norm sph_tp', np.sqrt(np.sum(sph_tp**2)))
    
    #print('sph tp', sph_tp)
    Yx, Yy, Yz = np.abs(sph_tp) * xyz_rot
    # print("Yx shape", Yx.shape)
    # print('xyz norms', np.sqrt(np.sum(xyz**2, axis=0)))
    # print('Y real 10, 10', Y.real[20, 20])
    # print('sph [10, 10]', sph[20, 20])

    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('bwr'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(sph_tp),
                    rstride=2, cstride=2)

    # Draw a set of x, y, z axes for reference.
    ax_lim = 0.5
    ax.plot([-ax_lim, ax_lim], [0, 0], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [-ax_lim, ax_lim], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [0, 0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    ax.set_title(r'$Y_{{{}}}$'.format(el3))
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('off')

# %%
rot_mat = utils.random_rotation_matrix()
print('rot mat shape', rot_mat.shape)
xyz_rot = np.einsum('ij,jkl->ikl', rot_mat, xyz)
fig = plt.figure(figsize=plt.figaspect(1.))
ax1 = fig.add_subplot(131, projection='3d')
el1, m1 = 1, 1
plot_Y(ax1, el1, m1, equiv=False)
ax2 = fig.add_subplot(132, projection='3d')
el2, m2 = 2, 2
plot_Y(ax2, el2, m2, equiv=False)
ax3 = fig.add_subplot(133, projection='3d')
el3, m3 = 3, 3
plot_tp_Y(ax3, el1, m1, el2, m2, el3, m3, equiv=False)
#plt.savefig('Y{}_{}_no_rot.svg'.format(l, m))
plt.show()

# %%
rot_mat = utils.random_rotation_matrix()
print('rot mat shape', rot_mat.shape)
xyz_rot = np.einsum('ij,jkl->ikl', rot_mat, xyz)

fig = plt.figure(figsize=plt.figaspect(1.))
ax1 = fig.add_subplot(131, projection='3d')
el1, m1 = 1, 1
plot_Y(ax1, el1, m1, equiv=False)
ax2 = fig.add_subplot(132, projection='3d')
el2, m2 = 2, 2
plot_Y(ax2, el2, m2, equiv=False)
ax3 = fig.add_subplot(133, projection='3d')
el3, m3 = 3, 3
plot_tp_Y(ax3, el1, m1, el2, m2, el3, m3, equiv=False)
#plt.savefig('Y{}_{}_rot.svg'.format(l, m))
plt.show()

# %%
period = 1000
time = np.linspace(0, period, num=period)

y = np.exp(-0.00015 * (time-period*0.75)**2) + np.exp(-0.00005 * (time + -period*0.25)**2)
# print(time)
# print(y)

def cn(n):
   c = y*np.exp(-1j*2*n*np.pi*time/period)
   return c.sum()/c.size

def fn(x, n):
    return 2*cn(n)*np.exp(1j*2*n*np.pi*x/period)

def f(x, Nh):
   f = np.array([fn(x, i) for i in range(1,Nh+1)])
   return f.sum()


y2 = np.array([f(t,6).real for t in time])
y2 = y2 - y2.min()


fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(111)
for i in range(1, 7):
    ax.plot(time, [fn(t, i) + i for t in time], 'k')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.ylabel('basis functions', fontsize=16)
plt.savefig('figures/fourier_functions.svg')
fig = plt.figure(figsize=(7, 1))
ax = fig.add_subplot(111)
ax.plot(time, y, 'k', label='original')
ax.plot(time, y2, label='representation')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.legend()
plt.savefig('figures/fourier_basis.svg')

# %%
rot_mat = utils.random_rotation_matrix()
print('rot mat shape', rot_mat.shape)
xyz_rot = np.einsum('ij,jkl->ikl', rot_mat, xyz)

fig = plt.figure(figsize=plt.figaspect(1.))
el1, m1 = 2, -1
sph = compute_Y(el1, m1, theta, phi, equiv=False)
Yxyz = np.abs(sph) * xyz
print('Yxyz shape', Yxyz.shape)
print('Yxyz 0 ', Yxyz[0])
print('Yxyz 1 ', Yxyz[1])
print('max Yxyz 0 ', np.max(Yxyz[0]))

print('sph shape', sph.shape)
print('sph ', np.min(sph * xyz[2]))
print('sph ', np.max(sph * xyz[2]))

Z = (sph*xyz[2])[:, :50]
Z = np.concatenate([(sph*xyz[2])[:50, :50], (sph*xyz[2])[:50, :50]], axis=0)
cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# fig.colorbar(cp)
# ax2 = fig.add_subplot(132, projection='3d')
# el2, m2 = 2, 2
# plot_Y(ax2, el2, m2, equiv=False)
# ax3 = fig.add_subplot(133, projection='3d')
# el3, m3 = 3, 3
# plot_tp_Y(ax3, el1, m1, el2, m2, el3, m3, equiv=False)
# #plt.savefig('Y{}_{}_rot.svg'.format(l, m))
# plt.show()
# print(xyz[2])
plt.savefig('figures/Y{}_no_rad.svg'.format(el1))

# %%
def gauss(x, y, a):
    return np.exp(-a * (x**2 + y**2))

fig = plt.figure(figsize=plt.figaspect(1.))
# Z = (sph*xyz[2])[:, :50]
Z = np.concatenate([(sph*xyz[2])[:50, :50], (sph*xyz[2])[:50, :50]], axis=0)
Z = Z * gauss(Yxyz[0][:, :50], Yxyz[1][:, :50], 5)
# cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr')
cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# fig.colorbar(cp)
plt.savefig('figures/Y{}_rad.svg'.format(el1))

# %%
fig = plt.figure(figsize=plt.figaspect(1.))
el1, m1 = 1, 0
sph = compute_Y(el1, m1, theta, phi, equiv=False)
Yxyz = np.abs(sph) * xyz
print('Yxyz shape', Yxyz.shape)
print('Yxyz 0 ', Yxyz[0])
print('Yxyz 1 ', Yxyz[1])
print('max Yxyz 0 ', np.max(Yxyz[0]))

print('sph shape', sph.shape)
print('sph ', np.min(sph * xyz[2]))
print('sph ', np.max(sph * xyz[2]))

plt.xlim([-0.5, 0.5])
plt.ylim([-0.5, 0.5])
Z = (sph*xyz[2])[:, :50]
vmin = np.min(Z)
vmax = np.max(Z)
# Z = np.concatenate([(sph*xyz[2])[:50, :50], (sph*xyz[2])[:50, :50]], axis=0)
cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
plt.savefig('figures/Y{}_no_rad.svg'.format(el1))

fig = plt.figure(figsize=plt.figaspect(1.))
# Z = (sph*xyz[2])[:, :50]
plt.xlim([-0.5, 0.5])
plt.ylim([-0.5, 0.5])
Z = Z * gauss(Yxyz[0][:, :50], Yxyz[1][:, :50], 50)
# cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr')
cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# fig.colorbar(cp)
plt.savefig('figures/Y{}_rad.svg'.format(el1))

# %%
fig = plt.figure(figsize=plt.figaspect(1.))
el1, m1 = 0, 0
sph = compute_Y(el1, m1, theta, phi, equiv=False)
Yxyz = np.abs(sph) * xyz
print('Yxyz shape', Yxyz.shape)
print('Yxyz 0 ', Yxyz[0])
print('Yxyz 1 ', Yxyz[1])
print('max Yxyz 0 ', np.max(Yxyz[0]))

print('sph shape', sph.shape)
print('sph ', np.min(sph * xyz[2]))
print('sph ', np.max(sph * xyz[2]))

plt.xlim([-0.5, 0.5])
plt.ylim([-0.5, 0.5])
Z = (sph*xyz[2])[:, :50]
# print("Z shape", Z.shape)
# Z = (sph*xyz[2])[sph * xyz[2] > 0].reshape(100, 50)
# print("Z shape", Z.shape)
# Z = np.concatenate([(sph*xyz[2])[:50, :50], (sph*xyz[2])[:50, :50]], axis=0)
cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
plt.savefig('figures/Y{}_no_rad.svg'.format(el1))

fig = plt.figure(figsize=plt.figaspect(1.))
# Z = (sph*xyz[2])[:, :50]
plt.xlim([-0.5, 0.5])
plt.ylim([-0.5, 0.5])
Z = Z * gauss(Yxyz[0][:, :50], Yxyz[1][:, :50], 10)
# cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr')
cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# fig.colorbar(cp)
plt.savefig('figures/Y{}_rad.svg'.format(el1))

# %%
rot_mat = utils.random_rotation_matrix()
print('rot mat shape', rot_mat.shape)
xyz_rot = np.einsum('ij,jkl->ikl', rot_mat, xyz)

fig = plt.figure(figsize=plt.figaspect(1.))
el1, m1 = 3, -1
sph = compute_Y(el1, m1, theta, phi, equiv=False)
Yxyz = np.abs(sph) * xyz
print('Yxyz shape', Yxyz.shape)
print('Yxyz 0 ', Yxyz[0])
print('Yxyz 1 ', Yxyz[1])
print('max Yxyz 0 ', np.max(Yxyz[0]))

print('sph shape', sph.shape)
print('sph ', np.min(sph * xyz[2]))
print('sph ', np.max(sph * xyz[2]))

Z = (sph*xyz[2])[:, :50]
Z = np.concatenate([(sph*xyz[2])[:50, :50], (sph*xyz[2])[:50, :50]], axis=0)
Z = Z * gauss(Yxyz[0][:, :50], Yxyz[1][:, :50], 5)
vmin = np.min(Z)
vmax = np.max(Z)
cp = plt.contourf(Yxyz[0][:, :50], Yxyz[1][:, :50], Z, levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# fig.colorbar(cp)
# ax2 = fig.add_subplot(132, projection='3d')
# el2, m2 = 2, 2
# plot_Y(ax2, el2, m2, equiv=False)
# ax3 = fig.add_subplot(133, projection='3d')
# el3, m3 = 3, 3
# plot_tp_Y(ax3, el1, m1, el2, m2, el3, m3, equiv=False)
# #plt.savefig('Y{}_{}_rot.svg'.format(l, m))
# plt.show()
# print(xyz[2])
plt.savefig('figures/Y{}_no_rad.svg'.format(el1))
