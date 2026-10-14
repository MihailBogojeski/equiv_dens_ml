import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.modules.spherical_harmonic_layers import SphericalLinear, unpack_orders
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
from equiv_dens.utils.orbitals import combine_orbital_basis, \
    get_max_order, get_n_electrons, gto_norm, pyscf_gto_factor, \
    PackedLayout, CoeffsDict, eval_packed_density
from equiv_dens.utils.base import calculate_distances_and_directions, batch_compressed_atoms, \
    bohr_to_angstrom, to_bohr
import time


def linear(x, lin):
    """nn.Linear lin on x as matmul + bias: F.linear (addmm) of a strided view does not trace with data-dependent
    (unbacked) atom counts."""
    return torch.matmul(x, lin.weight.t()) + lin.bias


class CoeffsIntegralConstraint(nn.Module):
    def __init__(self, integral_scale=False, remove_atom_density=False):
        super().__init__()
        if integral_scale:
            self.register_parameter('integral_scale', nn.Parameter(torch.ones(size=(1,))))
        else:
            self.register_buffer('integral_scale', torch.ones(size=(1,)))
        self.remove_atom_density = remove_atom_density

    def forward(self, atoms):
        n_electrons = get_n_electrons(atoms['batch_atom_numbers'])
        p = atoms['packed_coeffs'][0]
        coeff = p['coeff'][..., 0]  # [B, S0] (the L=0 shells)
        pmask = p['width'] != 0
        scale = p['scale']
        norms = 1 / gto_norm(0, torch.where(pmask, p['width'], torch.ones_like(p['width']))) / pyscf_gto_factor
        if self.remove_atom_density:
            # NOTE: only consistent for single-primitive shells (density fitting basis)
            v = torch.where(scale[..., 0] != 0, coeff, torch.zeros_like(coeff))
            w = (scale * norms)[..., 0]
            w_dot_w = torch.einsum('bi,bi->b', w, w)
            v = v - (torch.einsum('bi,bi->b', v, w) / w_dot_w)[:, None] * w
            p['coeff'] = v.unsqueeze(-1)
        else:
            coeffs_sum = torch.sum(coeff.unsqueeze(-1) * scale * norms, dim=(1, 2)).unsqueeze(-1)
            p['coeff'] = p['coeff'] * (n_electrons / coeffs_sum).unsqueeze(-1)
        return atoms


class DensityCoeffsNetwork(nn.Module):
    """
    Neural network for computing density coefficients from spherical harmonic features in a rotationally equivariant way
    """

    def __init__(self,
                 orbital_basis=None,  # orbitals of atoms, defines layout and shape of output matrix
                 order=1,  # maximum order of spherical harmonics features
                 num_features=32,
                 positive_coeffs=False,
                 integral_constraint=None,
                 integral_scale=False,
                 clebsch_gordan=None,
                 verbose=0,
                 compressed_extraction=False,
                 timing=False,
                 memory=False,
                 init_coeffs=None,
                 coeff_weights=None,
                 pred_radial_coeffs=True,
                 init_radial_coeffs=None,
                 ml_width_min=0,
                 ml_width_max=2,
                 scale_sph_order=True,
                 normalize=0,
                 parity=False,
                 remove_atom_density=False,
                 core_basis_ratio=0,
                 linear_out=False,
                 nonmixing=False,
                 nonmixing_bias=True,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

        # store hyperparameter values
        self.orbital_basis = orbital_basis
        print('orbitals_basis')
        self.order = order
        self.num_features = num_features
        self.positive_coeffs = positive_coeffs
        if integral_constraint in ('coeffs_in_coeff_net', 'coeffs_in_coeffs_net'):
            self.integral_constraint = CoeffsIntegralConstraint(integral_scale, remove_atom_density)
        else:
            self.integral_constraint = None
        self.verbose = verbose
        self.compressed_extraction = compressed_extraction
        self.timing = timing
        self.memory = memory
        self.init_coeffs = init_coeffs
        self.pred_radial_coeffs = pred_radial_coeffs
        self.scale_sph_order = scale_sph_order
        self.normalize = normalize
        self.coeff_weights = coeff_weights
        self.core_basis_ratio = core_basis_ratio
        self.radial_coeffs = init_radial_coeffs
        self.ml_width_offset = ml_width_min
        self.ml_width_scale = (ml_width_max - ml_width_min) / 2
        self.linear_out = linear_out
        self.nonmixing = nonmixing

        # if core basis ratio > 0, reduce orbital basis to only the highest width s orbitals
        if self.core_basis_ratio > 0:
            core_basis = {}
            for z in self.orbital_basis.keys():
                core_basis[z] = [orb for orb in self.orbital_basis[z] if orb[2] == 0]
                num_basis = round(len(core_basis[z]) * self.core_basis_ratio)
                core_basis[z] = core_basis[z][:num_basis]
            self.orbital_basis = core_basis
        # extract nuclear charges from orbitals, determine maximum order, and
        # build the occupation mask (for extracting occupied orbitals in energy prediction)
        self.orbitals_max_order = get_max_order(self.orbital_basis)
        # for calculating nucleus - nucleus repulsion

        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan

        # error checking
        # if (2 * self.order) < self.orbitals_max_order:
        #     print("An orbital with L={} was found, but the neural network was initialized with L={}".format(
        #         self.orbitals_max_order, self.order))
        #     print("The neural network MUST have an order of at least 1/2 of the maximum order of all orbitals!")
        #     quit()

        self.spherical_spec, self.radial_spec, self.radial_count = combine_orbital_basis(self.orbital_basis, self.orbitals_max_order)
        if self.verbose > 0:
            print('spherical_spec', self.spherical_spec)
            print('radial_spec', self.radial_spec)
            print('radial count', self.radial_count)

        if self.compressed_extraction:
            self.sph_counts, self.rad_counts, self.r_max, self.sph_dict, self.rad_dict = self.compute_orbital_features_num_compressed()
        else:
            self.sph_counts, self.rad_counts, self.r_max, self.sph_dict, self.rad_dict = self.compute_orbital_features_num()
        if self.verbose > 0:
            print('rad_counts', self.rad_counts)
            print('sph counts', self.sph_counts)
            print('rad_dict', self.rad_dict)
            print('sph_dict', self.sph_dict)
            print('max rad_counts', max(self.rad_counts))
        if self.init_coeffs is not None or self.core_basis_ratio > 0:
            self.output_bias = False
            self.output_zero_init = True
            mix_orders = True
        elif self.nonmixing:
            self.output_bias = nonmixing_bias
            self.output_zero_init = False
            mix_orders = False
        else:
            self.output_bias = True
            self.output_zero_init = False
            mix_orders = True

        self.spherical_output = SphericalLinear(self.order, self.num_features,
                                                self.orbitals_max_order,
                                                max(self.sph_counts), self.clebsch_gordan,
                                                mix_orders=mix_orders, bias=self.output_bias,
                                                zero_init=self.output_zero_init, normalize=self.normalize,
                                                parity=parity)
        if self.linear_out:
            self.linear_output = SphericalLinear(self.orbitals_max_order, max(self.sph_counts),
                                                 self.orbitals_max_order, max(self.sph_counts),
                                                 clebsch_gordan=None, mix_orders=False,
                                                 bias=self.output_bias, zero_init=True, normalize=self.normalize)
        if self.pred_radial_coeffs:
            self.radial_width = nn.ModuleList([nn.Linear(self.num_features, self.rad_counts[L])
                                               for L in range(self.orbitals_max_order + 1)])
            self.radial_scale = nn.ModuleList([nn.Linear(self.num_features, self.rad_counts[L])
                                               for L in range(self.orbitals_max_order + 1)])
        if init_coeffs is not None:
            self.init_L0_coeffs()
        if coeff_weights is not None:
            self.init_coeff_weights()

        self.init_radial_coeffs()
        self.layout = PackedLayout(self.orbital_basis, self.radial_coeffs, round_f32=True)
        self.init_channel_tables()
        self.reset_parameters()

    def reset_parameters(self):
        """
        Sets the initial L=0 coefficients for the model, which are used as baseline for the
        predicted coefficients to speed up convergence

        outputs:
            init_sph: Initial L=0 spherical harmonic coefficients
            init_scale: Initial L=0 radial scale coefficients
            init_width: Initial L=0 radial width coefficients
        """
        if self.pred_radial_coeffs:
            for L in range(len(self.radial_width)):
                nn.init.zeros_(self.radial_width[L].weight)
                nn.init.zeros_(self.radial_scale[L].weight)

    def init_radial_coeffs(self):
        init_width = {}
        init_scale = {}
        for z in self.orbital_basis.keys():
            # print('init coeffs i', i)
            rad_count = [0] * (self.orbitals_max_order + 1)
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                key = (z, L)
                if self.radial_coeffs is not None:
                    init_width[key] = torch.zeros(1, 1, self.r_max[key], orb[1])
                else:
                    init_width[key] = torch.ones(1, 1, self.r_max[key], orb[1])
                init_scale[key] = torch.zeros(1, 1, self.r_max[key], orb[1])
            if self.radial_coeffs is not None:
                for j in range(len(self.orbital_basis[z])):
                    # print('init coeffs j', j)
                    orb = self.orbital_basis[z][j]
                    L = orb[2]
                    key = (z, L)
                    # print('init coeffs orb', orb)
                    n_coeff = len(self.radial_coeffs[z][j][0])
                    # print('rad_count L', rad_count[L])
                    # print('init coeffs ncoeff', n_coeff)
                    init_width[key][..., :n_coeff, rad_count[L]] += torch.Tensor(self.radial_coeffs[z][j][0])
                    init_scale[key][..., :n_coeff, rad_count[L]] += torch.Tensor(self.radial_coeffs[z][j][1])
                    rad_count[L] += 1

        for key in init_width.keys():
            self.register_buffer('init_radial_width_{}_{}'.format(key[0], key[1]), init_width[key])
            self.register_buffer('init_radial_scale_{}_{}'.format(key[0], key[1]), init_scale[key])

        return init_width, init_scale

    def init_radial_width(self, key):
        return getattr(self, 'init_radial_width_{}_{}'.format(key[0], key[1]))

    def init_radial_scale(self, key):
        return getattr(self, 'init_radial_scale_{}_{}'.format(key[0], key[1]))

    def init_L0_coeffs(self):
        init_sph = {}
        init_width = {}
        init_scale = {}
        for z in self.orbital_basis.keys():
            for j in range(len(self.orbital_basis[z])):
                orb = self.orbital_basis[z][j]
                L = orb[2]
                key = (z, L)
                if L == 0:
                    init_sph[key] = self.init_basis['spherical_coeffs'][z]
                    init_width[key] = self.init_basis['radial_width'][z]
                    init_scale[key] = self.init_basis['radial_scale'][z]
                    self.register_buffer('init_sph_{}_{}'.format(key[0], key[1]), init_sph[key])
                    self.register_buffer('init_width_{}_{}'.format(key[0], key[1]), init_width[key])
                    self.register_buffer('init_scale_{}_{}'.format(key[0], key[1]), init_scale[key])

        return init_sph, init_width, init_scale

    def init_sph(self, key):
        return getattr(self, 'init_sph_{}_{}'.format(key[0], key[1]))

    def init_width(self, key):
        return getattr(self, 'init_width_{}_{}'.format(key[0], key[1]))

    def init_scale(self, key):
        return getattr(self, 'init_scale_{}_{}'.format(key[0], key[1]))

    def init_coeff_weights(self):
        coeff_weights = {}
        for z in self.orbital_basis.keys():
            # print('init coeffs i', i)
            rad_count = [0] * (self.orbitals_max_order + 1)
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                key = (z, L)
                coeff_weights[key] = torch.zeros(1, 1, self.r_max[key], orb[1])
            for j in range(len(self.orbital_basis[z])):
                # print('init coeffs j', j)
                orb = self.orbital_basis[z][j]
                L = orb[2]
                key = (z, L)
                # print('init coeffs orb', orb)
                if self.coeff_weights is not None:
                    n_coeff = len(self.coeff_weights[z][j])
                    # print('rad_count L', rad_count[L])
                    # print('init coeffs ncoeff', n_coeff)
                    coeff_weights[key][..., :n_coeff, rad_count[L]] += torch.Tensor(self.coeff_weights[z][j])
                    rad_count[L] += 1

        for key in coeff_weights.keys():
            self.register_buffer('coeff_weights_{}_{}'.format(key[0], key[1]), coeff_weights[key])

        return coeff_weights

    def coeff_weight(self, key):
        return getattr(self, 'coeff_weights_{}_{}'.format(key[0], key[1]))

    """
    Collects spherical harmonics features into orbital coefficients of the appropriate size

    inputs:
        fs: List of spherical harmonics features of different orders, each element of shape [batch_size, 3, 2*order + 1, num_features]
    outputs:
        matrix: Array of orbital coefficients of shape [batch_size, num_orbitals]
    """

    def init_channel_tables(self):
        """Per element shell e of each L (see PackedLayout): spherical feature channel, radial feature channels
        (with mask) and the DF coefficient weight. Non-persistent buffers, gathered by the layout of each batch."""
        self.chan_counts = [int(c) for c in self.sph_counts]
        for L in range(self.orbitals_max_order + 1):
            E, P = len(self.layout.eshell[L]), self.layout.P[L]
            sc = np.zeros(E, dtype=np.int64)
            rc = np.zeros((E, P), dtype=np.int64)
            pm = np.zeros((E, P), dtype=bool)
            w = np.zeros(E)
            for e, (z, k, j) in enumerate(self.layout.eshell[L]):
                counts = self.radial_count[z][L]
                sc[e] = self.sph_dict[(z, L)][k]
                rc[e, :counts[k]] = int(self.rad_dict[(z, L)][0]) + sum(counts[:k]) + np.arange(counts[k])
                pm[e, :counts[k]] = True
                if self.coeff_weights is not None:
                    w[e] = np.sum(self.coeff_weights[z][j])
            self.register_buffer(f'chan_sph_{L}', torch.from_numpy(sc), persistent=False)
            self.register_buffer(f'chan_rad_{L}', torch.from_numpy(rc), persistent=False)
            self.register_buffer(f'chan_pmask_{L}', torch.from_numpy(pm), persistent=False)
            self.register_buffer(f'shell_weight_{L}', torch.from_numpy(w), persistent=False)

    def batch_layout(self, atoms):
        """Packed layout of the batch (outside of a compiled region; inside it is passed in as atoms['coeff_layout'])."""
        zs = atoms['batch_atom_numbers'].max(0)[0].tolist()
        ref = atoms['positions']
        return zs, self.layout.get(zs, ref.device, ref.dtype)

    def channels(self, lay):
        """Feature channels (spherical and radial) that make up each packed shell."""
        return {'sph': [getattr(self, f'chan_sph_{L}')[e] for L, e in enumerate(lay['e'])],
                'rad': [getattr(self, f'chan_rad_{L}')[e] for L, e in enumerate(lay['e'])],
                'pmask': [getattr(self, f'chan_pmask_{L}')[e] for L, e in enumerate(lay['e'])],
                'counts': self.chan_counts, 'P': self.layout.P}

    def output_features(self, fs):
        """Spherical and radial output features [1, n, ...] of the atom features fs [1, n, (order + 1)^2, F]."""
        out_sph = self.spherical_output(fs)
        if self.linear_out:
            out_sph = self.linear_output(out_sph)
        out_sph = unpack_orders(out_sph)
        if self.scale_sph_order:
            for L in range(len(out_sph)):
                out_sph[L] = out_sph[L] * 10**(-L)
        if self.positive_coeffs:
            out_sph[0] = F.softplus(out_sph[0])
        out_width = []
        out_scale = []
        if self.pred_radial_coeffs:
            f0 = fs[..., 0, :]  # the L=0 features
            for L in range(len(self.radial_width)):
                out_width.append(torch.tanh(linear(f0, self.radial_width[L])).unsqueeze(-2))
                if self.positive_coeffs:
                    out_scale.append(torch.tanh(linear(f0, self.radial_scale[L])).unsqueeze(-2))
                else:
                    out_scale.append(linear(f0, self.radial_scale[L]).unsqueeze(-2))
        return out_sph, out_width, out_scale

    def extract_coefficients(self, sph_fs, rad_width, rad_scale, atoms, lay, chan):
        """Gathers the predicted features of the atoms [1, N + 1, ...] (last row: empty atom slot) into the packed
        coefficients [B, S_L, ...] (see PackedLayout)."""
        ref = sph_fs[0]
        B, A = atoms['batch_atom_numbers'].shape
        N = ref.shape[1] - 1
        # row of every (molecule, atom slot): its atom in the compressed features, or N for an empty slot
        row = torch.full((B * A,), N, dtype=torch.long, device=ref.device)
        row = row.index_put((atoms['batch_idx_pos'],), torch.arange(N, device=ref.device)).view(B, A)
        packed = {}
        for L in range(self.orbitals_max_order + 1):
            atom = lay['atom'][L]
            r = row[:, atom]  # [B, S_L]
            coeff = sph_fs[L][0].permute(0, 2, 1)[r, chan['sph'][L]]
            if self.pred_radial_coeffs:
                width = rad_width[L][0, :, 0][r[..., None], chan['rad'][L]] * chan['pmask'][L]
                scale = rad_scale[L][0, :, 0][r[..., None], chan['rad'][L]] * chan['pmask'][L]
            else:
                width = coeff.new_zeros(B, atom.shape[0], self.layout.P[L])
                scale = torch.zeros_like(width)
            if self.radial_coeffs is not None:
                atom_mask = atoms['batch_atom_mask'].to(dtype=ref.dtype)[:, atom].unsqueeze(-1)
                # the original range of the radial width is [-1, 1]
                width = ((width + 1) * self.ml_width_scale + self.ml_width_offset) * lay['width0'][L]
                if self.positive_coeffs:
                    scale = (((scale + 1) * self.ml_width_scale + self.ml_width_offset) * lay['scale0'][L]) * atom_mask
                else:
                    scale = (scale + lay['scale0'][L]) * atom_mask
            packed[L] = {'atom': atom, 'coeff': coeff, 'width': width, 'scale': scale}
        return packed

    def df_weights(self, lay):
        """Weights of the DF coefficients (sum of the coefficient weights of the shell primitives) in PySCF order."""
        w = torch.cat([getattr(self, f'shell_weight_{L}')[e].repeat_interleave(2 * L + 1)
                       for L, e in enumerate(lay['e'])])
        return w[lay['perm']].unsqueeze(0)

    """
    Counts how many features of each order are needed to collect the orbital coefficients
    outputs:
        matrix: Number of features required for each orbital order
    """

    def compute_orbital_features_num(self):
        # counts the number of orbitals of each order across all atoms for the given basis
        sph_counts = [0 for _ in range(self.orbitals_max_order + 1)]
        rad_counts = [0 for _ in range(self.orbitals_max_order + 1)]
        # contains maximum number of radial components for each order across all atoms for the given basis
        r_max = {}
        spherical_dict = {}
        radial_dict = {}
        for z in self.spherical_spec.keys():
            rad_c = self.radial_count[z]
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                n = orb[1]
                key = (z, L)
                if key not in spherical_dict:
                    # print('lcounts range', rad_counts[L], rad_counts[L] + n)
                    spherical_dict[key] = torch.arange(sph_counts[L], sph_counts[L] + n)
                    sph_counts[L] += n
            for j in range(len(self.radial_spec[z])):
                orb = self.radial_spec[z][j]
                L = orb[2]
                n = orb[1]
                key = (z, L)
                if key not in radial_dict:
                    # print('lcounts range', rad_counts[L], rad_counts[L] + n)
                    radial_dict[key] = torch.arange(rad_counts[L], rad_counts[L] + n)
                    rad_counts[L] += n
                    if len(rad_c[L]) < 1:
                        max_rad_c = 0
                    else:
                        max_rad_c = max(rad_c[L])
                    if key in r_max.keys():
                        r_max[key] = max(max_rad_c, r_max[key])
                    else:
                        r_max[key] = max_rad_c

        return sph_counts, rad_counts, r_max, spherical_dict, radial_dict

    """
    Counts how many features of each order are needed to collect the orbital coefficients
    outputs:
        matrix: Number of features required for each orbital order
    """

    def compute_orbital_features_num_compressed(self):
        L_counts = [0 for _ in range(self.orbitals_max_order + 1)]
        r_max = [0 for _ in range(self.orbitals_max_order + 1)]
        orbital_dict = {}
        for z in self.radial_spec.keys():
            for j in range(len(self.radial_spec[z])):
                orb = self.radial_spec[z][j]
                rad_c = self.radial_count[z][j]
                L = orb[2]
                n = orb[1]
                key = (z, L)
                if key not in orbital_dict:
                    orbital_dict[key] = range(n)
                    if L_counts[L] < n:
                        L_counts[L] = n
                    r_max[L] = max(rad_c[L], r_max[L])
        return L_counts, r_max, orbital_dict

    """
    Computes the spherical harmonics coefficients for the density

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
    outputs:
        C: Spherical harmonics coefficients
    """

    def forward(self, atoms):
        if self.memory:
            print('density coeffs forward start:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        # start = time.time()
        # the output layers act on the features of the real atoms and, separately, on one zero row that stands for the
        # empty atom slots of the padded layout (index N); the slots gather from these rows (see extract_coefficients)
        fs = atoms['sph_repr']
        outs = [self.output_features(f) for f in (fs, fs.new_zeros(1, 1, *fs.shape[2:]))]
        out_sph, out_width, out_scale = [[torch.cat(pair, dim=1) for pair in zip(*parts)] for parts in zip(*outs)]
        if self.memory:
            print('density coeffs forward outputs:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        if 'coeff_layout' in atoms:  # compiled region: the layout comes in as data
            lay = atoms.pop('coeff_layout')
        else:
            zs, lay = self.batch_layout(atoms)
            atoms['slot_z'] = zs
        chan = self.channels(lay)
        packed = self.extract_coefficients(out_sph, out_width, out_scale, atoms, lay, chan)
        if 'packed_coeffs' not in atoms:
            atoms['packed_coeffs'] = packed
            atoms['packed_chan'] = chan
            if self.integral_constraint is not None:
                atoms = self.integral_constraint(atoms)
            atoms['df_coeffs'] = self.layout.to_df_vector(atoms['packed_coeffs'], lay)
            if self.coeff_weights is not None:
                atoms['df_weights'] = self.df_weights(lay).to(fs[0].dtype)
        else:
            # core coefficients are added to the existing ones
            atoms.pop('packed_chan', None)
            old = atoms['packed_coeffs']
            for L, p in packed.items():
                if L not in old:
                    old[L] = p
                    continue
                P = max(p['width'].shape[-1], old[L]['width'].shape[-1])
                old[L] = {'atom': torch.cat([p['atom'], old[L]['atom']]),
                          **{k: torch.cat([F.pad(p[k], (0, P - p[k].shape[-1])),
                                           F.pad(old[L][k], (0, P - old[L][k].shape[-1]))], dim=1)
                             for k in ('width', 'scale')},
                          'coeff': torch.cat([p['coeff'], old[L]['coeff']], dim=1)}
        if self.memory:
            print('density coeffs forward end:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)

        atoms = CoeffsDict(atoms, self.layout)
        return atoms


class DFDensityCoeffs(nn.Module):
    """
    Module to set density coefficients to density fitting values and compute all required properties
    """

    def __init__(self, orbital_basis, radial_coeffs, dtype, require_grads=True):
        super().__init__()
        self.orbital_basis = orbital_basis
        self.radial_coeffs = radial_coeffs
        self.dtype = dtype
        self.require_grads = require_grads
        self.layout = PackedLayout(orbital_basis, radial_coeffs)

        # compute properties that are fixed for a given basis
        self.orbitals_max_order = get_max_order(self.orbital_basis)
        self.spherical_spec, _, _ = combine_orbital_basis(self.orbital_basis, self.orbitals_max_order)
        self.sph_counts = [0 for _ in range(self.orbitals_max_order + 1)]
        self.spherical_dict = {}
        for z in self.spherical_spec.keys():
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                n = orb[1]
                key = (z, L)
                if key not in self.spherical_dict:
                    self.spherical_dict[key] = torch.arange(self.sph_counts[L], self.sph_counts[L] + n)
                    self.sph_counts[L] += n

    def forward(self, atoms):
        # set density coefficients to DF coeffs
        dev = atoms['batch_atom_numbers'].device
        zs = atoms['batch_atom_numbers'].max(0)[0].tolist()
        atoms['packed_coeffs'] = self.layout.from_df_vector(atoms['df_coeffs'].to(dtype=self.dtype, device=dev), zs,
                                                            self.require_grads)
        atoms['slot_z'] = zs
        atoms['sph_dict'] = self.spherical_dict
        dij, uij = calculate_distances_and_directions(
            atoms['positions'], atoms['idx_i'], atoms['idx_j'])
        atoms['distances'] = dij
        atoms['directions'] = uij
        sph = spherical_harmonics(self.orbitals_max_order, uij)
        for L in range(self.orbitals_max_order + 1):
            sph[L] = sph[L].unsqueeze(-1)  # unsqueeze for broadcasting
        atoms['sph'] = sph
        if 'density' in atoms:
            del atoms['density']
        return CoeffsDict(atoms, self.layout)


class FreeAtomDensityCoeffs(nn.Module):
    """
    Module to set density coefficients to free atom values and compute all required properties
    """

    def __init__(self, orbital_basis, radial_coeffs, atom_dens, dtype, require_grads=True):
        super().__init__()
        self.orbital_basis = orbital_basis
        self.radial_coeffs = radial_coeffs
        self.atom_dens = atom_dens
        self.dtype = dtype
        self.require_grads = require_grads
        self.layout = PackedLayout(orbital_basis, radial_coeffs)

        # compute properties that are fixed for a given basis
        self.orbitals_max_order = get_max_order(self.orbital_basis)
        self.spherical_spec, _, _ = combine_orbital_basis(self.orbital_basis, self.orbitals_max_order)
        self.sph_counts = [0 for _ in range(self.orbitals_max_order + 1)]
        self.spherical_dict = {}
        for z in self.spherical_spec.keys():
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                n = orb[1]
                key = (z, L)
                if key not in self.spherical_dict:
                    self.spherical_dict[key] = torch.arange(self.sph_counts[L], self.sph_counts[L] + n)
                    self.sph_counts[L] += n

    def forward(self, atoms):
        # set density coefficients to free atom density coeffs
        # assume all batch elements have identical Z vector (TODO)
        Z = atoms['batch_atom_numbers'][0]
        dev = atoms['batch_atom_numbers'].device
        num_coeffs_mol = sum([self.atom_dens[z.item()]['df_coeffs'].shape[-1] for z in Z])
        sph_coeffs_vec = torch.empty((len(atoms['batch_atom_numbers']), num_coeffs_mol), device=dev)
        sph_coeffs_vec[:] = torch.cat([self.atom_dens[z.item()]['df_coeffs'] for z in Z], -1)
        zs = atoms['batch_atom_numbers'].max(0)[0].tolist()
        atoms['packed_coeffs'] = self.layout.from_df_vector(sph_coeffs_vec.to(dtype=self.dtype), zs,
                                                            self.require_grads)
        atoms['slot_z'] = zs
        atoms['sph_dict'] = self.spherical_dict
        dij, uij = calculate_distances_and_directions(
            atoms['positions'], atoms['idx_i'], atoms['idx_j'])
        atoms['distances'] = dij
        atoms['directions'] = uij
        sph = spherical_harmonics(self.orbitals_max_order, uij)
        for L in range(self.orbitals_max_order + 1):
            sph[L] = sph[L].unsqueeze(-1)  # unsqueeze for broadcasting
        atoms['sph'] = sph
        if 'density' in atoms:
            del atoms['density']
        return CoeffsDict(atoms, self.layout)


class DensityExpansion(nn.Module):
    """
    Module for expanding density coefficients into a density sampled on a given grid.
    All atoms and shells are evaluated at once on chunks of grid points (chunk size set by chunk_bytes).
    The density gradient is obtained by autodiff of the density w.r.t. the grid coordinates.
    """

    def __init__(self, orbital_basis,
                 expansion_constraint=None,
                 integral_constraint=None,
                 verbose=0,
                 timing=False,
                 memory=False,
                 grid_scaling_factor=False,
                 remove_atom_density=False,
                 density_grad=False,
                 chunk_bytes=2**30,
                 ):
        super().__init__()
        self.orbital_basis = orbital_basis
        self.expansion_constraint = expansion_constraint
        self.integral_constraint = integral_constraint
        self.timing = timing
        self.memory = memory
        self.verbose = verbose
        self.remove_atom_density = remove_atom_density
        self.density_grad = density_grad
        self.chunk_bytes = chunk_bytes
        if grid_scaling_factor:
            self.register_buffer('grid_scaling_factor', torch.ones(size=(1,)))
        else:
            self.grid_scaling_factor = 0
        if self.verbose:
            print('expansion constraint', self.expansion_constraint)
            print('integral constraint', self.integral_constraint)

        self.orbitals_max_order_dict = get_max_order(self.orbital_basis, per_atom=True)
        self.orbitals_max_order = max(self.orbitals_max_order_dict.values())
        self.layout = PackedLayout(self.orbital_basis)

    @staticmethod
    def density_at(coords, positions, packed, Lmax):
        """Density [B, G] at coords [B, G, 3] (Angstrom) of the atoms at positions [B, A, 3]."""
        r = positions.unsqueeze(1) - coords.unsqueeze(2)  # [B, G, A, 3]
        # safe norm: no NaNs (also in the gradients) for grid points on a nucleus
        d = torch.sqrt((r * r).sum(-1, keepdim=True).clamp_min(1e-30))
        Y = spherical_harmonics(Lmax, r / d)
        return eval_packed_density(d.squeeze(-1) * to_bohr, Y, packed)

    def forward(self, atoms, eval_atoms=None, eval_L=None, density_grad=None):
        if density_grad is None:
            density_grad = self.density_grad
        if self.timing:
            start = time.time()
        if 'packed_coeffs' in atoms:
            packed = atoms['packed_coeffs']
        else:
            zs = atoms['batch_atom_numbers'].max(0)[0].tolist()
            packed = self.layout.pack_dicts(atoms['spherical_coeffs'], atoms['radial_width'],
                                            atoms['radial_scale'], zs)
        if eval_L is not None:
            packed = {L: p for L, p in packed.items() if L in eval_L}
        if eval_atoms is not None:
            sel = {}
            for L, p in packed.items():
                idx = torch.isin(p['atom'], torch.tensor(eval_atoms, device=p['atom'].device)).nonzero()[:, 0]
                sel[L] = {k: v[idx] if k == 'atom' else v[:, idx] for k, v in p.items()}
            packed = sel
        Lmax = max(packed)
        coords = atoms['coords']
        positions = atoms['batch_positions']
        B, G = coords.shape[:2]
        # memory estimate per grid point -> chunk size
        floats = B * (positions.shape[1] * (4 + (Lmax + 1)**2)
                      + sum(p['coeff'].shape[1] * (3 * p['width'].shape[-1] + 3 * p['coeff'].shape[-1] + 4)
                            for p in packed.values()))
        chunk = max(1, min(G, int(self.chunk_bytes // (floats * coords.element_size()))))
        create_graph = torch.is_grad_enabled()

        def eval_chunk(c):
            if not density_grad:
                return self.density_at(c, positions, packed, Lmax)
            with torch.enable_grad():
                c = c.detach().requires_grad_(True)
                d = self.density_at(c, positions, packed, Lmax)
                g, = torch.autograd.grad(d.sum(), c, create_graph=create_graph)
            if not create_graph:
                d, g = d.detach(), g.detach()
            return d, g

        out = []
        for i in range(0, G, chunk):
            c = coords[:, i:i + chunk]
            # with several chunks only one chunk graph is kept in memory, the rest is recomputed in the backward pass
            out.append(checkpoint(eval_chunk, c, use_reentrant=False) if create_graph and chunk < G else eval_chunk(c))
        if density_grad:
            dens, grads = zip(*out)
        else:
            dens = out
        atoms['density'] = torch.cat(dens, dim=1)
        if density_grad:
            atoms['density_grad'] = bohr_to_angstrom(torch.cat(grads, dim=1))
        if self.verbose > 0:
            print('Density shape', atoms['density'].shape)
            print('sum neg integrals', torch.sum((atoms['density'] * atoms['coord_weights'])[atoms['density'] < 0], dim=-1))
        if self.expansion_constraint == 'sq':
            atoms['density'] = atoms['density']**2
        if self.expansion_constraint == 'abs':
            atoms['density'] = torch.sqrt(atoms['density']**2)
        if self.expansion_constraint == 'sp':
            atoms['density'] = F.softplus(atoms['density'], beta=100000000) + 1e-30
        if self.integral_constraint == 'grid':
            grid_scaling = get_n_electrons(atoms['batch_atom_numbers']) / torch.sum(atoms['density'] * atoms['coord_weights'], dim=1, keepdim=True)
            if self.verbose > 0:
                print('grid_scaling', grid_scaling)
                print('grid_scaling_factor', (1 - self.grid_scaling_factor))
            grid_scaling = grid_scaling * (1 - self.grid_scaling_factor) + self.grid_scaling_factor
            atoms['density'] = (atoms['density'] * grid_scaling)
        if self.timing:
            print('density expansion time:', time.time() - start)
        return atoms
