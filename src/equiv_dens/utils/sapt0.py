import numpy as np
import equiv_dens.utils.base as utils
from equiv_dens.utils import orbitals
import torch
import scipy as sp
from pyscf.dft import numint
from pyscf import df, lib, dft, gto
import pyscf.gto.mole
from pyscf.scf import hf
import scipy
from ase.data import chemical_symbols
import warnings
import time
import gc
import copy


def calculate_sapt0_ref(mols_1, mols_2,
                        dm_1=None, dm_2=None,
                        collect_calc_dict=False):

    calc_dict = []
    sapt0s = []
    for i in range(len(mols_1)):
        # water molecule
        m1_mol = mols_1[i]
        if not m1_mol._built:
            m1_mol.build()
        # benzene molecule
        m2_mol = mols_2[i]
        if not m2_mol._built:
            m2_mol.build()
        # combined molecule
        m12_mol = gto.M(atom=(m1_mol.atom + m2_mol.atom), basis=m1_mol.basis)

        if dm_1 is None:
            m1_mf = dft.RKS(m1_mol)
            m1_mf.chkfile = False
            m1_mf.xc = 'pbe'
            m1_mf.kernel()
            m1_dm = m1_mf.make_rdm1()
        else:
            m1_dm = dm_1[i]

        if dm_2 is None:
            m2_mf = dft.RKS(m2_mol)
            m2_mf.chkfile = False
            m2_mf.xc = 'pbe'
            m2_mf.kernel()
            m2_dm = m2_mf.make_rdm1()
        else:
            m2_dm = dm_2[i]

        m12_dm = np.zeros((m1_dm.shape[0] + m2_dm.shape[0], m1_dm.shape[1] + m2_dm.shape[1]))
        m12_dm[:m1_dm.shape[0], :m1_dm.shape[0]] = m1_dm
        m12_dm[m1_dm.shape[0]:, m1_dm.shape[0]:] = m2_dm

        vj, _ = hf.get_jk(m1_mol, m1_dm)
        coul_en_m1 = np.einsum('ij,ji->', vj, m1_dm).real * .5
        vj, _ = hf.get_jk(m2_mol, m2_dm)
        coul_en_m2 = np.einsum('ij,ji->', vj, m2_dm).real * .5
        vj, _ = hf.get_jk(m12_mol, m12_dm)
        coul_en_m12 = np.einsum('ij,ji->', vj, m12_dm).real * .5

        coulomb_en_els = coul_en_m12 - coul_en_m1 - coul_en_m2

        inter_nuc_en = m12_mol.energy_nuc() - m1_mol.energy_nuc() - m2_mol.energy_nuc()

        m_nuc = m12_mol.intor('int1e_nuc')
        m12_e_ne = np.einsum('ij,ji', m12_dm, m_nuc)
        m_nuc = m1_mol.intor('int1e_nuc')
        m1_e_ne = np.einsum('ij,ji', m1_dm, m_nuc)
        m_nuc = m2_mol.intor('int1e_nuc')
        m2_e_ne = np.einsum('ij,ji', m2_dm, m_nuc)

        m12_ne_inter = m12_e_ne - m1_e_ne - m2_e_ne

        sapt0_m12 = coulomb_en_els + inter_nuc_en + m12_ne_inter
        print('sapt0 coulomb energy', sapt0_m12)
        sapt0s.append(sapt0_m12)

        if collect_calc_dict:
            calc_dict['m1_mol'] = m1_mol.pack()
            calc_dict['m1_dm'] = m1_dm
            calc_dict['m1_coul_en'] = coul_en_m1
            calc_dict['m1_ne_en'] = m1_e_ne
            calc_dict['m2_mol'] = m2_mol.pack()
            calc_dict['m2_dm'] = m2_dm
            calc_dict['m2_coul_en'] = coul_en_m2
            calc_dict['m2_ne_en'] = m2_e_ne
            calc_dict['m12_mol'] = m12_mol.pack()
            calc_dict['m12_dm'] = m12_dm
            calc_dict['inter_coul_en'] = coulomb_en_els
            calc_dict['inter_nuc_en'] = inter_nuc_en
            calc_dict['inter_ne_el'] = m12_ne_inter
            calc_dict['coul_sapt0'] = sapt0_m12

    if collect_calc_dict:
        return sapt0s, calc_dict
    else:
        return sapt0s


def evaluate_ml_monomer(mols, model,
                        orbital_basis_num,
                        cutoff=7.937658158457616,
                        density_expansion=False,
                        grid_spec=None, grid_fn=None,
                        atom_dens_dict=None,
                        use_gpu=False,
                        batch_size=1,
                        collate=True,
                        split_atom_densities=False,
                        ):
    m_res_arr = []
    n_samples = mols['atom_numbers'].shape[0]
    n_batches = np.ceil(n_samples / batch_size).astype(int)

    for i in range(n_batches):
        max_idx = min((i + 1) * batch_size, n_samples)
        m_input = {key: mols[key][i * batch_size:max_idx] for key in mols.keys() if isinstance(mols[key], np.ndarray)}
        atom_nums, pos = utils.compress_batch_atoms(m_input['atom_numbers'], {'positions': m_input['positions']})
        m_input['atom_numbers'] = atom_nums
        m_input['positions'] = pos['positions']

        m_samp = orbitals.model_input_from_atoms(m_input,
                                                 density_expansion=density_expansion,
                                                 pyscf_grid=True,
                                                 skip_compress=True,
                                                 grid_spec=grid_spec,
                                                 grid_sampling_fn=grid_fn,
                                                 cutoff=cutoff,
                                                 dtype=torch.float32,
                                                 atom_dens_type="mo_coeffs",
                                                 free_atom_densities=atom_dens_dict,
                                                 split_atom_densities=split_atom_densities,
                                                 basis=None,
                                                 all_atom_coeffs=True,
                                                 )

        if use_gpu:
            for key in m_samp.keys():
                if isinstance(m_samp[key], torch.Tensor):
                    m_samp[key] = m_samp[key].cuda()
            model = model.cuda()

        res_m = model(m_samp)
        res_m['ml_df_coeffs_basis'] = []

        anum = torch.max(res_m["batch_atom_numbers"], dim=0)[0]
        atom_types, _ = utils.create_ghost_atom_types(anum)

        for idx in range(res_m['batch_positions'].shape[0]):
            ml_basis = orbitals.ml_basis_to_pyscf_basis(res_m, atom_types, idx)
            res_m['ml_df_coeffs_basis'].append(ml_basis)

        m_res_arr.append(res_m)
    relevant_batch_keys = ['atom_mo_coeffs', 'atom_mo_coeffs_occ', 'atom_df_coeffs',
                           'atom_df_coeffs_occ', 'batch_atom_numbers', 'batch_positions']
    relevant_fix_keys = ['ml_df_coeffs_basis', 'atom_mo_coeffs_basis', 'atom_df_coeffs_basis']
    if collate:
        m_res_arr = orbitals.collate_ml_outs(m_res_arr, relevant_batch_keys,
                                            relevant_fix_keys, orbital_basis_num)

    return m_res_arr


def evaluate_ml_dimer(mols_1, mols_2, model,
                      orbital_basis_num,
                      cutoff=7.937658158457616,
                      density_expansion=False,
                      grid_spec=None, grid_fn=None,
                      atom_dens_dict=None,
                      use_gpu=False,
                      batch_size=1,
                      collate=True,
                      split_atom_densities=False,
                      ):
    m1_res_arr = []
    m2_res_arr = []
    m12_res_arr = []
    n_samples = mols_1['atom_numbers'].shape[0]
    n_batches = np.ceil(n_samples / batch_size).astype(int)

    for i in range(n_batches):
        max_idx = min((i + 1) * batch_size, n_samples)
        m1_input = {key: mols_1[key][i * batch_size:max_idx] for key in mols_1.keys() if isinstance(mols_1[key], np.ndarray)}
        m2_input = {key: mols_2[key][i * batch_size:max_idx] for key in mols_2.keys() if isinstance(mols_2[key], np.ndarray)}

        atom_nums1, pos1 = utils.compress_batch_atoms(m1_input['atom_numbers'], {'positions': m1_input['positions']})
        m1_input['atom_numbers'] = atom_nums1
        m1_input['positions'] = pos1['positions']

        atom_nums2, pos2 = utils.compress_batch_atoms(m2_input['atom_numbers'], {'positions': m2_input['positions']})
        m2_input['atom_numbers'] = atom_nums2
        m2_input['positions'] = pos2['positions']

        print('batch', i, '/', n_batches)
        print('calc key', mols_1['calc_key'][i * batch_size])
        print('m1 num', m1_input['atom_numbers'].shape)
        print('m2 num', m2_input['atom_numbers'].shape)

        m12_input = {key: np.concatenate([m1_input[key], m2_input[key]], axis=1) for key in m1_input.keys()}

        m12_samp = orbitals.model_input_from_atoms(m12_input,
                                                   density_expansion=density_expansion,
                                                   pyscf_grid=True,
                                                   skip_compress=True,
                                                   grid_spec=grid_spec,
                                                   grid_sampling_fn=grid_fn,
                                                   cutoff=cutoff,
                                                   dtype=torch.float32,
                                                   atom_dens_type="mo_coeffs",
                                                   free_atom_densities=atom_dens_dict,
                                                   split_atom_densities=False,
                                                   basis=None,
                                                   all_atom_coeffs=True,
                                                   )

        if density_expansion:
            coord_params = {'coords': m12_samp['coords'],
                            'coord_weights': m12_samp['coord_weights']}
        else:
            coord_params = None

        m1_samp = orbitals.model_input_from_atoms(m1_input,
                                                  density_expansion=density_expansion,
                                                  pyscf_grid=True,
                                                  skip_compress=True,
                                                  grid_spec=grid_spec,
                                                  grid_sampling_fn=grid_fn,
                                                  cutoff=cutoff,
                                                  dtype=torch.float32,
                                                  atom_dens_type="mo_coeffs",
                                                  free_atom_densities=atom_dens_dict,
                                                  split_atom_densities=False,
                                                  basis=None,
                                                  all_atom_coeffs=True,
                                                  coord_params=coord_params,
                                                  )

        m2_samp = orbitals.model_input_from_atoms(m2_input,
                                                  density_expansion=density_expansion,
                                                  pyscf_grid=True,
                                                  skip_compress=True,
                                                  grid_spec=grid_spec,
                                                  grid_sampling_fn=grid_fn,
                                                  cutoff=cutoff,
                                                  dtype=torch.float32,
                                                  atom_dens_type="df_coeffs",
                                                  free_atom_densities=atom_dens_dict,
                                                  split_atom_densities=False,
                                                  basis=None,
                                                  all_atom_coeffs=True,
                                                  coord_params=coord_params,
                                                  )

        if use_gpu:
            for key in m12_samp.keys():
                if isinstance(m12_samp[key], torch.Tensor):
                    m12_samp[key] = m12_samp[key].cuda()
                    m1_samp[key] = m1_samp[key].cuda()
                    m2_samp[key] = m2_samp[key].cuda()
            model = model.cuda()

        res_m1 = model(m1_samp)
        res_m2 = model(m2_samp)

        res_m12 = m12_samp
        res_m12['spherical_coeffs'] = res_m1['spherical_coeffs'] + res_m2['spherical_coeffs']
        res_m12['radial_scale'] = res_m1['radial_scale'] + res_m2['radial_scale']
        res_m12['radial_width'] = res_m1['radial_width'] + res_m2['radial_width']

        if density_expansion:
            res_m12 = model.property_models['density'](res_m12)
            res_m12['density'] += res_m12['atom_density']

        df_coeffs12 = orbitals.coeffs_dict_to_vector(res_m12,
                                                     orbital_basis_num,
                                                     res_m12['batch_atom_numbers'],
                                                     radial_coeffs=False,
                                                     convert_to_pyscf=True)['spherical_coeffs'].detach()
        res_m12['df_coeffs'] = df_coeffs12

        res_m1['ml_df_coeffs_basis'] = []
        res_m2['ml_df_coeffs_basis'] = []
        res_m12['ml_df_coeffs_basis'] = []

        anum1 = torch.max(res_m1["batch_atom_numbers"], dim=0)[0]
        atom_types1, _ = utils.create_ghost_atom_types(anum1)
        for idx in range(res_m1['batch_positions'].shape[0]):
            ml_basis = orbitals.ml_basis_to_pyscf_basis(res_m1, atom_types1, idx)
            res_m1['ml_df_coeffs_basis'].append(ml_basis)

        anum2 = torch.max(res_m2["batch_atom_numbers"], dim=0)[0]
        atom_types2, _ = utils.create_ghost_atom_types(anum2)
        for idx in range(res_m2['batch_positions'].shape[0]):
            ml_basis = orbitals.ml_basis_to_pyscf_basis(res_m2, atom_types2, idx)
            res_m2['ml_df_coeffs_basis'].append(ml_basis)

        anum12 = torch.max(res_m12["batch_atom_numbers"], dim=0)[0]
        atom_types12, _ = utils.create_ghost_atom_types(anum12)
        for idx in range(res_m12['batch_positions'].shape[0]):
            ml_basis = orbitals.ml_basis_to_pyscf_basis(res_m12, atom_types12, idx)
            res_m12['ml_df_coeffs_basis'].append(ml_basis)


        m1_res_arr.append(res_m1)
        m2_res_arr.append(res_m2)
        m12_res_arr.append(res_m12)


    relevant_batch_keys = ['atom_mo_coeffs', 'atom_mo_coeffs_occ', 'atom_df_coeffs',
                           'atom_df_coeffs_occ', 'batch_atom_numbers', 'batch_positions']
    relevant_fix_keys = ['ml_df_coeffs_basis', 'atom_mo_coeffs_basis', 'atom_df_coeffs_basis']
    
    if collate:
        m1_res_arr = orbitals.collate_ml_outs(m1_res_arr, relevant_batch_keys,
                                            relevant_fix_keys, orbital_basis_num)
        m2_res_arr = orbitals.collate_ml_outs(m2_res_arr, relevant_batch_keys,
                                            relevant_fix_keys, orbital_basis_num)
        m12_res_arr = orbitals.collate_ml_outs(m12_res_arr, relevant_batch_keys,
                                            relevant_fix_keys, orbital_basis_num)

    return m1_res_arr, m2_res_arr, m12_res_arr


def calculate_sapt0_ml(res_1, res_2, res, orbital_basis_num, precalc_basis=True, use_df=False):

    auxmol = utils.npy_to_pyscf(res['batch_positions'].numpy(force=True),
                                res['batch_atom_numbers'].numpy(force=True),
                                basis=res['atom_df_coeffs_basis'],
                                build=True, skip_zero=False)
    mol = utils.npy_to_pyscf(res['batch_positions'].numpy(force=True),
                             res['batch_atom_numbers'].numpy(force=True),
                             basis=res['atom_mo_coeffs_basis'],
                             build=True, skip_zero=False)

    auxmol_2 = utils.npy_to_pyscf(res_2['batch_positions'].numpy(force=True),
                                  res_2['batch_atom_numbers'].numpy(force=True),
                                  basis=res_2['atom_df_coeffs_basis'],
                                  build=True, skip_zero=False)
    mol_2 = utils.npy_to_pyscf(res_2['batch_positions'].numpy(force=True),
                               res_2['batch_atom_numbers'].numpy(force=True),
                               basis=res_2['atom_mo_coeffs_basis'],
                               build=True, skip_zero=False)

    auxmol_1 = utils.npy_to_pyscf(res_1['batch_positions'].numpy(force=True),
                                  res_1['batch_atom_numbers'].numpy(force=True),
                                  basis=res_1['atom_df_coeffs_basis'],
                                  build=True, skip_zero=False)
    mol_1 = utils.npy_to_pyscf(res_1['batch_positions'].numpy(force=True),
                               res_1['batch_atom_numbers'].numpy(force=True),
                               basis=res_1['atom_mo_coeffs_basis'],
                               build=True, skip_zero=False)

    if precalc_basis:
        df_coeffs_ml = res['ml_df_coeffs']
        df_coeffs_ml_1 = res_1['ml_df_coeffs']
        df_coeffs_ml_2 = res_2['ml_df_coeffs']
    else:
        df_coeffs_ml = orbitals.coeffs_dict_to_vector(res, orbital_basis_num,
                                                      res['batch_atom_numbers'], radial_coeffs=False,
                                                      convert_to_pyscf=True)['spherical_coeffs'].detach()
        df_coeffs_ml_2 = orbitals.coeffs_dict_to_vector(res_2, orbital_basis_num,
                                                        res_2['batch_atom_numbers'], radial_coeffs=False,
                                                        convert_to_pyscf=True)['spherical_coeffs'].detach()
        df_coeffs_ml_1 = orbitals.coeffs_dict_to_vector(res_1, orbital_basis_num,
                                                        res_1['batch_atom_numbers'],
                                                        radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()
    auxmol_ml = []
    auxmol_ml_1 = []
    auxmol_ml_2 = []
    for i in range(len(auxmol_1)):
        if precalc_basis:
            auxmol_ml += utils.npy_to_pyscf(res['batch_positions'][[i]].numpy(force=True),
                                           res['batch_atom_numbers'][[i]].numpy(force=True),
                                           basis=res['ml_df_coeffs_basis'][i],
                                           build=True, skip_zero=False, ml_basis=True)
            auxmol_ml_1 += utils.npy_to_pyscf(res_1['batch_positions'][[i]].numpy(force=True),
                                             res_1['batch_atom_numbers'][[i]].numpy(force=True),
                                             basis=res_1['ml_df_coeffs_basis'][i],
                                             build=True, skip_zero=False, ml_basis=True)
            auxmol_ml_2 += utils.npy_to_pyscf(res_2['batch_positions'][[i]].numpy(force=True),
                                             res_2['batch_atom_numbers'][[i]].numpy(force=True),
                                             basis=res_2['ml_df_coeffs_basis'][i],
                                             build=True, skip_zero=False, ml_basis=True)
        else:
            auxmol_ml.append(orbitals.ml_basis_to_auxmol(res, i, skip_zero=False))
            auxmol_ml_1.append(orbitals.ml_basis_to_auxmol(res_1, i, skip_zero=False))
            auxmol_ml_2.append(orbitals.ml_basis_to_auxmol(res_2, i, skip_zero=False))

    sapt0_ml_elst = calculate_sapt0_elst(res, mol, auxmol, auxmol_ml, df_coeffs_ml,
                                         res_1, mol_1, auxmol_1, auxmol_ml_1, df_coeffs_ml_1,
                                         res_2, mol_2, auxmol_2, auxmol_ml_2, df_coeffs_ml_2,
                                         use_df)

    sapt0_ml_efield12, sapt0_ml_efield21 = calculate_efield(res_1, mol_1, auxmol_1, auxmol_ml_1, df_coeffs_ml_1,
                                                            res_2, mol_2, auxmol_2, auxmol_ml_2, df_coeffs_ml_2,
                                                            use_df)

    # dens_ml_ovlp = calculate_ovlp(res_1, mol_1, auxmol_ml_1, df_coeffs_ml_1,
    #                               res_2, mol_2, auxmol_ml_2, df_coeffs_ml_2)

    return sapt0_ml_elst, sapt0_ml_efield12, sapt0_ml_efield21  #, dens_ml_ovlp


def calculate_sapt0_elst(res, mol, auxmol, auxmol_ml, df_coeffs_ml,
                         res_1, mol_1, auxmol_1, auxmol_ml_1, df_coeffs_ml_1,
                         res_2, mol_2, auxmol_2, auxmol_ml_2, df_coeffs_ml_2, use_df=False, use_less_mem=False):
    """ The use_less_mem flag will dramatically reduce memory requirements at the cost of longer calculation time"""
    sapt0_ml_elst = []
    for i in range(len(auxmol)):
        print('start atom mo dm')
        m1_dm = hf.make_rdm1(mo_coeff=res_1['atom_mo_coeffs'][i].numpy(force=True), mo_occ=res_1['atom_mo_coeffs_occ'][i].numpy(force=True)) # these are for both channels
        m2_dm = hf.make_rdm1(mo_coeff=res_2['atom_mo_coeffs'][i].numpy(force=True), mo_occ=res_2['atom_mo_coeffs_occ'][i].numpy(force=True))
        m12_dm = hf.make_rdm1(mo_coeff=res['atom_mo_coeffs'][i].numpy(force=True), mo_occ=res['atom_mo_coeffs_occ'][i].numpy(force=True))
        if use_df:
            print('start coulomb integrals joined ml')
            auxmol_ml_joined, df_coeffs_ml_joined = orbitals.join_free_atom_and_ml_basis(auxmol_ml[i], auxmol[i],
                                                                    df_coeffs_ml[i], res['atom_df_coeffs'][i])
            auxmol_ml_1_joined, df_coeffs_ml_1_joined = orbitals.join_free_atom_and_ml_basis(auxmol_ml_1[i], auxmol_1[i],
                                                                    df_coeffs_ml_1[i], res_1['atom_df_coeffs'][i])
            auxmol_ml_2_joined, df_coeffs_ml_2_joined = orbitals.join_free_atom_and_ml_basis(auxmol_ml_2[i], auxmol_2[i],
                                                                    df_coeffs_ml_2[i], res_2['atom_df_coeffs'][i])
            coulomb_en_sum = orbitals.calculate_int2c2e(auxmol_ml_joined, df_coeffs_ml_joined)
            coulomb_en_sum_1 = orbitals.calculate_int2c2e(auxmol_ml_1_joined, df_coeffs_ml_1_joined)
            coulomb_en_sum_2 = orbitals.calculate_int2c2e(auxmol_ml_2_joined, df_coeffs_ml_2_joined)
        else:
            # calculate coulomb of joined density analytically for DF basis
            print('start coulomb integrals ml')
            coulomb_en_ml = orbitals.calculate_int2c2e(auxmol_ml[i], df_coeffs_ml[i])
            coulomb_en_ml_1 = orbitals.calculate_int2c2e(auxmol_ml_1[i], df_coeffs_ml_1[i])
            coulomb_en_ml_2 = orbitals.calculate_int2c2e(auxmol_ml_2[i], df_coeffs_ml_2[i])

            if use_less_mem:
                # DW: I calculate the cross integral, rather than calculate the dimer - mon1 - mon2
                # I'll set coulomb_en_atom = cross integral and coulomb_en_atom_1 = 0 so that
                # is compatible with the rest of the code
                coulomb_en_atom = orbitals.calc_ee_cross_atom_by_atom_fast(mol_1[i], m1_dm, mol_2[i], m2_dm)
                coulomb_en_atom_1 = 0
                coulomb_en_atom_2 = 0
            else:
                vj, _ = hf.get_jk(mol[i], m12_dm)
                # m12_dm should be *.5 to get one spin channel, but this implictly incorportates coulomb_en = 2 * int rho_1channel* rho_1channel/r_ij
                coulomb_en_atom = np.einsum('ij,ji->', vj, m12_dm).real * .5
                vj_1, _ = hf.get_jk(mol_1[i], m1_dm)
                coulomb_en_atom_1 = np.einsum('ij,ji->', vj_1, m1_dm).real * .5
                vj_2, _ = hf.get_jk(mol_2[i], m2_dm)
                coulomb_en_atom_2 = np.einsum('ij,ji->', vj_2, m2_dm).real * .5

            print('start coulomb integrals mix')
            coulomb_en_mix = orbitals.calculate_int2c2e(auxmol_ml[i], df_coeffs_ml[i],
                                                        auxmol[i], res['atom_df_coeffs'][i])
            # df's are two channel, 
            coulomb_en_mix_1 = orbitals.calculate_int2c2e(auxmol_ml_1[i], df_coeffs_ml_1[i],
                                                        auxmol_1[i], res_1['atom_df_coeffs'][i])
            coulomb_en_mix_2 = orbitals.calculate_int2c2e(auxmol_ml_2[i], df_coeffs_ml_2[i],
                                                        auxmol_2[i], res_2['atom_df_coeffs'][i])

            coulomb_en_sum = coulomb_en_ml + coulomb_en_atom + 2 * coulomb_en_mix
            coulomb_en_sum_1 = coulomb_en_ml_1 + coulomb_en_atom_1 + 2 * coulomb_en_mix_1
            coulomb_en_sum_2 = coulomb_en_ml_2 + coulomb_en_atom_2 + 2 * coulomb_en_mix_2

        coulomb_en_sum_els = coulomb_en_sum - coulomb_en_sum_1 - coulomb_en_sum_2

        inter_nuc_en = auxmol_ml[i].energy_nuc() - auxmol_ml_1[i].energy_nuc() - auxmol_ml_2[i].energy_nuc()

        print('start coulomb integrals nuc')
        m1_m_nuc = mol_1[i].intor('int1e_nuc')
        m1_e_ne_mo = np.einsum('ij,ji', m1_dm, m1_m_nuc)
        m2_m_nuc = mol_2[i].intor('int1e_nuc')
        m2_e_ne_mo = np.einsum('ij,ji', m2_dm, m2_m_nuc)
        m12_m_nuc = mol[i].intor('int1e_nuc')
        m12_e_ne_mo = np.einsum('ij,ji', m12_dm, m12_m_nuc)

        m12_e_ne_df = orbitals.calculate_1e_intor(auxmol_ml[i], 'int1e_nuc', df_coeffs_ml[i],
                                                  coeffs_type='df_coeffs') * 0.5
        m12_e_nuc = m12_e_ne_df + m12_e_ne_mo
        m2_e_ne_df = orbitals.calculate_1e_intor(auxmol_ml_2[i], 'int1e_nuc', df_coeffs_ml_2[i],
                                                 coeffs_type='df_coeffs') * 0.5
        m2_e_nuc = m2_e_ne_df + m2_e_ne_mo
        m1_e_ne_df = orbitals.calculate_1e_intor(auxmol_ml_1[i], 'int1e_nuc', df_coeffs_ml_1[i],
                                                 coeffs_type='df_coeffs') * 0.5
        m1_e_nuc = m1_e_ne_df + m1_e_ne_mo
        m12_e_off = m12_e_nuc - m1_e_nuc - m2_e_nuc

        sapt0_coul = coulomb_en_sum_els + inter_nuc_en + m12_e_off
        sapt0_coul = utils.hartree_to_kcal(sapt0_coul)
        sapt0_ml_elst.append(sapt0_coul.numpy(force=True))

    return np.array(sapt0_ml_elst)



def calculate_efield(res_1, mol_1, auxmol_1, auxmol_ml_1, df_coeffs_ml_1,
                     res_2, mol_2, auxmol_2, auxmol_ml_2, df_coeffs_ml_2, use_df=False):
    elefs_ml_12 = []
    elefs_ml_21 = []
    alpha = 1000000
    coeff = 2 * np.sqrt(4/3) * alpha**(5/2)/(np.pi)

    for i in range(len(mol_1)):
        if use_df:
            auxmol_ml_1_joined, df_coeffs_ml_1_joined = orbitals.join_free_atom_and_ml_basis(auxmol_ml_1[i], auxmol_1[i],
                                                                    df_coeffs_ml_1[i], res_1['atom_df_coeffs'][i])
            auxmol_ml_2_joined, df_coeffs_ml_2_joined = orbitals.join_free_atom_and_ml_basis(auxmol_ml_2[i], auxmol_2[i],
                                                                    df_coeffs_ml_2[i], res_2['atom_df_coeffs'][i])
        c1 = mol_1[i].atom_coords(unit='Bohr')
        c2 = mol_2[i].atom_coords(unit='Bohr')

        z = mol_1[i].atom_charges()
        at_diffs = c2[None, :] - c1[:, None]
        at_dists = np.linalg.norm(at_diffs, axis=-1, keepdims=True)
        pos_e_field_1 = np.sum(np.reshape(z, (-1, 1, 1)) * at_diffs / at_dists**3, axis=0)

        z = mol_2[i].atom_charges()
        at_diffs = c1[None, :] - c2[:, None]
        at_dists = np.linalg.norm(at_diffs, axis=-1, keepdims=True)
        pos_e_field_2 = np.sum(np.reshape(z, (-1, 1, 1)) * at_diffs / at_dists**3, axis=0)

        # calculate electric field values from ml densities and compare to reference
        fakemol1 = gto.fakemol_for_charges(c2)
        fakemol1._bas[:,1] = 1 #set l = 1
        fakemol1._env[-1] = coeff
        fakemol1._env[-2] = alpha

        fakemol2 = gto.fakemol_for_charges(c1)
        fakemol2._bas[:,1] = 1 #set l = 1
        fakemol2._env[-1] = coeff
        fakemol2._env[-2] = alpha

        if use_df:
            print('start joined df intor cross')
            int_df = gto.mole.intor_cross('int2c2e', auxmol_ml_1_joined, fakemol1)
            elef_ml1 = -np.einsum('ip, i -> p', int_df, df_coeffs_ml_1_joined.numpy(force=True))
            elef_ml1 = np.reshape(elef_ml1, (-1, 3))
            int_df = gto.mole.intor_cross('int2c2e', auxmol_ml_2_joined, fakemol2)
            elef_ml2 = -np.einsum('ip, i -> p', int_df, df_coeffs_ml_2_joined.numpy(force=True))
            elef_ml2 = np.reshape(elef_ml2, (-1, 3))
        else:
            print('start atom mo dm')
            m1_dm = hf.make_rdm1(mo_coeff=res_1['atom_mo_coeffs'][i].numpy(force=True), mo_occ=res_1['atom_mo_coeffs_occ'][i].numpy(force=True))
            m2_dm = hf.make_rdm1(mo_coeff=res_2['atom_mo_coeffs'][i].numpy(force=True), mo_occ=res_2['atom_mo_coeffs_occ'][i].numpy(force=True))
            print('start df intor cross')
            int_df = gto.mole.intor_cross('int2c2e', auxmol_ml_1[i], fakemol1)
            int_df = gto.mole.intor_cross('int2c2e', auxmol_ml_2[i], fakemol2)
            print('start mo intor cross')
            int_mo = df.incore.aux_e2(mol_1[i], fakemol1)
            elef_ml_df1 = -np.einsum('ip, i -> p', int_df, df_coeffs_ml_1[i].numpy(force=True))
            elef_ml_df1 = np.reshape(elef_ml_df1, (-1, 3))
            elef_ml_mo1 = -np.einsum('ijp,ij->p', int_mo, m1_dm)
            elef_ml_mo1 = np.reshape(elef_ml_mo1, (-1, 3))
            elef_ml1 = elef_ml_df1 + elef_ml_mo1

            int_mo = df.incore.aux_e2(mol_2[i], fakemol2)
            elef_ml_df2 = -np.einsum('ip, i -> p', int_df, df_coeffs_ml_2[i].numpy(force=True))
            elef_ml_df2 = np.reshape(elef_ml_df2, (-1, 3))
            elef_ml_mo2 = -np.einsum('ijp,ij->p', int_mo, m2_dm)
            elef_ml_mo2 = np.reshape(elef_ml_mo2, (-1, 3))
            elef_ml2 = elef_ml_df2 + elef_ml_mo2
        elef_ml1 = pos_e_field_1 - elef_ml1
        elefs_ml_12.append(elef_ml1)

        elef_ml2 = pos_e_field_2 - elef_ml2
        elefs_ml_21.append(elef_ml2)

    elefs_ml_12 = np.array(elefs_ml_12)
    elefs_ml_21 = np.array(elefs_ml_21)

    return elefs_ml_12, elefs_ml_21


def calculate_ovlp(res_1, mol_1, auxmol_ml_1, df_coeffs_ml_1,
                   res_2, mol_2, auxmol_ml_2, df_coeffs_ml_2):
    dens_ml_ovlp = []
    for i in range(len(mol_1)):
        nbas1 = mol_1[i].nbas
        nbas2 = mol_2[i].nbas
        m1_dm = hf.make_rdm1(mo_coeff=res_1['atom_mo_coeffs'][i].numpy(force=True), mo_occ=res_1['atom_mo_coeffs_occ'][i].numpy(force=True))
        m2_dm = hf.make_rdm1(mo_coeff=res_2['atom_mo_coeffs'][i].numpy(force=True), mo_occ=res_2['atom_mo_coeffs_occ'][i].numpy(force=True))

        # overlap integral between free atom densities
        atmc, basc, envc = gto.mole.conc_env(mol_1[i]._atm, mol_1[i]._bas, mol_1[i]._env,
                                             mol_2[i]._atm, mol_2[i]._bas, mol_2[i]._env)
        shls_slice = (0, nbas1, 0, nbas1, nbas1, nbas1 + nbas2, nbas1, nbas1 + nbas2)
        ovlp_mo = gto.moleintor.getints('int4c1e_sph', atmc, basc, envc, shls_slice, None, 0)
        ovlp_int_mo = np.einsum('ijkp, ij, kp -> ', ovlp_mo, m1_dm, m2_dm)

        # overlap integral between ML densities
        ovlp_ml = gto.mole.intor_cross('int1e_ovlp', auxmol_ml_1[i], auxmol_ml_2[i])
        ovlp_int_ml = np.einsum('i, ij, j -> ', df_coeffs_ml_1[i].numpy(force=True), ovlp_ml, df_coeffs_ml_2[i].numpy(force=True))
        # %%
        # overlap integral based between two monomer densities,
        # one represented using a DF basis, and another using an MO basis
        # first monomer is in MO basis, second in DF basis
        ovlp_mo_ml = df.incore.aux_e2(mol_1[i], auxmol_ml_2[i], intor='int3c1e')
        ovlp_int_mo_ml = np.einsum('ijk, ij, k -> ', ovlp_mo_ml, m1_dm, df_coeffs_ml_2[i].numpy(force=True))
        # %%
        # overlap integral based between two monomer densities,
        # one represented using a DF basis, and another using an MO basis
        # first monomer is in DF basis, second in MO basis
        ovlp_int_ml_mo = df.incore.aux_e2(mol_2[i], auxmol_ml_1[i], intor='int3c1e')
        ovlp_int_ml_mo = np.einsum('ijk, ij, k -> ', ovlp_int_ml_mo, m2_dm, df_coeffs_ml_1[i].numpy(force=True))

        ovlp_total = ovlp_int_mo_ml + ovlp_int_ml_mo + ovlp_int_ml + ovlp_int_mo
        dens_ml_ovlp.append(ovlp_total)

    dens_ml_ovlp = np.array(dens_ml_ovlp)

    return dens_ml_ovlp

