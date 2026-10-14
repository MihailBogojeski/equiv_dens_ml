"""Export audited local water QC labels without importing Torch.

Historical input files are explicitly trusted by --trusted-legacy. Outputs use
safe numeric NPZ and checksummed native JSON/NPZ bundles. The spline/storage
compatibility loader is isolated to this export; it never patches SciPy.
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
class LegacySpline:
 def __setstate__(self,state):self.state=state
class StorageLoader(pickle.Unpickler):
 def find_class(self,module,name):
  if module=='torch' and name in ('DoubleStorage','FloatStorage','LongStorage','IntStorage'):
   return np.dtype({'DoubleStorage':'f8','FloatStorage':'f4','LongStorage':'i8','IntStorage':'i4'}[name])
  return super().find_class(module,name)
 def persistent_load(self, pid):
  kind,dtype,key,location,count,view=pid
  if kind!='storage' or location!='cpu' or view is not None:raise ValueError(pid)
  self.storages[key]=np.empty(count,dtype=dtype);return self.storages[key]
def storage_from_bytes(data):
 f=io.BytesIO(data);magic=pickle.load(f);protocol=pickle.load(f);info=pickle.load(f)
 if protocol!=1001 or not info['little_endian']:raise ValueError('storage format')
 u=StorageLoader(f);u.storages={};result=u.load();keys=pickle.load(f)
 for key in keys:
  x=u.storages[key];count=struct.unpack('<q',f.read(8))[0]
  if count!=x.size:raise ValueError('storage size')
  x[:]=np.frombuffer(f.read(count*x.dtype.itemsize),dtype=x.dtype)
 if f.read():raise ValueError('trailing storage data')
 return result
def rebuild_tensor(storage,offset,size,stride,*unused):
 return np.ndarray(tuple(size),dtype=storage.dtype,buffer=storage,offset=offset*storage.dtype.itemsize,strides=tuple(s*storage.dtype.itemsize for s in stride)).copy()
class Loader(pickle.Unpickler):
 def find_class(self,module,name):
  if module.startswith('scipy.interpolate') and name=='BSpline':return LegacySpline
  if module=='torch.storage' and name=='_load_from_bytes':return storage_from_bytes
  if module=='torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor'):return rebuild_tensor
  return super().find_class(module,name)
def load(path):
 with open(path,'rb') as f:
  v=np.lib.format.read_magic(f)
  h=np.lib.format.read_array_header_1_0(f) if v==(1,0) else np.lib.format.read_array_header_2_0(f)
  return Loader(f).load().item()
def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
 Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def identity(points,numbers,round_distances=False):
 points=np.asarray(points,dtype=np.float64)
 if round_distances:
  # H exchange/translation/rotation insensitive water fingerprint.
  oxygen=points[np.asarray(numbers)==8][0]; hydrogen=points[np.asarray(numbers)==1]
  values=sorted(np.linalg.norm(hydrogen-oxygen,axis=1))+[float(np.linalg.norm(hydrogen[1]-hydrogen[0]))]
  return tuple(np.round(values,6))
 return hashlib.sha256(np.asarray(numbers,dtype=np.int64).tobytes()+points.tobytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',default='scratch/water/prepared')
 parser.add_argument('--count',type=int,default=1000)
 parser.add_argument('--grid-level',type=int,default=2)
 parser.add_argument('--seed',type=int,default=4212)
 parser.add_argument('--trusted-legacy',action='store_true',required=True)
 args=parser.parse_args()
 if args.count<=0 or args.grid_level<0:parser.error('positive count and nonnegative grid level required')
 root=Path(__file__).resolve().parents[2];os.chdir(root)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'protocol.json').exists():raise FileExistsError('use a fresh output directory; protocol already frozen')
 files=[f'datasets/h2o_small_{sp}_{suffix}.npy' for sp in ('train','valid','test') for suffix in ('augccpvdz','dft_augccpvdz_df_hm_dm_oe_calc')]
 files+=['datasets/augccpvqzjkfit_orbital_basis_df.npy','datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy','datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy','notebooks/water_pyscf_calc.ju.py']
 protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'question':'Export real source-consistent density/gradient/E/F with unchanged MO SAD and disjoint splits','count':args.count,'grid_level':args.grid_level,'seed':args.seed,'validation_seed':args.seed+10000,'train_batch_size':5,'validation_batch_size':10,'precision':'float64','units':{'positions':'Angstrom','coords':'Angstrom','coord_weights':'Bohr^3','density':'electrons/Bohr^3','density_grad':'electrons/Bohr^4','energy':'Hartree total','forces':'Hartree/Angstrom'},'weight_policy':'uniform without replacement; original weight times Nfull/count','coordinate_policy':'common charge centroid; model atom order H,H,O; QC AO order O,H,H','normalization_policy':'train-only statistics; raw labels unchanged','fresh_qc_indices':{'train':[0],'valid':[0]},'gradient_fd_indices':[0,50,99],'gates':{'density_dm_max_abs':1e-10,'gradient_fd_max_abs':1e-4,'electron_full_abs':0.02,'fresh_energy_abs':2e-6,'fresh_force_max_abs':2e-5},'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'research_infra_commit':subprocess.check_output(['git','-C','../research-infra','rev-parse','HEAD'],text=True).strip(),'tool_sha256':digest(__file__),'input_sha256':{f:digest(f) for f in files},'hostname':platform.node(),'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
 write_json(out/'protocol.json',protocol)
 print('Frozen protocol',digest(out/'protocol.json'),flush=True)
 from pyscf import dft, gto, lib, __version__ as pyscf_version
 from equiv_dens_jax.preparation import BOHR_ANGSTROM as bohr,load_legacy_qc_results,make_pyscf_grid,sample_qc_results,sample_sad_coeff_density
 from equiv_dens_jax.serialization import save_tree,load_tree
 from equiv_dens_jax.data import load_batches
 lib.num_threads(1)
 atomic=load('datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy')
 atomic={z:{k:atomic[z][k] for k in ('mo_basis','mo_coeff','mo_occ')} for z in (1,8)}
 save_tree(out/'atomic_mo',atomic)
 atomic=load_tree(out/'atomic_mo')
 atomic_checks={}
 for z,e in atomic.items():
  mol=gto.M(atom=[(z,[0,0,0])],basis=e['mo_basis'],spin=z%2,verbose=0)
  dm=(e['mo_coeff']*e['mo_occ'])@e['mo_coeff'].T
  points,w=make_pyscf_grid(mol,level=args.grid_level,coords_unit='angstrom')
  rho,_=sample_sad_coeff_density(np.zeros((1,1,3)),np.array([[z]]),points[None],atomic,coefficient_type='mo_coeffs')
  count=float(np.sum(rho*w));trace=float(np.einsum('ij,ji',dm,mol.intor('int1e_ovlp')))
  if abs(trace-z)>1e-8 or abs(count-z)>.02:raise ValueError('source free-atom MO electron count')
  atomic_checks[z]={'nao':mol.nao_nr(),'occupation_sum':float(e['mo_occ'].sum()),'trace_ds':trace,'full_grid_electrons':count,'mo_coeff_shape':list(e['mo_coeff'].shape),'mo_occ_shape':list(e['mo_occ'].shape)}
 basis=np.load('datasets/augccpvqzjkfit_orbital_basis_df.npy',allow_pickle=True).item()
 radial=np.load('datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy',allow_pickle=True).item()
 basis={z:[tuple(map(int,x)) for x in basis[symbol]] for z,symbol in [(1,'H'),(8,'O')]}
 radial={z:radial[symbol] for z,symbol in [(1,'H'),(8,'O')]}
 save_tree(out/'basis',{'orbital_basis':basis,'radial_basis':radial})
 identities={};shape_identity={}
 for sp in ('train','valid','test'):
  a=np.load(f'datasets/h2o_small_{sp}_augccpvdz.npy',allow_pickle=True).item()
  identities[sp]=[identity(p,z) for p,z in zip(a['positions'],a['atom_numbers'])]
  shape_identity[sp]=[identity(p,z,True) for p,z in zip(a['positions'],a['atom_numbers'])]
  if len(set(identities[sp]))!=len(identities[sp]):raise ValueError('within split exact duplicate')
 split_audit={f'{x}/{y}':{'exact':len(set(identities[x])&set(identities[y])),'distances_rounded_1e-6_angstrom':len(set(shape_identity[x])&set(shape_identity[y]))} for x,y in [('train','valid'),('train','test'),('valid','test')]}
 if any(v['exact'] or v['distances_rounded_1e-6_angstrom'] for v in split_audit.values()):raise ValueError('cross-split duplicate')
 report={'atomic_checks':atomic_checks,'split_overlap':split_audit,'protocol_sha256':digest(out/'protocol.json'),'runtime':{'python':sys.version,'numpy':np.__version__,'pyscf':pyscf_version},'splits':{}}
 permutation=np.array([1,2,0]);all_data={}
 for sp in ('train','valid'):
  records=load_legacy_qc_results(f'datasets/h2o_small_{sp}_dft_augccpvdz_df_hm_dm_oe_calc.npy',trusted_legacy=True)
  atoms=np.load(f'datasets/h2o_small_{sp}_augccpvdz.npy',allow_pickle=True).item()
  rows=[];checks=[];support=[]
  for index,(m,c) in enumerate(records):
   mol=gto.Mole(**dict(m));mol.verbose=0;mol.build()
   pos=mol.atom_coords(unit='Angstrom');num=mol.atom_charges();shift=np.sum(pos*num[:,None],axis=0)/num.sum()
   if not np.array_equal(num,[8,1,1]) or not np.allclose(pos[permutation],atoms['positions'][index],atol=1e-12,rtol=0):raise ValueError('QC geometry/source atom permutation')
   energy_archive_error=abs(float(c['energy'])-float(atoms['energy'][index,0]))
   force_archive_error=float(np.max(np.abs(np.asarray(c['forces'])[permutation]-atoms['forces'][index])))
   if energy_archive_error>1e-10 or force_archive_error>1e-10:raise ValueError('QC/geometry labels differ beyond roundoff')
   if mol.nao_nr()!=41 or mol.cart or m['basis']!='augccpvdz':raise ValueError('source AO basis contract')
   aux=gto.M(atom=m['atom'],basis=c['auxbasis'],verbose=0)
   if aux.nao_nr()!=294:raise ValueError('auxiliary basis size')
   for atom,z in enumerate(num):
    shells=aux._bas[aux._bas[:,0]==atom]
    if list(shells[:,1])!=[x[2] for x in basis[int(z)]]:raise ValueError('basis shell order')
    for shell_idx,rad in zip(np.flatnonzero(aux._bas[:,0]==atom),radial[int(z)]):
     if not np.allclose(aux.bas_exp(shell_idx),rad[0],atol=1e-12,rtol=0) or not np.allclose(aux._libcint_ctr_coeff(shell_idx).ravel(),rad[1],atol=1e-10,rtol=1e-12):raise ValueError('auxiliary radial width/scale')
   full_coords,full_weights=make_pyscf_grid(mol,level=args.grid_level,coords_unit='angstrom')
   if len(full_coords)<args.count:raise ValueError('requested count exceeds full grid')
   seed=args.seed+(10000 if sp=='valid' else 0)+index
   indices=np.random.default_rng(seed).choice(len(full_coords),args.count,replace=False)
   coords=full_coords[indices];weights=full_weights[indices]
   pack=sample_qc_results([(m,c)],coords[None],density_grad=True)[0]
   ao=dft.numint.eval_ao(mol,coords/bohr,deriv=1)
   dm_pack=dft.numint.eval_rho(mol,ao,c['density_matrix'],xctype='GGA',hermi=1).T
   dm_error=float(np.max(np.abs(dm_pack-pack)))
   if dm_error>1e-10:raise ValueError('saved DM and MO density disagreement')
   full_pack=sample_qc_results([(m,c)],full_coords[None],density_grad=False)[0]
   electrons=float(full_pack@full_weights)
   if abs(electrons-10)>.02:raise ValueError('full-grid molecular electron count')
   sad,_=sample_sad_coeff_density(pos[None],num[None],coords[None],atomic,coefficient_type='mo_coeffs',density_grad=True)
   row={'positions':(pos[permutation]-shift)[None],'atom_numbers':num[permutation][None].astype(np.int64),'coords':(coords-shift)[None],'coord_weights':(weights*len(full_coords)/args.count)[None],'coord_weights_original':weights[None],'density':pack[None,:,0],'density_grad':pack[None,:,1:],'atom_density':sad[...,0],'atom_density_grad':sad[...,1:],'energy':np.array([[float(c['energy'])]]),'forces':np.asarray(c['forces'])[permutation][None],'idx':np.array([[index]],dtype=np.int64),'pos_shift':(-shift)[None,None,:],'grid_indices':indices[None],'full_grid_count':np.array([[len(full_coords)]],dtype=np.int64)}
   check={'index':index,'energy_archive_abs':energy_archive_error,'force_archive_max_abs':force_archive_error,'density_dm_max_abs':dm_error,'full_grid_count':len(full_coords),'full_grid_electrons':electrons,'sampled_grid_electrons':float(row['density'][0]@row['coord_weights'][0]),'original_subset_electrons':float(pack[:,0]@weights),'force_sum_max_abs':float(np.max(np.abs(np.sum(c['forces'],axis=0))))}
   if index in [0,50,99]:
    fd=[];delta=1e-6
    # Full sampled grid verifies derivatives also near dense core nodes.
    for axis in range(3):
     plus=coords.copy();minus=coords.copy();plus[:,axis]+=delta*bohr;minus[:,axis]-=delta*bohr
     rp=sample_qc_results([(m,c)],plus[None])[0];rm=sample_qc_results([(m,c)],minus[None])[0]
     fd.append((rp-rm)/(2*delta))
    fd_error=float(np.max(np.abs(np.stack(fd,axis=-1)-pack[:,1:])))
    check['gradient_fd_max_abs']=fd_error
    check['gradient_fd_step_bohr']=delta
    if fd_error>1e-4:raise ValueError('density gradient finite difference')
   if index==0:
    mf=dft.RKS(mol);mf.chkfile=None;mf.xc='pbe';mf.conv_tol=1e-10;mf.kernel()
    fresh_force=-mf.nuc_grad_method().kernel()/bohr
    check['fresh_scf_converged']=bool(mf.converged);check['fresh_energy_abs']=abs(float(mf.e_tot)-float(c['energy']));check['fresh_force_max_abs']=float(np.max(np.abs(fresh_force-c['forces'])))
    if not mf.converged or check['fresh_energy_abs']>2e-6 or check['fresh_force_max_abs']>2e-5:raise ValueError('fresh QC energy/force gate '+str(check))
   for key,value in row.items():
    if not np.isfinite(value).all():raise ValueError('nonfinite '+key)
   if np.min(row['density']) < -1e-12 or np.min(row['atom_density']) < -1e-12:raise ValueError('negative density')
   rows.append(row);checks.append(check)
   support.append({'positions_qc_angstrom':pos,'mo_coeff':c['mo_coeff'],'mo_occ':c['mo_occ'],'density_matrix':c['density_matrix'],'df_coeff_qc_ao':c['df_coeff']})
   if index%10==0:print(sp,index,'electrons',electrons,'DM',dm_error,flush=True)
  arrays={key:np.concatenate([row[key] for row in rows],axis=0) for key in rows[0]}
  np.savez_compressed(out/f'{sp}.npz',**arrays)
  np.savez_compressed(out/f'{sp}_qc_support.npz',**{key:np.stack([row[key] for row in support]) for key in support[0]})
  size=5 if sp=='train' else 10
  batches=[{key:value[start:start+size] for key,value in arrays.items()} for start in range(0,len(rows),size)]
  save_tree(out/sp,{'batches':batches})
  loaded=load_batches(out/sp)
  if len(loaded)!=len(batches) or not np.array_equal(loaded[0]['density'],batches[0]['density']):raise ValueError('native reload')
  all_data[sp]=arrays
  report['splits'][sp]={'count':len(rows),'batch_count':len(batches),'shapes':{key:list(value.shape) for key,value in arrays.items()},'records':checks,'geometry_identities_sha256':identities[sp]}
 train=all_data['train']
 stats={'energy_offset_hartree':float(train['energy'].mean()),'energy_std_hartree':float(train['energy'].std()),'density_rms':float(np.sqrt(np.mean(train['density']**2))),'density_residual_rms':float(np.sqrt(np.mean((train['density']-train['atom_density'])**2))),'density_grad_rms':float(np.sqrt(np.mean(train['density_grad']**2))),'density_grad_residual_rms':float(np.sqrt(np.mean((train['density_grad']-train['atom_density_grad'])**2))),'force_rms_hartree_per_angstrom':float(np.sqrt(np.mean(train['forces']**2))),'fitting_split':'train','sampled_points_per_geometry':args.count}
 write_json(out/'normalization.json',stats);report['normalization']=stats
 report['torch_imported']='torch' in sys.modules
 if report['torch_imported']:raise RuntimeError('unexpected Torch import')
 report['output_sha256']={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='report.json'}
 report['passed']=True;write_json(out/'report.json',report)
 print(json.dumps({'passed':True,'output':str(out.resolve()),'normalization':stats,'atomic':atomic_checks,'split_overlap':split_audit}),flush=True)

if __name__=='__main__':main()
