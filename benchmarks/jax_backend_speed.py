"""Matched native backend instrument: portable initialized tensors, FP64 SGD."""
from __future__ import annotations
import argparse, hashlib, json, os, resource, subprocess, time
from pathlib import Path
import numpy as np

CASES = {'tiny': ('tiny',4,1,1,32), 'wide': ('wide',16,2,2,256),
         'mixed': ('batch',16,2,3,512)}
MODES = ('density_update','energy_force_update','ef','calculator')
LR = 1e-5


def numpy_tree(value):
    if hasattr(value,'detach'):
        return value.detach().cpu().numpy().copy()
    if hasattr(value,'dtype') and hasattr(value,'shape'):
        return np.asarray(value).copy()
    if isinstance(value,dict): return {k:numpy_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [numpy_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(numpy_tree(v) for v in value)
    return value


def generate(root):
    import torch
    from benchmarks.fixtures import fixture_spec,build_model,make_atoms,tensor_hash
    from benchmarks.precision import use_full_precision
    from equiv_dens_jax.convert import export_module
    from equiv_dens_jax.serialization import save_tree
    use_full_precision(); torch.set_num_threads(1)
    for case,(layout,f,m,l,g) in CASES.items():
        spec=fixture_spec(layout,torch.float64,num_features=f,num_modules=m,order=l)
        model=build_model(spec,'ef').eval(); model.no_force_props=['density']
        atoms=make_atoms(layout,torch.float64,grid_points=g)
        pred=model({k:v.clone() if torch.is_tensor(v) else v for k,v in atoms.items()})
        for field in ('density','density_grad','energy','forces'):
            offset=torch.linspace(.011,.029,pred[field].numel(),dtype=torch.float64).reshape(pred[field].shape)
            atoms[field]=pred[field].detach()+offset
        node,state,skipped=export_module(model)
        save_tree(root/case,{'model':node,'state':state,'inputs':numpy_tree(atoms),
                  'torch_state':numpy_tree(model.state_dict()),'spec':spec,
                  'metadata':{'case':case,'input_hash':tensor_hash(atoms),
                              'state_hash':tensor_hash(model.state_dict()),'skipped':skipped}})
        print('generated',case,flush=True)


def host_frame(inputs,index):
    from equiv_dens_jax.data import prepare_atoms
    z=inputs['batch_atom_numbers']; r=inputs['batch_positions'].copy()
    delta=.001*np.sin(index+np.arange(r.size).reshape(r.shape)*.37)
    r=np.where(z[...,None]>0,r+delta,0.)
    center=(r*z[...,None]).sum(1)/z.sum(1)[:,None]
    r=np.where(z[...,None]>0,r-center[:,None],0.)
    atoms=prepare_atoms({'atom_numbers':z,'positions':r})
    # Supplied deterministic grid is transferred, without PySCF/SCF grid generation.
    atoms['coords']=inputs['coords'].copy(); atoms['coord_weights']=inputs['coord_weights'].copy()
    return atoms


def provenance():
    return {'host':os.uname().nodename,'allocation':os.getenv('SLURM_JOB_ID'),
            'runtime':os.getenv('APPTAINER_CONTAINER'),'source_revision':subprocess.check_output(
            ['git','rev-parse','HEAD'],text=True).strip(),
            'precision':'float64','learning_rate':LR,'optimizer':'SGD, no momentum/decay/EMA/Lookahead',
            'instrument_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def worker(args):
    from equiv_dens_jax.serialization import load_tree,save_tree
    bundle=load_tree(args.fixture); inputs=bundle['inputs']; mode=args.mode
    update=mode.endswith('_update'); density=mode=='density_update'
    fields=('density','density_grad') if density else ('energy','forces')
    weights=(1.,.1) if density else (.2,.3)
    prefix='density_repr_model.' if density else 'property_models.energy.'
    started=time.perf_counter()
    if args.backend=='torch':
        import torch
        from benchmarks.fixtures import build_model
        from benchmarks.precision import use_full_precision
        use_full_precision(); torch.set_num_threads(1)
        model=build_model(bundle['spec'],'density' if density else 'ef').cuda().train(update)
        model.load_state_dict({k:torch.as_tensor(v,device='cuda') for k,v in bundle['torch_state'].items()},strict=True)
        for name,p in model.named_parameters(): p.requires_grad_(update and name.startswith(prefix))
        active={k:p for k,p in model.named_parameters() if p.requires_grad}
        opt=torch.optim.SGD(active.values(),lr=LR) if update else None
        def arrays(tree):
            if isinstance(tree,np.ndarray): return torch.as_tensor(tree,device='cuda')
            if isinstance(tree,dict): return {k:arrays(v) for k,v in tree.items()}
            return tree
        data=arrays(inputs)
        def sync(value=None): torch.cuda.synchronize()
        def execute(index=0):
            atoms=arrays(host_frame(inputs,index)) if mode=='calculator' else dict(data)
            if update: opt.zero_grad(set_to_none=True)
            out=model(atoms); result={f:out[f] for f in fields}
            if update:
                loss=sum(w*(out[f]-data[f]).square().mean() for f,w in zip(fields,weights))
                loss.backward(); opt.step(); result['loss']=loss.detach()
            if mode=='calculator': return numpy_tree(result)
            return result
        def parameters(): return {k:p for k,p in model.named_parameters()}
        def gradients(): return {k:p.grad for k,p in active.items() if p.grad is not None}
        def memory(): return {'live_bytes':torch.cuda.memory_allocated(),
                              'peak_live_bytes':torch.cuda.max_memory_allocated(),
                              'reserved_bytes':torch.cuda.memory_reserved()}
        torch.cuda.reset_peak_memory_stats()
        backend_meta={'version':torch.__version__,'matmul_precision':'highest; TF32 disabled',
                      'device':torch.cuda.get_device_name()}
    else:
        import jax
        jax.config.update('jax_enable_x64',True)
        jax.config.update('jax_default_matmul_precision','highest')
        import jax.numpy as jnp
        from equiv_dens_jax.pipeline import Pipeline
        pipeline=Pipeline(bundle['model'],bundle['state'])
        data=pipeline.prepare(inputs)
        params=pipeline.state['params']; buffers=pipeline.state['buffers']
        active_names=tuple(k for k in params if k.startswith(prefix))
        def core(parameters,atoms):
            out=pipeline.forward({'params':parameters,'buffers':buffers},atoms,training=update,
                                 properties=('density',) if density else ('energy',),forces=not density)
            result={f:out[f] for f in fields}
            if update:
                result['loss']=sum(w*jnp.mean((out[f]-atoms[f])**2) for f,w in zip(fields,weights))
            return result
        if update:
            @jax.jit
            def operation(parameters,atoms):
                active={k:parameters[k] for k in active_names}
                def objective(value):
                    result=core(dict(parameters,**value),atoms)
                    return result['loss'],result
                (_,result),grad=jax.value_and_grad(objective,has_aux=True)(active)
                new=dict(parameters)
                new.update({k:parameters[k]-LR*v for k,v in grad.items()})
                return new,result,grad
        else: operation=jax.jit(core)
        last_grad={}
        def sync(value=None):
            if value is not None:
                jax.block_until_ready((value,params,last_grad) if update else value)
            else: jax.effects_barrier()
        def execute(index=0):
            nonlocal params,last_grad
            atoms=pipeline.prepare(host_frame(inputs,index)) if mode=='calculator' else data
            if update: params,result,last_grad=operation(params,atoms)
            else: result=operation(params,atoms)
            if mode=='calculator': return numpy_tree(result)
            return result
        def parameters(): return params
        def gradients(): return last_grad
        def memory():
            stats=jax.devices()[0].memory_stats() or {}
            return {'live_bytes':stats.get('bytes_in_use'),'peak_live_bytes':stats.get('peak_bytes_in_use'),
                    'native_stats':stats}
        backend_meta={'version':jax.__version__,'matmul_precision':'highest','device':str(jax.devices()[0])}
    setup_seconds=time.perf_counter()-started
    cold_start=time.perf_counter(); initial=execute(0); sync(initial)
    cold_seconds=time.perf_counter()-cold_start
    # First call diagnostics precede warmup/timing and compare exact initialized tensors.
    save_tree(args.output.with_suffix(''),{'outputs':numpy_tree(initial),
        'one_step_params':numpy_tree(parameters()) if update else {},
        'gradients':numpy_tree(gradients()) if update else {}})
    for i in range(args.warmup): sync(execute(i+1))
    before=memory(); times=[]
    for i in range(args.iterations):
        t=time.perf_counter(); result=execute(i+args.warmup+1); sync(result)
        times.append((time.perf_counter()-t)*1000.)
    final=numpy_tree(result)
    if not all(np.isfinite(v).all() for v in final.values()): raise AssertionError('Nonfinite outputs')
    if update:
        final_params=numpy_tree(parameters())
        if not all(np.isfinite(v).all() for v in final_params.values()):
            raise AssertionError('Nonfinite updated parameters')
        for name,value in final_params.items():
            if not name.startswith(prefix) and not np.array_equal(value,bundle['state']['params'][name]):
                raise AssertionError('Frozen parameter changed: '+name)
        if not any(not np.array_equal(value,bundle['state']['params'][name])
                   for name,value in final_params.items() if name.startswith(prefix)):
            raise AssertionError('Requested parameter stage did not update')
    record={'backend':args.backend,'mode':mode,'case':bundle['metadata']['case'],
            'times_ms':times,'setup_seconds':setup_seconds,'cold_first_call_seconds':cold_seconds,
            'memory_before_timing':before,'memory_after_timing':memory(),
            'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'metadata':bundle['metadata'],'backend_metadata':backend_meta,'provenance':provenance()}
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'case':record['case'],'mode':mode,'backend':args.backend,
                      'median_ms':float(np.median(times)),'cold_seconds':cold_seconds}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',type=Path)
    parser.add_argument('--backend',choices=('torch','jax'))
    parser.add_argument('--fixture',type=Path); parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--output',type=Path); parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--iterations',type=int,default=10)
    args=parser.parse_args()
    if args.generate: generate(args.generate)
    else: worker(args)

if __name__=='__main__': main()
