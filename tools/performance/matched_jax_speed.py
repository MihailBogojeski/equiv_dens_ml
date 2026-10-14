"""Serial fresh-runtime alternating pairs; censor failed cells without ranking."""
import argparse,hashlib,json,os,subprocess,time,statistics
from pathlib import Path
import numpy as np
from equiv_dens_jax.serialization import load_tree


def compare(a,b):
    rows=[]
    def visit(x,y,path=''):
        if isinstance(x,dict):
            for k in x:
                if k in y: visit(x[k],y[k],path+'/'+k)
                elif x[k] is not None: raise AssertionError('Missing '+path+'/'+k)
        elif x is not None:
            x,y=np.asarray(x),np.asarray(y)
            if x.shape!=y.shape: raise AssertionError('Shape mismatch '+path)
            error=float(np.max(np.abs(x-y),initial=0)); scale=float(np.max(np.abs(x),initial=0))
            passed=bool(np.allclose(x,y,rtol=1e-8,atol=1e-8))
            rows.append({'field':path,'max_abs':error,'max_scale':scale,'pass':passed})
    visit(a,b)
    return {'pass':bool(rows) and all(r['pass'] for r in rows),'rows':rows}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['tiny','wide','mixed'])
    p.add_argument('--modes',nargs='+',default=['density_update','energy_force_update','ef','calculator'])
    p.add_argument('--timeout',type=int,default=600);args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    subprocess.run(['tools/runtime/run.sh','torch','python','-m','benchmarks.jax_backend_speed',
                    '--generate',str(root/'fixtures')],check=True)
    registry=[]
    for case in args.cases:
        for mode in args.modes:
            cell={'case':case,'mode':mode,'rounds':[],'status':'running'};registry.append(cell)
            for pair in range(5):
                row={'pair':pair,'arms':{}}
                cell['rounds'].append(row)
                for backend in (('torch','jax') if pair%2==0 else ('jax','torch')):
                    output=root/f'{case}-{mode}-{pair}-{backend}.json'
                    command=['tools/runtime/run.sh',backend,'env',
                             'SLURM_JOB_ID='+os.environ.get('SLURM_JOB_ID',''),
                             'JAX_PLATFORMS=cuda','JAX_ENABLE_X64=true',
                             'XLA_PYTHON_CLIENT_PREALLOCATE=false','XLA_PYTHON_CLIENT_ALLOCATOR=default',
                             'python','-u','-m','benchmarks.jax_backend_speed',
                             '--backend',backend,'--fixture',str(root/'fixtures'/case),'--mode',mode,'--output',str(output)]
                    print('START',case,mode,pair,backend,flush=True)
                    t=time.monotonic()
                    with output.with_suffix('.log').open('w') as log:
                        try:
                            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                            status='ok' if result.returncode==0 else f'exit-{result.returncode}'
                        except subprocess.TimeoutExpired: status='timeout'
                    row['arms'][backend]={'status':status,'wall_seconds':time.monotonic()-t,'file':str(output)}
                    if status!='ok': cell['status']=status;break
                if cell['status']!='running': break
                gate=compare(load_tree(Path(row['arms']['torch']['file']).with_suffix('')),
                             load_tree(Path(row['arms']['jax']['file']).with_suffix('')))
                row['gate']=gate
                (root/f'{case}-{mode}-{pair}-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
                if not gate['pass']:cell['status']='numerical-gate-failed';break
                tdata=json.loads(Path(row['arms']['torch']['file']).read_text())
                jdata=json.loads(Path(row['arms']['jax']['file']).read_text())
                row['torch_ms']=statistics.median(tdata['times_ms']);row['jax_ms']=statistics.median(jdata['times_ms'])
                row['ratio_torch_over_jax']=row['torch_ms']/row['jax_ms']
                print('PAIR',case,mode,pair,row['ratio_torch_over_jax'],flush=True)
                (root/'register.json').write_text(json.dumps(registry,indent=2)+'\n')
            if cell['status']=='running':cell['status']='complete'
            (root/'register.json').write_text(json.dumps(registry,indent=2)+'\n')
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
