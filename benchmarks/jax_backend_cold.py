"""Cache-disabled cold probe using the immutable matched instrument unchanged."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path


def main():
    started=time.perf_counter()
    import jax
    jax.config.update('jax_enable_compilation_cache',False)
    setup_extra=time.perf_counter()-started
    from benchmarks import jax_backend_speed
    jax_backend_speed.main()
    import sys
    output=Path(sys.argv[sys.argv.index('--output')+1])
    data=json.loads(output.read_text())
    data['setup_seconds']+=setup_extra
    data['cold_cache_control']={'enable_compilation_cache':jax.config.jax_enable_compilation_cache,
       'configured_directory':os.getenv('JAX_COMPILATION_CACHE_DIR'),
       'preimport_setup_seconds_added':setup_extra,
       'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
