"""Summarize completed, numerically gated paired GPU measurements."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument('folder',type=Path);a=p.parse_args()
    cells=json.loads((a.folder/'register.json').read_text()); summaries=[]
    def resolve(name):
        path=Path(name)
        return path if path.is_absolute() else a.folder/path
    for cell in cells:
        summary={'case':cell['case'],'mode':cell['mode'],'status':cell['status'],
                 'completed_pairs':sum('ratio_torch_over_jax' in r for r in cell['rounds'])}
        if cell['status']=='complete':
            ratios=[r['ratio_torch_over_jax'] for r in cell['rounds']]
            summary.update(ratio_median=statistics.median(ratios),ratio_min=min(ratios),ratio_max=max(ratios))
            for backend in ('torch','jax'):
                records=[json.loads(resolve(r['arms'][backend]['file']).read_text()) for r in cell['rounds']]
                samples=[t for r in records for t in r['times_ms']]
                quartiles=statistics.quantiles(samples,n=4,method='inclusive')
                summary[backend]={'median_ms':statistics.median(samples),'q25_ms':quartiles[0],
                    'q75_ms':quartiles[2],'min_ms':min(samples),'max_ms':max(samples),
                    'setup_median_seconds':statistics.median(r['setup_seconds'] for r in records),
                    'fresh_process_first_call_median_seconds':statistics.median(r['cold_first_call_seconds'] for r in records),
                    'first_observed_call_seconds':records[0]['cold_first_call_seconds'],
                    'peak_live_bytes_max':max((r['memory_after_timing']['peak_live_bytes'] or 0) for r in records),
                    'peak_host_rss_kib_max':max(r['host_peak_rss_kib'] for r in records)}
            savings=(summary['torch']['median_ms']-summary['jax']['median_ms'])/1000
            extra_cold=summary['jax']['fresh_process_first_call_median_seconds']-summary['torch']['fresh_process_first_call_median_seconds']
            extra_total=extra_cold+summary['jax']['setup_median_seconds']-summary['torch']['setup_median_seconds']
            summary['break_even_calls_fresh_process_first_call']=max(0,extra_cold)/savings if savings>0 else None
            summary['break_even_calls_fresh_process_setup_and_first_call']=max(0,extra_total)/savings if savings>0 else None
            cold_path=a.folder.parent/'cold-isolated'/f"wide-{cell['mode']}-cold.json"
            if not cold_path.exists():
                cold_path=a.folder/'cold'/f"wide-{cell['mode']}-cold.json"
            if cell['case']=='wide' and cold_path.exists():
                isolated=json.loads(cold_path.read_text())
                summary['isolated_cold']=isolated
                extra=isolated['setup_seconds']+isolated['cold_first_call_seconds']-summary['torch']['setup_median_seconds']-summary['torch']['fresh_process_first_call_median_seconds']
                summary['break_even_calls_isolated_setup_and_compile']=max(0,extra)/savings if savings>0 else None
            rows=[]
            for pair in cell['rounds']:
                gate=pair['gate']
                if 'rows' not in gate:
                    gate=json.loads(resolve(pair['gate_path']).read_text())
                rows.extend(gate['rows'])
            summary['gate_max_abs']=max(r['max_abs'] for r in rows)
        summaries.append(summary)
    (a.folder/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    for s in summaries:
        if s['status']=='complete':
            print(f"{s['case']:6} {s['mode']:20} Torch {s['torch']['median_ms']:.3f}ms JAX {s['jax']['median_ms']:.3f}ms ratio {s['ratio_median']:.2f}x [{s['ratio_min']:.2f},{s['ratio_max']:.2f}] first-call {s['jax']['fresh_process_first_call_median_seconds']:.1f}s peak {s['jax']['peak_live_bytes_max']/2**20:.1f}MiB")
        else: print(s['case'],s['mode'],s['status'],s['completed_pairs'])

if __name__=='__main__':main()
