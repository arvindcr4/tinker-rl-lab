"""Review27-v3: Gaussian likelihood and explicit model definitions.
Information criteria include residual variance; changepoint search penalties remain approximate.
"""
import math
import numpy as np
ANALYSIS_VERSION = 'review27-v3'
FAMILY_SPEC = {
 'constant': {'equation':'mu', 'mean_parameters':1},
 'linear': {'equation':'a+b*t', 'mean_parameters':2},
 'saturation': {'equation':'R*(-expm1(-lambda*t))', 'mean_parameters':2},
 'power_law': {'equation':'a*t**b', 'mean_parameters':2},
 'log_time': {'equation':'a+b*log10(t)', 'mean_parameters':2},
 'ar1': {'equation':'c+phi*y_previous', 'mean_parameters':2},
 'continuous_3seg': {'equation':'a+b*t+d*max(t-c1,0)+e*max(t-c2,0)', 'mean_parameters':6},
}
def information_criteria(rss,n,k):
    """k includes variance; ordinary AIC/BIC, not AICc. No short-trace victory."""
    if not math.isfinite(rss) or rss<0 or n<=k: return float('nan'),float('nan')
    # Common deterministic roundoff floor, recorded rather than log(0).
    dev=n*(math.log(2*math.pi*max(rss/n,1e-15))+1)
    return dev+2*k,dev+k*math.log(n)
def signed_phases(slopes):
    a,b,c=slopes
    return bool(a>=0 and b>a and b>0 and c>=0 and c<b and c<=a)

def best_candidate(scores):
    valid={k:v for k,v in scores.items() if math.isfinite(v)}
    return min(valid,key=valid.get) if valid else 'no_valid_candidate'

def write_manifest(script, trace_dir, output_dir):
    """Provenance for explicitly regenerated v3 analyses only."""
    import hashlib, json, subprocess, importlib.metadata
    from pathlib import Path
    script=Path(script)
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(trace_dir).glob('*.json'))}
    try: revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=script.parent,text=True).strip()
    except Exception: revision=None
    data={'analysis_version':ANALYSIS_VERSION,'interpretation':'exploratory; no frozen evaluation protocol',
          'input_inventory_sha256':inputs,'code_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
          'revision':revision,'family_spec':FAMILY_SPEC,'variance_parameter_included':True,
          'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','scipy']},
          'input_inventory_scope':'available traces; analysis-specific included/excluded rows remain in result tables',
          'seed':42 if script.stem.endswith(('73','77','89','93','97')) else None,
          'uncertainty': {'iter89':'fixed-design circular residual blocks around continuous trend; conditional exploratory uncertainty', 'iter93':'recursive iid innovation bootstrap conditional on fitted AR1 model', 'iter77':'parametric predictive interval includes new observation noise'},
          'equivalence':'not assessed; no prespecified margin'}
    path=Path(output_dir)/(script.stem+'_v3_manifest.json')
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def configure_paths(namespace, argv=None):
    """Explicit locations; defaults preserve historical Modal inputs, never guess evidence."""
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Exploratory review27-v3 analysis; does not establish prospective validation')
    for flag,key in [('trace-dir','TRACE_DIR'),('output-dir','RESULTS_DIR'),('figure-dir','FIG_DIR'),('paper-figure-dir','PAPER_FIG')]:
        parser.add_argument('--'+flag,type=Path,default=namespace.get(key))
    args=parser.parse_args(argv)
    for flag,key in [('trace_dir','TRACE_DIR'),('output_dir','RESULTS_DIR'),('figure_dir','FIG_DIR'),('paper_figure_dir','PAPER_FIG')]:
        value=getattr(args,flag)
        if value is not None:
            namespace[key]=value.resolve()
            if key!='TRACE_DIR': namespace[key].mkdir(parents=True,exist_ok=True)
    if not namespace['TRACE_DIR'].is_dir(): parser.error('trace directory does not exist')
    return args
