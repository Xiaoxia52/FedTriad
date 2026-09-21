"""18 frozen PSL runs with full risk-fusion validation at 1,15,...,300."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parent
CALLER_DIRECTORY = Path.cwd()
REPOSITORY = ROOT.parents[1]
os.chdir(ROOT)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def export_results():
    from fedtriad_curves.curve_eval import ROUNDS
    rows = []
    directories = sorted((ROOT / 'runs').glob('*'))
    for directory in directories:
        if not directory.is_dir():
            continue
        result = read(directory / 'results.json')
        if result.get('status') != 'complete':
            raise RuntimeError('Incomplete run: ' + str(directory))
        points = sorted((directory / 'curve_points').glob('round_*.json'))
        if [read(p)['round'] for p in points] != ROUNDS:
            raise RuntimeError('Missing curve points: ' + str(directory))
        for point in points:
            item = read(point)
            for policy in ('global_class_risk','uniform'):
                metrics = item['client_validation'][policy]['aggregate']
                rows.append({k: item[k] for k in ('dataset','alpha','seed','round')} | {
                    'method': 'FedTriad' if policy == 'global_class_risk' else 'Uniform P/S/L',
                    'macro_f1': metrics['macro_f1_mean'], 'accuracy': metrics['accuracy_mean'],
                    'nll': metrics['nll_mean']})
    if len(directories) != 18 or len(rows) != 18 * 21 * 2:
        raise RuntimeError('Expected 18 completed runs and 378 full-FedTriad points')
    out = ROOT / 'results_export'
    out.mkdir(exist_ok=True)
    def csv_write(path, entries):
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(entries[0])); writer.writeheader(); writer.writerows(entries)
    csv_write(out / 'curves_per_seed.csv', rows)
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ('dataset','alpha','round','method'))
        groups.setdefault(key,[]).append(row)
    summaries=[]
    for key, entries in sorted(groups.items()):
        if sorted(e['seed'] for e in entries) != [0,1,2]:
            raise RuntimeError('Seed coverage mismatch')
        r=dict(zip(('dataset','alpha','round','method'),key))
        for metric in ('macro_f1','accuracy','nll'):
            vals=[e[metric] for e in entries]
            r[metric+'_mean']=statistics.mean(vals)
            r[metric+'_sample_std']=statistics.stdev(vals)
        summaries.append(r)
    csv_write(out / 'curves_summary.csv',summaries)
    archive=ROOT / 'FedTriad_CURVES_RESULTS.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('runs','results_export','protocol'):
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file() and p.suffix in ('.json','.csv','.log'):
                    z.write(p,p.relative_to(ROOT))
        for name in ('README.md','requirements.txt','PACKAGE_MANIFEST.json'):
            p=ROOT/name
            if p.exists(): z.write(p,p.name)
        for p in sorted(ROOT.glob('*.py')): z.write(p,p.name)
        for p in sorted((ROOT/'fedtriad_curves').glob('*.py')): z.write(p,p.relative_to(ROOT))
    print('RETURN THIS FILE: ' + str(archive), flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default=str(REPOSITORY / 'data'),help='Directory containing the three original NPZ files; defaults to repository/data')
    p.add_argument('--job','--jobs','--jobx',dest='jobs',type=int,default=6)
    p.add_argument('--preview',action='store_true')
    p.add_argument('--export-only',action='store_true')
    args=p.parse_args()
    data_directory = Path(args.data_dir).expanduser()
    if not data_directory.is_absolute():
        data_directory = CALLER_DIRECTORY / data_directory
    data_directory = data_directory.resolve()
    if not 1 <= args.jobs <= 9: p.error('job must be 1..9')
    if args.export_only: export_results(); return
    import numpy as np
    import torch
    from fedtriad_curves.data import inspect_npz, locate, prepare
    from fedtriad_curves.partition import build_partition
    from fedtriad_curves.suite import expand_suite, execute_suite
    from launcher_base import _prepare_resume_outputs
    spec=read(ROOT/'protocol/suite.json')
    spec['variants']=['psl_uniform']
    spec['output_root']='runs'
    # Relative paths keep experiment IDs portable when moving the whole package.
    for ds in spec['datasets']:
        spec['datasets'][ds]={'data_file':'','data_root':str(data_directory)}
    configs=expand_suite(spec)
    print('18 jobs; 300 rounds; full fusion at 1,15,30,...,300; job=%s' % args.jobs,flush=True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA PyTorch and NVIDIA GPU required')
    print('GPU: '+torch.cuda.get_device_name(0),flush=True)
    prepared={}
    for c in configs:
        if c.dataset not in prepared:
            actual=inspect_npz(locate(c))
            expected=read(ROOT/'protocol'/ (c.dataset+'_data.json'))
            if actual['arrays'] != expected['arrays']:
                raise RuntimeError('Original NPZ content mismatch: '+c.dataset)
            prepared[c.dataset]=prepare(c)
        cache,_,data_id=prepared[c.dataset]
        partition=build_partition(np.load(cache/'train_labels.npy'),np.load(cache/'val_labels.npy'),c,data_id)
        refs=list((ROOT/'protocol').glob('%s_fedtriad_psl_uniform_a%s_seed%s_*.json' % (c.dataset,c.alpha,c.seed)))
        if len(refs)!=1: raise RuntimeError('Missing frozen partition')
        frozen=read(refs[0])
        for field in ('train','val','local_test','calibration'):
            if partition[field]!=frozen[field]: raise RuntimeError('Partition mismatch '+c.output_dir+' '+field)
    print('Preflight OK: original NPZ arrays and all 18 partitions verified.',flush=True)
    if args.preview: return
    _prepare_resume_outputs(configs)
    execute_suite(spec,resume=True,jobs=args.jobs)
    export_results()

if __name__ == '__main__':
    main()
