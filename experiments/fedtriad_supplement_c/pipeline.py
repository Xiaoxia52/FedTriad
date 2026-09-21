"""Disk-backed queue, bounded GPU concurrency, resume and evidence summaries."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / 'runs/supplement_c_300r'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def alive(pid):
    if not pid:
        return False
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    h = kernel.OpenProcess(0x00100000, False, pid)
    if not h:
        return ctypes.get_last_error() == 5
    try:
        return kernel.WaitForSingleObject(h, 0) == 0x102
    finally:
        kernel.CloseHandle(h)


def acquire(path):
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text(encoding='utf-8'))
            except ValueError:
                raise RuntimeError('Unrecognized lock file: ' + str(path))
            if alive(pid):
                raise RuntimeError(f'Another queue is active: PID {pid}')
            path.unlink()
            continue
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        return
    raise RuntimeError('Cannot acquire queue lock')


def child_command(*args):
    return [sys.executable, '-B', '-u', '-X', 'utf8', str(PROJECT/'RUN_SUPPLEMENT_C.py'), *args]


def gpu_info():
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    parts = next(csv.reader(r.stdout.splitlines()))
    return {'index': int(parts[0]), 'name': parts[1].strip(),
            'total_mib': int(parts[2]), 'free_mib': int(parts[3]), 'used_mib': int(parts[4])}


def _dataset_file(data_dir, dataset):
    root = Path(data_dir).resolve()
    candidates = (root / (dataset + '.npz'), root / dataset / (dataset + '.npz'))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'Missing {dataset}.npz under {root}; pass --data-dir pointing to the NPZ directory'
    )


def _validate_risk_endpoint(risk_output, psl_run, psl_result):
    """Bind one risk endpoint to its actual PSL run and checkpoint.

    The published risk JSON stores the run directory, selected round, and
    checkpoint digest; newer summaries may also carry a partition identity.
    The PSL result, config, and partition files are checked first, then those
    endpoint fields are matched to the concrete run before the checkpoint
    bytes are hashed.
    """
    risk_output = Path(risk_output).resolve()
    if not risk_output.is_dir():
        raise RuntimeError('Risk-output directory not found: ' + str(risk_output))
    psl_run = Path(psl_run).resolve()
    psl_result = dict(psl_result)
    partition_path = psl_run / 'partition.json'
    config_path = psl_run / 'config.json'
    checkpoint_result = psl_run / 'results.json'
    if (not partition_path.is_file() or not config_path.is_file() or
            not checkpoint_result.is_file()):
        raise RuntimeError('PSL reference is missing config/partition/results: ' + str(psl_run))
    partition = read(partition_path)
    config = read(config_path)
    if not partition.get('partition_id') or partition.get('partition_id') != psl_result.get('partition_id'):
        raise RuntimeError('PSL partition identity mismatch: ' + str(psl_run))
    if psl_result.get('status') != 'complete' or psl_result.get('variant') != 'FedTriad-psl_uniform':
        raise RuntimeError('Reference is not a complete PSL result: ' + str(psl_run))
    expected = {
        'dataset': config.get('dataset'),
        'alpha': config.get('alpha'),
        'seed': config.get('seed'),
        'partition_id': partition['partition_id'],
        'selected_round': psl_result.get('selected_personal_round'),
    }
    if (not expected['dataset'] or expected['alpha'] is None or
            expected['seed'] is None or expected['selected_round'] is None or
            psl_result.get('dataset') != expected['dataset'] or
            psl_result.get('seed') != expected['seed']):
        raise RuntimeError('PSL reference config/results identity mismatch: ' + str(psl_run))
    matches = []
    for path in sorted(risk_output.glob('*.json')):
        try:
            value = read(path)
        except (OSError, ValueError):
            continue
        run_directory = str(value.get('run_directory', '')).replace('\\', '/')
        if (value.get('dataset') == expected['dataset'] and
                value.get('alpha') == expected['alpha'] and
                value.get('seed') == expected['seed'] and
                # v2 risk summaries predate a top-level partition_id.  The
                # run basename binds them to the partition-checked PSL run;
                # reject the field when a newer summary does provide it.
                (value.get('partition_id') is None or
                 value.get('partition_id') == expected['partition_id']) and
                value.get('selected_round') == expected['selected_round'] and
                Path(run_directory).name == psl_run.name and
                'memory' in value):
            matches.append((path, value))
    if len(matches) != 1:
        raise RuntimeError(
            'Expected one risk endpoint for %s, found %s' % (psl_run.name, len(matches))
        )
    risk_path, risk = matches[0]
    run_directory = str(risk.get('run_directory', '')).replace('\\', '/')
    if not run_directory or Path(run_directory).name != psl_run.name:
        raise RuntimeError('Risk endpoint run_directory does not bind to PSL run: ' + str(risk_path))
    if (risk.get('partition_id') is not None and
            risk.get('partition_id') != expected['partition_id']):
        raise RuntimeError('Risk endpoint partition does not bind to PSL run: ' + str(risk_path))
    if risk.get('selected_round') != expected['selected_round']:
        raise RuntimeError('Risk endpoint selected round does not match PSL checkpoint: ' + str(risk_path))
    checkpoint = risk.get('checkpoint')
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Risk endpoint has no checkpoint metadata: ' + str(risk_path))
    checkpoint_name = str(checkpoint.get('file', ''))
    if (not checkpoint_name or Path(checkpoint_name).name != checkpoint_name or
            '..' in Path(checkpoint_name).parts):
        raise RuntimeError('Unsafe risk checkpoint path: ' + str(risk_path))
    checkpoint_path = psl_run / checkpoint_name
    if not checkpoint_path.is_file():
        raise RuntimeError('Risk checkpoint is missing from PSL run: ' + str(checkpoint_path))
    actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if str(checkpoint.get('sha256', '')).lower() != actual_hash:
        raise RuntimeError('Risk checkpoint hash mismatch: ' + str(checkpoint_path))
    return risk_path, risk


def preflight(output, data_dir, base_runs, cache_root, risk_output):
    import numpy as np
    import torch
    from fedtriad.data import inspect_npz
    from fedtriad.partition import validate_partition
    from experiments.fedtriad_supplement_c.experiment import implementation_id
    if not torch.cuda.is_available():
        raise RuntimeError('This Python environment cannot use CUDA')
    if torch.cuda.get_device_properties(0).total_memory < 5 * 1024**3:
        raise RuntimeError('At least 5 GiB total GPU memory required for this tested profile')
    roots = Path(base_runs).resolve()
    cache_root = Path(cache_root).resolve()
    risk_output = Path(risk_output).resolve()
    if not roots.is_dir():
        raise RuntimeError('Base runs directory not found: ' + str(roots))
    jobs, datasets, matched_settings = [], {}, set()
    streams = {v: {} for v in ('p_only', 'psl_uniform')}
    for d in roots.iterdir():
        if not d.is_dir() or not (d/'config.json').exists():
            continue
        c = read(d/'config.json')
        if c.get('triad_variant') not in streams:
            continue
        key = (c['dataset'], c['alpha'], c['seed'])
        if key in streams[c['triad_variant']]:
            raise RuntimeError('Duplicate original run: ' + str(key))
        streams[c['triad_variant']][key] = (d,c)
    expected = {(d,a,s) for d in ('bloodmnist','organamnist','pathmnist')
                for a in (.1,.5) for s in (0,1,2)}
    for stream in streams.values():
        if set(stream) != expected:
            raise RuntimeError('Expected exactly 18 original runs per structural configuration')
    for key in sorted(expected):
        pd,pc = streams['p_only'][key]
        sd,sc = streams['psl_uniform'][key]
        part = read(pd/'partition.json')
        spart = read(sd/'partition.json')
        # Historical path/mtime changes alter data_id and partition_id even when
        # every sample index and its order are identical. Check semantics directly.
        partition_fields = (set(part) | set(spart)) - {'partition_id', 'data_id'}
        if any(part.get(k) != spart.get(k) for k in partition_fields):
            raise RuntimeError('Original P and PSL partitions differ: ' + str(key))
        fields = ['dataset','seed','split_seed','alpha','rounds','clients','participation_rate',
                  'batch_size','local_epochs','lr','momentum','weight_decay','image_size',
                  'width','feature_dim','lr_schedule','lr_decay_fractions','lr_decay_gamma',
                  'selection_metric','eval_every']
        if any(pc[k]!=sc[k] for k in fields):
            raise RuntimeError('Original P and PSL protocols differ: ' + str(key))
        if pc['rounds'] != 300 or pc['smoke_samples'] != 0 or pc['eval_every'] != 1:
            raise RuntimeError('Non-final source configuration')
        for d in (pd,sd):
            if read(d/'results.json')['status'] != 'complete':
                raise RuntimeError('Incomplete original reference: ' + str(d))
        _validate_risk_endpoint(risk_output, sd, read(sd/'results.json'))
        ds = pc['dataset']
        if ds not in datasets:
            source = _dataset_file(data_dir, ds)
            meta = inspect_npz(source)
            caches=[]
            for m in cache_root.glob('*/manifest.json'):
                cm=read(m)
                if (cm.get('dataset')==ds and cm.get('arrays')==meta['arrays'] and
                    cm.get('smoke_samples')==0 and cm.get('resize')=='none'):
                    caches.append(m.parent)
            if not caches:
                raise RuntimeError('Original full-data cache not found: '+ds)
            cache=caches[0]
            for split in ('train','val','test'):
                for kind in ('images','labels'):
                    arr=np.load(cache/(split+'_'+kind+'.npy'),mmap_mode='r',allow_pickle=False)
                    if len(arr)!=meta['arrays'][split+'_'+kind]['shape'][0]:
                        raise RuntimeError('Cache count mismatch')
            datasets[ds]={'cache':str(cache),'data':str(source),'arrays':meta['arrays']}
        validate_partition(part, datasets[ds]['arrays']['train_labels']['shape'][0],
                           datasets[ds]['arrays']['val_labels']['shape'][0])
        for variant,(d,c) in [('p_clip5',(pd,pc)),('p3_ensemble_clip5',(sd,sc))]:
            config=dict(c)
            config.update(data_file=datasets[ds]['data'],device='cuda:0',workers=0,cpu_threads=2)
            name=f'{ds}_{variant}_a{key[1]}_seed{key[2]}'
            config['output_dir']=str(output/name)
            original_partition = part if variant == 'p_clip5' else spart
            jobs.append({'key':name,'variant':variant,'config':config,
                         'partition_file':str(d/'partition.json'),'partition_id':original_partition['partition_id'],
                         'cache':datasets[ds]['cache'],'reference_run':str(d),
                         'init_seeds':[c['seed']+1000003*i for i in range(1 if variant=='p_clip5' else 3)]})
    # All 18 single-P jobs first, followed by all 18 ensemble jobs.
    jobs.sort(key=lambda j:(j['variant']!='p_clip5',j['config']['dataset'],j['config']['alpha'],j['config']['seed']))
    value={'protocol':'scheme-c-v1','jobs':jobs,'datasets':datasets,
           'base_runs':str(roots),'risk_output':str(risk_output),
           'implementation_id':implementation_id(PROJECT),
           'runtime':{'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
                      'gpu':torch.cuda.get_device_name(0)},'total_jobs':36,
           'matched_dimensions':['original partition','client participation','batch order',
              'rounds and local epochs','per-branch sample visits','backbone','joint clip threshold'],
           'unmatched_dimensions':['communication volume','persistent model count','official inference model count'],
           'selection':'per-round mean client validation Macro-F1 for the combined predictor'}
    target=output/'queue.json'
    if target.exists() and read(target)!=value:
        raise RuntimeError('Queue or runtime changed; refusing incompatible resume')
    write(target,value)
    print(f'Preflight OK: {len(jobs)} jobs, original partitions verified, GPU {value["runtime"]["gpu"]}',flush=True)


def worker(queue_file, key, output, stop_after):
    from experiments.fedtriad_supplement_c.experiment import execute
    q=read(queue_file)
    job=next(j for j in q['jobs'] if j['key']==key)
    try:
        execute(job,output,PROJECT,stop_after=stop_after)
    except Exception as e:
        detail=traceback.format_exc()
        print(detail,flush=True)
        write(Path(output)/'failure.json',{'error':str(e),'traceback':detail,'time':time.time()})
        if 'out of memory' in str(e).lower():
            return 86
        return 1
    return 0


def csv_write(path, rows):
    if not rows:
        raise RuntimeError('No eligible results')
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(output):
    q=read(output/'queue.json')
    rows=[]
    for job in q['jobs']:
        p=output/job['key']/'results.json'
        if not p.exists() or read(p).get('status')!='complete':
            raise RuntimeError('Cannot mark complete: '+job['key'])
        r=read(p)
        if r['trained_rounds']!=300 or r['implementation_id']!=q['implementation_id']:
            raise RuntimeError('Unexpected completion identity')
        local=r['local_test']['aggregate'];official=r['official_test']
        rows.append({'dataset':r['dataset'],'alpha':r['alpha'],'seed':r['seed'],
            'method':r['variant'],'selected_round':r['selected_round'],
            'local_f1':local['macro_f1_mean'],'local_nll':local['nll_mean'],
            'worst_client_f1':local['macro_f1_min'],
            'official_f1':official['macro_f1'],'official_nll':official['nll'],
            'source':str(p)})
    # Attach original reference endpoints, all selected by client validation.
    base_runs = Path(q['base_runs'])
    risk_output = Path(q['risk_output'])
    for job in q['jobs']:
        if job['variant']!='p_clip5':
            continue
        references = {}
        for variant,label in [('p_only','P_original_unclipped'),('psl_uniform','PSL_uniform_original')]:
            candidates=[p for p in base_runs.glob(
                f"{job['config']['dataset']}_fedtriad_{variant}_a{job['config']['alpha']}_seed{job['config']['seed']}_*") if (p/'results.json').exists()]
            if len(candidates)!=1:
                raise RuntimeError('Ambiguous original reference')
            reference_run = candidates[0]
            p=reference_run/'results.json';r=read(p)
            references[variant] = (reference_run, r)
            local=r['local_test_personalized']['aggregate']
            official=r['official_test_personalized_ensemble']['metrics']
            rows.append({'dataset':job['config']['dataset'],'alpha':job['config']['alpha'],
                'seed':job['config']['seed'],'method':label,
                'selected_round':r['selected_personal_round'],
                'local_f1':local['macro_f1_mean'],'local_nll':local['nll_mean'],
                'worst_client_f1':local['macro_f1_min'],
                'official_f1':official['macro_f1'],'official_nll':official['nll'],'source':str(p)})
        psl_run, psl_result = references['psl_uniform']
        p, r = _validate_risk_endpoint(risk_output, psl_run, psl_result)
        local=r['local_test']['global_class_risk']['aggregate']
        official=r['official_test_personalized_ensemble']['global_class_risk']
        rows.append({'dataset':r['dataset'],'alpha':r['alpha'],'seed':r['seed'],
            'method':'PSL_global_risk_original','selected_round':r['selected_round'],
            'local_f1':local['macro_f1_mean'],'local_nll':local['nll_mean'],
            'worst_client_f1':local['macro_f1_min'],
            'official_f1':official['macro_f1'],'official_nll':official['nll'],'source':str(p)})
    fields=('local_f1','local_nll','worst_client_f1','official_f1','official_nll')
    aggregates=[]
    for dataset,alpha,method in sorted({(r['dataset'],r['alpha'],r['method']) for r in rows}):
        group=[r for r in rows if (r['dataset'],r['alpha'],r['method'])==(dataset,alpha,method)]
        if sorted(r['seed'] for r in group)!=[0,1,2]:
            raise RuntimeError('Missing or duplicate seeds')
        aggregates.append({'dataset':dataset,'alpha':alpha,'method':method,'seeds':3,
            **{f+'_mean':statistics.mean(r[f] for r in group) for f in fields},
            **{f+'_sd':statistics.stdev(r[f] for r in group) for f in fields}})
    csv_write(output/'per_run_comparison.csv',rows)
    csv_write(output/'comparison_by_setting.csv',aggregates)
    overall=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        group=[r for r in rows if r['method']==method]
        overall.append({'method':method,'runs':len(group),**{f:statistics.mean(r[f] for r in group) for f in fields}})
    csv_write(output/'comparison_overall.csv',overall)
    lines=['# Scheme C completed','',
      '36 new full 300-round runs. Three seeds per dataset/alpha; F1 values below are percent.',
      'Original reference rows retain the original runtime; see queue.json and per-run results for provenance.',
      'Three-P controls match branch work and joint clipping, not communication/storage/official-inference budgets.','',
      '| Method | Local F1 | Local NLL | Official F1 | Official NLL |',
      '|---|---:|---:|---:|---:|']
    for r in overall:
        lines.append(f"| {r['method']} | {100*r['local_f1']:.2f} | {r['local_nll']:.4f} | {100*r['official_f1']:.2f} | {r['official_nll']:.4f} |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(output/'ALL_DONE.json',{'status':'complete','new_runs':36,'rounds_per_run':300,
        'comparison_rows':len(rows),'completed_at':time.strftime('%Y-%m-%d %H:%M:%S'),
        'report':str(output/'REPORT.md')})
    print('ALL 36 RUNS COMPLETE. Results: '+str(output/'REPORT.md'),flush=True)


def supervise(args, output):
    q=read(output/'queue.json')
    jobs=q['jobs']
    if args.verify:
        jobs=[j for j in jobs if j['config']['dataset']=='pathmnist' and
              j['config']['alpha']==.1 and j['config']['seed']==0]
    done=[];pending=[]
    for j in jobs:
        p=output/j['key']/'results.json'
        if p.exists() and read(p).get('status')=='complete':
            r=read(p)
            expected={k:j[k] for k in ('key','variant','config','partition_id','init_seeds')}
            expected['protocol']='scheme-c-v1'
            if r['identity']!=expected or r['implementation_id']!=q['implementation_id']:
                raise RuntimeError('Incompatible completed result: '+j['key'])
            done.append(j['key'])
        else:
            pending.append(j)
    concurrency=args.job
    info=gpu_info()
    if info['free_mib']<3600 and concurrency==2:
        concurrency=1
        print(f"Only {info['free_mib']} MiB free; starting with --job 1",flush=True)
    if info['free_mib']<1600:
        raise RuntimeError('Less than 1600 MiB GPU memory free; close other GPU workloads and rerun')
    live=[];last_status=0;oom_count={};peak_used=info['used_mib']
    try:
        while pending or live:
            while pending and len(live)<concurrency:
                j=pending.pop(0);folder=output/j['key'];folder.mkdir(exist_ok=True)
                log=(folder/'worker.log').open('a',encoding='utf-8')
                cmd=child_command('--worker',j['key'],'--output',str(output))
                if args.verify:
                    cmd+=['--stop-after',str(args.verify_rounds)]
                process=subprocess.Popen(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                live.append((j,process,log))
                print(f"START {j['key']} PID={process.pid} [{len(done)}/{len(jobs)} complete]",flush=True)
            for item in live[:]:
                j,p,log=item;code=p.poll()
                if code is None:
                    continue
                live.remove(item);log.close()
                if code==0:
                    r=read(output/j['key']/'results.json')
                    expected='validation_only' if args.verify else 'complete'
                    if r['status']!=expected:
                        raise RuntimeError('Worker returned wrong completion status')
                    done.append(j['key']);print('DONE '+j['key'],flush=True)
                elif code==86 and concurrency>1 and not oom_count.get(j['key']):
                    concurrency=1;oom_count[j['key']]=1;pending.insert(0,j)
                    print('CUDA OOM: reducing --job to 1; retry from last saved round, batch size unchanged.',flush=True)
                else:
                    raise RuntimeError(f"Worker failed ({code}): {output/j['key']/'worker.log'}; rerun after resolving cause")
            info=gpu_info();peak_used=max(peak_used,info['used_mib'])
            if time.time()-last_status>=30:
                running=[]
                for j,p,_ in live:
                    f=output/j['key']/'progress.json'
                    running.append({'job':j['key'],'pid':p.pid,'progress':read(f) if f.exists() else 'starting'})
                write(output/'STATUS.json',{'state':'running','completed':len(done),'total':len(jobs),
                     'job':concurrency,'gpu':info,'peak_observed_used_mib':peak_used,'running':running})
                print(f"PROGRESS {len(done)}/{len(jobs)}; GPU {info['used_mib']}/{info['total_mib']} MiB; "+
                      '; '.join(f"{x['job']}: {x['progress']}" for x in running),flush=True)
                last_status=time.time()
            if live:
                time.sleep(2)
        write(output/'STATUS.json',{'state':'validation_complete' if args.verify else 'complete',
              'completed':len(done),'total':len(jobs),'peak_observed_used_mib':peak_used,'job':concurrency})
    finally:
        for _,p,log in live:
            if p.poll() is None:
                p.terminate()
            p.wait();log.close()
    if args.verify:
        print('GPU validation finished; these short runs are excluded from final results.',flush=True)
    else:
        summarize(output)


def main():
    parser=argparse.ArgumentParser(description='Scheme C: 18 P clip=5 + 18 three-P ensembles, 300 rounds, automatic resume and summaries')
    parser.add_argument('--job','--jobs',type=int,default=2,choices=(1,2),help='concurrent GPU workers; 2 for 8 GiB laptop')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--verify',action='store_true',help='full-data short GPU check in separate validation folder')
    parser.add_argument('--verify-rounds',type=int,default=1,choices=(1,2))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-dir',type=Path,default=PROJECT/'data',
                        help='Directory containing bloodmnist.npz, organamnist.npz and pathmnist.npz')
    parser.add_argument('--base-runs',type=Path,
                        default=PROJECT/'runs/fedtriad_3datasets_300r_ablations',
                        help='Completed base FedTriad runs used for matched partitions and references')
    parser.add_argument('--cache-root',type=Path,default=PROJECT/'.cache/medmnist',
                        help='Prepared base-data cache containing manifest.json files')
    parser.add_argument('--risk-output',type=Path,
                        default=PROJECT/'output/final-global-class-risk',
                        help='Directory containing the completed PSL global-class-risk JSON endpoints')
    parser.add_argument('--preflight',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    parser.add_argument('--stop-after',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    output=(args.output or (PROJECT/'validation/supplement_c_gpu_check' if args.verify else OUT)).resolve()
    data_dir=args.data_dir.resolve()
    base_runs=args.base_runs.resolve()
    cache_root=args.cache_root.resolve()
    risk_output=args.risk_output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.preflight:
        preflight(output,data_dir,base_runs,cache_root,risk_output);return 0
    if args.worker:
        return worker(output/'queue.json',args.worker,output/args.worker,args.stop_after)
    lock=output/'suite.lock';acquire(lock)
    sleep_state=False
    try:
        subprocess.run(child_command('--preflight','--output',str(output),
            '--data-dir',str(data_dir),'--base-runs',str(base_runs),
            '--cache-root',str(cache_root),'--risk-output',str(risk_output)),
            cwd=PROJECT,check=True)
        if args.preview:
            for j in read(output/'queue.json')['jobs']:
                p=output/j['key']/'results.json'
                label='SKIP' if p.exists() and read(p).get('status')=='complete' else 'RUN'
                print(label+' '+j['key'],flush=True)
            return 0
        if os.name=='nt':
            # Keep the system awake while this user-started training command runs.
            sleep_state=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        supervise(args,output)
        return 0
    except BaseException as e:
        write(output/'QUEUE_ERROR.json',{'error':str(e),'time':time.strftime('%Y-%m-%d %H:%M:%S')})
        raise
    finally:
        if sleep_state:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)
