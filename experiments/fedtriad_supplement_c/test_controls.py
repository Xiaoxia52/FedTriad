"""Correctness tests for topology, clipping and checkpoint continuation."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fedtriad.config import Config
from fedtriad.models import MedicalCNN, average, state
from fedtriad.algorithm import Federation
from experiments.fedtriad_supplement_c.experiment import ParallelControl, execute
from experiments.fedtriad_supplement_c import pipeline

PROJECT = Path(__file__).resolve().parents[2]


class ControlsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(prefix='fedtriad_c_test_')
        self.root = Path(self.tmp.name)
        self.cache = self.root/'cache'
        self.cache.mkdir()
        rng = np.random.default_rng(47)
        for split,n in [('train',30),('val',10),('test',10)]:
            np.save(self.cache/(split+'_images.npy'),rng.integers(0,256,(n,28,28,3),dtype=np.uint8))
            np.save(self.cache/(split+'_labels.npy'),np.arange(n,dtype=np.int64)%8)
        self.part={'train':[list(range(k*6,k*6+4)) for k in range(5)],
                   'local_test':[list(range(k*6+4,k*6+6)) for k in range(5)],
                   'val':[[2*k,2*k+1] for k in range(5)],
                   'partition_id':'synthetic-test-only'}
        self.partition_file=self.root/'partition.json'
        self.partition_file.write_text(json.dumps(self.part),encoding='utf-8')
        self.cfg=Config(width=4,feature_dim=8,batch_size=4,rounds=3,device='cpu')

    def tearDown(self):
        gc.collect()  # Release mocked-call cycles holding Windows mmap handles.
        self.tmp.cleanup()

    def test_topology_joint_clipping_initialization_and_schedule(self):
        from experiments.fedtriad_supplement_c import experiment
        for variant in ('p_clip5','p3_ensemble_clip5'):
            model=ParallelControl(self.cfg,self.part,self.cache,variant,torch.device('cpu'))
            torch.manual_seed(self.cfg.seed)
            original=MedicalCNN(3,8,4,8)
            for k,v in original.state_dict().items():
                self.assertTrue(torch.equal(v,model.servers[0].state_dict()[k]))
            if model.members==3:
                self.assertFalse(torch.equal(model.servers[0].head.weight,model.servers[1].head.weight))
                self.assertFalse(torch.equal(model.servers[1].head.weight,model.servers[2].head.weight))
            original_fed=Federation(self.cfg,self.cache,self.part,torch.device('cpu'))
            self.assertEqual(model.active_clients(),original_fed.active_clients())
            anchors=[state(m) for m in model.servers]
            captured=[]
            real_train=experiment.train_client
            real_clip=torch.nn.utils.clip_grad_norm_
            clip_calls=[]
            def capture_clip(parameters,max_norm,**kw):
                parameters=list(parameters)
                clip_calls.append((len(parameters),max_norm))
                return real_clip(parameters,max_norm,**kw)
            def capture_train(p,s,l,*args):
                members=[m for m in (p,s,l) if m is not None]
                for a,m in zip(anchors,members):
                    for k,v in a.items():
                        self.assertTrue(torch.equal(v,m.state_dict()[k]),'P copy was not reset')
                result=real_train(p,s,l,*args)
                captured.append(([state(m) for m in members],result['examples']))
                return result
            with patch.object(experiment,'train_client',side_effect=capture_train), \
                 patch('torch.nn.utils.clip_grad_norm_',side_effect=capture_clip):
                diag=model.train_round()
            expected_params=len(list(model.servers[0].parameters()))*model.members
            self.assertTrue(all(count==expected_params and limit==5 for count,limit in clip_calls))
            self.assertEqual(diag['execution_order'],original_fed.serial_order(original_fed.active_clients()))
            for i,server in enumerate(model.servers):
                expected=average([v[0][i] for v in captured],[v[1] for v in captured])
                for k,v in expected.items():
                    self.assertTrue(torch.equal(v,server.state_dict()[k]),'Wrong member aggregation')
            self.assertEqual(model.branch_optimizer_steps,3*model.members)
            self.assertEqual(model.train_examples,12*model.members)
            self.assertEqual(self.cfg.learning_rate(.01,0),.01)

    def test_exact_resume_and_complete_skip(self):
        for variant in ('p_clip5','p3_ensemble_clip5'):
            members=1 if variant=='p_clip5' else 3
            job={'key':'synthetic_'+variant,'variant':variant,'config':self.cfg.as_dict(),
                 'partition_id':self.part['partition_id'],'partition_file':str(self.partition_file),
                 'init_seeds':[self.cfg.seed+1000003*i for i in range(members)],
                 'cache':str(self.cache),'reference_run':'synthetic_test_only'}
            a=self.root/(variant+'_continuous');b=self.root/(variant+'_resumed')
            full=execute(job,a,PROJECT,device='cpu')
            first=execute(job,b,PROJECT,device='cpu',stop_after=1)
            self.assertEqual(first['status'],'validation_only')
            resumed=execute(job,b,PROJECT,device='cpu')
            self.assertEqual(full['local_test'],resumed['local_test'])
            self.assertEqual(full['official_test'],resumed['official_test'])
            self.assertEqual(full['selected_round'],resumed['selected_round'])
            ac=torch.load(a/'last.pt',weights_only=False)
            bc=torch.load(b/'last.pt',weights_only=False)
            for left,right in zip(ac['model']['servers'],bc['model']['servers']):
                for k,v in left.items():
                    self.assertTrue(torch.equal(v,right[k]),'Resume differs from uninterrupted training')
            selected=max(range(len(bc['scores'])),key=lambda i:bc['scores'][i]['client_validation_macro_f1'])+1
            self.assertEqual(resumed['selected_round'],selected)
            mtime=(b/'last.pt').stat().st_mtime_ns
            execute(job,b,PROJECT,device='cpu')
            self.assertEqual(mtime,(b/'last.pt').stat().st_mtime_ns)
            invalid=copy.deepcopy(job);invalid['config']['lr']=.123
            with self.assertRaises(ValueError):
                execute(invalid,b,PROJECT,device='cpu')

    def test_summary_completion_gate_and_reference_endpoints(self):
        project=self.root/'summary_fixture'
        output=project/'runs/supplement_c_300r'
        source=project/'runs/fedtriad_3datasets_300r_ablations'
        # Deliberately keep the risk directory outside the project tree: the
        # queue must preserve and consume an explicitly supplied endpoint root.
        riskroot=self.root/'external-risk-output'
        jobs=[]
        local=lambda x:{'macro_f1_mean':x,'nll_mean':1-x,'macro_f1_min':x-.1}
        official=lambda x:{'macro_f1':x,'nll':1-x}
        for ds in ('bloodmnist','organamnist','pathmnist'):
            for alpha in (.1,.5):
                for seed in (0,1,2):
                    c={'dataset':ds,'alpha':alpha,'seed':seed}
                    pdir=source/f'{ds}_fedtriad_p_only_a{alpha}_seed{seed}_fixture'
                    sdir=source/f'{ds}_fedtriad_psl_uniform_a{alpha}_seed{seed}_fixture'
                    for d,x in [(pdir,.5),(sdir,.6)]:
                        variant='FedTriad-psl_uniform' if d == sdir else 'FedTriad-p_only'
                        partition_id=f'{d.name}-partition'
                        pipeline.write(d/'config.json',{**c,'triad_variant':'psl_uniform' if d == sdir else 'p_only'})
                        pipeline.write(d/'partition.json',{'partition_id':partition_id})
                        checkpoint=d/'best_personal.pt'
                        checkpoint.parent.mkdir(parents=True,exist_ok=True)
                        checkpoint.write_bytes(f'checkpoint:{d.name}'.encode('ascii'))
                        pipeline.write(d/'results.json',{**c,'status':'complete','variant':variant,
                            'partition_id':partition_id,'selected_personal_round':17,
                            'local_test_personalized':{'aggregate':local(x)},
                            'official_test_personalized_ensemble':{'metrics':official(x)}})
                    pipeline.write(riskroot/(sdir.name+'.json'),{**c,'memory':{},
                        'run_directory':str(Path('runs')/sdir.name),
                        'partition_id':partition_id,'selected_round':17,
                        'checkpoint':{'file':'best_personal.pt',
                                      'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                        'local_test':{'global_class_risk':{'aggregate':local(.7)}},
                        'official_test_personalized_ensemble':{'global_class_risk':official(.7)}})
                    for variant,x in [('p_clip5',.4),('p3_ensemble_clip5',.8)]:
                        key=f'{ds}_{variant}_a{alpha}_seed{seed}'
                        jobs.append({'key':key,'variant':variant,'config':c,'reference_run':str(pdir)})
                        pipeline.write(output/key/'results.json',{
                            **c,'variant':variant,'status':'complete','trained_rounds':300,
                            'implementation_id':'fixture','selected_round':17,
                            'local_test':{'aggregate':local(x)},'official_test':official(x)})
        pipeline.write(output/'queue.json',{
            'jobs': jobs,
            'implementation_id': 'fixture',
            'base_runs': str(source),
            'risk_output': str(riskroot),
        })
        with patch.object(pipeline,'PROJECT',project):
            missing=output/jobs[-1]['key']/'results.json'
            held=pipeline.read(missing)
            pipeline.write(missing,{**held,'status':'running'})
            with self.assertRaises(RuntimeError):pipeline.summarize(output)
            self.assertFalse((output/'ALL_DONE.json').exists())
            pipeline.write(missing,held)
            pipeline.summarize(output)
        import csv
        with (output/'comparison_overall.csv').open(encoding='utf-8-sig') as handle:
            rows=list(csv.DictReader(handle))
        expected={'p_clip5':.4,'p3_ensemble_clip5':.8,'P_original_unclipped':.5,
                  'PSL_uniform_original':.6,'PSL_global_risk_original':.7}
        self.assertEqual(len(rows),5)
        for row in rows:
            self.assertEqual(int(row['runs']),18)
            self.assertAlmostEqual(float(row['local_f1']),expected[row['method']])
            self.assertAlmostEqual(float(row['official_f1']),expected[row['method']])
        self.assertEqual(pipeline.read(output/'ALL_DONE.json')['comparison_rows'],90)

        psl_run=source/'bloodmnist_fedtriad_psl_uniform_a0.1_seed0_fixture'
        psl_result=pipeline.read(psl_run/'results.json')
        risk_path=riskroot/(psl_run.name+'.json')
        risk=pipeline.read(risk_path)
        pipeline.write(risk_path,{**risk,'run_directory':'runs/not-the-psl-run'})
        with self.assertRaises(RuntimeError):
            pipeline._validate_risk_endpoint(riskroot,psl_run,psl_result)
        pipeline.write(risk_path,risk)
        pipeline.write(risk_path,{**risk,'checkpoint':{**risk['checkpoint'],'sha256':'0'*64}})
        with self.assertRaises(RuntimeError):
            pipeline._validate_risk_endpoint(riskroot,psl_run,psl_result)


if __name__=='__main__':
    unittest.main()
