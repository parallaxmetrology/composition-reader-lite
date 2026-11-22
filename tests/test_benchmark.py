import json
from pathlib import Path
import pytest
from composition_reader.benchmark import metrics,run_benchmark
from test_measurements import picture


def test_confusion_metrics_and_undefined_cases():
    r=metrics([(True,True),(True,False),(False,True),(False,False)])
    assert r['mcc']==0 and r['f1']==.5 and r['balanced_accuracy']==.5
    assert metrics([])['accuracy'] is None
    assert metrics([(True,True)])['specificity'] is None


def test_benchmark_runs_real_images_and_omits_unknowns(tmp_path):
    (tmp_path/'center.png').write_bytes(picture())
    (tmp_path/'thirds.png').write_bytes(picture(((1/3,1/3),)))
    m={'schema':'composition-benchmark/1','dataset':'engineering fixtures','split':'development',
       'label_policy':'Known drawn blob locations; no external category claim.',
       'thresholds':{'centre-cross':.8,'rule-of-thirds':.8},'analysis_options':{'null_samples':0,'stability':False},
       'items':[{'image':'center.png','labels':{'centre-cross':True,'rule-of-thirds':False}},
                {'image':'thirds.png','labels':{'centre-cross':False,'rule-of-thirds':None}}]}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    r=run_benchmark(path,tmp_path/'results')
    assert r['complete'] and r['metrics']['centre-cross']['accuracy']==1
    assert r['metrics']['centre-cross']['n']==2 and r['metrics']['rule-of-thirds']['n']==1
    assert len(list((tmp_path/'results').glob('*.json')))==3
    m['items'].append(m['items'][0]);path.write_text(json.dumps(m))
    r=run_benchmark(path,tmp_path/'duplicates')
    assert not r['complete'] and 'Duplicate' in r['failures'][0]['error']


def test_benchmark_requires_explicit_policy_and_thresholds(tmp_path):
    p=tmp_path/'manifest.json';p.write_text(json.dumps({'schema':'composition-benchmark/1'}))
    with pytest.raises(ValueError,match='label mapping'):run_benchmark(p,tmp_path/'results')
