"""Evaluate explicitly mapped labels without treating unknown labels as negatives."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from .engine import analyze, implementation_hash
from .registry import DETECTORS


def metrics(rows):
    tp=sum(y and p for y,p in rows);tn=sum(not y and not p for y,p in rows)
    fp=sum(not y and p for y,p in rows);fn=sum(y and not p for y,p in rows)
    def ratio(a,b):return a/b if b else None
    tpr=ratio(tp,tp+fn);tnr=ratio(tn,tn+fp)
    denominator=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return {'n':len(rows),'tp':tp,'tn':tn,'fp':fp,'fn':fn,
        'precision':ratio(tp,tp+fp),'recall':tpr,'specificity':tnr,
        'f1':ratio(2*tp,2*tp+fp+fn),'accuracy':ratio(tp+tn,len(rows)),
        'balanced_accuracy':(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
        'mcc':ratio(tp*tn-fp*fn,denominator)}


def run_benchmark(manifest_path,output):
    path=Path(manifest_path).resolve();data=path.read_bytes();m=json.loads(data)
    if m.get('schema')!='composition-benchmark/1':raise ValueError('Unsupported benchmark schema.')
    if not m.get('label_policy'):raise ValueError('Declare the label mapping policy before evaluation.')
    if m.get('split') not in ['development','holdout']:raise ValueError('Declare development or holdout split.')
    thresholds=m.get('thresholds',{})
    if not thresholds or any(k not in DETECTORS or isinstance(v,bool) or not isinstance(v,(int,float)) or not 0<=v<=1 for k,v in thresholds.items()):
        raise ValueError('Thresholds must name existing detectors and contain values in [0,1].')
    items=m.get('items',[])
    if not items:raise ValueError('Benchmark has no images.')
    for item in items:
        if not item.get('image') or not isinstance(item.get('labels'),dict):raise ValueError('Each item needs an image and explicit label mapping.')
        for id,label in item['labels'].items():
            if id not in thresholds:raise ValueError(f'No predeclared threshold for {id}.')
            if label is not None and not isinstance(label,bool):raise ValueError('Labels must be true, false, or null (unknown).')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    observations={k:[] for k in thresholds};predictions=[];failures=[];seen=set()
    for i,item in enumerate(items):
        image=(path.parent/item['image']).resolve()
        try:
            raw=image.read_bytes();sha=hashlib.sha256(raw).hexdigest()
            if sha in seen:raise ValueError('Duplicate image bytes; remove duplicates before evaluation.')
            seen.add(sha)
            report,_=analyze(raw,name=image.name,**m.get('analysis_options',{'null_samples':32,'stability':False}))
            report_path=f'{i:05d}-{sha[:12]}.json';(output/report_path).write_text(json.dumps(report,indent=2)+'\n')
            by_id={h['id']:h for h in report['hypotheses']}
            values={}
            for id,label in item['labels'].items():
                if label is None:continue
                h=by_id[id];positive=bool(h['eligible'] and h['fit']>=thresholds[id])
                observations[id].append((label,positive))
                values[id]={'label':label,'predicted':positive,'fit':h['fit'],'eligible':h['eligible']}
            predictions.append({'image':item['image'],'sha256':sha,'report':report_path,'results':values})
        except Exception as error:failures.append({'image':item['image'],'error':str(error)})
    result={'schema':'composition-benchmark-result/1','dataset':m.get('dataset','unnamed'),
        'split':m['split'],'label_policy':m['label_policy'],'thresholds':thresholds,
        'decision_rule':'eligible and raw within-method fit >= predeclared threshold; distinct from UI supported status',
        'manifest_sha256':hashlib.sha256(data).hexdigest(),'implementation_sha256':implementation_hash(),
        'metrics':{id:metrics(rows) for id,rows in observations.items()},
        'predictions':predictions,'failures':failures,'complete':not failures,
        'interpretation':'Metrics apply only to the supplied labels and split. Unknown labels are omitted. No threshold fitting is performed.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
