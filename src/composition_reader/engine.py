from __future__ import annotations
import base64, hashlib, io, json, os, platform, tempfile, time
from pathlib import Path
from importlib.metadata import version as _version, PackageNotFoundError
def version(name):
    # The server image installs opencv-python-headless, so accept either distribution.
    for n in ([name,'opencv-python-headless'] if name=='opencv-python' else [name]):
        try:return _version(n)
        except PackageNotFoundError:pass
    return 'unknown'
import cv2
import numpy as np
from PIL import Image, ImageFilter
from parallax_notan import construction_read
from parallax_notan.notan import value_map
import vtl
from .extract import load_image, extract
from .geometry import PHI
from .hypotheses import evaluate, quick_reads, symmetry_read, NAMES
from .rings import ring_read, reference_rings
from .model import native, SCHEMA, ENGINE
from .registry import contracts

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'data/analyses' if (ROOT/'pyproject.toml').is_file() and (ROOT/'vendor/SOURCES.json').is_file() else Path.home()/'.composition-reader/analyses'
DATA_DIR=Path(__file__).parent/'data'
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'composition-reader-matplotlib'))

def _source_hash():
    digest=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(p.name.encode());digest.update(p.read_bytes())
    for p in sorted(DATA_DIR.glob('*.json')):
        digest.update(p.name.encode());digest.update(p.read_bytes())
    return digest.hexdigest()

# Freeze at process startup: source files may change while a development server
# still has older modules loaded. Never stamp that older runtime with a newer hash.
_IMPLEMENTATION_HASH=_source_hash()

def implementation_hash():
    return _IMPLEMENTATION_HASH

def png(array):
    a=np.asarray(array)
    if a.dtype!=np.uint8:a=np.uint8(np.clip(a,0,1)*255)
    im=Image.fromarray(a);out=io.BytesIO();im.save(out,format='PNG');return out.getvalue()

def frame_context(aspect):
    ratios=[('root-2','Root 2',2**.5),('root-3','Root 3',3**.5),('root-4','Double square',2),
            ('root-5','Root 5',5**.5),('phi-rectangle','Golden rectangle',PHI),('three-two-rectangle','3:2 rectangle',1.5)]
    r=max(aspect,1/aspect)
    return [{'id':id,'name':name,'ratio':v,'relative_error':abs(r/v-1),'match':abs(r/v-1)<=.015,
             'orientation':'landscape' if aspect>=1 else 'portrait','meaning':'Frame proportion only; internal construction is a separate hypothesis.'}
            for id,name,v in ratios]


def held_back_reason(h):
    """Plain statement of why a fitting hypothesis was not reported as a reading."""
    c=h.get('control') or {};st=h.get('stability') or {};p=h.get('parameters') or {}
    if h['status']=='sensitive':
        parts=[]
        if st.get('max_fit_change',0)>.15:parts.append(f'its fit changes by up to {st["max_fit_change"]:.2f}')
        if st.get('max_contact_weight_change',0)>.1:parts.append(f'the share of subject weight pinned by the frame changes by up to {st["max_contact_weight_change"]:.2f}')
        if not parts:parts.append('the selected evidence does not persist')
        return 'Held back: under slight blur or resampling '+' and '.join(parts)+', so the reading depends on extraction details.'
    reasons=[]
    if c and h['fit']<.5:
        reasons.append(f'its fit of {int(h["fit"]*1000)/1000:.3f} is under the 0.500 a reading needs')
    if c.get('two_axis_informative') is False:
        reasons.append('the subject fills or touches the frame, so its position is constrained and the control cannot test it')
    tail=c.get(c.get('decision_tail_key','family_max_tail_fraction'))
    if tail is not None and tail>.1:
        reasons.append(f'randomized controls match or beat it {tail:.0%} of the time (a reading needs 10% or less)')
    if c.get('minimum_excess_fit') is not None and c.get('excess_fit',1)<c['minimum_excess_fit']:
        reasons.append(f'it is only {c["excess_fit"]:+.2f} above the control average (needs +{c["minimum_excess_fit"]:.2f})')
    guards={'whole_subject_corroboration':'the whole subject does not corroborate the face alignment',
            'whole_foreground_corroboration':'the whole foreground does not corroborate the object alignment',
            'object_identity_guard':'the detected object is not confident enough to anchor the reading',
            'main_subject_guard':'the restored main subject does not pass the stricter coincidence guard'}
    reasons+=[text for key,text in guards.items() if isinstance(p.get(key),dict) and p[key].get('passed') is False]
    if not reasons:reasons.append('it does not meet this detector’s decision rule')
    return 'Held back: '+'; '.join(reasons)+'.'


def near_misses(hypotheses,reported,limit=3):
    """The strongest fits that were not reported, each with the rule that held it back."""
    pool=[h for h in hypotheses if h['status'] in ('ambiguous','sensitive') and h['eligible']
          and h['family'] not in ('diagnostic','relations') and h['id'] not in reported and h['fit']>=.45]
    return [{'hypothesis_id':h['id'],'name':h['name'],'fit':h['fit'],'status':h['status'],'reason':held_back_reason(h)}
            for h in sorted(pool,key=lambda h:-h['fit'])[:limit]]


def apply_stability(rgb,source,hypotheses,border='raw',excluded=None,anchors=None):
    variants={};subject_variants={}
    # Re-extract after blur and down/up sampling; manual anchors stay user-declared.
    smooth=np.array(Image.fromarray(np.uint8(rgb*255)).filter(ImageFilter.GaussianBlur(1.4)))/255
    h,w=rgb.shape[:2];small=cv2.resize(rgb,(max(32,round(w*.75)),max(32,round(h*.75))),interpolation=cv2.INTER_AREA)
    sampled=cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR)
    for label,im in [('blur_1.4px',smooth),('resample_75_percent',sampled)]:
        ss=extract(im,source,border,excluded,anchors)
        subject_variants[label]={'rgb':im,'subjects':ss.subjects or []}
        variants[label]={r['id']:r for r in quick_reads(ss)+[symmetry_read(ss),ring_read(ss)]}
    for hyp in hypotheses:
        vals={k:v[hyp['id']]['fit'] for k,v in variants.items() if hyp['id'] in v}
        if vals:
            delta=max(abs(v-hyp['fit']) for v in vals.values())
            hyp['stability']={'variants':vals,'max_fit_change':delta,'stable':delta<=.15,
                              'scope':'extraction sensitivity at working resolution; not observer agreement'}
            contact_delta=0.
            if hyp['family']=='placement':
                original=np.array(hyp['parameters'].get('constrained_axis_weight',[0.,0.]))
                contact_variants={k:v[hyp['id']]['parameters'].get('constrained_axis_weight',[0.,0.]) for k,v in variants.items() if hyp['id'] in v}
                contact_delta=max(float(np.max(np.abs(np.array(v)-original))) for v in contact_variants.values())
                hyp['stability'].update(frame_contact_variants=contact_variants,max_contact_weight_change=contact_delta,contact_stable=contact_delta<=.1)
                hyp['stability']['stable']=delta<=.15 and contact_delta<=.1
                if contact_delta>.1:hyp['explanation']+=' Frame-contact constraints change under blur/resampling; this placement reading is sensitive.'
            if (delta>.15 or contact_delta>.1) and (hyp['status']=='supported' or ((hyp['id'] in ('bilateral-symmetry','concentric-rings') or hyp['family']=='placement') and hyp['status']=='descriptive')):hyp['status']='sensitive'
    return subject_variants



def analyze(data: bytes, *, name='image', border='raw', null_samples=64, stability=True, excluded=None,
            anchors=None, crop=None, progress=None):
    start=time.monotonic();progress=progress or (lambda *args:None)
    if border not in ['raw','protected']:raise ValueError('Unknown boundary mode.')
    if not isinstance(null_samples,int) or not 0<=null_samples<=256:raise ValueError('Control samples must be an integer from 0 to 256.')
    if anchors is not None and len(anchors)>24:raise ValueError('At most 24 manual anchors are supported.')
    progress('Reading image',.05)
    rgb,source=load_image(data)
    source['name']=Path(name).name
    if crop:
        if len(crop)!=4:raise ValueError('Crop requires four coordinates.')
        x0,y0,x1,y1=map(float,crop)
        if not(0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('Crop coordinates must define a rectangle inside the image.')
        h,w=rgb.shape[:2];bounds=[round(x0*w),round(y0*h),round(x1*w),round(y1*h)]
        if bounds[2]-bounds[0]<32 or bounds[3]-bounds[1]<32:raise ValueError('Crop is too small at working resolution.')
        rgb=rgb[bounds[1]:bounds[3],bounds[0]:bounds[2]]
        source['crop_in_upright_frame']=list(crop);source['working_crop_pixels']=bounds
    source['analysis_size']=[rgb.shape[1],rgb.shape[0]]
    progress('Extracting masses and line systems',.15)
    scene=extract(rgb,source,border,excluded,anchors)
    progress('Testing composition hypotheses and geometric controls',.3)
    hypotheses=evaluate(scene,null_samples=null_samples)
    progress('Checking stability',.65)
    if stability and not scene.summary['blank']:
        apply_stability(rgb,source,hypotheses,border,excluded,anchors)
    progress('Reading kernel coordinates',.8)
    kernel_rgb,_=load_image(data,max_side=1536)
    if crop:
        kh,kw=kernel_rgb.shape[:2]
        kernel_rgb=kernel_rgb[round(crop[1]*kh):round(crop[3]*kh),round(crop[0]*kw):round(crop[2]*kw)]
    source['kernel_input_size']=[kernel_rgb.shape[1],kernel_rgb.shape[0]]
    # Vendor APIs are called unchanged. The adapter records their coordinate conventions.
    with np.errstate(all='ignore'):
        kernel=vtl.measure(Image.fromarray(np.uint8(np.clip(kernel_rgb*255,0,255))))
        color=vtl.measure_color(Image.fromarray(np.uint8(np.clip(kernel_rgb*255,0,255))))
        construction=construction_read(rgb)
    # Rank only within each family; raw indices across families are not commensurate.
    family_order=['placement','armature','direction','arrangement','symmetry','curve','space','relations','diagnostic']
    hypotheses.sort(key=lambda x:(family_order.index(x['family']),-x['fit']))
    options={'border':border,'null_samples':null_samples,'stability':stability,'excluded':excluded or [],
             'anchors':anchors,'crop':crop}
    report={'schema':SCHEMA,'engine_version':ENGINE,'implementation_sha256':implementation_hash(),'detector_contracts':contracts(),'source':source,'options':options,
            'coordinates':{'geometry':'x right, y down; normalized to current analysis frame [0,1]',
                           'distances':'Euclidean pixels divided by image diagonal',
                           'vtl':'centroid offsets centered at (0,0); input independently decoded at up to 1536 px; see vendor documentation'},
            'summary':scene.summary,'warnings':scene.warnings,
            'regions':scene.regions,'subjects':scene.subjects or [],'subject_selection':scene.subject_metadata,
            'lines':scene.lines,'reference_rings':reference_rings(scene) if not scene.summary['blank'] else None,'hypotheses':hypotheses,
            'frame_context':frame_context(scene.aspect),'kernel':kernel,'color':color,'construction':construction,
            'validation':{'type':'engineering and known-answer validation',
                     'external_category_accuracy':None,'confidence_calibrated':False,
                     'notice':'Detector support values are exploratory measurements, not calibrated category probabilities or quality scores.'},
            'environment':{'python':platform.python_version(),**{p:version(p) for p in ['numpy','scipy','scikit-image','Pillow','opencv-python','onnxruntime']}},
            'elapsed_seconds':time.monotonic()-start}
    visible=[r for r in hypotheses if r['status'] in ['supported','descriptive'] and r['family'] not in ['diagnostic','relations']]
    report['reading']=[{'hypothesis_id':r['id'],'text':r['explanation']} for r in sorted(visible,key=lambda r:(r['status']!='supported',-r['fit']))[:5]]
    if not report['reading']:report['reading']=[{'hypothesis_id':None,'text':'No clear supported arrangement under the current extraction. Inspect the regions or add your own anchors.'}]
    report['near_misses']=near_misses(hypotheses,{r['hypothesis_id'] for r in report['reading']})
    field_images={
        'image':png(rgb),'mass':png(scene.mass/max(float(scene.mass.max()),1e-9)),
        'notan':png(value_map(rgb,mass=scene.mass)),
        'chroma':png(np.clip(scene.chroma/.25,0,1)),
        'edges':png(scene.edge.astype(float)),
        'subjects':png(scene.subject_map if scene.subject_map is not None else np.zeros_like(scene.mass)),
    }
    progress('Saving report',.98)
    return native(report),field_images


def compare_reports(a,b):
    if a.get('schema')!=SCHEMA or b.get('schema')!=SCHEMA:raise ValueError('Report schema is not supported.')
    ha={r['id']:r for r in a['hypotheses']};hb={r['id']:r for r in b['hypotheses']}
    delta=[{'id':id,'name':ha[id]['name'],'family':ha[id]['family'],'before':ha[id]['fit'],
            'after':hb[id]['fit'],'change':hb[id]['fit']-ha[id]['fit']} for id in ha.keys()&hb.keys()]
    delta.sort(key=lambda r:-abs(r['change']))
    warnings=[]
    if a.get('implementation_sha256')!=b.get('implementation_sha256'):warnings.append('Engine implementations differ. Compare reports made with the same implementation for controlled image comparisons.')
    if a['options']!=b['options']:warnings.append('Analysis settings differ; changes can reflect extraction or anchor choices as well as the images.')
    if a['source'].get('analysis_size',a['source']['working_size'])!=b['source'].get('analysis_size',b['source']['working_size']):warnings.append('Analysis dimensions differ. Coordinates refer to each image’s own frame.')
    if a['options'].get('crop')!=b['options'].get('crop'):warnings.append('Frames differ: this comparison measures reframing, not displacement within a shared source frame.')
    return {'schema':'composition-comparison/1','before':a['source']['name'],'after':b['source']['name'],
            'hypotheses':delta,'warnings':warnings,'interpretation':'Changes in geometric support; positive does not mean better composition.'}
