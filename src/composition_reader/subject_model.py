"""Pinned local inference. Never downloads a model or uploads an image during analysis."""
from __future__ import annotations
import hashlib,json,os,threading,urllib.request
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image

METADATA_PATH=Path(__file__).parent/'data/subject-model.json'
SPEC=json.loads(METADATA_PATH.read_text())
_lock=threading.Lock()

def model_directory():
    root=Path(__file__).resolve().parents[2]
    default=root/'data/models' if (root/'pyproject.toml').exists() else Path.home()/'.composition-reader/models'
    return Path(os.environ.get('COMPOSITION_MODELS',str(default))).expanduser()

def checked_model():
    path=model_directory()/SPEC['file']
    if not path.is_file():raise RuntimeError('Subject model is not installed. Run composition-reader install-model once, then retry. Images stay local.')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SPEC['sha256']:
        raise RuntimeError('Subject model checksum mismatch. Restore the pinned model with composition-reader install-model.')
    return path

def install_model():
    dest=model_directory()/SPEC['file'];dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==SPEC['sha256']:return dest
    with urllib.request.urlopen(SPEC['url'],timeout=60) as response:raw=response.read(SPEC['bytes']+1)
    if len(raw)!=SPEC['bytes'] or hashlib.sha256(raw).hexdigest()!=SPEC['sha256']:
        raise RuntimeError('Downloaded model failed size/checksum verification; existing file was not replaced.')
    temp=dest.with_suffix('.download');temp.write_bytes(raw);temp.replace(dest)
    return dest

@lru_cache(maxsize=2)
def _session(directory):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    path=checked_model()
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])

def foreground(rgb):
    # Session creation is synchronized; ORT sessions themselves support concurrent run().
    with _lock:session=_session(str(model_directory()))
    im=Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).resize((320,320),Image.Resampling.LANCZOS)
    array=np.asarray(im,dtype=np.float32);array/=max(float(array.max()),1e-6)
    array=(array-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    predicted=session.run([session.get_outputs()[0].name],{session.get_inputs()[0].name:array.transpose(2,0,1)[None]})[0][0,0]
    if not np.isfinite(predicted).all():raise RuntimeError('Foreground model produced non-finite output.')
    # Preserve absolute network output. Stretching every map to [0,1] would turn
    # uncertain texture into a seemingly confident subject.
    import cv2
    return np.clip(cv2.resize(predicted,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR),0,1)
