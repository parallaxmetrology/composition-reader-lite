from __future__ import annotations
import argparse,json
from pathlib import Path
from .engine import analyze,compare_reports,ROOT
from .store import Store

def main():
    p=argparse.ArgumentParser(description='Composition Reader: analyze, compare and inspect visual structure.')
    sub=p.add_subparsers(dest='command',required=True)
    serve=sub.add_parser('serve');serve.add_argument('--port',type=int,default=7820);serve.add_argument('--data-dir');serve.add_argument('--open',action='store_true',help='Open the browser after the local server is ready')
    read=sub.add_parser('analyze');read.add_argument('image',type=Path);read.add_argument('--out',type=Path,default=Path('analysis.json'))
    read.add_argument('--border',choices=['raw','protected'],default='raw');read.add_argument('--null-samples',type=int,default=64);read.add_argument('--no-stability',action='store_true')
    batch=sub.add_parser('batch');batch.add_argument('folder',type=Path);batch.add_argument('--out',type=Path,default=Path('batch-results'));batch.add_argument('--null-samples',type=int,default=32)
    compare=sub.add_parser('compare');compare.add_argument('before',type=Path);compare.add_argument('after',type=Path);compare.add_argument('--out',type=Path)
    benchmark=sub.add_parser('benchmark');benchmark.add_argument('manifest',type=Path);benchmark.add_argument('--out',type=Path,default=Path('benchmark-results'))
    sub.add_parser('install-model',help='Download and verify the local foreground model')
    a=p.parse_args()
    if a.command=='install-model':
        from .subject_model import install_model
        print(install_model())
    elif a.command=='serve':
        import uvicorn
        from .server import create_app
        print(f'Composition Reader: http://127.0.0.1:{a.port}',flush=True)
        if a.open:
            import threading,time,urllib.request,webbrowser
            def open_when_ready():
                url=f'http://127.0.0.1:{a.port}'
                for _ in range(100):
                    try:
                        with urllib.request.urlopen(url+'/api/health',timeout=1) as response:
                            if json.load(response).get('ok'):
                                webbrowser.open(url);return
                    except OSError:pass
                    time.sleep(.1)
            threading.Thread(target=open_when_ready,daemon=True).start()
        uvicorn.run(create_app(a.data_dir),host='127.0.0.1',port=a.port)
    elif a.command=='benchmark':
        from .benchmark import run_benchmark
        result=run_benchmark(a.manifest,a.out)
        print(json.dumps({'summary':str(a.out/'summary.json'),'metrics':result['metrics'],'failures':result['failures']},indent=2))
        if not result['complete']:raise SystemExit(1)
    elif a.command=='compare':
        r=compare_reports(json.loads(a.before.read_text()),json.loads(a.after.read_text()));s=json.dumps(r,indent=2)
        if a.out:a.out.write_text(s+'\n')
        else:print(s)
    elif a.command=='analyze':
        r,fields=analyze(a.image.read_bytes(),name=a.image.name,border=a.border,null_samples=a.null_samples,stability=not a.no_stability)
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
        assetdir=a.out.with_suffix('');assetdir.mkdir(exist_ok=True)
        for k,v in fields.items():(assetdir/(k+'.png')).write_bytes(v)
        print(f'{a.out}: {len(r["hypotheses"])} hypotheses, {r["elapsed_seconds"]:.1f}s')
    elif a.command=='batch':
        a.out.mkdir(parents=True,exist_ok=True);index=[]
        for path in sorted(a.folder.iterdir()):
            if path.suffix.lower() not in ['.jpg','.jpeg','.png','.webp','.tif','.tiff']:continue
            try:
                r,_=analyze(path.read_bytes(),name=path.name,null_samples=a.null_samples,stability=False)
                output=a.out/(path.name+'.json');output.write_text(json.dumps(r,indent=2)+'\n');index.append({'image':path.name,'report':output.name,'status':'complete'})
                print(path.name,'complete',flush=True)
            except Exception as e:index.append({'image':path.name,'status':'failed','error':str(e)})
        (a.out/'index.json').write_text(json.dumps(index,indent=2)+'\n')

if __name__=='__main__':main()
