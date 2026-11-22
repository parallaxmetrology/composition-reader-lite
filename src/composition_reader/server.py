from __future__ import annotations
import base64, binascii, csv, html, io, json, os, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
from .engine import analyze,compare_reports,ROOT,DEFAULT_DATA
from .store import Store
from .model import ENGINE

class AnalyzeRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    image: str|None = None
    source_id: str|None = None
    example: str|None = None
    name: str = Field(default='image',max_length=200)
    border: Literal['raw','protected']='raw'
    null_samples: int = Field(default=64,ge=0,le=256)
    stability: bool = True
    excluded: list[str] = Field(default_factory=list,max_length=64)
    anchors: list[tuple[float,float]]|None=Field(default=None,max_length=24)
    crop: tuple[float,float,float,float]|None=None

    @field_validator('anchors')
    @classmethod
    def valid_anchors(cls,points):
        if points is not None and any(not(0<=x<=1 and 0<=y<=1) for x,y in points):
            raise ValueError('Anchors must lie inside the image.')
        return points

    @field_validator('crop')
    @classmethod
    def valid_crop(cls,crop):
        if crop is not None and not(0<=crop[0]<crop[2]<=1 and 0<=crop[1]<crop[3]<=1):
            raise ValueError('Crop must be a rectangle inside the image.')
        return crop


def create_app(data_dir=None):
    store=Store(data_dir or os.environ.get('COMPOSITION_DATA',str(DEFAULT_DATA)))
    pool=ThreadPoolExecutor(max_workers=1)
    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=True,cancel_futures=True)
    app=FastAPI(title='Composition Reader',version=ENGINE,lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=[h for h in os.environ.get('COMPOSITION_ALLOWED_HOSTS','127.0.0.1,localhost,[::1]').split(',') if h])
    jobs={};lock=threading.Lock()
    public=os.environ.get('COMPOSITION_PUBLIC')=='1';owned={}  # public site: each browser sees only its own readings
    examples=ROOT/'examples';web=Path(__file__).parent/'web'

    @app.middleware('http')
    async def local_boundary(request:Request,call_next):
        # No permissive CORS. Reject browser cross-origin writes and bound request size.
        if request.method in ['POST','PUT','DELETE']:
            origin=request.headers.get('origin')
            if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
                return Response('Cross-origin writes are not allowed.',status_code=403)
            try:length=int(request.headers.get('content-length','0') or 0)
            except ValueError:return Response('Invalid content length.',status_code=400)
            if length>42_000_000:
                return Response('Request too large.',status_code=413)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        if public and 'cr' not in request.cookies:response.set_cookie('cr',uuid.uuid4().hex,max_age=86400*30,httponly=True,samesite='lax',secure=request.url.scheme=='https')
        return response

    @app.get('/api/health')
    def health():return {'ok':True,'engine':ENGINE,'public':public,**({} if public else {'data_dir':str(store.root)})}

    @app.get('/api/examples')
    def get_examples():
        return [{'name':p.name,'url':'/api/examples/'+p.name} for p in sorted(examples.glob('*')) if p.suffix.lower() in ['.png','.jpg','.jpeg']]

    @app.get('/api/examples/{name}')
    def example_image(name:str):
        if Path(name).name!=name or Path(name).suffix.lower() not in ['.png','.jpg','.jpeg']:raise HTTPException(404)
        p=examples/name
        if not p.is_file():raise HTTPException(404)
        return FileResponse(p)

    @app.post('/api/analyze',status_code=202)
    def submit(req:AnalyzeRequest,request:Request):
        owner=request.cookies.get('cr')
        if sum(x is not None for x in [req.image,req.source_id,req.example])!=1:raise HTTPException(400,'Supply exactly one image source.')
        try:
            if req.image:
                data=base64.b64decode(req.image.split(',')[-1],validate=True)
            elif req.source_id:
                data=(store.folder(req.source_id)/'source.bin').read_bytes()
            else:
                if Path(req.example).name!=req.example or Path(req.example).suffix.lower() not in ['.png','.jpg','.jpeg']:raise ValueError('Invalid example.')
                data=(examples/req.example).read_bytes()
            if len(data)>30_000_000:raise ValueError('Image exceeds 30 MB.')
        except (ValueError,FileNotFoundError,binascii.Error) as e:raise HTTPException(400,str(e))
        with lock:
            if sum(j['status'] in ['queued','running'] for j in jobs.values())>=4:raise HTTPException(429,'Four analyses are already queued. Wait for one to finish.')
            id=uuid.uuid4().hex;jobs[id]={'id':id,'status':'queued','stage':'Waiting for analyzer','progress':0}
        def run():
            jobs[id].update(status='running')
            def progress(stage,value):jobs[id].update(stage=stage,progress=value)
            try:
                options=req.model_dump(exclude={'image','source_id','example'})
                report,fields=analyze(data,progress=progress,**options)
                saved=store.save(report,fields,data)
                if public and owner:owned.setdefault(owner,set()).add(saved['id'])
                jobs[id].update(status='complete',progress=1,stage='Complete',analysis_id=saved['id'])
            except Exception as e:
                import traceback;traceback.print_exc()
                jobs[id].update(status='failed',stage='Analysis failed',error=str(e))
        pool.submit(run)
        return {'job_id':id}

    @app.get('/api/jobs/{id}')
    def job(id:str):
        if id not in jobs:raise HTTPException(404,'Job not found.')
        return jobs[id]

    @app.get('/api/analyses')
    def analyses(request:Request):
        rows=store.list()
        if public:
            mine=owned.get(request.cookies.get('cr'),set());rows=[r for r in rows if r['id'] in mine]
        return rows

    @app.get('/api/analyses/{id}')
    def get(id:str):
        try:return store.get(id)
        except (ValueError,FileNotFoundError):raise HTTPException(404,'Analysis not found.')

    @app.get('/api/analyses/{id}/assets/{name}')
    def asset(id:str,name:str):
        if name not in ['image.png','mass.png','notan.png','chroma.png','edges.png','subjects.png','sky.png']:raise HTTPException(404)
        try:p=store.folder(id)/name
        except ValueError:raise HTTPException(404)
        if not p.is_file():raise HTTPException(404)
        return FileResponse(p)

    @app.get('/api/analyses/{id}/session')
    def session(id:str):
        r=get(id);data=(store.folder(id)/'source.bin').read_bytes()
        session={'schema':'composition-session/1','source_name':r['source']['name'],'options':r['options'],
                 'image_base64':base64.b64encode(data).decode(),'report':r}
        return Response(json.dumps(session),media_type='application/json',headers={'Content-Disposition':f'attachment; filename="composition-{id[:8]}.json"'})

    @app.get('/api/compare')
    def compare(before:str,after:str):return compare_reports(get(before),get(after))

    @app.get('/api/compare.csv')
    def compare_csv(before:str,after:str):
        r=compare_reports(get(before),get(after));out=io.StringIO();writer=csv.DictWriter(out,fieldnames=['id','name','family','before','after','change']);writer.writeheader();writer.writerows(r['hypotheses'])
        return Response(out.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="composition-comparison.csv"'})

    @app.get('/api/analyses/{id}/report.html')
    def html_report(id:str):
        from .report import render_html
        r=get(id);content=render_html(r,(store.folder(id)/'image.png').read_bytes())
        return HTMLResponse(content,headers={'Content-Disposition':f'attachment; filename="composition-{id[:8]}.html"'})

    app.mount('/',StaticFiles(directory=web,html=True),name='workbench')
    app.state.store=store
    return app
