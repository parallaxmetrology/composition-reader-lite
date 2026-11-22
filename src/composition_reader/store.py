from __future__ import annotations
import json, os, re, time, uuid
from pathlib import Path
from .model import native

class Store:
    def __init__(self,path):
        self.root=Path(path).resolve();self.root.mkdir(parents=True,exist_ok=True)
    def folder(self,id):
        if not re.fullmatch(r'[a-f0-9]{32}',id):raise ValueError('Invalid analysis identifier.')
        return self.root/id
    def save(self,report,fields,original):
        id=uuid.uuid4().hex;folder=self.folder(id);folder.mkdir()
        report={**report,'id':id,'created_at':time.time(),'assets':{k:f'/api/analyses/{id}/assets/{k}.png' for k in fields}}
        for key,data in fields.items():(folder/(key+'.png')).write_bytes(data)
        (folder/'source.bin').write_bytes(original)
        tmp=folder/'report.tmp';tmp.write_text(json.dumps(native(report),indent=2,allow_nan=False));tmp.replace(folder/'report.json')
        return report
    def get(self,id):return json.loads((self.folder(id)/'report.json').read_text())
    def list(self):
        out=[]
        for p in self.root.glob('*/report.json'):
            try:
                r=json.loads(p.read_text());out.append({'id':r['id'],'name':r['source']['name'],'created_at':r['created_at'],
                'image':r['assets']['image'],'elapsed_seconds':r['elapsed_seconds'],'blank':r['summary']['blank']})
            except (ValueError,KeyError):continue
        return sorted(out,key=lambda r:-r['created_at'])
