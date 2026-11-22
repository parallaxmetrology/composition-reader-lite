import base64
import io
import json
import time
from PIL import Image
from fastapi.testclient import TestClient
from composition_reader.server import create_app


def test_complete_service_session_comparison_and_exports(tmp_path):
    im=Image.new('RGB',(64,64),'white');out=io.BytesIO();im.save(out,format='PNG')
    image=base64.b64encode(out.getvalue()).decode()
    with TestClient(create_app(tmp_path),base_url='http://localhost') as client:
        assert client.get('/').status_code==200
        assert client.get('/api/health').json()['ok']
        submit=client.post('/api/analyze',json={'image':image,'name':'<script>study.png','null_samples':0,'stability':False})
        assert submit.status_code==202
        job_id=submit.json()['job_id'];deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            job=client.get('/api/jobs/'+job_id).json()
            if job['status'] in ['complete','failed']:break
            time.sleep(.05)
        assert job['status']=='complete',job
        id=job['analysis_id'];base='/api/analyses/'+id
        report=client.get(base).json()
        assert report['summary']['blank']
        assert client.get(report['assets']['image']).headers['content-type']=='image/png'
        assert client.get(report['assets']['subjects']).headers['content-type']=='image/png'
        assert 'subject_selection' in report and 'subjects' in report
        session=client.get(base+'/session').json()
        assert session['image_base64']==image and session['options']==report['options']
        html=client.get(base+'/report.html').text
        assert '&lt;script&gt;' in html and '<script>study.png' not in html
        compare=client.get('/api/compare',params={'before':id,'after':id}).json()
        assert all(h['change']==0 for h in compare['hypotheses'])
        assert client.get('/api/compare.csv',params={'before':id,'after':id}).text.startswith('id,name,family,')
        assert len(client.get('/api/analyses').json())==1
        assert client.get(base+'/assets/source.bin').status_code==404
    # Reports survive a server restart.
    with TestClient(create_app(tmp_path),base_url='http://localhost') as client:
        assert client.get(base).json()['id']==id


def test_invalid_requests_and_local_boundary(tmp_path):
    with TestClient(create_app(tmp_path),base_url='http://localhost') as client:
        assert client.post('/api/analyze',json={}).status_code==400
        assert client.post('/api/analyze',json={'image':'!'}).status_code==400
        for values in [{'null_samples':999},{'anchors':[[2,.5]]},{'crop':[0,0,0,1]},{'deep':'unexpected'},{'unknown':True}]:
            assert client.post('/api/analyze',json={'example':'crow_0.png',**values}).status_code==422
        assert client.post('/api/analyze',json={'example':'../crow_0.png'}).status_code==400
        assert client.post('/api/analyze',json={},headers={'Origin':'https://foreign.example'}).status_code==403
        assert client.get('/api/health',headers={'Host':'foreign.example'}).status_code==400
        assert client.post('/api/analyze',content=b'{}',headers={'Content-Length':'43000000'}).status_code==413
        assert client.get('/api/analyses/not-an-id').status_code==404
