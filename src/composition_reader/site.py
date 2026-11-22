"""Public-site entry point (Railway). Same engine as the local app.

Images are analysed in memory and written only to a temporary folder that is
emptied after COMPOSITION_RETENTION_MINUTES (default 60). Nothing is kept.
"""
from __future__ import annotations
import os, shutil, tempfile, threading, time
from pathlib import Path

def prune(root: Path, minutes: float):
    cutoff = time.time() - minutes * 60
    for d in root.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass

def main():
    import uvicorn
    from .server import create_app
    root = Path(os.environ.get('COMPOSITION_DATA') or Path(tempfile.gettempdir()) / 'composition-site')
    root.mkdir(parents=True, exist_ok=True)
    minutes = float(os.environ.get('COMPOSITION_RETENTION_MINUTES', '60'))
    def loop():
        while True:
            prune(root, minutes)
            time.sleep(300)
    threading.Thread(target=loop, daemon=True).start()
    os.environ.setdefault('COMPOSITION_ALLOWED_HOSTS', '*'); os.environ['COMPOSITION_PUBLIC'] = '1'
    port = int(os.environ.get('PORT', '8080'))
    uvicorn.run(create_app(str(root)), host='0.0.0.0', port=port,
                proxy_headers=True, forwarded_allow_ips='*')

if __name__ == '__main__':
    main()
