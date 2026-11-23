"""Portable, self-contained evidence report; no server or external assets required."""
import base64
import html
import json


def render_html(report, image_bytes):
    r=report;esc=html.escape
    image=base64.b64encode(image_bytes).decode()
    width=600;height=width*r['source'].get('analysis_size',r['source']['working_size'])[1]/r['source'].get('analysis_size',r['source']['working_size'])[0]
    def svg(geometry,hypothesis):
        shapes=[]
        for g in geometry:
            if g['kind']=='point':
                x,y=g['point'];shapes.append(f'<circle cx="{x*width}" cy="{y*height}" r="5"/>')
            elif g.get('points'):
                coords=' '.join(f'{x*width},{y*height}' for x,y in g['points'])
                tag='polygon' if g.get('closed') else 'polyline'
                shapes.append(f'<{tag} points="{coords}"/>')
        anchors=[]
        if hypothesis['family']=='placement':
            subjects=r.get('subjects',r.get('regions',[]))
            for subject in subjects:
                if not subject.get('included',True):continue
                x,y=subject['center'];contour=subject.get('contour',[])
                if contour:
                    coordinates=' '.join(f'{a*width},{b*height}' for a,b in contour)
                    anchors.append(f'<polygon points="{coordinates}" fill="none"/>')
                anchors.append(f'<circle cx="{x*width}" cy="{y*height}" r="4"/><text x="{x*width+7}" y="{y*height-7}" fill="#a5c9ff" stroke="none" font-size="12">{esc(subject["id"])}</text>')
                for ex,ey in subject.get('eyes',[]):anchors.append(f'<circle cx="{ex*width}" cy="{ey*height}" r="3" fill="#d36eac"/>')
        return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Measured geometry"><use href="#source-image"/><g fill="none" stroke="#ffe099" stroke-width="2">{"".join(shapes)}</g><g fill="#a5c9ff" stroke="#a5c9ff" stroke-width="1.5">{"".join(anchors)}</g></svg>'
    cards=[]
    for h in r['hypotheses']:
        control=h.get('control');sensitivity=h.get('stability')
        note=(f'{esc(control["method"])}. {control["samples"]} samples; method tail {control["tail_fraction"]:.3f}; {esc(control.get("comparison_label","Family maximum"))} tail {control.get("adjusted_tail_fraction",control["family_max_tail_fraction"]):.3f}.' if control else 'Descriptive measurement; no randomized control for this method.')
        if sensitivity:note+=f' Blur/resampling maximum change: {sensitivity["max_fit_change"]:.3f}.'
        if sensitivity and 'max_contact_weight_change' in sensitivity:note+=f' Frame-contact weight change: {sensitivity["max_contact_weight_change"]:.3f} (sensitive above 0.10).'
        evidence=''.join(f'<li>{esc(str(e.get("element_id","Field")))}: {esc(e["measure"])} = {e["value"]:.4f} {esc(e["unit"])}</li>' for e in h['evidence'])
        limits=''.join(f'<li>{esc(s)}</li>' for s in h['limitations'])
        cards.append(f'<section><div>{svg(h["geometry"],h)}</div><div><h2>{esc(h["name"])}</h2><p><b>{esc(h["status"])} · {h["fit"]:.3f}</b></p><p>{esc(h["explanation"])}</p><p>{note}</p><details><summary>Measured evidence</summary><ul>{evidence}</ul></details><ul class="limitations">{limits}</ul></div></section>')
    summary='<h2>What the reader found</h2><ul>'+''.join(f'<li>{esc(x["text"])}</li>' for x in r.get('reading',[]))+'</ul>'
    if r.get('near_misses'):summary+='<h3>Close, but held back</h3><ul>'+''.join(f'<li><b>{esc(x["name"])}</b> · fit {x["fit"]:.2f}. {esc(x["reason"])}</li>' for x in r['near_misses'])+'</ul>'
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Composition Reader — {esc(r['source']['name'])}</title>
<style>body{{font:16px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:24px;color:#263139}}h1{{font-size:36px}}h2{{font-size:22px}}section{{display:grid;grid-template-columns:1fr 1fr;gap:30px;border-top:1px solid #ccd2d7;padding:28px 0;break-inside:avoid}}svg{{width:100%;max-height:560px;background:#16222b}}.limitations,small{{color:#52616e}}.reading-guide table{{border-collapse:collapse;width:100%;font-size:13px}}.reading-guide th,.reading-guide td{{padding:10px;border-bottom:1px solid #ccd2d7;text-align:left;vertical-align:top}}.reading-guide th{{min-width:125px}}.reading-guide dt{{font-weight:600;margin-top:12px}}.reading-guide dd{{margin:5px 0 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}@media(max-width:650px){{section{{grid-template-columns:1fr}}}}@media print{{body{{margin:0;padding:0;font-size:11px}}section{{gap:20px}}h2{{font-size:17px}}.raw{{display:none}}}}</style>
<h1>Composition Reader</h1><p>{esc(r['source']['name'])} · engine {esc(r['engine_version'])}</p>
<p>Geometric support values describe measured structure. They are not category probabilities or aesthetic grades. Multiple readings may coexist.</p>
{summary}
<small>Source SHA-256: {esc(r['source']['sha256'])}<br>Analysis frame: {esc(str(r['source'].get('analysis_size')))} · options: {esc(json.dumps(r['options']))}</small>
<svg width="0" height="0" style="position:absolute"><defs><image id="source-image" href="data:image/png;base64,{image}" width="{width}" height="{height}"/></defs></svg>
{''.join(cards)}<details class="raw"><summary>Complete reproducible analysis</summary><pre>{esc(json.dumps(r,indent=2))}</pre></details></html>'''
