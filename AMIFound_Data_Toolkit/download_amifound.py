#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMIFound original source downloader/entrypoint index.

Based on the PDF's Tables E35-E37 (benchmark sources) and A2 (diagnosis).
No claim is made that original sources are the prebuilt 3,765,591 HQ-LQ pairs.
Honors source-side authentication, permissions and licenses.

Examples:
  python download_amifound.py list --scope benchmark
  python download_amifound.py list --modality PET
  python download_amifound.py plan --id B072
  python download_amifound.py download --id B072 --accept-terms
  python download_amifound.py download --modality Ultrasound --accept-terms
  python download_amifound.py download --all --auto-only --accept-terms
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

MANIFEST = Path(__file__).resolve().with_name('amifound_sources.csv')


def load_rows(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    must = {'id', 'scope', 'modality', 'dataset', 'access_url', 'download_mode'}
    if not rows or not must.issubset(rows[0]):
        raise RuntimeError(f'Invalid manifest: {path}')
    return rows


def slugify(s):
    # Avoid path injection when constructing output directories.
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', s.strip()).strip('._')
    return (safe or 'unnamed')[:90]


def selection(rows, args):
    out = [r for r in rows if args.scope == 'all' or r['scope'] == args.scope]
    if args.modality and args.modality.lower() != 'all':
        out = [r for r in out if r['modality'].lower() == args.modality.lower()]
    if args.id:
        ids = {v.upper() for v in args.id}
        out = [r for r in out if r['id'].upper() in ids]
    if args.name:
        out = [r for r in out if args.name.lower() in r['dataset'].lower()]
    if args.auto_only:
        out = [r for r in out if r['download_mode'] in {'kaggle_dataset','kaggle_competition','zenodo_record'}]
    return out


def output_dir(base: Path, row):
    return base / row['scope'] / row['modality'] / f"{row['id']}_{slugify(row['dataset'])}"


def kaggle_ref(url: str, mode: str):
    u = urlparse(url)
    p = [part for part in u.path.strip('/').split('/') if part]
    if mode == 'kaggle_competition':
        if len(p) >= 2 and p[0] in ('c', 'competitions'):
            return p[1]
        raise ValueError(f'Cannot identify Kaggle competition in {url}')
    if p and p[0] == 'datasets':
        p = p[1:]
    if len(p) < 2 or p[0] in {'c', 'competitions'}:
        raise ValueError(f'Cannot identify Kaggle dataset in {url}')
    return f'{p[0]}/{p[1]}'


def download_kaggle(row, target: Path):
    if shutil.which('kaggle') is None:
        raise RuntimeError('Kaggle CLI not found. Install: python -m pip install kaggle; configure API credentials.')
    mode = row['download_mode']
    ref = kaggle_ref(row['access_url'], mode)
    cmd = (['kaggle', 'competitions', 'download', '-c', ref, '-p', str(target)]
           if mode == 'kaggle_competition' else
           ['kaggle', 'datasets', 'download', '-d', ref, '-p', str(target)])
    print('RUN:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)  # No shell expansion or implicit bypass of terms.
    return 'Kaggle CLI completed; downloaded archives are not automatically extracted.'


def requests_lib():
    try:
        import requests
        return requests
    except ImportError as e:
        raise RuntimeError('Install dependency: python -m pip install requests') from e


def verify_md5(path, checksum):
    if not checksum or not checksum.startswith('md5:'):
        return True
    h = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest().lower() == checksum.split(':', 1)[1].lower()


def stream_file(requests, url, path, checksum='', size=None):
    # Resume safely using byte ranges; if server ignores Range, restart from 0.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    if path.exists():
        if (size is None or path.stat().st_size == size) and verify_md5(path, checksum):
            print('  SKIP verified:', path.name)
            return
        print('  Re-fetching unverified file:', path.name)
        path.rename(temp)
    for attempt in range(1, 5):
        try:
            existing = temp.stat().st_size if temp.exists() else 0
            headers = {'Range': f'bytes={existing}-'} if existing else {}
            with requests.get(url, headers=headers, stream=True, timeout=(20, 180),
                              allow_redirects=True) as resp:
                if resp.status_code == 416:
                    if size is not None and existing == size and verify_md5(temp, checksum):
                        temp.rename(path); return
                    temp.unlink(missing_ok=True)
                    raise RuntimeError('Server rejected the resume range; removed partial file for retry.')
                resp.raise_for_status()
                append = existing > 0 and resp.status_code == 206
                with temp.open('ab' if append else 'wb') as f:
                    for block in resp.iter_content(chunk_size=4*1024*1024):
                        if block: f.write(block)
            if size is not None and temp.stat().st_size != size:
                raise RuntimeError(f'File-size mismatch: expected {size}, got {temp.stat().st_size}')
            if not verify_md5(temp,checksum):
                temp.unlink(missing_ok=True)
                raise RuntimeError(f'MD5 mismatch: {path.name}')
            temp.replace(path)
            print('  OK:', path.name)
            return
        except Exception as exc:
            if attempt == 4: raise
            print(f'  Retry {attempt}/3 for {path.name}: {exc}')
            time.sleep(min(2**attempt, 10))


def download_zenodo(row, target):
    requests = requests_lib()
    m = re.search(r'zenodo\.org/records?/(\d+)', row['access_url'])
    if not m: raise ValueError('Invalid Zenodo record URL')
    record_id = m.group(1)
    api = f'https://zenodo.org/api/records/{record_id}'
    response = requests.get(api,timeout=30)
    response.raise_for_status()
    record = response.json()
    files = record.get('files', [])
    if not files: raise RuntimeError(f'Zenodo record {record_id} returned no downloadable files')
    n=0
    for item in files:
        key = item.get('key')
        content_url = item.get('links', {}).get('content') or item.get('links', {}).get('self')
        if not content_url or not key:
            raise RuntimeError(f'Incomplete file metadata in record {record_id}')
        # Preserve subdirectories where safe, never allow .. or absolute paths.
        sub = Path(key)
        if sub.is_absolute() or any(p in ('..','') for p in sub.parts):
            raise RuntimeError(f'Unsafe Zenodo filename: {key}')
        stream_file(requests, content_url, target/sub,
                    checksum=item.get('checksum', ''), size=item.get('size'))
        n+=1
    (target/'zenodo_record_metadata.json').write_text(json.dumps(record, ensure_ascii=False, indent=2),encoding='utf-8')
    return f'Zenodo record {record_id}: {n} files fetched/verified.'


def log_result(base, row, status, message):
    data = dict(id=row['id'], dataset=row['dataset'], status=status, message=message,
                when=datetime.now(timezone.utc).isoformat())
    with (base/'download_log.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(data,ensure_ascii=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description='Download/index AMIFound original-source datasets from NMI PDF.')
    parser.add_argument('command', choices=['list', 'plan', 'download'])
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--out', type=Path, default=Path('./AMIFound_raw_sources'))
    parser.add_argument('--scope', choices=['all','benchmark','diagnosis'],default='all')
    parser.add_argument('--modality',choices=['all','CT','MRI','PET','X-ray','Endoscopy','Ultrasound','Fundus'],default='all')
    parser.add_argument('--id',action='append',help='Source ID, e.g. B072; may be repeated')
    parser.add_argument('--name',help='Case-insensitive name substring, e.g. BUSI')
    parser.add_argument('--auto-only',action='store_true',help='Only entries with supported API/CLI downloads')
    parser.add_argument('--all',action='store_true',help='Allow bulk downloading (may require vast disk space)')
    parser.add_argument('--accept-terms', action='store_true',help='Confirm you have read and will obey each provider license, permissions, and usage terms')
    args=parser.parse_args()
    rows=load_rows(args.manifest)
    selected=selection(rows,args)
    if not selected:
        parser.error('No sources match the filters')
    if args.command == 'list':
        print(f'{"ID":<6} {"SCOPE":<10} {"MODALITY":<12} {"MODE":<20} DATASET')
        for r in selected:
            print(f"{r['id']:<6} {r['scope']:<10} {r['modality']:<12} {r['download_mode']:<20} {r['dataset']}")
        print(f'\n{len(selected)} sources. Use plan to view official links; download requires --accept-terms.')
        return 0
    if args.command == 'plan':
        for r in selected:
            print(f"{r['id']} | {r['scope']} | {r['modality']} | {r['dataset']} | {r['download_mode']}")
            print(f"    Official source: {r['access_url'] or '[not given in PDF]'}")
            print(f"    Output folder: {output_dir(args.out,r)}")
        return 0
    if not args.accept_terms:
        parser.error('Download requires --accept-terms; a provider may require separate registration/approval.')
    if not args.id and not args.name and not args.all:
        parser.error('Specify --id, --name or --all before downloading; bulk download may require terabytes.')
    base=args.out.resolve()
    base.mkdir(parents=True,exist_ok=True)
    unavailable=[]
    failures=[]
    completed=0
    for i,row in enumerate(selected,1):
        target=output_dir(base,row)
        mode=row['download_mode']
        print(f'\n[{i}/{len(selected)}] {row["id"]} | {row["modality"]} | {row["dataset"]}')
        if mode not in {'kaggle_dataset','kaggle_competition','zenodo_record'}:
            print('  MANUAL (no direct dataset URL or requires provider-specific process):', row['access_url'] or '[unlisted]')
            unavailable.append(row)
            log_result(base,row,'manual','Check original webpage and license; no direct API download configured')
            continue
        target.mkdir(parents=True,exist_ok=True)
        marker=target/'.download_complete.json'
        if marker.exists():
            print('  SKIP: previous completed download. Remove .download_complete.json to force a recheck.')
            completed+=1;continue
        try:
            result=(download_kaggle(row,target) if mode.startswith('kaggle_') else download_zenodo(row,target))
            marker.write_text(json.dumps({'id':row['id'],'url':row['access_url'],'completed_utc':datetime.now(timezone.utc).isoformat(),'message':result},indent=2),encoding='utf-8')
            log_result(base,row,'downloaded',result)
            print('  DONE:', result)
            completed+=1
        except Exception as e:
            msg=f'{type(e).__name__}: {e}'
            print('  ERROR:',msg, file=sys.stderr)
            log_result(base,row,'failed',msg)
            failures.append((row['id'],msg))
    if unavailable:
        with (base/'manual_access_list.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()));writer.writeheader();writer.writerows(unavailable)
        print(f'\n{len(unavailable)} entries require manual access. See: {base / "manual_access_list.csv"}')
    print(f'\nSummary: completed or skipped={completed}, manual={len(unavailable)}, failed={len(failures)}')
    return 1 if failures else 0

if __name__=='__main__':
    sys.exit(main())
