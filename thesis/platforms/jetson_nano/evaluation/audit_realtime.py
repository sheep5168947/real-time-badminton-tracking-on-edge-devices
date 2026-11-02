"""Independent audit of an unchanged Nano benchmark (Python 3.6 compatible).

Instrumentation only: timestamps and per-frame observations are added to the
source in memory. No threshold, crop, model, KF or decision logic is changed.
The original source bytes, hashes, legacy scores and corrected scores are saved.
"""
import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def metric_counts(gt, predictions, threshold=15.0):
    counts = dict(tp=0, tn=0, wrong_location=0, false_positive_absent=0,
                  missed_visible=0, unknown_visibility=0)
    rows = []
    for fid, item in sorted(gt.items()):
        pred = predictions.get(fid)
        dist = None
        if item['vis'] not in (0, 1):
            category = 'unknown_visibility'
        elif item['vis'] == 1:
            if pred is None:
                category = 'missed_visible'
            else:
                dist = math.hypot(pred[0]-item['x'], pred[1]-item['y'])
                category = 'tp' if math.isfinite(dist) and dist <= threshold else 'wrong_location'
        else:
            category = 'tn' if pred is None else 'false_positive_absent'
        counts[category] += 1
        rows.append(dict(frame=fid, visibility=item['vis'], gt_x=item['x'], gt_y=item['y'],
                         pred_x=None if pred is None else float(pred[0]),
                         pred_y=None if pred is None else float(pred[1]),
                         distance_px=dist, category=category))
    return counts, rows


def metrics(c):
    tp = c['tp']
    fp = c['wrong_location'] + c['false_positive_absent']
    fn = c['missed_visible'] + c['wrong_location']
    visible = tp + fn
    return dict(precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/visible if visible else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                legacy_recall=tp/(tp+c['missed_visible']) if tp+c['missed_visible'] else None,
                visible_frames=visible, corrected_fp=fp, corrected_fn=fn)


def percentiles(values):
    if not values:
        return dict(n=0)
    vals = sorted(values)
    def q(p):
        return vals[int(round((len(vals)-1)*p))]
    return dict(n=len(vals), mean=sum(vals)/len(vals), p50=q(.5),
                p95=q(.95), p99=q(.99), maximum=vals[-1])


def instrument(source):
    changes = [
        ('    predictions = {}\n', '    predictions = {}\n    audit_arrivals = {}\n    audit_trace = []\n    audit_read_start = time.perf_counter()\n'),
        ('            ret, frame = cap.read()\n',
         '            audit_before_read = time.perf_counter()\n            ret, frame = cap.read()\n            audit_after_read = time.perf_counter()\n'),
        ('            stats["ingested"] += 1\n',
         '            audit_arrivals[f_idx] = (audit_before_read, audit_after_read)\n            stats["ingested"] += 1\n'),
        ('        stats["processed"] += 1\n',
         '        audit_enter = time.perf_counter()\n        stats["processed"] += 1\n'),
        ('        target_mid_fid = fid_buffer[1]\n',
         '        target_mid_fid = fid_buffer[1]\n        audit_mode = "rescue" if is_lost else "local"\n'),
        ('            predictions[target_mid_fid] = pred_pt\n',
         '            predictions[target_mid_fid] = pred_pt\n'
         '        audit_trace.append((target_mid_fid, tuple(fid_buffer), audit_enter, time.perf_counter(), audit_mode, pred_pt is not None))\n'),
        ('        "TP": TP, "FP": FP, "FN": FN,\n',
         '        "_audit": (gt_dict, predictions, audit_arrivals, audit_trace, audit_read_start, total_time),\n        "TP": TP, "FP": FP, "FN": FN,\n')]
    for before, after in changes:
        if source.count(before) != 1:
            raise RuntimeError('Source changed; ambiguous instrumentation anchor: '+repr(before))
        source = source.replace(before, after)
    ast.parse(source)
    return source


def read_system():
    paths = ['/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq',
             '/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq',
             '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq',
             '/sys/kernel/debug/clock/override.emc/rate']
    out = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    for path in paths:
        try:
            out[path] = Path(path).read_text().strip()
        except OSError:
            pass
    temps = {}
    for p in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            temps[(p/'type').read_text().strip()] = int((p/'temp').read_text())
        except (OSError, ValueError):
            pass
    out['temperatures_mC'] = temps
    return out


def git_info(root):
    out = {}
    for key, args in [('head', ['rev-parse', 'HEAD']), ('status', ['status', '--short'])]:
        out[key] = subprocess.check_output(['git', '-C', str(root)] + args, universal_newlines=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/nano/Desktop/tracknet-series-pytorch'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only', help='One match/clip key, e.g. match1/1_01_00')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    source = args.source or root/'benchmark_nano_all146_matches.py'
    output.mkdir(parents=True, exist_ok=False)
    original = source.read_bytes()
    (output/'source_original.py').write_bytes(original)
    instrumented = instrument(original.decode('utf-8-sig'))
    (output/'source_instrumented.py').write_text(instrumented, encoding='utf-8')
    ns = dict(__name__='audit_original', __file__=str(source))
    exec(compile(instrumented, str(source), 'exec'), ns)
    import numpy as np
    import cv2
    import tensorrt as trt
    import pycuda.driver as cuda

    clips = []
    missing_gt = []
    for m in range(1, 24):
        folder = root/'raw/part1'/('match'+str(m))
        for video in sorted((folder/'video').glob('*.mp4')):
            gt = folder/'csv'/(video.stem+'_ball.csv')
            if not gt.exists():
                missing_gt.append(str(video))
                continue
            clips.append(('match'+str(m)+'/'+video.stem, video, gt))
    all_keys = [x[0] for x in clips]
    if args.only:
        clips = [x for x in clips if x[0] == args.only]
        assert clips, 'Requested clip does not exist'
    if args.limit:
        clips = clips[:args.limit]
    eng128 = root/'weights/miniroi_128x128_augmented.engine'
    eng256 = root/'weights/miniroi_256x144_augmented.engine'
    old_csv = root/'report_materials/nano_realtime_full_146videos_results.csv'
    old_rows = {}
    if old_csv.exists():
        shutil.copyfile(str(old_csv), str(output/'previous_results.csv'))
        with old_csv.open() as f:
            for row in csv.DictReader(f):
                key = row['Match']+'/'+row['Clip']
                assert key not in old_rows, 'Duplicate key in previous CSV'
                old_rows[key] = row
    manifest = dict(start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    original_source=str(source), source_sha256=digest(source),
                    auditor_sha256=digest(Path(__file__)), git=git_info(root),
                    engine_sha256={str(p):digest(p) for p in (eng128, eng256)},
                    python=platform.python_version(), numpy=np.__version__,
                    opencv=cv2.__version__, tensorrt=trt.__version__,
                    discovered_keys=all_keys, selected_keys=[x[0] for x in clips],
                    missing_gt=missing_gt, previous_only=sorted(set(old_rows)-set(all_keys)),
                    missing_previous=sorted(set(all_keys)-set(old_rows)),
                    thresholds=dict(track=.18, coarse=.16, verify=.30, distance_original_pixels=15),
                    instrumentation='timestamps and observation lists only; unchanged source decision logic',
                    timing_scope='decoded video 30Hz replay; excludes physical sensor, rendering and encoding',
                    system_start=read_system(), clip_metadata=[])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    totals = dict(tp=0, tn=0, wrong_location=0, false_positive_absent=0, missed_visible=0, unknown_visibility=0)
    summaries = []
    cuda_ctx = cuda.Device(0).make_context()
    try:
        model128 = ns['FastTRTModel'](str(eng128), cuda_ctx)
        model256 = ns['FastTRTModel'](str(eng256), cuda_ctx)
        buf128 = cuda.pagelocked_empty((1,9,128,128), dtype=np.float32)
        buf256 = cuda.pagelocked_empty((1,9,144,256), dtype=np.float32)
        buf128.fill(0)
        buf256.fill(0)
        for _ in range(10):
            model128.infer(buf128)
            model256.infer(buf256)
        for i, (key, video, gt_path) in enumerate(clips, 1):
            cap = cv2.VideoCapture(str(video))
            meta = dict(key=key, video_bytes=video.stat().st_size,
                        video_sha256=digest(video), gt_sha256=digest(gt_path),
                        width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                        source_fps=cap.get(cv2.CAP_PROP_FPS), header_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            print('START {}/{} {}'.format(i,len(clips),key), flush=True)
            res = ns['run_realtime_clip'](str(video),str(gt_path), model128,model256,buf128,buf256,
                                          t_track=.18,t_coarse=.16,t_verify=.30)
            gt, predictions, arrivals, trace, read_start, proc_total = res.pop('_audit')
            counts, rows = metric_counts(gt,predictions)
            trace_map = {t[0]:t for t in trace}
            assert len(trace_map)==len(trace), 'Repeated target frame IDs'
            assert res['processed']+res['dropped']==res['ingested'], 'Frame accounting mismatch'
            assert len(arrivals)==res['ingested']
            assert res['TP']==counts['tp']
            assert res['FP']==counts['wrong_location']+counts['false_positive_absent']
            assert res['FN']==counts['missed_visible']
            e2e, proc, gaps = [], [], 0
            for fid, ids, enter, finished, mode, found in trace:
                e2e.append(1000*(finished-arrivals[fid][1]))
                proc.append(1000*(finished-enter))
                gaps += ids[1]!=ids[0]+1 or ids[2]!=ids[1]+1
            for row in rows:
                t = trace_map.get(row['frame'])
                row['evaluated_by_model'] = t is not None
                row['input_frame_ids'] = list(t[1]) if t else None
                row['mode'] = t[4] if t else None
                row['mid_read_end_s'] = arrivals[row['frame']][1]-read_start if row['frame'] in arrivals else None
                row['output_s'] = t[3]-read_start if t else None
                row['prediction_age_ms'] = (t[3]-arrivals[row['frame']][1])*1000 if t else None
            name = key.replace('/','__')
            with gzip.open(str(output/(name+'.frames.jsonl.gz')), 'wt', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
            with gzip.open(str(output/(name+'.arrivals.jsonl.gz')), 'wt', encoding='utf-8') as f:
                for fid,(begin,end) in sorted(arrivals.items()):
                    f.write(json.dumps(dict(frame=fid,read_start_s=begin-read_start,read_end_s=end-read_start))+'\n')
            old = old_rows.get(key)
            summary = dict(key=key, legacy=res, counts=counts, corrected=metrics(counts),
                           prediction_age_ms=percentiles(e2e), full_processing_ms=percentiles(proc),
                           evaluated_targets=len(trace), temporal_gap_windows=gaps,
                           header_matches_ingested=meta['header_frames']==res['ingested'],
                           gt_frame_count=len(gt), gt_outside_video=sum(fid<0 or fid>=res['ingested'] for fid in gt),
                           old_counts=None if old is None else {k:int(old[k]) for k in ('TP','FP','FN')},
                           legacy_counts_match_previous=None if old is None else all(res[k]==int(old[k]) for k in ('TP','FP','FN')),
                           system=read_system())
            for k,v in counts.items():
                totals[k] += v
            summaries.append(summary)
            manifest['clip_metadata'].append(meta)
            (output/(name+'.summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
            aggregate = dict(completed=len(summaries), intended=len(clips), counts=totals,corrected=metrics(totals),
                             zero_drop_clips=sum(s['legacy']['dropped']==0 for s in summaries),
                             total_ingested=sum(s['legacy']['ingested'] for s in summaries),
                             total_dropped=sum(s['legacy']['dropped'] for s in summaries),
                             legacy_exact_matches=sum(s['legacy_counts_match_previous'] is True for s in summaries),
                             summaries=summaries)
            (output/'aggregate.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print('DONE '+key+' '+json.dumps({k:summary[k] for k in ('counts','corrected','legacy_counts_match_previous')}),flush=True)
        manifest['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        manifest['source_unchanged']=digest(source)==manifest['source_sha256']
        manifest['git_end']=git_info(root)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('COMPLETE '+str(output),flush=True)
    finally:
        cuda_ctx.pop()


if __name__=='__main__':
    main()
