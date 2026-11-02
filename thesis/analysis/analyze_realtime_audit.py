"""Recompute all metrics from independently saved per-frame observations."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

from thesis.platforms.jetson_nano.evaluation.audit_realtime import metric_counts, metrics, percentiles


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--pc-root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    run=args.run
    manifest=json.loads((run/'manifest.json').read_text())
    aggregate=json.loads((run/'aggregate.json').read_text())
    categories=collections.Counter()
    modes=collections.defaultdict(collections.Counter)
    all_age=[]
    accepted_age=[]
    intervals=[]
    visible_distances=[]
    per_clip=[]
    hashes=[]
    for s in aggregate['summaries']:
        stem=s['key'].replace('/','__')
        rows=[json.loads(x) for x in gzip.open(str(run/(stem+'.frames.jsonl.gz')),'rt')]
        arrivals=[json.loads(x) for x in gzip.open(str(run/(stem+'.arrivals.jsonl.gz')),'rt')]
        gt={x['frame']:dict(vis=x['visibility'],x=x['gt_x'],y=x['gt_y']) for x in rows}
        preds={x['frame']:(x['pred_x'],x['pred_y']) for x in rows if x['pred_x'] is not None}
        counts, rescored=metric_counts(gt,preds)
        assert counts==s['counts'],s['key']
        assert [x['category'] for x in rows]==[x['category'] for x in rescored]
        assert len(arrivals)==s['legacy']['ingested']
        assert [x['frame'] for x in arrivals]==list(range(len(arrivals)))
        counts=collections.Counter(counts)
        categories.update(counts)
        longest=cur=0
        previous_fid=None
        for x in rows:
            modes[x['mode'] or 'no_temporal_window'][x['category']]+=1
            if previous_fid is not None and x['frame']!=previous_fid+1:
                cur=0
            if x['visibility']==1 and x['category']!='tp':
                cur+=1
            else:
                cur=0
            longest=max(longest,cur)
            previous_fid=x['frame']
            if x['prediction_age_ms'] is not None:
                all_age.append(x['prediction_age_ms'])
                if x['pred_x'] is not None:
                    accepted_age.append(x['prediction_age_ms'])
            if x['visibility']==1 and x['distance_px'] is not None:
                visible_distances.append(x['distance_px'])
        intervals.extend(1000*(b['read_end_s']-a['read_end_s']) for a,b in zip(arrivals,arrivals[1:]))
        m=metrics(counts)
        per_clip.append(dict(key=s['key'],**m,longest_visible_failure_run_frames=longest,
                             dropped=s['legacy']['dropped'],fps=s['legacy']['fps'],
                             tp=counts['tp'],wrong_location=counts['wrong_location'],
                             missed_visible=counts['missed_visible'],
                             prediction_age_p95_ms=s['prediction_age_ms'].get('p95'),
                             legacy_exact_match=s['legacy_counts_match_previous']))
    assert dict(categories)==aggregate['counts']
    for meta in manifest['clip_metadata']:
        match,clip=meta['key'].split('/')
        base=args.pc_root/'dataset/raw/part1'
        if not base.exists():
            base=args.pc_root/'raw/part1'
        gt=base/match/'csv'/(clip+'_ball.csv')
        hashes.append(dict(key=meta['key'],gt_hash_matches_pc=hashlib.sha256(gt.read_bytes()).hexdigest()==meta['gt_sha256']))
    frames=sum(s['legacy']['ingested'] for s in aggregate['summaries'])
    dropped=sum(s['legacy']['dropped'] for s in aggregate['summaries'])
    targets=sum(s['evaluated_targets'] for s in aggregate['summaries'])
    proc_seconds=sum(s['legacy']['processed']/s['legacy']['fps'] for s in aggregate['summaries'])
    sensitivity={}
    for threshold in (5,10,15,20):
        alternative=dict(categories)
        alternative['tp']=sum(d<=threshold for d in visible_distances)
        alternative['wrong_location']=len(visible_distances)-alternative['tp']
        sensitivity[str(threshold)]=metrics(alternative)
    report=dict(completed=aggregate['completed'],intended=aggregate['intended'],counts=dict(categories),
                corrected=metrics(categories),ingested_frames=frames,dropped_frames=dropped,
                global_drop_rate=dropped/frames,zero_drop_clips=aggregate['zero_drop_clips'],
                processed_fps_weighted=(frames-dropped)/proc_seconds,
                model_decision_fps_weighted=targets/proc_seconds,
                mean_clip_fps=statistics.mean(s['legacy']['fps'] for s in aggregate['summaries']),
                input_fps_range=[min(s['legacy']['fps'] for s in aggregate['summaries']),
                                 max(s['legacy']['fps'] for s in aggregate['summaries'])],
                gt_labelled_frames=sum(categories.values()),
                unlabelled_input_frames=frames-sum(categories.values()),
                evaluated_target_frames=targets,temporal_gap_windows=sum(s['temporal_gap_windows'] for s in aggregate['summaries']),
                header_ingestion_mismatches=[s['key'] for s in aggregate['summaries'] if not s['header_matches_ingested']],
                legacy_exact_matches=aggregate['legacy_exact_matches'],
                prediction_age_ms=percentiles(all_age),accepted_prediction_age_ms=percentiles(accepted_age),
                read_interval_ms=percentiles(intervals),
                note='Ages cover GT-labelled frames only, include middle-frame wait, queue and processing; exclude physical camera/rendering. prediction_age_ms includes no-detection decisions.',
                modes={k:dict(v) for k,v in modes.items()},
                visible_output_error_px=percentiles(visible_distances),
                distance_tolerance_sensitivity=sensitivity,
                gt_hashes=hashes,per_clip=per_clip,
                fps_weighting='Total processed / sum of per-clip measured processing wall time; no startup/hash/serialization time.')
    (run/'independent_analysis.json').write_text(json.dumps(report,indent=2))
    with (run/'independent_per_clip.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_clip[0]));w.writeheader();w.writerows(per_clip)
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_clip','gt_hashes')},indent=2))


if __name__=='__main__':
    main()
