"""Independent full-tracker ablation, matching realtime_student_pc_with_events.

Offline consecutive frames; this does not benchmark real-time frame dropping/FPS.
Score measurements separately from the displayed KF-predicted positions.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[2]
BASE=PROJECT
sys.path.insert(0,str(PROJECT))
from thesis.platforms.pc import student_realtime_events as rt


def outcome(point,label,tolerance):
    if label is None:
        return None,None
    visible=int(label['Visibility'])==1
    if point is None:
        return ('FN' if visible else 'TN'),None
    if not visible:
        return 'FP',None
    distance=float(np.hypot(point[0]-float(label['X']),point[1]-float(label['Y'])))
    return ('TP' if distance<=tolerance else 'FP+FN'),distance


def self_test():
    gt={'Visibility':'1','X':'10','Y':'20'}
    assert outcome((10,20),gt,15)==('TP',0.0)
    assert outcome((25,20),gt,15)==('TP',15.0)
    assert outcome((26,20),gt,15)==('FP+FN',16.0)
    assert outcome(None,gt,15)==('FN',None)
    assert outcome(None,{'Visibility':'0'},15)==('TN',None)
    assert outcome((0,0),{'Visibility':'0'},15)==('FP',None)
    assert outcome((0,0),None,15)==(None,None)
    print('Metric boundary, absence, missing-label, and localization-error checks passed.')


class Tracker:
    def __init__(self,use_mask):
        self.use_mask=use_mask
        self.lost=True
        self.misses=0
        self.kf=None
        self.last=(256,144)

    def step(self,infer):
        measurement=None
        conf=0.0
        local_used=global_used=False
        if not self.lost and self.kf is not None:
            local_used=True
            pred=self.kf.predict()
            px,py=int(pred[0,0]),int(pred[1,0])
            x,y,conf=infer((px,py),self.use_mask)
            if conf>=.55:
                self.last=(px-64+x,py-64+y)
                self.kf.correct(np.array([[self.last[0]],[self.last[1]]],np.float32))
                measurement=self.last
                self.misses=0
                status='local_hit'
            else:
                self.misses+=1
                if self.misses>=3:
                    self.lost=True
                else:
                    self.last=(px,py)
                status='kf_predict' if not self.lost else 'lost'
        if self.lost:
            global_used=True
            x,y,conf=infer(None,self.use_mask)
            if conf>.55:
                self.kf=rt.init_kalman_filter(x,y)
                self.last=(x,y)
                self.lost=False
                self.misses=0
                measurement=self.last
                status='global_hit'
            else:
                status='global_search'
        displayed=self.last if measurement is not None or not self.lost else None
        return measurement,displayed,conf,status,local_used,global_used


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--manifest',type=Path,help='Evaluation manifest CSV; required except for --self-test')
    parser.add_argument('--output-prefix',default='mask_ablation')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.manifest is None:
        parser.error('--manifest is required for evaluation')
    self_test()
    root=BASE/'model_results/evaluation'/datetime.now().strftime(args.output_prefix+'_%Y%m%d_%H%M%S')
    root.mkdir(parents=True)
    weights=PROJECT/'checkpoints_v4_student/epoch_30.pth'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=rt.MiniROIUNet().to(device)
    model.load_state_dict(torch.load(weights,map_location=device))
    model.eval()
    with args.manifest.open(encoding='utf-8-sig',newline='') as f:
        sequences=[r for r in csv.DictReader(f) if r['mode']=='offline']
    if not sequences:
        raise ValueError('Manifest has no offline sequences')
    thresholds=[15,20,30,50]
    config=dict(weights=str(weights),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                device=str(device),torch_version=torch.__version__,opencv_version=cv2.__version__,
                videos=len(sequences),mode='offline_consecutive_frames',mask_threshold=10,
                local_conf=.55,global_conf=.55,lost_duration=3,roi=128,
                resize='native -> 1280x720 -> 512x288',alignments=['middle','latest'],
                thresholds_native_px=thresholds,independent_tracking_states=True,
                outputs=['measurement','display_with_kf'],
                protocol='Wrong-location detection counts as FP and FN. No-label frames excluded. '
                'First two inputs have no output. GT is never used by tracker. '
                'Raw inference may be cached for identical inputs within one frame; no speed comparison.')
    (root/'protocol.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    counters=defaultdict(lambda:defaultdict(float))
    per_sequence=[]
    fields=['match','video','input_last_frame','middle_frame','variant','status','confidence',
            'local_used','global_used','measurement_x','measurement_y','display_x','display_y',
            'middle_visibility','latest_visibility','measurement_middle_distance_px',
            'measurement_latest_distance_px','measurement_middle_outcome15',
            'measurement_latest_outcome15','display_middle_outcome15','display_latest_outcome15']
    output=(root/'predictions.csv').open('w',encoding='utf-8-sig',newline='')
    writer=csv.DictWriter(output,fieldnames=fields)
    writer.writeheader()
    try:
        for seq_index,sequence in enumerate(sequences,1):
            match,name=sequence['match'],sequence['video']
            video=Path(sequence['video_path']) if sequence.get('video_path') else PROJECT/'dataset/raw/part1'/match/'video'/name
            labels_path=Path(sequence['labels_path']) if sequence.get('labels_path') else video.parent.parent/'csv'/f'{video.stem}_ball.csv'
            if not video.is_absolute():
                video=PROJECT/video
            if not labels_path.is_absolute():
                labels_path=PROJECT/labels_path
            with labels_path.open(encoding='utf-8-sig',newline='') as f:
                labels={int(r['Frame']):r for r in csv.DictReader(f)}
            cap=cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise OSError(str(video))
            expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width,height=cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            states={'mask':Tracker(True),'no_mask':Tracker(False)}
            frames=deque(maxlen=3)
            fid=-1
            seq_counters=defaultdict(lambda:defaultdict(float))
            try:
                while True:
                    ok,frame=cap.read()
                    if not ok:
                        break
                    fid+=1
                    frames.append(cv2.resize(cv2.resize(frame,(1280,720)),(512,288)))
                    if len(frames)<3:
                        continue
                    cache={}
                    def infer(center,use_mask):
                        if center not in cache:
                            images=list(frames) if center is None else [rt.crop_and_pad(im,*center,128)[0] for im in frames]
                            x=np.concatenate(images,axis=-1).astype(np.float32)/255
                            x=torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(device)
                            with torch.inference_mode():
                                raw=model(x).squeeze().cpu().numpy()
                            gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images[:2]]
                            _,mask=cv2.threshold(cv2.absdiff(gray[1],gray[0]),10,255,cv2.THRESH_BINARY)
                            mask=cv2.resize(mask.astype(np.float32)/255,(raw.shape[1],raw.shape[0]))
                            cache[center]=(raw,mask)
                        raw,mask=cache[center]
                        return rt.get_shuttle_center_local(raw*mask if use_mask else raw)
                    for variant,state in states.items():
                        measurement,display,conf,status,local_used,global_used=state.step(infer)
                        native=lambda point: None if point is None else (point[0]*width/512,point[1]*height/288)
                        measurement,display=native(measurement),native(display)
                        row=dict(match=match,video=name,input_last_frame=fid,middle_frame=fid-1,
                                 variant=variant,status=status,confidence=conf,local_used=local_used,global_used=global_used,
                                 measurement_x=measurement[0] if measurement else '',measurement_y=measurement[1] if measurement else '',
                                 display_x=display[0] if display else '',display_y=display[1] if display else '')
                        for tag,index in [('middle',fid-1),('latest',fid)]:
                            label=labels.get(index)
                            row[f'{tag}_visibility']=label['Visibility'] if label else ''
                            for output_type,point in [('measurement',measurement),('display',display)]:
                                for tolerance in thresholds:
                                    result,distance=outcome(point,label,tolerance)
                                    if tolerance==15:
                                        row[f'{output_type}_{tag}_outcome15']=result or ''
                                        if output_type=='measurement':
                                            row[f'measurement_{tag}_distance_px']=distance if distance is not None else ''
                                    key=(variant,output_type,tag,tolerance)
                                    for counter in [counters[key],seq_counters[key]]:
                                        if result is None:
                                            counter['missing_gt']+=1
                                            continue
                                        counter['frames']+=1
                                        counter['gt_visible']+=int(label['Visibility'])==1
                                        counter['predicted']+=point is not None
                                        counter['local_passes']+=local_used
                                        counter['global_passes']+=global_used
                                        if result=='FP+FN':
                                            counter['FP']+=1
                                            counter['FN']+=1
                                            counter['wrong_location']+=1
                                        else:
                                            counter[result]+=1
                                        if distance is not None:
                                            counter['visible_prediction_distance_sum']+=distance
                                            counter['visible_prediction_count']+=1
                        writer.writerow(row)
            finally:
                cap.release()
            if fid+1!=expected:
                raise RuntimeError(f'Incomplete decoding: {video}: {fid+1}/{expected}')
            for key,counter in seq_counters.items():
                per_sequence.append(dict(match=match,video=name,**finalize(key,counter)))
            output.flush()
            (root/'progress.json').write_text(json.dumps(dict(completed=seq_index,total=len(sequences),last_video=str(video))),encoding='utf-8')
            print(f'[{seq_index}/{len(sequences)}] {match}/{name}: {fid-1} output frames',flush=True)
    finally:
        output.close()
    metrics=[finalize(key,counter) for key,counter in counters.items()]
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    for name,records in [('metrics.csv',metrics),('metrics_by_video.csv',per_sequence)]:
        with (root/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    print('RESULTS:',root,flush=True)
    for r in metrics:
        if r['tolerance_px']==15:
            print(json.dumps(r),flush=True)


def finalize(key,counter):
    variant,output_type,alignment,tolerance=key
    tp,fp,fn=[int(counter[k]) for k in ['TP','FP','FN']]
    return dict(variant=variant,output_type=output_type,alignment=alignment,tolerance_px=tolerance,
                frames=int(counter['frames']),gt_visible=int(counter['gt_visible']),predicted=int(counter['predicted']),
                TP=tp,FP=fp,FN=fn,TN=int(counter['TN']),wrong_location=int(counter['wrong_location']),
                precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                mean_error_visible_predictions_px=counter['visible_prediction_distance_sum']/counter['visible_prediction_count'] if counter['visible_prediction_count'] else None,
                local_passes=int(counter['local_passes']),global_passes=int(counter['global_passes']),
                missing_gt=int(counter['missing_gt']))


if __name__=='__main__':
    main()
