"""Recorded supplemental workload; does not repair/skip the failed first-frame case."""
import json
import sys
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

REPO=Path('/mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain')
sys.path.insert(0,str(REPO/'examples/umi_pretrain/tools'))
from profile_umi_pipeline import raw_dataset, loader_case
from starVLA.training.trainer_utils.umi_checkpoint import write_json

def main():
    root=Path('/mnt/workspace/Native_Policy/user/wyt/runs/umi_performance_v1')
    report=json.loads((root/'loader_benchmark.json').read_text())
    plan=OmegaConf.to_container(OmegaConf.load(REPO/'examples/umi_pretrain/train_files/umi_training_qwenpi_engineering.yaml'),resolve=True)
    index='/mnt/workspace/Native_Policy/user/wyt/data_preparation/roban_umi_access_v1'
    raw=raw_dataset(plan,index)
    sequence=[]
    for first in report['sequences']['cross_file_stress']:
        pos=int(np.searchsorted(raw._cumulative,first,side='right'))
        _,a,b=raw._ranges[pos]
        sequence.append(first+int(b-a)//2)
    source=raw.read_lowdim(126293227)
    failure=dict(index=126293227,episode=34300,camera=source['cameras'][1],
                 tolerance_seconds=raw.decode_tolerance_seconds,trace=source['trace'])
    import av
    camera=source['cameras'][1]
    target=camera['from_timestamp']
    with av.open(str(raw._path(camera['video_path'],video=True))) as container:
        stream=container.streams.video[0]
        container.seek(int(target/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
        around=[]
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            sec=float(frame.pts*stream.time_base)
            if abs(sec-target)<0.1:
                around.append(dict(seconds=sec,offset=sec-target,inside_episode=target<=sec<camera['to_timestamp']))
            if sec>target+0.1:
                break
        failure['nearby_video_timestamps']=around
    write_json(root/'video_boundary_failure.json',failure)
    raw.close()
    report['supplementary_sequence']=sequence
    report['supplementary_note']='Same 16 physical-file anchors, shifted deterministically to each first valid range midpoint. Original failing first-frame sequence and all failures are retained, not filtered or fixed. No tolerance changes.'
    report['supplementary_cases']=[]
    for workers in (0,2,4):
        result=loader_case(plan,index,sequence,workers,180)
        result['workload']='cross_file_range_midpoints'
        report['supplementary_cases'].append(result)
        write_json(root/'loader_benchmark.json',report)
        print({k:result[k] for k in ('error','delivered_windows','stable_windows_per_second','locality')},flush=True)

if __name__=='__main__':
    main()
