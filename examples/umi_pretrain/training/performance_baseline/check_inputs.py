"""Check the observed numerical-trajectory caveat without additional model updates."""
import hashlib
import json
from itertools import islice
from pathlib import Path
import numpy as np
from starVLA.training.trainer_utils.umi_training_data import make_loader

ROOT=Path('/mnt/workspace/Native_Policy/user/wyt/runs/umi_performance_v1')

def digest(sample):
    h=hashlib.sha256()
    h.update(sample['lang'].encode())
    for image in sample['image']:
        h.update(str((image.mode,image.size)).encode())
        h.update(image.tobytes())
    for key in ('state','action'):
        value=np.asarray(sample[key])
        h.update(str((key,value.shape,str(value.dtype))).encode())
        h.update(value.tobytes())
    return h.hexdigest()

def main():
    output={}
    for mode in ('decoded_replay','end_to_end'):
        plan=json.loads((ROOT/f'{mode}_plan.json').read_text())
        output[mode]=[]
        for stage in plan['stages']:
            count=stage['updates']*plan['training']['batch_size']*plan['training']['gradient_accumulation_steps']
            loader=make_loader(plan,stage,ROOT/'input_equivalence'/mode)
            iterator=iter(loader)
            try:
                for batch in islice(iterator,count):
                    for sample in batch:
                        output[mode].append(dict(stage=stage['name'],index=sample['umi_metadata']['dataset_index'],sha256=digest(sample)))
            finally:
                if hasattr(iterator,'_shutdown_workers'): iterator._shutdown_workers()
                loader.dataset.close()
    output['identical']=output['decoded_replay']==output['end_to_end']
    output['note']='Exact bytes of PIL pixels/mode/size, language and normalized state/action; metadata-only timing/PID fields excluded. No model updates.'
    (ROOT/'input_equivalence.json').write_text(json.dumps(output,indent=2))
    assert output['identical']
    print('PASS exact model-facing input bytes for',len(output['decoded_replay']),'windows')

if __name__=='__main__':main()
