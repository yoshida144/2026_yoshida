# Evaluation entrypoint; invoked through train_ML_CSC.py --evaluate-events.
"""Evaluate identical inputs/checkpoints with dense and sparse-event inference."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from model import ML_CSC_Reconstruction, ML_CSC_Image_Classification
from utils import build_loaders, set_execution

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--task',choices=['csr','classification'],default='csr')
    parser.add_argument('--batches',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='event_evaluation.json')
    parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args()
    if args.batches<1 or args.batch_size<1 or args.warmup<0:
        parser.error('batches/batch-size must be >=1; warmup must be >=0')
    torch.set_num_threads(1);torch.manual_seed(43)
    params=json.loads(Path(args.config).read_text())
    if args.synthetic:params['synthetic']=True
    params['batch_size']=args.batch_size
    print('Loading evaluation dataset...',flush=True)
    _,loader,info=build_loaders(params,evaluation_only=True)
    device=torch.device(args.device)
    cls=args.task=='classification'
    model=(ML_CSC_Image_Classification(info['input_channels'],info['num_classes'],params['layer_specs'],params['method'])
           if cls else ML_CSC_Reconstruction(info['input_channels'],params['layer_specs'],params['method'])).to(device)
    model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True),strict=True)
    model.eval()
    accumulator={m:dict(seconds=0.,mse_sum=0.,raw_squared_error=0.,elements=0,correct=0,samples=0,
        spikes=0,opportunities=0,dense_recurrent_macs_estimate=0,event_recurrent_macs=0) for m in ('dense','event')}
    max_output_difference=0.; mismatched_code_elements=0
    with torch.no_grad():
        for i,(clean,labels) in enumerate(loader):
            if i>=args.batches:break
            clean,labels=clean.to(device),labels.to(device)
            model_input=clean
            outputs={}
            # Warm up both paths once on the first batch; reset happens inside forward.
            if i==0:
                for _ in range(args.warmup):
                    for mode in accumulator:
                        print(f'Warmup: {mode}',flush=True)
                        set_execution(model,mode); model(model_input)
            for mode,metrics in accumulator.items():
                print(f'Batch {i+1}/{min(args.batches,len(loader))}: {mode}, {len(clean)} images',flush=True)
                set_execution(model,mode);sync(device);start=time.perf_counter()
                final,codes,_=model(model_input)
                reconstruction=model.decode(len(codes)-1,clean,codes)
                sync(device);metrics['seconds']+=time.perf_counter()-start
                error=reconstruction-clean
                metrics['mse_sum']+=error.square().sum().item()
                std=clean.new_tensor(info['data_std']).view(1,-1,1,1)
                metrics['raw_squared_error']+=(error*std).square().sum().item()
                metrics['elements']+=clean.numel();metrics['samples']+=len(clean)
                if cls:metrics['correct']+=int((final.argmax(1)==labels).sum())
                for layer in model.feature_extractor.layers:
                    stats=layer.last_stats
                    for key in ('spikes','opportunities','dense_recurrent_macs_estimate','event_recurrent_macs'):
                        metrics[key]+=stats[key] or 0
                outputs[mode]=(final,codes,reconstruction)
            max_output_difference=max(max_output_difference,float((outputs['dense'][2]-outputs['event'][2]).abs().max()))
            mismatched_code_elements+=sum(int((a!=b).sum()) for a,b in zip(outputs['dense'][1],outputs['event'][1]))
    for mode,m in accumulator.items():
        m['mse_normalized']=m['mse_sum']/m['elements']
        raw=m['raw_squared_error']/m['elements']
        m['psnr_db_unclipped_unit_range']=float(-10*torch.tensor(max(raw,1e-30)).log10())
        m['seconds_per_image']=m['seconds']/m['samples']
        m['spike_rate']=m['spikes']/max(m['opportunities'],1)
        m['accuracy']=m['correct']/m['samples'] if cls else None
    dense=accumulator['event']['dense_recurrent_macs_estimate']
    result={'synthetic':params.get('synthetic',False),'task':args.task,'device':str(device),
        'batch_size':args.batch_size,'warmup':args.warmup,'metrics':accumulator,'max_reconstruction_difference':max_output_difference,
        'mismatched_code_elements':mismatched_code_elements,
        'recurrent_mac_reduction_estimate':1-accumulator['event']['event_recurrent_macs']/dense if dense else None,
        'notes':['Reduction compares recurrent linear operations within the same LIF model, not original LISTA.',
                 'Dense counts are standard full-kernel MAC estimates including padding; event counts are actual valid contributions.',
                 'Excludes Wy input convolution, final synthesis/decode, pointwise LIF/budget selection work, scans, indexing and memory traffic.',
                 'Python event kernels may be slower; no measured energy or power claim.',
                 'Threshold decisions may differ near a boundary because floating-point addition order changes.',
                 'Use original model=lista configs and retrain separately to establish restoration/accuracy baseline.']}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))



if __name__ == '__main__':
    main()
