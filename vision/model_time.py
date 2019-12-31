import os
import torch
import torch.nn as nn
import argparse
import time
import csv
import models
torch.cuda.manual_seed_all(11)
torch.random.manual_seed(11)
def main(args):
    print(args)
    g = args.group
    N = args.N
    h = args.h
    w = args.w
    c1 = args.c1
    c2 = args.c2
    input = torch.randn(N, c1, h, w).cuda()
    model = models.__dict__[args.model](pretrained=False, groups=args.group).cuda()
    print('Profiling...')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
#        model(input) # Warmup CUDA memory allocator and profiler
        for i in list(range(args.repeat)):
            with torch.autograd.profiler.record_function("label-"+str(i)):
                model(input)
    dir_name = './trace/'+args.gpu_type+'/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    prof.export_chrome_trace('./trace/'+args.gpu_type+'/'+
        args.model+'.'+str(args.group)+'.'+'trace')
    time = 0
    for i in prof.key_averages():
        if i.key.startswith('label-'):
            print(i.key)
            print(i.cuda_time)
            time += i.cuda_time
    time = time / args.repeat
    write_to_csv(args, time)
    print('Done!')

def write_to_csv(args, time_act):
    print('Calculating...')
    P = args.P
    Bandwidth = args.Bandwidth
    N = args.N
    group = args.group
    type_byte = args.type_byte
    data = [['', P, Bandwidth, args.model, N, group, type_byte, time_act]]
    print('Writing...')
    dir_name = './csvs/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    with open(args.filename, 'a+', newline='') as f:
        w = csv.writer(f)
        w.writerows(data)

def parse_args():
    parser = argparse.ArgumentParser(description='MEASURE TIME')
    parser.add_argument('--N', default=64, type=int)
    parser.add_argument('--h', default=224, type=int)
    parser.add_argument('--w', default=None, type=int)
    parser.add_argument('--c1', default=3, type=int)
    parser.add_argument('--c2', default=None, type=int)
    parser.add_argument('--kernel-size', default=1, type=int)
    parser.add_argument('--group', default=1, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--P', default=15.7, type=float)
    parser.add_argument('--Bandwidth', default=0.9, type=float)
    parser.add_argument('--type_byte', default=4, type=int)
    parser.add_argument('--filename', default='output.csv', type=str)
    parser.add_argument('--repeat', default=5, type=int)
    parser.add_argument('--gpu-type', default='V100-16GB', type=str)
    parser.add_argument('--model', default='shufflenet_v2_x1_5', type=str)
    args = parser.parse_args()
    if args.w is None:
        args.w = args.h
    if args.c2 is None:
        args.c2 = args.c1
    return args
    

if __name__ == '__main__':
    args = parse_args()
    print('Creating csv...')
    with open(args.filename, 'a+', newline='') as f:
        w = csv.writer(f)
        w.writerows([[args.gpu_type, 'P', 'Bandwidth', 'model',
            'N', 'group', 'type_byte', 'time_act']])
    main(args)
