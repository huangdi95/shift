import os
import torch
import torch.nn as nn
import argparse
import time
import csv
#from new_conv import NewConv
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
    model_list = [nn.Conv2d(c1, c2, 1, groups=g) for i in range(args.num_layers)]
    model = nn.Sequential(*model_list).cuda()
    print(model)
    print('Profiling...')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(input) # Warmup CUDA memory allocator and profiler
        for i in list(range(args.repeat)):
            with torch.autograd.profiler.record_function("label-"+str(i)):
                model(input)
    prof.export_chrome_trace('./trace/'+
        str(args.num_layers)+'.'+str(args.N)+'.'+
        str(args.h)+'.'+str(args.w)+'.'+str(args.c1)+'.'
        +str(args.c2)+'.'+str(args.g)+'.'+'trace')
    time = 0
    for i in prof.key_averages():
        if i.key.startswith('label-'):
            print(i.key)
            print(i.cuda_time)
            time += i.cuda_time
    time = time / args.repeat
    write_to_csv(args, time)
    print('Done!')
##    print(input)
#    torch.cuda.synchronize()
#    start = time.time()
#    out = model(input)
#    torch.cuda.synchronize()
#    print(out.shape)
#    end = time.time()
#    print(end - start)

def write_to_csv(args, time_act):
    print('Calculating...')
    P = args.P
    Bandwidth = args.Bandwidth
    N = args.N
    h = args.h
    w = args.w
    c1 = args.c1
    c2 = args.c2
    group = args.group
    num_layers = args.num_layers
    type_byte = args.type_byte
    input_size = N * h * w * c1 * num_layers * type_byte
    weight_size = (c1 * c2 * num_layers * type_byte) / group
    output_size = N * h * w * c2 * num_layers * type_byte
    FLOPs = (N * h * w * c2 * c1 * num_layers) / group
    IO_or = max(input_size + weight_size, output_size)
    IO_and = input_size + weight_size + output_size
    IO = IO_or
    time_compute = FLOPs / (P * 1e6)
    time_io = IO / (Bandwidth * 1e6)
    time_total = max(time_compute, time_io)
    #TODO: IO also has its efficiency!!!
    efficiency = time_compute / time_act
    P_act = efficiency * P
    g_balance = ((N*Bandwidth*h*w/type_byte-P_act)*c1*c2) / (N*P_act*h*w*(c1+c2))
    data = [[P, Bandwidth, N, h, w, c1, c2, group, num_layers, type_byte,
        input_size, weight_size, output_size, FLOPs, IO,
        time_compute, time_io, time_total, time_act, efficiency,
        P_act, g_balance]]
    print('Writing...')
    with open(args.filename, 'a+', newline='') as f:
        w = csv.writer(f)
        w.writerows(data)

def parse_args():
    parser = argparse.ArgumentParser(description='MEASURE TIME')
    parser.add_argument('--N', default=64, type=int)
    parser.add_argument('--h', default=28, type=int)
    parser.add_argument('--w', default=None, type=int)
    parser.add_argument('--c1', default=128, type=int)
    parser.add_argument('--c2', default=None, type=int)
    parser.add_argument('--kernel-size', default=1, type=int)
    parser.add_argument('--group', default=1, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--P', default=15.7, type=float)
    parser.add_argument('--Bandwidth', default=0.9, type=float)
    parser.add_argument('--type_byte', default=4, type=int)
    parser.add_argument('--filename', default='output.csv', type=str)
    parser.add_argument('--repeat', default='10', type=int)
    args = parser.parse_args()
    if args.w is None:
        args.w = args.h
    if args.c2 is None:
        args.c2 = args.c1
    return args

def mat(args):
    g = args.group
    N = args.N
    h = args.h
    w = args.w
    c2 = args.c1
    c1 = args.c2
    input1 = torch.randn(N*h*w, c1).cuda()
    input2_0 = torch.randn(int(c1/2), int(c2/2)).cuda()
    input2_3 = torch.randn(int(c1/2), int(c2/2)).cuda()
    zero = torch.zeros(int(c1/2), int(c2/2)).cuda()
    input2_02 = torch.cat((input2_0, zero), 0)
    input2_13 = torch.cat((zero, input2_3), 0)
    input2 = torch.cat((input2_02, input2_13), 1)
    input1 = input1.transpose(0, 1)
    input2 = input2.transpose(0, 1).to_sparse()
    print(input2)
    torch.cuda.synchronize()
    start = time.time()
    out = torch.sparse.mm(input2, input1)
    torch.cuda.synchronize()
    end = time.time()
    print(end - start)
    torch.cuda.synchronize()
    start = time.time()
    out = torch.mm(input2, input1)
    torch.cuda.synchronize()
    end = time.time()
    print(end - start)
    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
#    mat(args)
