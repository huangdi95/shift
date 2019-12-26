import os
import re
import torch

table = ['Self CPU total %', 'Self CPU total', 'CPU total %', 'CPU total', 'CPU time avg', 'CUDA total %', 'CUDA total', 'CUDA time avg', 'Number of Calls']
prof = {}
filename = 'log_1'
f = open(filename, 'r')
flag = 0
for line in f.readlines():
    if '--------------------' in line:
        flag += 1
        continue
    if flag != 2:
        continue
    ele = line.split()
    print(ele)
    name = ele[0]
    prof.update({name: {}})
    for i in list(range(len(ele)-1)):
        print(i)
        prof[name].update({table[i]: ele[i+1]})
f.close()
print(prof['label-z']['CUDA total'])
