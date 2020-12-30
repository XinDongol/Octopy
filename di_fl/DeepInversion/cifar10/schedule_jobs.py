import re
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--filename', '-f',type=str)
parser.add_argument('--gpus', '-g', nargs='+', type=int)
args = parser.parse_args()

f1 = open(args.filename, 'r')
all_command = [l for l in f1]
print('Spliting %d commands into %d gpus.'%(len(all_command), len(args.gpus)))

comm_group= [list(i) for i in np.array_split(all_command, len(args.gpus))]
print([len(i) for i in comm_group])

for idx, gpu in enumerate(args.gpus):
    comms = comm_group[idx]
    with open("%d_gpu.sh"%idx, "w") as file:
        for c in comms:
            file.write('CUDA_VISIBLE_DEVICES=%d '%gpu+re.sub(r".+?(?=python)", "", c))
        