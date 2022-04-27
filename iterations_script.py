import argparse
import time
import numpy as np
import pandas as pd
from torch.cuda import is_available
from staghunt.mrftools.mrftools import TorchMatrixBeliefPropagator
from staghunt import TorchStagHuntModel
import os

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='path to the output file')
parser.add_argument('--size', type=int, help='size of the grid')
parser.add_argument('--M', type=int, help='number of agents')
parser.add_argument('--h', type=int, help='horizon')
parser.add_argument('--niter', type=int, help='number of repeats')
args = parser.parse_args()

cuda = is_available()

model = TorchStagHuntModel(is_cuda=cuda)

df = pd.DataFrame(columns=['lmb', 'num_iter'])
for i in range(args.niter):

    model.new_game_sample(size=(args.size, args.size), num_agents=args.M)
    for lmb in np.arange(1, 101) / 10:
        # with open('C:\\Users\\asfan\\PycharmProjects\\staghunt\\jobs\\iterations.txt', 'a+') as f:
        with open('/home/asfandyar/jobs/iterations.txt', 'a') as f:
            f.write(str(i) + '-' + str(lmb) + '\t' + str(time.time()) + '\n')
        start_time = time.time()
        model.lmb = lmb
        model.horizon = args.h
        model.build_model()
        print(model.lmb)
        bp = TorchMatrixBeliefPropagator(model.mrf, is_cuda=cuda, var_on=False)
        bp.set_max_iter(3000)
        niter = bp.infer(display='none')
        model.reset_game()
        df = df.append({'lmb': lmb,
                        'num_iter': niter,
                        'time': time.time() - start_time,
                        'iter': i,
                        'lmb': lmb}, ignore_index=True)

df.to_pickle(args.out)
