import time
import random
import argparse
import pandas as pd
import numpy as np
from torch.cuda import is_available, init
from staghunt import StagHuntGame, MatrixStagHuntModel, TorchStagHuntModel

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='path to the output file')
parser.add_argument('--lmb', type=float, help='value of lambda')
args = parser.parse_args()

random.seed(30)  # reproducibility

model = StagHuntGame()
model.lmb = args.lmb
model.horizon = 20

loopy = MatrixStagHuntModel()
matrix = MatrixStagHuntModel()
cpu = TorchStagHuntModel(is_cuda=False, var_on=False)
if is_available():
    gpu = TorchStagHuntModel(is_cuda=True, var_on=False)
    init()
n_iter = 5
data = pd.DataFrame(columns=['LMB', 'N', 'M', 'Horizon', 'loopy_time', 'matrix_time', 'cpu_time', 'gpu_time'])
for n in range(5, 10):
    N = n*n
    for M in range(2, int(0.5*2*N/7)):  # number of ranges up to ~80% of the grid capacity

        for i in range(n_iter):
            print("size: " + str((n, n)) + "\t" + "M: " + str(M) + "iter: ", str(i) + "/" + str(n_iter))
            model.new_game_sample(size=(n, n), num_agents=M)
            conf = model.get_game_config()

            loopy.set_game_config(conf)
            matrix.set_game_config(conf)
            cpu.set_game_config(conf)
            if is_available():
                gpu.set_game_config(conf)

            loopy.build_model()
            matrix.fast_build_model()
            cpu.fast_build_model()
            if is_available():
                gpu.fast_build_model()

            t0 = time.time()
            loopy.run_game(inference_type='slow')
            t1 = time.time()
            loopy_time = t1 - t0
            loopy.reset_game()

            t0 = time.time()
            matrix.run_game(inference_type='matrix')
            t1 = time.time()
            matrix_time = t1 - t0
            matrix.reset_game()

            t0 = time.time()
            cpu.run_game()
            t1 = time.time()
            cpu_time = t1 - t0
            cpu.reset_game()

            if is_available():
                t0 = time.time()
                gpu.run_game()
                t1 = time.time()
                gpu_time = t1 - t0
                gpu.reset_game()
            else:
                gpu_time = 0

            data = data.append({'LMB': model.lmb,
                                'N': N,
                                'M': M,
                                'Horizon': model.horizon,
                                'loopy_time': loopy_time,
                                'matrix_time': matrix_time,
                                'cpu_time': cpu_time,
                                'gpu_time': gpu_time}, ignore_index=True)

print("Save results to: " + args.out)
data.to_pickle(args.out)
