import random
import argparse
import pandas as pd
import numpy as np
from torch.cuda import is_available, init
from torch import float64
from staghunt import TorchStagHuntModel
from mrftools import TorchMatrixBeliefPropagator

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='path to the output file')
parser.add_argument('--size', type=int, help='size of the grid')
parser.add_argument('--M', type=int, help='number of agents')
args = parser.parse_args()

random.seed(30)  # reproducibility

cuda = is_available()
model = TorchStagHuntModel(is_cuda=cuda, var_on=False)
model.horizon = 10

if cuda:
    init()

n_iter = 5
data = pd.DataFrame(columns=['LMB', 'N', 'M', 'Horizon', 'energy_functional', 'energy', 'bethe_entropy'])

for lmb in np.arange(1, 20)/10:
    ef, be, e = [], [], []
    print(lmb)
    for i in range(n_iter):

        # model.reset_game()
        model.new_game_sample(size=(args.size, args.size), num_agents=args.M)
        model.lmb = lmb
        model.build_model()

        bp = TorchMatrixBeliefPropagator(model.mrf, is_cuda=cuda, var_on=False, dtype=float64)
        bp.set_max_iter(5000)
        bp.infer(display='final')
        ef.append(bp.compute_energy_functional().item())
        be.append(bp.compute_bethe_entropy().item())
        e.append(bp.compute_energy().item())

    data = data.append({'LMB': model.lmb,
                        'N': args.size*args.size,
                        'M': args.M,
                        'Horizon': model.horizon,
                        'energy_functional': np.average(ef),
                        'energy': np.average(e),
                        'bethe_entropy': np.average(be)
                        }, ignore_index=True)

print("Save results to: " + args.out)
data.to_pickle(args.out)
