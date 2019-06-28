import argparse
import numpy as np
from staghunt import MatrixStagHuntModel

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='path to the output file')
parser.add_argument('--size', type=int, help='size of the grid')
parser.add_argument('--M', type=int, help='number of agents')
parser.add_argument('--h', type=int)
parser.add_argument('--lmb', type=float)
args = parser.parse_args()


model = MatrixStagHuntModel()
model.INF = -np.inf
model.new_game_sample(size=(args.size, args.size), num_agents=args.M)
model.lmb = args.lmb
model.horizon = args.h

model.fast_build_model()
model.run_game(inference_type='matrix')
model.display(file=args.out)
