import random
import argparse
import pandas as pd
import numpy as np
from staghunt import MatrixStagHuntModel

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='path to the output file')
args = parser.parse_args()

random.seed(30)  # reproducibility

matrix = MatrixStagHuntModel()
matrix.MIN = -np.inf

df = pd.DataFrame(columns=['N', 'M', 'S', 'Horizon', 'Zero', '-Inf', 'Total'])
for n in range(5, 25):

    N = n * n
    for M in range(2, int(0.5 * 2 * N / 7)):
        print(n, M)
        matrix.reset_game()
        matrix.new_game_sample(size=(n, n), num_agents=M)
        matrix.fast_build_model()
        unique, counts = np.unique(matrix.mrf.edge_pot_tensor, return_counts=True)
        unique_dict = dict(zip(unique, counts))
        df = df.append({'N': N,
                        'M': M,
                        'S': len(matrix.sPos),
                        'Horizon': matrix.horizon,
                        'Zero': unique_dict[0.],
                        '-Inf': unique_dict[-float('inf')],
                        'Total': np.prod(matrix.mrf.edge_pot_tensor.shape)}, ignore_index=True)
print("Save results to: " + args.out)
df.to_pickle(args.out)
