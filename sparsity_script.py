import random
import pandas as pd
import numpy as np
from staghunt import MatrixStagHuntModel

random.seed(30)  # reproducibility

matrix = MatrixStagHuntModel()
matrix.MIN = -np.inf

df = pd.DataFrame(columns=['Zero', '-Inf', 'Total'])
for n in range(5, 6):
    N = n * n
    for M in range(2, int(0.8 * 2 * N / 7)):
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

df.to_pickle('../results/sparsity.pkl')
