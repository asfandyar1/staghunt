import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.cuda import is_available
from staghunt import TorchStagHuntModel
import os
from datetime import datetime




tstart = datetime.now()
print(str(tstart))
cuda = is_available()
model = TorchStagHuntModel(is_cuda=cuda)

model.new_game_sample(size=(20, 20), num_agents=5)
model.lmb = 0.1
model.horizon = 8
model.build_model()
model.run_game(verbose=True)

tend = datetime.now()
print(str(tend))
c = tend - tstart
print(c)