import argparse
import torch as t
from staghunt import TorchStagHuntModel

parser = argparse.ArgumentParser()
parser.add_argument('--outa', type=str, help='path to the output file')
parser.add_argument('--outb', type=str, help='path to the output file')
parser.add_argument('--size', type=int, help='size of the grid')
parser.add_argument('--M', type=int, help='number of agents')
parser.add_argument('--h', type=int)
parser.add_argument('--lmb', type=float)
args = parser.parse_args()

cuda = t.cuda.is_available()

model = TorchStagHuntModel(is_cuda=cuda, var_on=False)
model.new_game_sample(size=(args.size, args.size), num_agents=args.M)
conf = model.get_game_config()
model.lmb = args.lmb
model.horizon = args.h

model.fast_build_model()
model.run_game()
model.display(file=args.outa)

model.reset_game()
del model

model = TorchStagHuntModel(is_cuda=cuda, var_on=False)
model.set_game_config(conf)
model.lmb = args.lmb
model.horizon = args.h

model.fast_build_model()
model.run_game()
model.display(file=args.outb)



