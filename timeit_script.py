import time
import random
from torch.cuda import is_available, init
from staghunt import StagHuntGame, MatrixStagHuntModel, TorchStagHuntModel

random.seed(30)  # reproducibility

model = StagHuntGame()

loopy = MatrixStagHuntModel()
matrix = MatrixStagHuntModel()
cpu = TorchStagHuntModel(is_cuda=False, var_on=False)
if is_available():
    gpu = TorchStagHuntModel(is_cuda=True, var_on=False)
    init()

data = {}
for n in range(5, 6):
    N = n*n
    for M in range(2, int(0.8*2*N/7)):  # number of ranges up to ~80% of the grid capacity
        experiment_data = {}

        model.new_game_sample(size=(n, n), num_agents=M)
        conf = model.get_game_config()
        experiment_data['conf'] = conf

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

        experiment_data['loopy_time'] = loopy_time
        experiment_data['matrix_time'] = matrix_time
        experiment_data['cpu_time'] = cpu_time
        if is_available():
            experiment_data['gpu_time'] = gpu_time

        data[(N, M)] = experiment_data

print(data)

# n_reps = 5
# n_instances = 10
# on_times = []
# off_times = []
# for i in range(n_instances):
#     on_model.new_game_sample(size=(10, 10), num_agents=5)
#     off_model.set_game_config(on_model.get_game_config())
#     on_rep_time = 0
#     off_rep_time = 0
#     for r in range(n_reps):
#         on_model.fast_build_model()
#         off_model.fast_build_model()
#         t0 = time.time()
#         on_model.run_game()
#         t1 = time.time()
#         off_model.run_game()
#         t2 = time.time()
#         on_model.reset_game()
#         off_model.reset_game()
#         on_rep_time += t1-t0
#         off_rep_time += t2-t1
#     on_times.append(on_rep_time/n_reps)
#     off_times.append(off_rep_time/n_reps)
# print(on_times)
# print(off_times)
