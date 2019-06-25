import unittest
import random
import numpy as np
import torch as t
from staghunt import StagHuntModel, MatrixStagHuntModel, TorchStagHuntModel, new_var


class TestTorchStagHunt(unittest.TestCase):
    """
    Unit test class for belief propagation implementation
    """

    def test_phi_q_factor(self):
        """
        Correctness of the phi_q factor efficient computation by comparing it with the explicit way of computing it
        :return: None
        """
        sh_model = StagHuntModel()
        sh_model.MIN = -float('inf')
        th_model = TorchStagHuntModel()
        size = (random.randint(5, 20), random.randint(5, 20))
        sh_model.size = size
        th_model.size = size

        sh_phi_q = sh_model.build_phi_q()
        th_phi_q = th_model.build_phi_q()

        assert t.equal(t.tensor(sh_phi_q, dtype=t.float64), th_phi_q)

    def test_ground_vs_pairwise(self):
        """
        TORCH: Correctness of the simplified pairwise model vs the ground truth model
        :return: None
        """
        ground_model, pairwise_model = setup_two_models(type1='torch', type2='torch', num_agents=2)

        ground_model.build_ground_model()
        pairwise_model.build_model()

        for i in range(ground_model.horizon - 1):
            print("Ground:   ", end='')
            ground_model.infer()
            ground_model.compute_probabilities()
            ground_model.move_next(break_ties='first')
            ground_model.update_model()

            print("Pairwise: ", end='')
            pairwise_model.infer()
            pairwise_model.compute_probabilities()
            pairwise_model.move_next(break_ties='first')
            pairwise_model.update_model()

            # compare marginals
            assert compare_beliefs(ground_model.bp.var_probabilities, pairwise_model.bp.var_probabilities)
            # compare computed conditional probabilities used to decide next step
            for key in ground_model.bp.conditional_probabilities:
                if not (key == (new_var('x', ground_model.horizon, 1), new_var('x', ground_model.horizon, 2))
                        or key == (new_var('x', ground_model.horizon, 2), new_var('x', ground_model.horizon, 1))):
                    c = ground_model.bp.conditional_probabilities[key]
                    d = pairwise_model.bp.conditional_probabilities[key]
                    assert np.allclose(c, d, equal_nan=True)

    def test_matrix_vs_torch(self):
        """
        TORCH: Consistency between python and torch matrix BP
        :return:
        """
        matrix_model, torch_model = setup_two_models(type1='python', type2='torch')

        matrix_model.MIN = -float('inf')
        matrix_model.build_model()
        torch_model.build_model()

        assert np.all(torch_model.mrf.unary_mat.numpy() == matrix_model.mrf.unary_mat), \
            "unary matrices are not equal"
        assert np.all(torch_model.mrf.edge_pot_tensor.numpy() == matrix_model.mrf.edge_pot_tensor), \
            "edge tensors are not equal"

        for i in range(matrix_model.horizon - 1):
            print("Matrix: ", end='')
            matrix_model.infer(inference_type='matrix')
            matrix_model.compute_probabilities()
            matrix_model.move_next(break_ties='first')
            matrix_model.update_model()

            print("Torch:  ", end='')
            torch_model.infer()
            torch_model.compute_probabilities()
            torch_model.move_next(break_ties='first')
            torch_model.update_model()

            assert matrix_model.aPos == torch_model.aPos, "Trajectories differ"

            # compare marginals
            assert compare_beliefs(matrix_model.bp.var_probabilities, torch_model.bp.var_probabilities), \
                "Marginal probabilities differ"
            # compare conditional probabilities
            assert compare_beliefs(matrix_model.bp.conditional_probabilities,
                                   torch_model.bp.conditional_probabilities), \
                "Conditional probabilities differ"

    def test_cpu_vs_cuda(self):
        """
        TORCH: Consistency between CUDA and CPU matrix BP
        :return:
        """
        if not t.cuda.is_available():
            print('\nCUDA is not available in this machine')
            return

        cpu_model, cuda_model = setup_two_models(type1='torch', type2='torch')

        cuda_model.is_cuda = True

        cpu_model.build_model()
        cuda_model.build_model()

        assert t.equal(cuda_model.mrf.unary_mat.cpu(), cpu_model.mrf.unary_mat), \
            "unary matrices are not equal"
        assert t.equal(cuda_model.mrf.edge_pot_tensor.cpu(), cpu_model.mrf.edge_pot_tensor), \
            "edge tensors are not equal"

        for i in range(cpu_model.horizon - 1):
            print("CPU: ", end='')
            cpu_model.infer()
            cpu_model.compute_probabilities()
            cpu_model.move_next(break_ties='first')
            cpu_model.update_model()

            print("CUDA:  ", end='')
            cuda_model.infer()
            cuda_model.compute_probabilities()
            cuda_model.move_next(break_ties='first')
            cuda_model.update_model()

            assert cpu_model.aPos == cuda_model.aPos, "Trajectories differ"

            # compare marginals
            assert compare_beliefs(cpu_model.bp.var_probabilities, cuda_model.bp.var_probabilities), \
                "Marginal probabilities differ"
            # compare conditional probabilities
            assert compare_beliefs(cpu_model.bp.conditional_probabilities, cuda_model.bp.conditional_probabilities), \
                "Conditional probabilities differ"

    def test_fast_build_tensors(self):
        """
        The fast-built unary matrix coincides with the one built from the potentials by the mrftools object
        :return:
        """
        fast_model, slow_model = setup_two_models(type1='torch', type2='torch')
        fast_model.fast_build_model()
        slow_model.build_model()

        for var, index_slow in slow_model.mrf.var_index.items():
            index_fast = fast_model.mrf.var_index[var]
            assert np.equal(slow_model.mrf.unary_mat[:, index_slow], fast_model.mrf.unary_mat[:, index_fast]).all()

        fast_model, slow_model = setup_two_models()
        fast_model.fast_build_model()
        slow_model.build_model()
        assert len(fast_model.mrf.message_index) == len(slow_model.mrf.message_index)
        for var, index_slow in slow_model.mrf.message_index.items():
            if var in fast_model.mrf.message_index.keys():
                index_fast = fast_model.mrf.message_index[var]
                assert np.equal(slow_model.mrf.edge_pot_tensor[:, :, index_slow],
                                fast_model.mrf.edge_pot_tensor[:, :, index_fast]).all()
            else:
                index_fast = fast_model.mrf.message_index[var[::-1]]
                assert np.equal(slow_model.mrf.edge_pot_tensor[:, :, index_slow].T,
                                fast_model.mrf.edge_pot_tensor[:, :, index_fast]).all()

    def test_fast_build_beliefs(self):
        """
        TORCH: Consistency between python and torch matrix BP
        :return:
        """
        fast_model, slow_model = setup_two_models(type1='torch', type2='torch')
        fast_model.fast_build_model()
        slow_model.build_model()

        for i in range(slow_model.horizon - 1):
            print("Fast: ", end='')
            fast_model.infer()
            fast_model.fast_move_next(break_ties='first')
            fast_model._clamp_agents()

            print("Slow:  ", end='')
            slow_model.infer()
            slow_model.compute_probabilities()
            slow_model.move_next(break_ties='first')
            slow_model.update_model()

            assert slow_model.aPos == fast_model.aPos, "Trajectories differ"


def setup_two_models(num_agents=None, type1='python', type2='python'):
    """
    Utility to set up two different instances of StagHuntMRF with the exact same random configuration
    :param num_agents: Number of agents
    :param type1: Type of the first model: python or torch
    :param type2: Type of the second model: python or torch
    :return: Two StagHuntMRF instances
    """
    if type1 == 'python':
        model_1 = MatrixStagHuntModel()
    else:
        model_1 = TorchStagHuntModel()

    if type2 == 'python':
        model_2 = MatrixStagHuntModel()
    else:
        model_2 = TorchStagHuntModel()

    # random.seed(1)
    lmb = random.uniform(0.1, 10)
    r_h = random.randint(-5, -1)
    r_s = random.randint(-10, -5)
    horizon = random.randint(4, 15)
    size = random.randint(5, 15)

    if not num_agents:
        num_agents = random.randint(2, size // 2)

    model_1.lmb = lmb
    model_1.r_h = r_h
    model_1.r_s = r_s
    model_1.horizon = horizon
    model_1.new_game_sample(size=(size, size), num_agents=num_agents)

    model_2.lmb = lmb
    model_2.r_h = r_h
    model_2.r_s = r_s
    model_2.horizon = horizon
    model_2.size = (size, size)
    model_2.set_game_config(game_conf=model_1.get_game_config())

    return model_1, model_2


def compare_beliefs(belief_dict_1, belief_dict_2):
    """
    Utility to compare two belief or probability dictionaries.
    :param belief_dict_1:
    :param belief_dict_2:
    :return: Boolean - True if beliefs coincide for every variable, false otherwise
    """
    check = []
    for key in belief_dict_1:
        if key[0] == 'x':
            a = belief_dict_1[key]
            if isinstance(a, t.Tensor):
                if a.is_cuda:
                    a = a.cpu().numpy()
                else:
                    a = a.numpy()
            b = belief_dict_2[key]
            if isinstance(b, t.Tensor):
                if b.is_cuda:
                    b = b.cpu().numpy()
                else:
                    b = b.numpy()
            check.append(np.allclose(a, b, equal_nan=True))
    return all(check)
