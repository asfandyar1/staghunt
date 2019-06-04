"""Test class to test stag hunt game implementation"""
import unittest

from staghunt import *


def setup_two_models(num_agents=2):
    """
    Utility to set up two different instances of StagHuntMRF with the exact same random configuration
    :return:
    """
    model_1 = StagHuntMRF()
    model_2 = StagHuntMRF()

    # random.seed(1)

    lmb = random.uniform(0.1, 10)
    r_h = random.randint(-5, -1)
    r_s = random.randint(-10, -5)
    horizon = random.randint(4, 15)
    size = random.randint(5, 10)

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


class TestStagHunt(unittest.TestCase):
    """
    Unit test class for belief propagation implementation
    """

    def test_ground_vs_pairwise(self):
        """
        Tests the correctness of the simplified pairwise model by comparing it with the ground truth model
        Asserts that the results of BP are the same in both of them, thus giving the same trajectories
        :return: None
        """
        ground_model, pairwise_model = setup_two_models()

        ground_model.build_ground_model()
        pairwise_model.build_model()

        for i in range(ground_model.horizon-1):
            ground_model.infer(inference_type='slow')
            ground_model.compute_probabilities()
            ground_model.move_next()
            ground_model.update_model()

            pairwise_model.infer(inference_type='slow')
            pairwise_model.compute_probabilities()
            pairwise_model.move_next()
            pairwise_model.update_model()

            # compare marginals
            for key in ground_model.bp.var_probabilities:
                a = ground_model.bp.var_probabilities[key]
                b = pairwise_model.bp.var_probabilities[key]
                assert np.allclose(a, b, equal_nan=True)

            # compare computed conditional probabilities used to decide next step
            for key in ground_model.bp.conditional_probabilities:
                if not(key == (new_var('x', ground_model.horizon, 1), new_var('x', ground_model.horizon, 2))
                        or key == (new_var('x', ground_model.horizon, 2), new_var('x', ground_model.horizon, 1))):
                    c = ground_model.bp.conditional_probabilities[key]
                    d = pairwise_model.bp.conditional_probabilities[key]
                    assert np.allclose(c, d, equal_nan=True)

    def test_slow_vs_matrix(self):
        """
        Tests the consistency of matrix BP versus slow BP that has been tested vs the ground truth model
        :return:
        """
        slow_model, matrix_model = setup_two_models()

        slow_model.build_model()
        matrix_model.build_model()

        for i in range(slow_model.horizon - 1):
            slow_model.infer(inference_type='slow')
            slow_model.compute_probabilities()
            slow_model.move_next()
            slow_model.update_model()
            slow_model.build_model()

            matrix_model.infer(inference_type='matrix')
            matrix_model.compute_probabilities()
            matrix_model.move_next()
            matrix_model.update_model()
            matrix_model.build_model()

            # compare marginals
            for key in slow_model.bp.var_probabilities:
                a = slow_model.bp.var_probabilities[key]
                b = matrix_model.bp.var_probabilities[key]
                assert np.allclose(a, b, equal_nan=True)

            # compare conditional probabilities
            for key in slow_model.bp.conditional_probabilities:
                a = slow_model.bp.conditional_probabilities[key]
                b = matrix_model.bp.conditional_probabilities[key]
                assert np.allclose(a, b, equal_nan=True)

    def test_clamp_vs_advance(self):
        """
        Is it the same to advance by clamping states than by shortening the model?
        :return:
        """
        clamp_model, advance_model = setup_two_models()

        clamp_model.build_model()
        advance_model.build_model()

        clamp_trajectories = []
        advance_trajectories = []
        for i in range(clamp_model.horizon-1):
            clamp_trajectories.append(clamp_model.aPos.copy())
            clamp_model.infer(inference_type='slow')
            clamp_model.compute_probabilities()
            clamp_model.move_next()
            clamp_model.update_model()

            advance_trajectories.append(advance_model.aPos.copy())
            advance_model.infer(inference_type='slow')
            advance_model.compute_probabilities()
            advance_model.move_next()
            advance_model.time = 1
            advance_model.horizon -= 1
            advance_model.build_model()

        clamp_trajectories.append(clamp_model.aPos.copy())
        advance_trajectories.append(advance_model.aPos.copy())

        assert np.all(advance_trajectories == clamp_trajectories)
