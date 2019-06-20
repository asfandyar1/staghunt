import numpy as np
from mrftools import MarkovNet, BeliefPropagator, MatrixBeliefPropagator
from .StagHuntModel import StagHuntModel
from .util import *


class MatrixStagHuntModel(StagHuntModel):

    def __init__(self):
        """
        MRF formulation of the game, using mrftools package
        """
        super().__init__()
        self.mrf = None
        self.bp = None

    def _set_agent_vars(self, mn):
        """
        Sets clamped unary potentials to x11,...,x1M
        Sets uniform unary potentials to x21,...,x2M,...,x(T-1)M
        Sets hare-reward unary potentials to xT1,...,xTM
        All unary factors involving agent variables are defined here
        :param mn: MarkovNet object
        :return: Modified MarkovNet object
        """
        for i in range(1, self.horizon + 1):
            for j, agent_pos in enumerate(self.aPos):
                agent_index = j + 1
                var_key = new_var('x', i, agent_index)
                if i == 1:  # t = 1 initial state -> clamp
                    factor = np.full(self.N, self.MIN, dtype=np.float64)
                    factor[self.get_index(agent_pos)] = self.NEU
                elif i < self.horizon:  # t = 2,...,T-1 -> uniform
                    factor = np.full(self.N, self.NEU, dtype=np.float64)
                else:  # t = T -> \prod_{k=1}^{k=H}phi_{h_k}
                    factor = np.full(self.N, self.NEU, dtype=np.float64)
                    for hare_pos in self.hPos:
                        # factor[self.get_index(hare_pos)] = np.exp(-self.r_h / self.lmb)
                        factor[self.get_index(hare_pos)] = -self.r_h / self.lmb
                # set factor
                mn.set_unary_factor(var_key, factor)
        return mn

    def _set_uncontrolled_dynamics(self, mn):
        """
        Sets the uncontrolled dynamics pairwise factors phi_q: (x11,x21),...,(x(T-1)1,xT1),...,(x(T-1)M,xTM)
        :param mn: MarkovNet object
        :return: Modified MarkovNet object
        """
        # build the phi_q factor, which is the same for every variable pair
        phi_q = np.full((self.N, self.N), self.NEU, dtype=np.float64)
        for i in range(self.N):
            for j in range(self.N):
                phi_q[i, j] = self.phi_q(self.get_pos(i), self.get_pos(j))
        # and set the factor forming the chains
        for i in range(1, self.horizon):
            for j in range(1, len(self.aPos) + 1):
                var_keys = (new_var('x', i, j), new_var('x', i + 1, j))
                mn.set_edge_factor(var_keys, phi_q)
        return mn

    def build_ground_model(self):
        """
        Builds the mrftools MarkovNet ground truth model based on the game definition
        Works only if the number of agents is exactly equal to 2
        :return: none - sets markov_net attribute
        """

        if not len(self.aPos) == 2:
            raise ValueError('Ground truth model can only be built when the number of agents is 2')

        mn = MarkovNet()
        mn = self._set_agent_vars(mn)
        mn = self._set_uncontrolled_dynamics(mn)

        # stag control factor -> ones and stag reward when both agents are in the position of a stag
        factor = np.full((self.N, self.N), self.NEU, dtype=np.float64)
        for stag_pos in self.sPos:
            s_ind = self.get_index(stag_pos)
            factor[s_ind, s_ind] = -self.r_s / self.lmb
        # one factor
        var_keys = (new_var('x', self.horizon, 1), new_var('x', self.horizon, 2))
        mn.set_edge_factor(var_keys, factor)

        mn.create_matrices()
        self.mrf = mn

    def build_model(self):
        """
        Builds the mrftools library MarkovNet model based on the game definition
        :return: none - sets markov_net attribute
        """
        mn = MarkovNet()
        self.time = 1

        mn = self._set_agent_vars(mn)
        mn = self._set_uncontrolled_dynamics(mn)

        # unary and pairwise factors involving auxiliary variables d_ij, u_ij, z_ij
        for i, agent_pos in enumerate(self.aPos):
            agent_index = i + 1
            for j, stag_pos in enumerate(self.sPos):
                stag_index = j + 1
                # declare d_ij variables and set uniform unary potentials
                var_key_d = 'd' + str(agent_index) + str(stag_index)
                mn.set_unary_factor(var_key_d, np.full(2, self.NEU, dtype=np.float64))
                # declare u_{ij} variables and set uniform unary potentials
                if agent_index > 1:
                    var_key_u = new_var('u', agent_index, stag_index)
                    mn.set_unary_factor(var_key_u, np.full(3, self.NEU, dtype=np.float64))
                    var_key_z = new_var('z', agent_index, stag_index)
                    if agent_index == 2:
                        mn.set_unary_factor(var_key_z, np.full(12, self.NEU, dtype=np.float64))
                        mn.set_edge_factor((var_key_z, new_var('d', agent_index-1, stag_index)),
                                           np.array(self.edge_factor(([0, 1], [0, 1], [0, 1, 2]), 0),
                                                    dtype=np.float64))
                        mn.set_edge_factor((var_key_z, var_key_d),
                                           np.array(self.edge_factor(([0, 1], [0, 1], [0, 1, 2]), 1),
                                                    dtype=np.float64))
                        mn.set_edge_factor((var_key_z, var_key_u),
                                           np.array(self.edge_factor(([0, 1], [0, 1], [0, 1, 2]), 2),
                                                    dtype=np.float64))
                    else:
                        mn.set_unary_factor(var_key_z, np.full(18, self.NEU, dtype=np.float64))
                        mn.set_edge_factor((var_key_z, var_key_d),
                                           np.array(self.edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 1),
                                                    dtype=np.float64))
                        mn.set_edge_factor((var_key_z, var_key_u),
                                           np.array(self.edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 2),
                                                    dtype=np.float64))
                        mn.set_edge_factor((var_key_z, new_var('u', agent_index-1, stag_index)),
                                           np.array(self.edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 0),
                                                    dtype=np.float64))

                # build and set phi_{s_j} potentials
                var_key_x = new_var('x', self.horizon, agent_index)
                # inefficient but obvious way to fill the potential phi_{s_j}
                phi_s = np.full((self.N, 2), self.MIN, dtype=np.float64)
                for x in range(phi_s.shape[0]):
                    for d in range(phi_s.shape[1]):
                        if d == kronecker_delta(x, self.get_index(stag_pos)):
                            phi_s[x, d] = self.NEU

                mn.set_edge_factor((var_key_x, var_key_d), phi_s)

        factor = np.full(3, self.NEU, dtype=np.float64)
        factor[2] = -self.r_s / self.lmb
        for j in range(len(self.sPos)):
            mn.set_unary_factor(new_var('u', len(self.aPos), j+1), factor)

        mn.create_matrices()
        self.mrf = mn

    def update_model(self):
        """
        Updates the mrf model by clamping the current position of the agents
        :return: None
        """
        for j, agent_pos in enumerate(self.aPos):
            agent_index = j + 1
            var_key = new_var('x', self.time, agent_index)
            factor = np.full(self.N, self.MIN, dtype=np.float64)
            factor[self.get_index(agent_pos)] = self.NEU
            self.mrf.set_unary_factor(var_key, factor)
        self.mrf.create_matrices()  # IMPORTANT

    def infer(self, inference_type=None, max_iter=30000):
        """
        Runs matrix inference on the current MRF. Sets the object bp to the resulting BeliefPropagator object.
        :param inference_type: Type of inference: slow - python loops BP OR matrix - sparse matrix BP
        :param max_iter: Max number of iterations of BP
        :return: None
        """

        if inference_type == 'matrix':
            bp = MatrixBeliefPropagator(self.mrf)
        else:
            bp = BeliefPropagator(self.mrf)  # DEFAULT: slow BP
        bp.set_max_iter(max_iter)
        bp.infer(display='final')
        bp.load_beliefs()
        self.bp = bp

    def compute_probabilities(self):
        """
        If the bp object is loaded, computes the conditional probabilities for every variable pair in pair_beliefs
        :return: None
        """

        if self.bp:
            # convert variable beliefs into probabilities
            self.bp.var_probabilities = {}
            for key in self.bp.var_beliefs:
                var_probabilities = np.exp(self.bp.var_beliefs[key])
                # fix zeroes with tolerance
                self.bp.var_probabilities[key] = round_by_tol(var_probabilities, self.TOL)

            # compute pair conditional probabilities from pair (joint) probabilities and var probabilities
            # P(x2 | x1) is stored in key (x1, x2)
            self.bp.conditional_probabilities = {}
            for key in self.bp.pair_beliefs:
                with np.errstate(divide='ignore', invalid='ignore'):
                    pair_prob = round_by_tol(np.exp(self.bp.pair_beliefs[key]), self.TOL)
                    cond_prob = np.transpose(np.transpose(pair_prob) / self.bp.var_probabilities[key[0]])
                    self.bp.conditional_probabilities[key] = cond_prob

    def move_next(self, break_ties='random'):
        """
        Look for the states of maximum probability and move the agents accordingly, breaking ties randomly.
        :return: none - state change
        """
        if not(self.bp.conditional_probabilities and self.bp.var_probabilities) or self.time == self.horizon:
            return

        for i in range(len(self.aPos)):
            var_key = (new_var('x', self.time, i + 1), new_var('x', self.time + 1, i + 1))
            trans_mat = self.bp.conditional_probabilities[var_key]
            i_from, i_to = np.unravel_index(np.nanargmax(trans_mat), trans_mat.shape)
            if break_ties == 'first':  # TESTING: need to be able to go always to the same destination (matrix VS torch)
                i_to = np.isclose(trans_mat, trans_mat[i_from, i_to]).nonzero()[1][0]
            elif break_ties == 'random':  # NORMALLY: we break ties randomly
                possibilities = np.isclose(trans_mat, trans_mat[i_from, i_to]).nonzero()[1]
                random_index = np.random.choice(possibilities.shape[0], 1, replace=False).item()
                i_to = possibilities[random_index]
            self.aPos[i] = self.get_pos(i_to)
        self.time += 1

    def run_game(self, inference_type='slow', verbose=True, break_ties='random'):
        """
        Run the inference to the horizon clamping the variables at every time step as decisions are taken
        :param inference_type: Type of inference: slow - python loops BP OR matrix - sparse matrix BP
        :param verbose: Prints info about the agents final positions
        :param break_ties: Way in which ties are broken, either random or first
        :return: None
        """
        for i in range(self.horizon - 1):
            self.infer(inference_type=inference_type)
            self.compute_probabilities()
            self.move_next(break_ties=break_ties)
            self.update_model()
        # Print trajectories and preys in final positions if verbose
        if verbose:
            for agent in range(1, len(self.aPos) + 1):
                trajectory = []
                for i in range(1, self.horizon + 1):
                    position_index = np.argmax(self.mrf.unary_potentials['x' + str(i) + str(agent)])
                    trajectory.append(self.get_pos(position_index))
                if trajectory[-1] in self.hPos:
                    sth = 'hare'
                elif trajectory[-1] in self.sPos:
                    sth = 'stag'
                else:
                    sth = 'nothing'
                print("->".join([str(el) for el in trajectory]), sth)
