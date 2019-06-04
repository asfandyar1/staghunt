import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
from mrftools import MarkovNet, BeliefPropagator, MatrixBeliefPropagator
from .util import *

# DEFAULT game configuration
SIZE = (5, 5)
S_POS = [(3, 3)]
H_POS = [(1, 1), (1, 5), (5, 1), (5, 5)]
A_POS = [(2, 1), (5, 3)]

# DEFAULT game parameters
T = 4
RS = -10
RH = -2
LAMBDA = 10

# MISC parameters
MIN = -np.float64(100)  # alternative to -inf
TOL = 10e-5  # tolerance to set number to zero
NEU = np.float64(0)  # neutral element


def phi_r1(d1, d2, u1):
    """
    Straightforward implementation of the phi_r1 factor
    :param d1: d_{1j} variable, j = 1,...,S
    :param d2: d_{2j} variable, j = 1,...,S
    :param u1: u_{1j} variable, j = 1,...,S
    :return: Indicator function of allowed configurations
    """
    condition = ((d1 == 0) and (d2 == 0) and (u1 == 0)) or \
                ((d1 == 1) and (d2 == 1) and (u1 == 2)) or \
                ((d1 != d2) and (u1 == 1))
    return NEU if condition else MIN


def phi_ri(u1, d2, u2):
    """
    Straightforward implementation of the phi_{r_{i-1}} factor, i = 3,...,M
    :param u1: d_{(i-1)j} variable, j = 1,...,S
    :param d2: d_{ij} variable, j = 1,...,S
    :param u2: u_{ij} variable, j = 1,...,S
    :return: Indicator function of allowed configurations
    """
    condition = ((d2 == 0) and (u1 == u2)) or \
                ((d2 == 1) and (u1 == 0) and (u2 == 1)) or \
                ((d2 == 1) and (u1 == 1) and (u2 == 2)) or \
                ((d2 == 1) and (u1 == 2) and (u2 == 2))
    return NEU if condition else MIN


def edge_factor(var_tuple, pos):
    """
    Straightforward implementation of the conversion from ternary to binary factors
    by means of an auxiliary variable z that lives in the cartesian product of the
    individual domains.
    :param var_tuple: contains lists with the possible values for each of the 3 variables
    :param pos: position in the tuple of the variable to factor with the aux var.
    :return: factor as a matrix
    """
    z_card = np.prod([len(v) for v in var_tuple])  # cardinality of new var z
    x_card = len(var_tuple[pos])  # cardinality of old var x1, x2, x3
    factor = np.full((z_card, x_card), MIN, dtype=float64)
    for i, el in enumerate(itertools.product(*var_tuple)):
        for j in range(x_card):
            if j == el[pos]:  # indicator function
                if z_card == 12:
                    factor[i, j] = phi_r1(*el)
                elif z_card == 18:
                    factor[i, j] = phi_ri(*el)
    return factor


def new_var(var_name, t, agent):
    """
    Naming of agent variables xij, where i in {1,...,T} and j in {1,...,M}
    :param var_name: variable name x, u, d,...
    :param t: time index
    :param agent: agent index
    :return: string with the name of the variable
    """
    return var_name + str(t) + str(agent)


class StagHuntGame:

    def __init__(self, size=SIZE, s_pos=None, h_pos=None, a_pos=None):
        """
        Initial configuration of the game.
        :param size: Size of the grid as a 2-tuple
        :param s_pos: Positions of the stags as a list of 2-tuples
        :param h_pos: Positions of the hares as a list of 2-tuples
        :param a_pos: Positions of the agents as a list of 2-tuples
        """
        if s_pos is None:
            self.sPos = S_POS.copy()
        if h_pos is None:
            self.hPos = H_POS.copy()
        if a_pos is None:
            self.aPos = A_POS.copy()
        self._size = size
        self.N = int(np.prod(size))
        self._r_h = RH
        self._r_s = RS
        self._lmb = LAMBDA
        self._horizon = T
        self.time = 1

    @property
    def r_h(self):
        return self._r_h

    @property
    def r_s(self):
        return self._r_s

    @property
    def lmb(self):
        return self._lmb

    @property
    def horizon(self):
        return self._horizon

    @property
    def size(self):
        return self._size

    @r_h.setter
    def r_h(self, value):
        self._r_h = value

    @r_s.setter
    def r_s(self, value):
        self._r_s = value

    @lmb.setter
    def lmb(self, value):
        self._lmb = value

    @horizon.setter
    def horizon(self, value):
        if not isinstance(value, int):
            raise TypeError('Horizon must be of type int')
        else:
            if value <=0:
                raise ValueError('Horizon must be greater than 0')

        self._horizon = value

    @size.setter
    def size(self, value):
        if not isinstance(value, tuple):
            raise TypeError('Size must be a tuple of int types')
        else:
            if not all([isinstance(e, int) and e > 0 for e in value]):
                raise TypeError('Size tuple elements must be positive integers')
        self._size = value
        self.N = np.prod(value)

    def reward(self, a_pos):
        """
        State dependent reward
        :param a_pos: position of the agents in the game
        :return: reward
        """
        if not a_pos:  # not informed
            a_pos = self.aPos

        r_h = RH * sum([1 for hunt in self.hPos if hunt in a_pos])
        r_s = RS * sum([1 for stag in self.sPos if a_pos.count(stag) > 1])
        return r_h + r_s

    def phi_q(self, x1, x2):
        """
        Binary factor phi_q for a given markov state transition
        :param x1: agent position at time t
        :param x2: agent position at time t+1
        :return: factor value, binary
        """
        x1_x, x1_y = x1
        x2_x, x2_y = x2
        ind = ((x2_x == x1_x) and (x2_y == x1_y)) or \
              (x2_x == x1_x - 1) and (x2_y == x1_y) and (x1_x > 0) or \
              (x2_x == x1_x) and (x2_y == x1_y - 1) and (x1_y > 0) or \
              (x2_x == x1_x + 1) and (x2_y == x1_y) and (x1_x < self.size[0]) or \
              (x2_x == x1_x) and (x2_y == x1_y + 1) and (x1_y < self.size[1])
        return NEU if ind else MIN

    def get_index(self, pos):
        """
        Converts the position expressed in cartesian coordinates into an index i in {0,...,N-1}
        :param pos: position (x,y) in {1,...,size_x} x {1,...,size_y}
        :return: index i in {0,...,N-1}
        """
        x, y = pos
        index = (y-1) * self.size[0] + x - 1  # i in {0,...,N-1}
        return index

    def get_pos(self, index):
        """
        Converts the position expressed as an index i in {1,...,N} into cartesian coordinates
        :param index: index i in {1,...,N}
        :return: position (x,y) in {1,...,size_x} x {1,...,size_y}
        """
        index = index + 1  # convert to {1,...,size_x} x {1,...,size_y}
        r = np.remainder(index, self.size[0])
        if r == 0:
            x = self.size[0]
            y = index // self.size[0]
        else:
            x = r
            y = index // self.size[0] + 1
        return x, y

    def new_game_sample(self, size, num_agents):
        """
        Generates a random game configuration of given size and number of agents
        Number of hares is set to 2*num_agents, and number of stags to num_agents//2
        :param size: size of the grid
        :type size: tuple
        :param num_agents:
        :type num_agents: int
        :return: none - changes state
        """
        # Number of things in the grid
        num_things = 3 * num_agents + num_agents // 2
        # Assert that size is compatible with the number of agents (there's no too many things and no space)
        assert np.prod(size) > num_things
        self.size = size
        locations = set()
        while len(locations) < num_things:
            locations.add((random.randint(1, size[0]), random.randint(1, size[1])))
        locations = list(locations)

        self.aPos = locations[:num_agents]
        locations = locations[num_agents:]
        self.hPos = locations[:2*num_agents]
        locations = locations[2*num_agents:]
        self.sPos = locations

    def get_game_config(self):
        """
        Utility to copy the game configuration from one instance to another
        :return: None
        """
        return self.aPos.copy(), self.hPos.copy(), self.sPos.copy()

    def set_game_config(self, game_conf):
        """
        Utility to set the game configuration coming from another instance
        :return: None
        """
        self.aPos,  self.hPos, self.sPos = game_conf


class StagHuntMRF(StagHuntGame):

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
                    factor = np.full(self.N, MIN, dtype=float64)
                    factor[self.get_index(agent_pos)] = NEU
                elif i < self.horizon:  # t = 2,...,T-1 -> uniform
                    factor = np.full(self.N, NEU, dtype=float64)
                else:  # t = T -> \prod_{k=1}^{k=H}phi_{h_k}
                    factor = np.full(self.N, NEU, dtype=float64)
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
        phi_q = np.full((self.N, self.N), NEU, dtype=float64)
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
        factor = np.full((self.N, self.N), NEU, dtype=float64)
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

        # uncontrolled dynamics pairwise factors phi_q
        # build the phi_q factor, which is the same for every variable pair
        phi_q = np.full((self.N, self.N), NEU, dtype=float64)
        for i in range(self.N):
            for j in range(self.N):
                phi_q[i, j] = self.phi_q(self.get_pos(i), self.get_pos(j))
        # and set the factor forming the chains
        for i in range(1, self.horizon):
            for j in range(1, len(self.aPos)+1):
                var_keys = (new_var('x', i, j), new_var('x', i + 1, j))
                mn.set_edge_factor(var_keys, phi_q)

        # unary and pairwise factors involving auxiliary variables d_ij, u_ij, z_ij
        for i, agent_pos in enumerate(self.aPos):
            agent_index = i + 1
            for j, stag_pos in enumerate(self.sPos):
                stag_index = j + 1
                # declare d_ij variables and set uniform unary potentials
                var_key_d = 'd' + str(agent_index) + str(stag_index)
                mn.set_unary_factor(var_key_d, np.full(2, NEU, dtype=float64))
                # declare u_{ij} variables and set uniform unary potentials
                if agent_index > 1:
                    var_key_u = new_var('u', agent_index, stag_index)
                    mn.set_unary_factor(var_key_u, np.full(3, NEU, dtype=float64))
                    var_key_z = new_var('z', agent_index, stag_index)
                    if agent_index == 2:
                        mn.set_unary_factor(var_key_z, np.full(12, NEU, dtype=float64))
                        mn.set_edge_factor((var_key_z, new_var('d', agent_index-1, stag_index)),
                                           edge_factor(([0, 1], [0, 1], [0, 1, 2]), 0))
                        mn.set_edge_factor((var_key_z, var_key_d),
                                           edge_factor(([0, 1], [0, 1], [0, 1, 2]), 1))
                        mn.set_edge_factor((var_key_z, var_key_u),
                                           edge_factor(([0, 1], [0, 1], [0, 1, 2]), 2))
                    else:
                        mn.set_unary_factor(var_key_z, np.full(18, NEU, dtype=float64))
                        mn.set_edge_factor((var_key_z, var_key_d),
                                           edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 1))
                        mn.set_edge_factor((var_key_z, var_key_u),
                                           edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 2))
                        mn.set_edge_factor((var_key_z, new_var('u', agent_index-1, stag_index)),
                                           edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 0))

                # build and set phi_{s_j} potentials
                var_key_x = new_var('x', self.horizon, agent_index)
                # inefficient but obvious way to fill the potential phi_{s_j}
                phi_s = np.full((self.N, 2), MIN, dtype=float64)
                for x in range(phi_s.shape[0]):
                    for d in range(phi_s.shape[1]):
                        if d == kronecker_delta(x, self.get_index(stag_pos)):
                            phi_s[x, d] = NEU

                mn.set_edge_factor((var_key_x, var_key_d), phi_s)

        factor = np.full(3, NEU, dtype=float64)
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
            factor = np.array(self.N * [MIN])
            factor[self.get_index(agent_pos)] = NEU
            self.mrf.set_unary_factor(var_key, factor)
        self.mrf.create_matrices()

    def infer(self, inference_type='slow', max_iter=30000):
        """
        Runs matrix inference on the current MRF. Sets the object bp to the resulting BeliefPropagator object.
        :param inference_type: Type of inference: slow - python loops BP OR matrix - sparse matrix BP
        :param max_iter: Max number of iterations of BP
        :return: None
        """
        if inference_type == 'slow':
            bp = BeliefPropagator(self.mrf)
        elif inference_type == 'matrix':
            bp = MatrixBeliefPropagator(self.mrf)
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
                self.bp.var_probabilities[key] = round_by_tol(var_probabilities, TOL)

            # compute pair conditional probabilities from pair (joint) probabilities and var probabilities
            # P(x2 | x1) is stored in key (x1, x2)
            self.bp.conditional_probabilities = {}
            for key in self.bp.pair_beliefs:
                with np.errstate(divide='ignore', invalid='ignore'):
                    pair_prob = round_by_tol(np.exp(self.bp.pair_beliefs[key]), TOL)
                    cond_prob = np.transpose(np.transpose(pair_prob) / self.bp.var_probabilities[key[0]])
                    self.bp.conditional_probabilities[key] = cond_prob

    def move_next(self):
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
            self.aPos[i] = self.get_pos(i_to)
        self.time += 1

    def run_game(self, inference_type='slow'):
        """
        Run the inference to the horizon clamping the variables at every time step as decisions are taken
        :param inference_type: Type of inference: slow - python loops BP OR matrix - sparse matrix BP
        :return: None
        """
        for i in range(self.horizon - 1):
            self.infer(inference_type=inference_type)
            self.compute_probabilities()
            self.move_next()
            self.update_model()

    def reset_game(self):
        """
        Reset the game to the state it began
        :return: None
        """
        a_pos = []
        for agent in range(1, len(self.aPos)+1):
            a_pos.append(self.get_pos(np.argmax(self.mrf.unary_potentials[new_var('x', 1, agent)])))
        self.aPos = a_pos
        self.bp = None
        self.mrf = None
        self.time = 1

    def display(self):
        """
        Prints the game state on the screen
        :return: None
        """
        if len(self.size) == 2:
            s_x, s_y = np.transpose(np.array(self.sPos))
            h_x, h_y = np.transpose(np.array(self.hPos))
            a_x, a_y = np.transpose(np.array(self.aPos))
            size_x, size_y = self.size

            plt.scatter(h_x, h_y, marker='d', s=112, facecolors='none', edgecolors='k')
            plt.scatter(s_x, s_y, marker='d', s=224, facecolors='none', edgecolors='k')
            plt.scatter(a_x, a_y, s=82, facecolors='none', edgecolors='k')

            if self.mrf:
                for a_ind in range(1, len(self.aPos) + 1):
                    trajectory = []
                    for t in range(1, self.time + 1):
                        a_pos = self.get_pos(np.argmax(self.mrf.unary_potentials[new_var('x', str(t), str(a_ind))]))
                        trajectory.append(list(a_pos))
                    plt.plot(np.transpose(trajectory)[0], np.transpose(trajectory)[1], linewidth=0.8, color='k')
                    plt.scatter([trajectory[0][0]], [trajectory[0][1]], facecolors='none', edgecolors='k')

            plt.xlim((0, size_x + 1))
            plt.ylim((0, size_y + 1))
            plt.xticks(np.arange(1, size_x + 1))
            plt.yticks(np.arange(1, size_y + 1))
            plt.grid(linestyle='dotted')
            plt.show()
