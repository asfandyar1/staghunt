import numpy as np
import matplotlib.pyplot as plt
import itertools
from mrftools import MarkovNet, BeliefPropagator, MatrixBeliefPropagator
from .util import *

# DEFAULT game configuration
SIZE = (5, 5)
SPOS = [(3, 3)]
HPOS = [(1, 1), (1, 5), (5, 1), (5, 5)]
APOS = [(2, 1), (5, 3)]

# DEFAULT game parameters
T = 4
RS = -10
RH = -2
LAMBDA = 10


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
    return 1 if condition else 0


def phi_ri(u1,d2,u2):
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
    return 1 if condition else 0


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
    factor = np.zeros((z_card, x_card))
    for i, el in enumerate(itertools.product(*var_tuple)):
        for j in range(x_card):
            if j == el[pos]: # indicator function
                if z_card == 12:
                    factor[i, j] = phi_r1(*el)
                elif z_card == 18:
                    factor[i, j] = phi_ri(*el)
    return factor


class StagHuntGame:

    def __init__(self, size=SIZE, s_pos=SPOS, h_pos=HPOS, a_pos=APOS):
        """
        Initial configuration of the game.
        :param size: Size of the grid as a 2-tuple
        :param s_pos: Positions of the stags as a list of 2-tuples
        :param h_pos: Positions of the hares as a list of 2-tuples
        :param a_pos: Positions of the agents as a list of 2-tuples
        """
        self.size = size
        self.sPos = s_pos
        self.hPos = h_pos
        self.aPos = a_pos
        self._r_h = RH
        self._r_s = RS
        self._lmb = LAMBDA
        self._horizon = T
        self.N = np.prod(size)

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
        self._horizon = value

    def display(self):
        if len(self.size) == 2:
            s_x, s_y = np.transpose(np.array(self.sPos))
            h_x, h_y = np.transpose(np.array(self.hPos))
            a_x, a_y = np.transpose(np.array(self.aPos))
            size_x, size_y = self.size

            plt.scatter(h_x, h_y, marker='d', s=112, facecolors='none', edgecolors='k')
            plt.scatter(s_x, s_y, marker='d', s=224, facecolors='none', edgecolors='k')
            plt.scatter(a_x, a_y, s=82, facecolors='none', edgecolors='k')

            plt.xlim((0, size_x + 1))
            plt.ylim((0, size_y + 1))
            plt.xticks(np.arange(1, size_x + 1))
            plt.yticks(np.arange(1, size_y + 1))
            plt.grid(linestyle='dotted')
            plt.show()

    def reward(self, a_pos):
        """
        State dependent reward
        :return: reward
        """
        if not a_pos:  # not informed
            a_pos = self.aPos

        r_h = RH * sum([1 for hunt in self.hPos if hunt in a_pos])
        r_s = RS * sum([1 for stag in self.sPos if a_pos.count(stag) > 1])
        return r_h + r_s

    def phi_q(self, x1, x2):
        """
        Value of the binary factor phi_q for a given markov state transition
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
        return 1 if ind else 0

    def phi_hk(self, x):
        """
        Value of the unary factor phi_hk for a given state x^T_i
        :param x:
        :return: factor value, real
        """
        return np.exp(self._r_h / LAMBDA) if x in HPOS else 1

    def get_index(self, pos):
        """
        Converts the position expressed in cartesian coordinates to an index i in {0,...,N-1}
        :param pos: position (x,y) in {1,...,size_x} x {1,...,size_y}
        :return: index i in {0,...,N-1}
        """
        x, y = pos
        index = (y-1) * self.size[0] + x - 1  # i in {0,...,N-1}
        return index

    def get_pos(self, index):
        """
        Converts the position expressed as an index i in {1,...,N} in cartesian coordinates
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


class StagHuntMRF(StagHuntGame):

    def __init__(self):
        """
        MRF formulation of the game, using mrftools package
        :param game: instance of StagHuntGame
        """
        super().__init__()
        self.mrf = None
        self.bp = None

    def build_model(self):
        """
        Builds the mrftools library MarkovNet model based on the game definition
        :return: none - sets markov_net attribute
        """
        mn = MarkovNet()

        # clamp initial state to localised unary potentials
        # set uniform unary potentials to x21,...,x2M,...,xTM
        # all unary factors involving agent variables are defined here
        for i in range(1, self.horizon + 1):
            for j, agent_pos in enumerate(self.aPos):
                agent_index = j + 1
                var_key = 'x' + str(i) + str(agent_index)
                if i == 1:  # t = 1 initial state -> clamp
                    factor = np.zeros(self.N)
                    factor[self.get_index(agent_pos)] = 1
                elif i < self.horizon:  # t = 2,...,T-1 -> uniform
                    factor = np.ones(self.N)
                else:  # t = T -> \prod_{k=1}^{k=H}phi_{h_k}
                    factor = np.ones(self.N)
                    for hare_pos in self.hPos:
                        factor[self.get_index(hare_pos)] = np.exp(-self.r_h / self.lmb)
                # set factor
                mn.set_unary_factor(var_key, factor)

        # uncontrolled dynamics pairwise factors phi_q
        # build the phi_q factor, which is the same for every variable pair
        phi_q = np.zeros((25, 25))
        for i in range(25):
            for j in range(25):
                phi_q[i, j] = self.phi_q(self.get_pos(i), self.get_pos(j))
        # and set the factor forming the chains
        for i in range(1, self.horizon):
            for j in range(1, len(self.aPos)+1):
                var_keys = ('x' + str(i) + str(j), 'x' + str(i+1) + str(j))
                mn.set_edge_factor(var_keys, phi_q)

        # unary and pairwise factors involving auxiliary variables d_ij, u_ij, z_ij
        for i, agent_pos in enumerate(self.aPos):
            agent_index = i + 1
            for j, stag_pos in enumerate(self.sPos):
                stag_index = j + 1
                # declare d_ij variables and set uniform unary potentials
                var_key_d = 'd' + str(agent_index) + str(stag_index)
                mn.set_unary_factor(var_key_d, np.ones(2))
                # declare u_{ij} variables and set uniform unary potentials
                if agent_index > 1:
                    var_key_u = 'u' + str(agent_index) + str(stag_index)
                    mn.set_unary_factor(var_key_u, np.ones(3))
                    var_key_z = 'z' + str(agent_index) + str(stag_index)
                    if agent_index == 2:
                        mn.set_unary_factor(var_key_z, np.ones(12))
                        mn.set_edge_factor((var_key_z, 'd' + str(agent_index-1) + str(stag_index)),
                                           edge_factor(([0, 1], [0, 1], [0, 1, 2]), 0))
                        mn.set_edge_factor((var_key_z, var_key_d),
                                           edge_factor(([0, 1], [0, 1], [0, 1, 2]), 1))
                        mn.set_edge_factor((var_key_z, var_key_u),
                                           edge_factor(([0, 1], [0, 1], [0, 1, 2]), 2))
                    else:
                        mn.set_unary_factor(var_key_z, np.ones(18))
                        mn.set_edge_factor((var_key_z, var_key_d),
                                           edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 1))
                        mn.set_edge_factor((var_key_z, var_key_u),
                                           edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 2))
                        mn.set_edge_factor((var_key_z, 'u' + str(agent_index - 1) + str(stag_index)),
                                           edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 0))

                # build and set phi_{s_j} potentials
                var_key_x = 'x' + str(self.horizon) + str(agent_index)
                # inefficient but obvious way to fill the potential phi_{s_j}
                phi_s = np.zeros((self.N, 2))
                for x in range(phi_s.shape[0]):
                    for d in range(phi_s.shape[1]):
                        if d == kronecker_delta(x, self.get_index(stag_pos)):
                            phi_s[x, d] = 1

                mn.set_edge_factor((var_key_x, var_key_d), phi_s)

        factor = np.ones(3)
        factor[2] = np.exp(-self.r_s / self.lmb)
        for j in range(len(self.sPos)):
            mn.set_unary_factor('u' + str(len(self.aPos)) + str(j+1), factor)

        mn.create_matrices()
        self.mrf = mn

    def slow_infer(self):
        """
        Runs slow inference on the current MRF. Sets the object bp to the resulting BeliefPropagator object
        :return: None
        """
        slow_bp = BeliefPropagator(self.mrf)
        slow_bp.set_max_iter(30000)
        slow_bp.infer(display='final')
        slow_bp.compute_beliefs()
        slow_bp.compute_pairwise_beliefs()
        self.bp = slow_bp

    def print_trajectories(self):
        """
        Aux print of all agent trajectories after BP
        :return: None
        """
        for i in range(1, len(self.aPos) + 1):
            for t in range(1, self.horizon):
                var_key = ('x' + str(t) + str(i), 'x' + str(t+1) + str(i))
                trans_mat = self.bp.pair_beliefs[var_key]
                index_trans = np.unravel_index(np.argmax(trans_mat), trans_mat.shape)
                print('Agent ' + str(i), self.get_pos(index_trans[0]), ' -> ', self.get_pos(index_trans[1]))


