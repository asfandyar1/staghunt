import numpy as np
import matplotlib.pyplot as plt
from mrftools import MarkovNet
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


class StagHuntGame:

    def __init__(self, size=SIZE, s_pos=SPOS, h_pos=HPOS, a_pos=APOS):
        self.size = size
        self.sPos = s_pos
        self.hPos = h_pos
        self.aPos = a_pos
        self._r_h = RH
        self._r_s = RS
        self._lmb = LAMBDA
        self._horizon = T
        self.N = np.prod(size)

        self.markov_net = None

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
        self.horizon = value

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

    def _get_index(self, pos):
        """
        Converts the position expressed in cartesian coordinates to an index i in {0,...,N-1}
        :param pos: position (x,y) in {1,...,size_x} x {1,...,size_y}
        :return: index i in {0,...,N-1}
        """
        x, y = pos
        index = (y-1) * self.size[0] + x - 1  # i in {0,...,N-1}
        return index

    def _get_pos(self, index):
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

    def build_model(self):
        """
        Builds the mrftools library MarkovNet model based on the game definition
        :return: none - sets markov_net attribute
        """
        mn = MarkovNet()

        # clamp initial state to localised unary potentials
        # set uniform unary potentials to x21,...,x2M,...,xTM
        # all unary factors involving agent variables are defined here
        for i in range(1, self.horizon+1):
            for j, agent_pos in enumerate(self.aPos):
                agent_index = j + 1
                var_key = 'x' + str(i) + str(agent_index)
                if i == 1:  # t = 1 initial state -> clamp
                    factor = np.zeros(self.N)
                    factor[self._get_index(agent_pos)] = 1
                elif i < self.horizon:  # t = 2,...,T-1 -> uniform
                    factor = np.ones(self.N)
                else: # t = T -> \prod_{k=1}^{k=H}phi_{h_k}
                    factor = np.ones(self.N)
                    for hare_pos in self.hPos:
                        factor[self._get_index(hare_pos)] = np.exp(-self.r_h / self.lmb)
                # set factor
                mn.set_unary_factor(var_key, factor)

        # uncontrolled dynamics pairwise factors phi_q
        # build the phi_q factor, which is the same for every variable pair
        phi_q = np.zeros((25, 25))
        for i in range(25):
            for j in range(25):
                phi_q[i, j] = self.phi_q(self._get_pos(i), self._get_pos(j))
        # set the factor and form the chains
        for i in range(1, self.horizon):
            for j in range(1, len(self.aPos)+1):
                var_keys = ('x' + str(i) + str(j), 'x' + str(i+1) + str(j))
                mn.set_edge_factor(var_keys, phi_q)

        # unary and pairwise factors involving auxiliary variables d_ij, u_ij
        for i, agent_pos in enumerate(self.aPos):
            agent_index = i + 1
            for j, stag_pos in enumerate(self.sPos):
                stag_index = j + 1
                var_key = 'd' + str(agent_index) + str(j + 1)
                mn.set_unary_factor(var_key, np.ones(2))
                agent_key = 'x' + str(self.horizon) + str(agent_index)

                # inefficient but obvious way to fill the potential
                phi_s = np.zeros((self.N, 2))
                for x in range(phi_s.shape[0]):
                    for d in range(phi_s.shape[1]):
                        print(d, x, self._get_index(stag_pos))
                        if d == kronecker_delta(x, self._get_index(stag_pos)):
                            phi_s[x, d] = 1

                mn.set_edge_factor((agent_key, var_key), phi_s)

        self.markov_net = mn
