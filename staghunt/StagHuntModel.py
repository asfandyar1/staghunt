import random
import itertools
import matplotlib.pyplot as plt
from .StagHuntGame import StagHuntGame
from .util import prod, new_var


# MODEL parameters
MIN = -float(200)  # alternative to -inf
TOL = 10e-8  # tolerance to set number to zero
NEU = float(0)  # neutral element


class StagHuntModel(StagHuntGame):
    """

    """
    def __init__(self):
        """
        MRF formulation of the game, using mrftools package
        """
        super().__init__()
        self.MIN = MIN
        self.TOL = TOL
        self.NEU = NEU
        self.mrf = None
        self.bp = None
        self.build = None

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
        return self.NEU if ind else self.MIN

    def build_phi_q(self):
        """
        Inefficient way to compute the pairwise factor between agent vars (uncontrolled dynamics)
        :return:
        """
        phi_q = [[self.NEU]*self.N for _ in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                phi_q[i][j] = self.phi_q(self.get_pos(i), self.get_pos(j))

        return phi_q

    def phi_r1(self, d1, d2, u1):
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
        return self.NEU if condition else self.MIN

    def phi_ri(self, u1, d2, u2):
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
        return self.NEU if condition else self.MIN

    def edge_factor(self, var_tuple, pos):
        """
        Straightforward implementation of the conversion from ternary to binary factors
        by means of an auxiliary variable z that lives in the cartesian product of the
        individual domains.
        :param var_tuple: contains lists with the possible values for each of the 3 variables
        :param pos: position in the tuple of the variable to factor with the aux var.
        :return: factor as a matrix
        """
        z_card = prod([len(v) for v in var_tuple])  # cardinality of new var z
        x_card = len(var_tuple[pos])  # cardinality of old var x1, x2, x3
        factor = [[self.MIN]*x_card for _ in range(z_card)]
        for i, el in enumerate(itertools.product(*var_tuple)):
            for j in range(x_card):
                if j == el[pos]:  # indicator function
                    if z_card == 12:
                        factor[i][j] = self.phi_r1(*el)
                    elif z_card == 18:
                        factor[i][j] = self.phi_ri(*el)
        return factor

    def _get_agent_pos(self, potential):
        """
        Util that returns the agent position from a given unary potential
        :param potential: unary potential of an agent variable
        :return: position of that agent in cartesian coordinates
        """
        potential = list(potential)
        a_index = max(range(len(potential)), key=lambda i: potential[i])
        a_pos = self.get_pos(a_index)
        return a_pos

    def _get_var_indices(self, var_array):
        return [self.mrf.var_index[var] for var in var_array]

    def reset_game(self):
        """
        Reset the game to the state it began, agents at locations of the first clamping, no bp nor mrf objects, time=1
        :return: None
        """
        if self.mrf:
            a_pos = []

            if len(self.mrf.unary_mat) and self.mrf.var_len is not None:
                for agent in range(1, len(self.aPos) + 1):
                    var_index = self._get_var_indices([new_var('x', 1, agent)])[0]
                    a_pos.append(self._get_agent_pos(self.mrf.unary_mat[:,var_index]))
            else:
                for agent in range(1, len(self.aPos) + 1):
                    a_pos.append(self._get_agent_pos(self.mrf.unary_potentials[new_var('x', 1, agent)]))
            self.aPos = a_pos
            del self.bp
            self.bp = None
            del self.mrf
            self.mrf = None
            self.time = 1
        self.build = None

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
        assert prod(size) > num_things
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

    def display(self, file=None):
        """
        Prints the game state on the screen
        :return: None
        """
        if len(self.size) == 2:
            s_x = [sPos[0] for sPos in self.sPos]  # stag x positions
            s_y = [sPos[1] for sPos in self.sPos]  # stag y positions
            h_x = [hPos[0] for hPos in self.hPos]  # hare x positions
            h_y = [hPos[1] for hPos in self.hPos]  # hare y positions
            a_x = [aPos[0] for aPos in self.aPos]  # agent x positions
            a_y = [aPos[1] for aPos in self.aPos]  # agent y positions
            size_x, size_y = self.size  # size of the grid

            plt.scatter(h_x, h_y, marker='d', s=112, facecolors='none', edgecolors='k')
            plt.scatter(s_x, s_y, marker='d', s=224, facecolors='none', edgecolors='k')
            plt.scatter(a_x, a_y, s=82, facecolors='none', edgecolors='k')

            if self.mrf and self.time > 1:  # have inference results so that we can draw trajectories
                for agent in range(1, len(self.aPos) + 1):
                    trajectory = []
                    for t in range(1, self.time + 1):
                        if len(self.mrf.unary_mat) and self.mrf.var_len is not None:
                            var_index = self._get_var_indices([new_var('x', t, agent)])[0]
                            a_pos = self._get_agent_pos(self.mrf.unary_mat[:, var_index])
                        else:
                            a_pos = self._get_agent_pos(self.mrf.unary_potentials[new_var('x', t, agent)])
                        trajectory.append(list(a_pos))
                    plt.plot(list(list(zip(*trajectory))[0]), list(list(zip(*trajectory))[1]), linewidth=0.8, color='k')
                    plt.scatter([trajectory[0][0]], [trajectory[0][1]], facecolors='none', edgecolors='k')

            plt.xlim((0, size_x + 1))
            plt.ylim((0, size_y + 1))
            plt.xticks(list(range(1, size_x + 1)))
            plt.yticks(list(range(1, size_y + 1)))
            plt.grid(linestyle='dotted')
            if file:
                plt.savefig(file, dpi=100, format='eps')
            else:
                plt.show()
