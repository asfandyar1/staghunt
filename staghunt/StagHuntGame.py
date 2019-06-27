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
        self.N = int(prod(size))
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
            if value <= 0:
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
        self.N = prod(value)

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
        r = index % self.size[0]
        if r == 0:
            x = self.size[0]
            y = index // self.size[0]
        else:
            x = r
            y = index // self.size[0] + 1
        return x, y

    def get_game_config(self):
        """
        Utility to copy the game configuration from one instance to another
        :return: None
        """
        return (self.size, self.aPos.copy(), self.hPos.copy(), self.sPos.copy(),
                self.r_h, self.r_s, self.lmb, self.horizon)

    def set_game_config(self, game_conf):
        """
        Utility to set the game configuration coming from another instance
        :return: None
        """
        self.size, self.aPos,  self.hPos, self.sPos, self.r_h, self.r_s, self.lmb, self.horizon = game_conf
