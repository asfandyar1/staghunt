import torch as t
from mrftools import TorchMarkovNet, TorchMatrixBeliefPropagator
from .StagHuntModel import StagHuntModel
from .util import *


class TorchStagHuntModel(StagHuntModel):

    def __init__(self, is_cuda=False, var_on=False):
        """
        MRF formulation of the game, using mrftools package
        """
        super().__init__()
        self.mrf = None
        self.bp = None
        self.is_cuda = is_cuda
        self.var_on = var_on
        self.MIN = -float('inf')
        self.dtype = t.float64

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
                    factor = t.tensor(self.N * [self.MIN], dtype=self.dtype)
                    factor[self.get_index(agent_pos)] = self.NEU
                elif i < self.horizon:  # t = 2,...,T-1 -> uniform
                    factor = t.tensor(self.N*[self.NEU], dtype=self.dtype)
                else:  # t = T -> \prod_{k=1}^{k=H}phi_{h_k}
                    factor = t.tensor(self.N*[self.NEU], dtype=self.dtype)
                    for hare_pos in self.hPos:
                        factor[self.get_index(hare_pos)] = -self.r_h / self.lmb
                # set factor
                mn.set_unary_factor(var_key, factor)
        return mn

    def build_phi_q(self):
        """
        Efficient way to compute the pairwise factor between agent vars (uncontrolled dynamics)
        :return:
        """
        phi_q = self.MIN * t.ones(self.N, self.N, requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            phi_q = phi_q.cuda()

        # diagonal
        phi_q[range(self.N), range(self.N)] = 0
        # n-diagonals
        phi_q[range(self.N - self.size[0]), range(self.size[0], self.N)] = 0
        phi_q[range(self.size[0], self.N), range(self.N - self.size[0])] = 0
        # diagonal starting at index 1
        index_1 = t.arange(self.N - 1)
        index_1 = index_1[(index_1 + 1) % self.size[0] != 0]
        index_2 = t.arange(1, self.N)
        index_2 = index_2[index_2 % self.size[0] != 0]
        phi_q[index_1, index_2] = 0
        phi_q[index_2, index_1] = 0

        return phi_q

    def _set_uncontrolled_dynamics(self, mn):
        """
        Sets the uncontrolled dynamics pairwise factors phi_q: (x11,x21),...,(x(T-1)1,xT1),...,(x(T-1)M,xTM)
        :param mn: MarkovNet object
        :return: Modified MarkovNet object
        """
        # build the phi_q factor, which is the same for every variable pair
        phi_q = self.build_phi_q()
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

        mn = TorchMarkovNet(is_cuda=self.is_cuda, var_on=self.var_on)
        mn = self._set_agent_vars(mn)
        mn = self._set_uncontrolled_dynamics(mn)

        # stag control factor -> ones and stag reward when both agents are in the position of a stag
        factor = self.NEU * t.ones(self.N, self.N, dtype=self.dtype)
        for stag_pos in self.sPos:
            s_ind = self.get_index(stag_pos)
            factor[s_ind, s_ind] = -self.r_s / self.lmb
        # one factor
        var_keys = (new_var('x', self.horizon, 1), new_var('x', self.horizon, 2))
        mn.set_edge_factor(var_keys, factor)

        mn.create_matrices()
        self.mrf = mn
        self.build = 1

    def build_model(self):
        """
        Builds the mrftools library MarkovNet model based on the game definition
        :return: none - sets markov_net attribute
        """
        mn = TorchMarkovNet(is_cuda=self.is_cuda, var_on=self.var_on)
        self.time = 1

        mn = self._set_agent_vars(mn)
        mn = self._set_uncontrolled_dynamics(mn)

        # unary and pairwise factors involving auxiliary variables d_ij, u_ij, z_ij
        for i, agent_pos in enumerate(self.aPos):
            agent_index = i + 1
            for j, stag_pos in enumerate(self.sPos):
                stag_index = j + 1
                # declare d_ij variables and set uniform unary potentials
                var_key_d = new_var('d', agent_index, stag_index)
                mn.set_unary_factor(var_key_d, t.tensor(2*[self.NEU], dtype=self.dtype))
                # declare u_{ij} variables and set uniform unary potentials
                if agent_index > 1:
                    var_key_u = new_var('u', agent_index, stag_index)
                    mn.set_unary_factor(var_key_u, t.tensor(3*[self.NEU], dtype=self.dtype))
                    var_key_z = new_var('z', agent_index, stag_index)
                    if agent_index == 2:
                        mn.set_unary_factor(var_key_z, t.tensor(12*[self.NEU], dtype=self.dtype))
                        mn.set_edge_factor((var_key_z, new_var('d', agent_index-1, stag_index)),
                                           t.tensor(self.edge_factor(([0, 1], [0, 1], [0, 1, 2]), 0),
                                                    dtype=self.dtype))
                        mn.set_edge_factor((var_key_z, var_key_d),
                                           t.tensor(self.edge_factor(([0, 1], [0, 1], [0, 1, 2]), 1),
                                                    dtype=self.dtype))
                        mn.set_edge_factor((var_key_z, var_key_u),
                                           t.tensor(self.edge_factor(([0, 1], [0, 1], [0, 1, 2]), 2),
                                                    dtype=self.dtype))
                    else:
                        mn.set_unary_factor(var_key_z, t.tensor(18*[self.NEU], dtype=self.dtype))
                        mn.set_edge_factor((var_key_z, var_key_d),
                                           t.tensor(self.edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 1),
                                                    dtype=self.dtype))
                        mn.set_edge_factor((var_key_z, var_key_u),
                                           t.tensor(self.edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 2),
                                                    dtype=self.dtype))
                        mn.set_edge_factor((var_key_z, new_var('u', agent_index-1, stag_index)),
                                           t.tensor(self.edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 0),
                                                    dtype=self.dtype))

                # build and set phi_{s_j} potentials
                var_key_x = new_var('x', self.horizon, agent_index)
                # inefficient but obvious way to fill the potential phi_{s_j}
                phi_s = self.MIN * t.ones(self.N, 2, dtype=self.dtype)
                for x in range(phi_s.shape[0]):
                    for d in range(phi_s.shape[1]):
                        if d == kronecker_delta(x, self.get_index(stag_pos)):
                            phi_s[x, d] = self.NEU

                mn.set_edge_factor((var_key_x, var_key_d), phi_s)

        factor = t.tensor(3*[self.NEU], dtype=self.dtype)
        factor[2] = -self.r_s / self.lmb
        for j in range(len(self.sPos)):
            mn.set_unary_factor(new_var('u', len(self.aPos), j+1), factor)

        mn.create_matrices()
        self.mrf = mn
        self.build = 1

    def _clamp_agents(self):
        """
        Util that clamps the position of the agents to the current time index
        :return:
        """
        factors = self.MIN * t.ones((self.N, len(self.aPos)), requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            factors = factors.cuda()
        for index, agent in enumerate(self.aPos):
            factors[self.get_index(agent), index] = self.NEU
        self.mrf.unary_mat[:, t.arange((self.time - 1)*len(self.aPos), self.time*len(self.aPos))] = factors

    def _fast_util(self, f_start, n_slices, chunk, from_cols, to_cols):
        if not chunk.shape == t.Size([0]):
            f_end = f_start + n_slices
            b_start = self.mrf.num_edges + f_start
            b_end = b_start + n_slices
            self.mrf.edge_pot_tensor[0:chunk.shape[0], 0:chunk.shape[1], f_start:f_end] = chunk
            self.mrf.edge_pot_tensor[0:chunk.shape[1], 0:chunk.shape[0], b_start:b_end] = chunk.transpose(1, 0)
            f_messages = list(range(f_start, f_end))
            b_messages = list(range(b_start, b_end))
            self.mrf.f_rows += f_messages
            self.mrf.t_rows += f_messages
            self.mrf.f_rows += b_messages
            self.mrf.t_rows += b_messages
            self.mrf.f_cols += from_cols
            self.mrf.t_cols += to_cols
            self.mrf.f_cols += to_cols
            self.mrf.t_cols += from_cols
            self.mrf.message_index.update({(self.mrf.var_list[from_cols[i]],
                                            self.mrf.var_list[to_cols[i]]): (i + f_start)
                                           for i in range(n_slices)})

    def fast_build_model(self):
        """
        Build the model by directly building the tensors
        :return:
        """
        if self.N < 18:
            print("Fast model building is only available for game sizes N >= 18")
            return

        self.mrf = TorchMarkovNet(is_cuda=self.is_cuda, var_on=self.var_on)
        self.mrf.matrix_mode = True

        self.mrf.max_states = self.N
        n_stag = len(self.sPos)
        n_agnt = len(self.aPos)
        n_vars = n_agnt * self.horizon + n_stag * (n_agnt + 2 * (n_agnt - 1))
        n_edges = n_agnt * (self.horizon - 1) + n_agnt * n_stag + 3 * (n_agnt - 1) * n_stag
        self.mrf.degrees = t.zeros(n_vars, requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            self.mrf.degrees = self.mrf.degrees.cuda()

        # VARIABLES OF THE MODEL
        self.mrf.var_list = []
        self.mrf.var_len = {}
        # agent vars
        self.mrf.var_list += [new_var('x', i + 1, j + 1) for i in range(self.horizon) for j in range(n_agnt)]
        self.mrf.var_len.update({new_var('x', i + 1, j + 1): self.N
                                 for i in range(self.horizon) for j in range(n_agnt)})
        # d vars
        self.mrf.var_list += [new_var('d', i + 1, j + 1) for j in range(n_stag) for i in range(n_agnt)]
        self.mrf.var_len.update({new_var('d', i + 1, j + 1): 2 for j in range(n_stag) for i in range(n_agnt)})
        # u vars
        self.mrf.var_list += [new_var('u', i + 2, j + 1) for j in range(n_stag) for i in range(n_agnt - 1)]
        self.mrf.var_len.update({new_var('u', i + 2, j + 1): 3 for j in range(n_stag) for i in range(n_agnt - 1)})
        # z vars
        self.mrf.var_list += [new_var('z', i + 2, j + 1) for i in range(n_agnt - 1) for j in range(n_stag)]
        self.mrf.var_len.update({new_var('z', 2, j + 1): 12 for j in range(n_stag)})
        self.mrf.var_len.update({new_var('z', i + 3, j + 1): 18 for i in range(n_agnt - 2) for j in range(n_stag)})
        # index
        self.mrf.var_index = {self.mrf.var_list[i]: i for i in range(len(self.mrf.var_list))}
        self.mrf.variables = set(self.mrf.var_list)

        # UNARY POTENTIALS MATRIX
        self.mrf.unary_mat = -float('inf') * t.ones((self.N, n_vars), requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            self.mrf.unary_mat = self.mrf.unary_mat.cuda()
        # clamped agent vars
        self._clamp_agents()
        # non-clamped agent vars
        col_start = n_agnt
        col_end = n_agnt * self.horizon
        factor = self.NEU * t.ones((self.N, col_end - col_start), requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            factor = factor.cuda()
        self.mrf.unary_mat[:, t.arange(n_agnt, col_end)] = factor
        self.mrf.unary_mat[[i for s in [n_agnt*[self.get_index(pos)] for pos in self.hPos] for i in s],
                           len(self.hPos)*list(range(col_end-col_start, col_end))] = -self.r_h / self.lmb
        # d vars
        col_start = col_end
        col_end = col_start + n_agnt*n_stag
        factor = self.NEU * t.ones((2, col_end - col_start), requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            factor = factor.cuda()
        self.mrf.unary_mat[0:2, col_start:col_end] = factor
        # u vars
        col_start = col_end
        col_end = col_start + n_stag*(n_agnt - 1)
        factor = self.NEU * t.ones((3, col_end - col_start), requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            factor = factor.cuda()
        self.mrf.unary_mat[0:3, col_start:col_end] = factor
        factor = (-self.r_s / self.lmb) * t.ones(n_stag, requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            factor = factor.cuda()
        self.mrf.unary_mat[2, self._get_var_indices([new_var('u', n_agnt, i+1) for i in range(n_stag)])] = factor
        # z vars
        col_start = col_end
        col_end = col_start + n_stag
        factor = self.NEU * t.ones((12, col_end - col_start), requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            factor = factor.cuda()
        self.mrf.unary_mat[0:12, col_start:col_end] = factor
        col_start = col_end
        col_end = col_start + n_stag*(n_agnt - 2)
        factor = self.NEU * t.ones((18, col_end - col_start), requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            factor = factor.cuda()
        self.mrf.unary_mat[0:18, col_start:col_end] = factor

        # EDGE POTENTIALS TENSOR
        self.mrf.num_edges = n_edges
        self.mrf.edge_pot_tensor = -float('inf') * t.ones((self.N, self.N, 2 * self.mrf.num_edges),
                                                          requires_grad=self.var_on, dtype=self.dtype)
        if self.is_cuda:
            self.mrf.edge_pot_tensor = self.mrf.edge_pot_tensor.cuda()
        self.mrf.message_index = {}
        # set up sparse matrix representation of adjacency
        self.mrf.f_rows, self.mrf.f_cols, self.mrf.t_rows, self.mrf.t_cols = [], [], [], []

        # phi_q potentials between x vars
        start = 0
        n_slices = n_agnt * (self.horizon - 1)
        factor = self.build_phi_q().repeat(n_agnt*(self.horizon - 1), 1, 1).transpose(0, 2)
        if self.is_cuda:
            factor = factor.cuda()
        self._fast_util(f_start=start, n_slices=n_slices,
                        chunk=factor,
                        from_cols=list(range(0, n_slices)), to_cols=list(range(n_agnt, n_agnt + n_slices)))

        # phi_s potentials between x vars and d vars
        start += n_slices
        n_slices = n_agnt * n_stag
        stack = t.stack((self.NEU * t.ones(self.N, requires_grad=self.var_on, dtype=self.dtype),
                         self.MIN * t.ones(self.N, requires_grad=self.var_on, dtype=self.dtype)))
        factor = stack.repeat((n_agnt*n_stag, 1, 1)).transpose(0, 1).transpose(1, 2)
        if self.is_cuda:
            factor = factor.cuda()
        s_index = [i for s in [n_agnt*[self.get_index(pos)] for pos in self.sPos] for i in s]
        factor[((n_agnt * n_stag) * [0], s_index, range(n_agnt * n_stag))] = self.MIN
        factor[((n_agnt * n_stag) * [1], s_index, range(n_agnt * n_stag))] = self.NEU
        self._fast_util(f_start=start, n_slices=n_slices, chunk=factor,
                        from_cols=n_stag * list(range(n_agnt * (self.horizon - 1), n_agnt * self.horizon)),
                        to_cols=list(range(n_agnt * self.horizon, n_agnt * (self.horizon + n_stag))))

        # factors between d_1j - z_2j, and d_2j and z_2j
        start += n_slices
        n_slices = 2 * n_stag
        stack = t.stack((t.tensor(self.edge_factor(([0, 1], [0, 1], [0, 1, 2]), 0),
                                  requires_grad=self.var_on, dtype=self.dtype),
                         t.tensor(self.edge_factor(([0, 1], [0, 1], [0, 1, 2]), 1),
                                  requires_grad=self.var_on, dtype=self.dtype)), dim=2)
        factor = stack.repeat(1, 1, n_stag)
        if self.is_cuda:
            factor = factor.cuda()
        self._fast_util(f_start=start, n_slices=n_slices, chunk=factor,
                        from_cols=[i for i in range(n_agnt * self.horizon, n_agnt * (self.horizon + n_stag))
                                   if i % n_agnt in {0, 1}],
                        to_cols=t.tensor(range(n_vars - (n_agnt - 1) * n_stag,
                                               n_vars - (n_agnt - 1) * n_stag + n_stag)).repeat_interleave(2).tolist())

        # factors between u_2j - z_2j
        start += n_slices
        n_slices = n_stag
        factor = t.tensor(self.edge_factor(([0, 1], [0, 1], [0, 1, 2]), 2),
                          requires_grad=self.var_on,
                          dtype=self.dtype).repeat(n_stag, 1, 1).transpose(0, 1).transpose(1, 2)
        if self.is_cuda:
            factor = factor.cuda()
        self._fast_util(f_start=start, n_slices=n_slices, chunk=factor,
                        from_cols=self._get_var_indices([new_var('u', 2, j+1) for j in range(n_stag)]),
                        to_cols=list(range(n_vars - (n_agnt - 1) * n_stag, n_vars - (n_agnt - 1) * n_stag + n_stag)))

        # factors between d_ij - z_ij, i>2
        start += n_slices
        n_slices = n_stag * (n_agnt - 2)
        if n_agnt == 2:
            factor = t.tensor([], requires_grad=self.var_on, dtype=self.dtype)
        else:
            factor = t.tensor(self.edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 1),
                              requires_grad=self.var_on,
                              dtype=self.dtype).repeat(n_stag * (n_agnt - 2), 1, 1).transpose(0, 1).transpose(1, 2)
        if self.is_cuda:
            factor = factor.cuda()
        self._fast_util(f_start=start, n_slices=n_slices, chunk=factor,
                        from_cols=[i for i in range(n_agnt * self.horizon, n_agnt * (self.horizon + n_stag))
                                   if i % n_agnt not in {0, 1}],
                        to_cols=self._get_var_indices([new_var('z', i+1, j+1)
                                                       for j in range(n_stag) for i in range(2, n_agnt)]))

        # factors between u_ij - z_ij, i>2
        start += n_slices
        n_slices = n_stag * (n_agnt - 2)
        if n_agnt == 2:
            factor = t.tensor([], requires_grad=self.var_on, dtype=self.dtype)
        else:
            factor = t.tensor(self.edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 2),
                              requires_grad=self.var_on,
                              dtype=self.dtype).repeat(n_stag * (n_agnt - 2), 1, 1).transpose(0, 1).transpose(1, 2)
        if self.is_cuda:
            factor = factor.cuda()
        self._fast_util(f_start=start, n_slices=n_slices, chunk=factor,
                        from_cols=self._get_var_indices([new_var('u', i + 1, j + 1)
                                                         for j in range(n_stag) for i in range(2, n_agnt)]),
                        to_cols=self._get_var_indices([new_var('z', i + 1, j + 1)
                                                       for j in range(n_stag) for i in range(2, n_agnt)]))

        # factors between u_ij - z_ij, i>2
        start += n_slices
        n_slices = n_stag * (n_agnt - 2)
        if n_agnt == 2:
            factor = t.tensor([], requires_grad=self.var_on, dtype=self.dtype)
        else:
            factor = t.tensor(self.edge_factor(([0, 1, 2], [0, 1], [0, 1, 2]), 0),
                              requires_grad=self.var_on,
                              dtype=self.dtype).repeat(n_stag * (n_agnt - 2), 1, 1).transpose(0, 1).transpose(1, 2)
        if self.is_cuda:
            factor = factor.cuda()
        self._fast_util(f_start=start, n_slices=n_slices, chunk=factor,
                        from_cols=self._get_var_indices([new_var('u', i, j + 1)
                                                         for j in range(n_stag) for i in range(2, n_agnt)]),
                        to_cols=self._get_var_indices([new_var('z', i + 1, j + 1)
                                                       for j in range(n_stag) for i in range(2, n_agnt)]))

        # generate a sparse matrix representation of the message indices to variables that receive messages
        if self.dtype == t.float64:
            self.mrf.message_to_map = t.sparse.DoubleTensor(
                t.tensor([self.mrf.t_rows, self.mrf.t_cols], dtype=t.long),
                t.ones(len(self.mrf.t_rows)).double(),
                t.Size([2 * self.mrf.num_edges, len(self.mrf.variables)])
            ).requires_grad_(self.var_on)
        else:
            self.mrf.message_to_map = t.sparse.FloatTensor(
                t.tensor([self.mrf.t_rows, self.mrf.t_cols], dtype=t.long),
                t.ones(len(self.mrf.t_rows)),
                t.Size([2 * self.mrf.num_edges, len(self.mrf.variables)])
            ).requires_grad_(self.var_on)

        if self.is_cuda:
            self.mrf.message_to_map = self.mrf.message_to_map.cuda()

        # store an array that lists which variable each message is sent to
        self.mrf.message_to = t.zeros(2 * self.mrf.num_edges, dtype=t.long)
        self.mrf.message_to[t.tensor(self.mrf.t_rows, dtype=t.long)] = t.tensor(self.mrf.t_cols, dtype=t.long)
        if self.is_cuda:
            self.mrf.message_to = self.mrf.message_to.cuda()
            self.mrf.message_to[t.tensor(self.mrf.t_rows,
                                         dtype=t.long).cuda()] = t.tensor(self.mrf.t_cols, dtype=t.long).cuda()

        # store an array that lists which variable each message is received from
        self.mrf.message_from = t.zeros(2 * self.mrf.num_edges, dtype=t.long)
        self.mrf.message_from[t.tensor(self.mrf.f_rows, dtype=t.long)] = t.tensor(self.mrf.f_cols, dtype=t.long)
        if self.is_cuda:
            self.mrf.message_from = self.mrf.message_from.cuda()
            self.mrf.message_from[t.tensor(self.mrf.f_rows,
                                           dtype=t.long).cuda()] = t.tensor(self.mrf.f_cols, dtype=t.long).cuda()

        self.build = 2

    def update_model(self):
        """
        Updates the mrf model by clamping the current position of the agents
        :return: None
        """
        for j, agent_pos in enumerate(self.aPos):
            agent_index = j + 1
            var_key = new_var('x', self.time, agent_index)
            factor = t.tensor(self.N * [self.MIN], dtype=self.dtype)
            factor[self.get_index(agent_pos)] = self.NEU
            self.mrf.set_unary_factor(var_key, factor)
        self.mrf.create_matrices()  # IMPORTANT

    def infer(self, max_iter=500):
        """
        Runs matrix inference on the current MRF. Sets the object bp to the resulting BeliefPropagator object.
        :param max_iter: Max number of iterations of BP
        :return: None
        """

        bp = TorchMatrixBeliefPropagator(self.mrf, is_cuda=self.is_cuda, var_on=self.var_on, dtype=self.dtype)
        bp.set_max_iter(max_iter)
        if self.dtype == t.float64:
            tolerance = 1e-8
        else:
            tolerance = 1e-4
        bp.infer(display='none', tolerance=tolerance)
        if self.build != 2:
            bp.load_beliefs()
        else:
            bp.compute_beliefs()
            bp.compute_pairwise_beliefs()
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
                var_probabilities = t.exp(self.bp.var_beliefs[key])
                # fix zeroes with tolerance
                # self.bp.var_probabilities[key] = t.from_numpy(round_by_tol(var_probabilities, TOL))
                self.bp.var_probabilities[key] = var_probabilities

            # compute pair conditional probabilities from pair (joint) probabilities and var probabilities
            # P(x2 | x1) is stored in key (x1, x2)
            self.bp.conditional_probabilities = {}
            for key in self.bp.pair_beliefs:
                # pair_prob = t.from_numpy(round_by_tol(t.exp(self.bp.pair_beliefs[key]), TOL))
                pair_prob = t.exp(self.bp.pair_beliefs[key])
                cond_prob = t.transpose(t.transpose(pair_prob, 0, 1) / self.bp.var_probabilities[key[0]], 0, 1)
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
            trans_mat[t.isnan(trans_mat)] = 0
            index = t.argmax(trans_mat).item()
            i_from, i_to = index // trans_mat.size()[0], index % trans_mat.size()[0]
            if break_ties == 'first':  # TESTING: need to be able to go always to the same destination (matrix VS torch)
                i_to = t.isclose(trans_mat, trans_mat[i_from, i_to]).nonzero().t()[1][0].item()
            elif break_ties == 'random':  # NORMALLY: we break ties randomly
                possibilities = t.isclose(trans_mat, trans_mat[i_from, i_to]).nonzero().t()[1]
                random_index = t.randint(0, list(possibilities.size())[0], (1,)).item()
                i_to = possibilities[random_index].item()
            self.aPos[i] = self.get_pos(i_to)
        self.time += 1

    def fast_move_next(self, break_ties='random'):
        trans_tensor = t.exp(self.bp.pair_belief_tensor)
        for i in range(len(self.aPos)):
            mes_key = (new_var('x', self.time, i + 1), new_var('x', self.time + 1, i + 1))
            mes_ind = self.mrf.message_index[mes_key]
            var_ind = self.mrf.var_index[new_var('x', self.time, i + 1)]
            trans_mat = trans_tensor[:, :, mes_ind]
            trans_mat = t.transpose(t.transpose(trans_mat, 0, 1) / t.exp(self.bp.belief_mat[:, var_ind]), 0, 1)
            trans_mat[t.isnan(trans_mat)] = 0
            index = t.argmax(trans_mat).item()
            i_from, i_to = index // trans_mat.size()[0], index % trans_mat.size()[0]
            if break_ties == 'first':  # TESTING: need to be able to go always to the same destination (matrix VS torch)
                i_to = t.isclose(trans_mat, trans_mat[i_from, i_to]).nonzero().t()[1][0].item()
            elif break_ties == 'random':  # NORMALLY: we break ties randomly
                possibilities = t.isclose(trans_mat, trans_mat[i_from, i_to]).nonzero().t()[1]
                random_index = t.randint(0, list(possibilities.size())[0], (1,)).item()
                i_to = possibilities[random_index].item()
            self.aPos[i] = self.get_pos(i_to)
        self.time += 1

    def run_game(self, verbose=True, break_ties='random'):
        """
        Run the inference to the horizon clamping the variables at every time step as decisions are taken
        :param verbose: Prints info about the agents final positions
        :param break_ties: Way in which ties are broken, either random or first
        :return: None
        """

        if not self.build:
            raise Exception("Model must be built before running the game")

        if self.build == 2:
            for i in range(self.horizon - 1):
                self.infer()
                self.fast_move_next(break_ties=break_ties)
                self._clamp_agents()
                if t.cuda.is_available():
                    t.cuda.empty_cache()
                    t.cuda.ipc_collect()
        else:
            for i in range(self.horizon - 1):
                self.infer()
                self.compute_probabilities()
                self.move_next(break_ties=break_ties)
                self.update_model()
        # Print trajectories and preys in final positions if verbose
        if verbose:
            for agent in range(1, len(self.aPos) + 1):
                trajectory = []
                for i in range(1, self.horizon + 1):
                    var_index = self.mrf.var_index[new_var('x', i, agent)]
                    position_index = t.argmax(self.mrf.unary_mat[:, var_index]).item()
                    trajectory.append(self.get_pos(position_index))
                if trajectory[-1] in self.hPos:
                    sth = 'hare'
                elif trajectory[-1] in self.sPos:
                    sth = 'stag'
                else:
                    sth = 'nothing'
                print("->".join([str(el) for el in trajectory]), sth)
