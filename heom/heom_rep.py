#%%
import math
import collections
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx

from . import q_func
#%%
class heom_state:
    """Construct a free-pole HEOM hierarchy and its sparse Liouvillian.

    ``C_list`` and ``gamma_list`` define
    ``C(t) = sum_k C_k exp(-gamma_k t)``.  A real pole uses one ``n_k``
    hierarchy coordinate.  A complex pole uses separate ``n_k`` and ``m_k``
    coordinates for the correlation and its conjugate.  Thus a complex pole
    need not be accompanied by a zero-coefficient conjugate pole in the input.
    """
    def __init__(
        self,
        K,
        L,
        H_s=None,
        H_c=None,
        C_list=None,
        gamma_list=None,
    ):
        if not isinstance(K, (int, np.integer)) or K < 0:
            raise ValueError("K must be a non-negative integer")
        if not isinstance(L, (int, np.integer)) or L < 0:
            raise ValueError("L must be a non-negative integer")

        if H_s is None:
            H_s = np.zeros((2, 2), dtype=np.complex128)
        else:
            H_s = np.asarray(H_s)
        if H_c is None:
            H_c = np.zeros_like(H_s)
        else:
            H_c = np.asarray(H_c)

        if H_s.ndim != 2 or H_s.shape[0] != H_s.shape[1]:
            raise ValueError("H_s must be a square matrix")
        if H_c.shape != H_s.shape:
            raise ValueError("H_c must have the same shape as H_s")

        if C_list is not None:
            C_list = np.asarray(C_list, dtype=np.complex128)
            if C_list.ndim != 1 or C_list.size != K + 1:
                raise ValueError("C_list must contain exactly K + 1 coefficients")
        if gamma_list is not None:
            gamma_list = np.asarray(gamma_list, dtype=np.complex128)
            if gamma_list.ndim != 1 or gamma_list.size != K + 1:
                raise ValueError("gamma_list must contain exactly K + 1 coefficients")

        self.K = K #truncation of Matsubara freq
        self.L = L #truncation of memory level
        self.H_s = H_s #system Hamiltonian
        self.H_c = H_c #coupling Hamiltonian
        self.C_list = C_list # coefficient C list
        self.gamma_list = gamma_list # coefficient gamma list
        if gamma_list is None:
            self.complex_modes = ()
        else:
            self.complex_modes = tuple(
                np.flatnonzero(~np.isclose(gamma_list.imag, 0.0, atol=1e-12))
            )
        self.has_complex_poles = bool(self.complex_modes)
        self.hierarchy_modes = (
            tuple(("n", k) for k in range(K + 1))
            + tuple(("m", k) for k in self.complex_modes)
        )
        self.nADO = math.comb(L + len(self.hierarchy_modes), L)
        self.system_size = H_s.shape[0]**2
        self.build_hierarchy() 

    def _make_node(self, n_vec, m_vec=None):
        n_vec = tuple(n_vec)
        if not self.has_complex_poles:
            return n_vec
        if m_vec is None:
            m_vec = (0,) * (self.K + 1)
        return n_vec, tuple(m_vec)

    def _split_node(self, node):
        if self.has_complex_poles:
            return node
        return node, (0,) * (self.K + 1)

    def _shift_node(self, node, branch, k, amount):
        n_vec, m_vec = map(list, self._split_node(node))
        target = n_vec if branch == "n" else m_vec
        target[k] += amount
        return self._make_node(n_vec, m_vec)

    def _occupation(self, node, branch, k):
        n_vec, m_vec = self._split_node(node)
        return (n_vec if branch == "n" else m_vec)[k]

    def _tier(self, node):
        n_vec, m_vec = self._split_node(node)
        return sum(n_vec) + sum(m_vec)

    def Gamma(self, node):
        #compute the coefficient related to decay rate (loop edge)
        if self.gamma_list is None:
            raise ValueError("gamma_list is required to construct the Liouvillian")
        n_vec, m_vec = self._split_node(node)
        return (
            np.dot(self.gamma_list, n_vec)
            + np.dot(np.conj(self.gamma_list), m_vec)
        )
    def couple_diag(self, node, normalized=False):
        # Construct the superoperator that acts on one ADO.
        # Normalization does not change a block whose source and target ADO
        # are the same, but accepting the option keeps the block API uniform.
        self._validate_normalized(normalized)
        L0 = -1j*(q_func.commutator_sup_op(self.H_s))
        # L0 is expected to be a scipy.sparse matrix
        L = L0 - self.Gamma(node) * sp.identity(L0.shape[0], dtype=L0.dtype)
        return L

    def _normalization_step(self, node, branch, k):
        r'''Return the single-coordinate factor in the ADO normalization.

        For an ``n`` coordinate this is ``sqrt(n_k C_k)`` and for an ``m``
        coordinate it is ``sqrt(m_k C_k^*)``.  Products of these step factors
        give

        ``sqrt(prod_k(n_k! m_k! C_k**n_k (C_k^*)**m_k))``.
        '''
        coefficient = (
            np.conj(self.C_list[k]) if branch == "m" else self.C_list[k]
        )
        return np.sqrt(self._occupation(node, branch, k) * coefficient)

    def _validate_normalized(self, normalized):
        if not isinstance(normalized, (bool, np.bool_)):
            raise TypeError("normalized must be a boolean")
        if normalized and self.C_list is None:
            raise ValueError(
                "C_list is required to construct normalized coupling blocks"
            )
        if normalized and np.any(self.C_list == 0):
            raise ValueError(
                "Normalized ADOs require every coefficient in C_list to be "
                "non-zero."
            )

    def couple_up(self, node=None, branch=None, k=None, normalized=False):
        '''
        Construct the block coupling an ADO to either an n or m successor.

        This corresponds to ``-i[H_c, rho_successor]``.  For normalized ADOs
        it is multiplied by ``sqrt((occupation + 1) C_k)`` on an n branch or
        by its conjugate-coefficient counterpart on an m branch.
        '''
        self._validate_normalized(normalized)
        coupling = -1j*q_func.commutator_sup_op(self.H_c)
        if not normalized:
            return coupling
        if node is None or branch is None or k is None:
            raise ValueError(
                "node, branch, and k are required for a normalized "
                "up-coupling"
            )
        successor = self._shift_node(node, branch, k, 1)
        return self._normalization_step(successor, branch, k) * coupling

    def _downward_core(self, node, branch, k):
        occupation = self._occupation(node, branch, k)
        if branch == "m":
            return (
                -occupation
                * np.conj(self.C_list[k])
                * q_func.right_sup_op(self.H_c)
            )

        core = occupation * self.C_list[k] * q_func.left_sup_op(self.H_c)
        if k not in self.complex_modes:
            core -= (
                occupation
                * np.conj(self.C_list[k])
                * q_func.right_sup_op(self.H_c)
            )
        return core

    def couple_down(self, node, branch, k, normalized=False):
        '''
        Construct a lowering block for an n or m hierarchy coordinate.

        Complex poles use separate terms ``-i n_k C_k H_c^L`` and
        ``+i m_k C_k^* H_c^R``.  A real pole has no m coordinate, so both
        actions are combined in its n lowering block.  With normalized ADOs,
        the unnormalized block is divided by the corresponding normalization
        step.  On separate n/m branches this gives the requested square-root
        lowering coefficients exactly.
        '''
        self._validate_normalized(normalized)
        core = self._downward_core(node, branch, k)
        if normalized:
            normalization_step = self._normalization_step(node, branch, k)
            if normalization_step == 0:
                raise ValueError(
                    "A normalized down-coupling requires a non-zero "
                    "occupation"
                )
            core = core / normalization_step
        return -1j * core

    def couple_markovian(
        self,
        node,
        branch,
        k,
        normalized=False,
        target_branch=None,
        target_k=None,
    ):
        r'''
        Construct one Markovian-terminator coupling through a virtual ADO.

        ``node`` labels a virtual ADO M one tier beyond the hierarchy cutoff,
        and ``branch, k`` select its predecessor.  Adiabatically eliminating
        M gives the superoperator

            -H_c^x K_M^{-1} A_{M, branch, k},

        where ``A`` is the lowering term without its factor ``-i`` and
        K_M = i H_s^x + Gamma(M) I.  The linear system is solved directly.
        '''
        self._validate_normalized(normalized)
        if self._occupation(node, branch, k) == 0:
            return sp.csr_array(
                (self.system_size, self.system_size),
                dtype=np.complex128,
            )

        identity = sp.identity(
            self.system_size,
            dtype=np.complex128,
            format="csr",
        )
        k_superoperator = (
            1j * q_func.commutator_sup_op(self.H_s)
            + self.Gamma(node) * identity
        ).toarray()
        source_coupling = self._downward_core(node, branch, k).toarray()

        try:
            eliminated_coupling = np.linalg.solve(
                k_superoperator,
                source_coupling,
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "The Markovian terminator requires an invertible K_M, but "
                f"K_M is singular for virtual ADO {node}."
            ) from error

        correction = (
            -q_func.commutator_sup_op(self.H_c) @ eliminated_coupling
        )
        if normalized:
            if target_branch is None or target_k is None:
                raise ValueError(
                    "target_branch and target_k are required for a "
                    "normalized Markovian coupling"
                )
            correction = correction * (
                self._normalization_step(node, target_branch, target_k)
                / self._normalization_step(node, branch, k)
            )
        return sp.csr_array(correction)
    
    def build_initial_state(self, rho0, as_sparse=True):
        """
        Constructs the initial state vector for the HEOM hierarchy.

        Assumes an initial product state between the system and a thermal bath,
        so only the all-zero primary ADO is non-zero at t=0.
        """
        rho0 = np.asarray(rho0, dtype=np.complex128)
        if rho0.shape != self.H_s.shape:
            raise ValueError(
                f"Initial density matrix rho0 has shape {rho0.shape}, "
                f"but expected shape {self.H_s.shape}."
            )
        total_size = self.nADO * self.system_size
        initial_heom_vec = np.zeros(total_size, dtype=np.complex128)

        # Vectorize the initial system density matrix using column-stacking.
        initial_heom_vec[:self.system_size] = rho0.reshape(-1, order="F")
        if as_sparse:
            return sp.csr_array(initial_heom_vec.reshape(-1, 1))
        return initial_heom_vec
    def build_Liouvillian(
        self,
        markovian_terminator=False,
        normalized=False,
    ):
        '''
        build the full Liouvillian for HEOM equation.
        The state vector is a column vector where all ADOs are stacked,
        i.e., [rho_0, rho_1, ...]^T.
        The Liouvillian L is a block matrix where L_ij is the operator
        that maps rho_j to a term in d/dt rho_i.

        If ``markovian_terminator`` is True, ADOs one tier beyond the cutoff
        are assumed stationary and adiabatically eliminated.  Their resulting
        correction is added between ADOs in the retained boundary tier.

        If ``normalized`` is True, construct the equation for ADOs divided by
        ``sqrt(prod_k(n_k! m_k! C_k**n_k (C_k^*)**m_k))``.  The primary ADO
        is unchanged, so physical reduced-density-matrix observables are the
        same as in the unnormalized hierarchy.  For complex coefficients the
        square root is interpreted consistently, one hierarchy step at a
        time; this is a complex similarity scaling rather than a positive
        norm.
        '''
        if (
            self.C_list is None
            or self.gamma_list is None
        ):
            raise ValueError(
                "C_list and gamma_list are required to construct "
                "the Liouvillian"
            )
        if not isinstance(markovian_terminator, (bool, np.bool_)):
            raise TypeError("markovian_terminator must be a boolean")
        self._validate_normalized(normalized)

        total_size = self.nADO * self.system_size
        liouvillian = sp.lil_array(
            (total_size, total_size),
            dtype=np.complex128,
        )
        markovian_couplings = {}

        for i in range(self.nADO):
            node_i = self.idx_to_node[i]
            start_i = i * self.system_size
            end_i = (i + 1) * self.system_size

            # Diagonal block: (-i[H_s, .] - Gamma(node_i)) rho_i
            liouvillian[start_i:end_i, start_i:end_i] = self.couple_diag(
                node_i,
                normalized=normalized,
            )

            # Up-coupling: contribution from rho_j to d/dt rho_i where j is a successor of i.
            # Both n and m successors enter as -i[H_c, rho_successor].
            for branch, k_mode in self.hierarchy_modes:
                successor = self._shift_node(node_i, branch, k_mode, 1)
                j_up = self.node_to_idx.get(successor)
                if j_up is None:
                    continue
                start_j_up = j_up * self.system_size
                end_j_up = (j_up + 1) * self.system_size
                liouvillian[start_i:end_i, start_j_up:end_j_up] = (
                    self.couple_up(
                        node_i,
                        branch,
                        k_mode,
                        normalized=normalized,
                    )
                )

            # Down-coupling: contribution from rho_j to d/dt rho_i where j is a predecessor of i.
            # n and m predecessors carry their respective left/right actions.
            for j_down, branch, k_mode in self.edge_down.get(i, []):
                start_j_down = j_down * self.system_size
                end_j_down = (j_down + 1) * self.system_size
                liouvillian[start_i:end_i, start_j_down:end_j_down] = self.couple_down(
                    node_i,
                    branch,
                    k_mode,
                    normalized=normalized,
                )

            # Markovian terminator: for each virtual successor M of node_i,
            # eliminate M and couple every retained predecessor M-e_k into
            # d/dt rho_i.  Several virtual paths can contribute to the same
            # block, so these terms must be accumulated rather than assigned.
            if markovian_terminator and self._tier(node_i) == self.L:
                for j_branch, j_mode in self.hierarchy_modes:
                    virtual_node = self._shift_node(
                        node_i, j_branch, j_mode, 1
                    )

                    for k_branch, k_mode in self.hierarchy_modes:
                        occupation = self._occupation(
                            virtual_node, k_branch, k_mode
                        )
                        if occupation == 0:
                            continue

                        source_node = self._shift_node(
                            virtual_node, k_branch, k_mode, -1
                        )
                        source_idx = self.node_to_idx[source_node]
                        start_source = source_idx * self.system_size
                        end_source = (source_idx + 1) * self.system_size

                        current_block = liouvillian[
                            start_i:end_i,
                            start_source:end_source,
                        ]
                        coupling_key = (
                            virtual_node,
                            j_branch,
                            j_mode,
                            k_branch,
                            k_mode,
                            normalized,
                        )
                        if coupling_key not in markovian_couplings:
                            markovian_couplings[coupling_key] = (
                                self.couple_markovian(
                                    virtual_node,
                                    k_branch,
                                    k_mode,
                                    normalized=normalized,
                                    target_branch=j_branch,
                                    target_k=j_mode,
                                )
                            )
                        liouvillian[
                            start_i:end_i,
                            start_source:end_source,
                        ] = current_block + markovian_couplings[coupling_key]

        liouvillian = liouvillian.tocsr()
        # Record how the current cached operator was assembled.  Downstream
        # solvers can then reject a same-shaped Liouvillian with incompatible
        # normalization or boundary termination.
        self.liouvillian = liouvillian
        self.liouvillian_options = {
            "markovian_terminator": bool(markovian_terminator),
            "normalized": bool(normalized),
        }
        return liouvillian
    def build_hierarchy(self):
        '''
        build hierarchy structure and index all ADOs
        '''
        node_to_idx = {} #hash map for converting ADO indices to node indices
        idx_to_node = [] #list for converting node indices to ADO indices
        edge_source = []; edge_target = []
        edge_up = {}; edge_down = {}
        zero_vec = (0,) * (self.K + 1)
        root = self._make_node(zero_vec, zero_vec)
        node_to_idx[root] = 0; idx_to_node.append(root)
        queue = collections.deque([root])
        #BFS algo
        while queue:
            node = queue.popleft()
            cur_idx = node_to_idx[node]
            tier = self._tier(node)
            if tier < self.L:
                for branch, k in self.hierarchy_modes:
                    next_node = self._shift_node(node, branch, k, 1)
                    if next_node not in node_to_idx:
                        queue.append(next_node)
                        next_idx = len(idx_to_node)
                        node_to_idx[next_node] = next_idx
                        idx_to_node.append(next_node)
                    else: 
                        next_idx = node_to_idx[next_node]
                    edge_source.append(cur_idx)
                    edge_target.append(next_idx)
                    edge_up[cur_idx] = edge_up.get(cur_idx, []) + [next_idx]
                    edge_down[next_idx] = edge_down.get(next_idx, []) + [
                        (cur_idx, branch, k)
                    ]
        self.edge_up = edge_up      # edges connecting ADOs to successors
        self.edge_down = edge_down  # edges connecting ADOs to predecessors
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.edge_index = np.array([edge_source, edge_target], dtype=np.int64) #for visualizing
    def visualize_state(self):
        # A directed graph is more appropriate for HEOM hierarchy.
        G = nx.Graph()
        G.add_nodes_from(range(self.nADO))
        G.add_edges_from(self.edge_index.T.tolist())

        # Display ADO labels instead of integer node indices.
        node_labels = {i: str(self.idx_to_node[i]) for i in range(self.nADO)}

        # Create a hierarchical layout based on the "tier" of the ADO.
        pos = {}
        nodes_by_tier = collections.defaultdict(list)
        for node_idx in G.nodes():
            tier = self._tier(self.idx_to_node[node_idx])
            nodes_by_tier[tier].append(node_idx)

        for tier, nodes in sorted(nodes_by_tier.items()):
            num_nodes_in_tier = len(nodes)
            # Space nodes horizontally within a tier
            x_positions = [i - (num_nodes_in_tier - 1) / 2.0 for i in range(num_nodes_in_tier)]
            for i, node_idx in enumerate(sorted(nodes)): # Sort for consistent ordering.
                pos[node_idx] = (x_positions[i], -tier)

        nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=500, node_color="skyblue", font_size=8, font_color="black")
        
        # Add tier labels (e.g., "l = 0", "l = 1") to the left of each level.
        if pos: # Check if the graph is not empty to avoid errors.
            # Find the minimum x-coordinate to position labels to the left of the graph.
            min_x = min(p[0] for p in pos.values())
            for tier in nodes_by_tier.keys():
                plt.text(min_x - 0.5, -tier, r'$l='+str(tier)+'$', fontsize=12, horizontalalignment='right', verticalalignment='center')

        plt.title(r"$K=$"+str(self.K)+", $L=$"+str(self.L))
        plt.show()
#%%
if __name__ == "__main__":
    # Define some dummy parameters for testing
    K_test = 1
    L_test = 2
    H_s_test = np.array([[1, 0.5], [0.5, -1]], dtype=np.complex128)
    H_c_test = np.array([[1, 0], [0, -1]], dtype=np.complex128) # e.g., sigma_z
    C_list_test = np.array([0.1, 0.2], dtype=np.complex128)
    gamma_list_test = np.array([0.05, 0.05], dtype=np.complex128)

    # Create the heom_state instance
    test_state = heom_state(
        K=K_test, L=L_test, H_s=H_s_test, H_c=H_c_test,
        C_list=C_list_test, gamma_list=gamma_list_test
    )
    
    print(f"Number of ADOs: {test_state.nADO}")
    print(f"System size (vectorized): {test_state.system_size}")
    
    # Build the Liouvillian
    L_heom = test_state.build_Liouvillian()
    print(f"Liouvillian shape: {L_heom.shape}")

    # Build the initial state, assuming system starts in |0> state
    rho0_test = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    initial_vec_sparse = test_state.build_initial_state(rho0_test)
    
    print(f"Initial sparse state vector shape: {initial_vec_sparse.shape}")
    print(f"Number of non-zero elements: {initial_vec_sparse.nnz}")
    
    # For inspection or use with solvers like solve_ivp, convert to dense
    initial_vec_dense = initial_vec_sparse.toarray().flatten()
    print("Initial dense vector (first 4 elements):", initial_vec_dense[:4])
    
    # Check that only the primary ADO part is non-zero
    assert np.all(initial_vec_dense[test_state.system_size:] == 0)
    print("\nCheck passed: Only the primary ADO is non-zero in the initial vector.")

    # Visualize the hierarchy
    test_state.visualize_state()



# %%
