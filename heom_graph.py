#%%
import torch
#from torch_geometric.nn import MessagePassing
import math
import collections
import networkx as nx
import matplotlib.pyplot as plt
#%%
class heom_spin_state:
    '''
    full HEOM state for a spin boson system encoded as a graph
    '''
    def __init__(self, K, L):
        self.K = K #truncation of Matsubara freq
        self.L = L #truncation of memory level
        self.nADO = math.comb(L+K+1,L)#tot number of nADO
        self.nodes = torch.zeros([self.nADO,8])
        self.build_graph() 
    def build_graph(self):
        node_to_idx = {} #hash map for converting ADO vector n to index in the node tensor
        idx_to_node = [] #list for converting node tensor index to ADO vector n
        edge_source = []; edge_target = []
        #initialize n vec
        n_vec = tuple([0]*(self.K+1))
        node_to_idx[n_vec] = 0; idx_to_node.append(n_vec)
        queue = collections.deque([n_vec])
        #BFS algo
        while queue:
            n_vec = queue.popleft()
            cur_idx = node_to_idx[n_vec]
            tier = sum(n_vec)
            if tier < self.L:
                for k in range(self.K+1):
                    #implement n^+ 
                    n_next = list(n_vec)
                    n_next[k] += 1
                    n_next = tuple(n_next)
                    if n_next not in queue:
                        queue.append(n_next)
                        next_idx = len(idx_to_node)
                        node_to_idx[n_next] = next_idx
                        idx_to_node.append(n_next)
                    else: 
                        next_idx = node_to_idx[n_next]
                    edge_source.append(cur_idx)
                    edge_target.append(next_idx)
        self.node_to_indx = node_to_idx
        self.idx_to_node = idx_to_node
        self.edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
    def visualize_state(self):
        # A directed graph is more appropriate for HEOM hierarchy.
        G = nx.Graph()
        G.add_nodes_from(range(self.nADO))
        G.add_edges_from(self.edge_index.T.tolist())

        # Create labels from n_vec to display on nodes, instead of node indices.
        node_labels = {i: str(self.idx_to_node[i]) for i in range(self.nADO)}

        # Create a hierarchical layout based on the "tier" of the ADO.
        pos = {}
        nodes_by_tier = collections.defaultdict(list)
        for node_idx in G.nodes():
            tier = sum(self.idx_to_node[node_idx])
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
test_state = heom_spin_state(2,3)
#%%
print(test_state.nADO)
print(test_state.node_to_indx)
print(test_state.idx_to_node)
print(test_state.edge_index)
test_state.visualize_state()



# %%
