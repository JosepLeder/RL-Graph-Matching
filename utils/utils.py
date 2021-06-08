import random
import numpy as np
import torch
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from torch_geometric.data import Data


def topological_reconstruct(st, graph):
    n_nodes = graph.shape[0]
    map2 = np.zeros([n_nodes, n_nodes])
    l1, l2 = [], []
    al_nodes = [True] * n_nodes
    l1.append(st)
    al_nodes[st] = False
    while l1:
        l2.append(l1[0])
        for i in range(n_nodes):
            if graph[l1[0], i] and al_nodes[i]:
                l1.append(i)
                al_nodes[i] = False
        del l1[0]
    for i1, i2 in enumerate(l2):
        for j1, j2 in enumerate(l2):
            map2[i1, j1] = graph[i2, j2]
    return map2, l2


def plot_graph(graph):

    # 绘制母图
    g = nx.DiGraph()
    nodes = range(graph.shape[0])
    g.add_nodes_from(nodes)
    for i in nodes:
        for j in nodes:
            if graph[i, j] == 1:
                g.add_edge(i, j)
    position = nx.circular_layout(g)
    nx.draw_networkx_nodes(g, position, nodelist=nodes, node_color="r")
    nx.draw_networkx_edges(g, position)
    nx.draw_networkx_labels(g, position)
    plt.show()


def plot_digraph(digraph):
    nodes = range(len(digraph.nodes))
    position = nx.circular_layout(digraph)
    nx.draw_networkx_nodes(digraph, position, nodelist=nodes, node_color="r")
    nx.draw_networkx_edges(digraph, position)
    nx.draw_networkx_labels(digraph, position)
    plt.show()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


def update_linear_schedule(optimizer, current, total_steps, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (current / float(total_steps)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def state2data(state, device):
    state['x'] = th.tensor(state['x'], dtype=th.float, device=device)
    state['edge_index'] = th.tensor(state['edge_index'], dtype=torch.long, device=device)
    return Data(x=state['x'], edge_index=state['edge_index'])


def state2tensor(state, device):
    return th.tensor(state, dtype=th.float).unsqueeze(0).to(device)


# define transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Create a queue to store transitions and use it for experience replay
class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    graph = np.load("/home/josep/code/python/rlcode/graph_matching_project/graph_matching/data/source.npy")
    for i in range(graph.shape[0]):
        graph[i][i] = 1
        print(graph[i])
    plot_graph(graph)
    g, idx = topological_reconstruct(0, graph)
    plot_graph(g)
    for i in g:
        print(i)
