import torch as th
import numpy as np
import networkx as nx

from torch import Tensor
from networkx.algorithms import isomorphism
from utils.utils import topological_reconstruct, plot_graph, plot_digraph


class GraphMatchingEnv(object):

    def __init__(self, args) -> None:
        self.args = args

        self.graph = np.load("/home/josep/code/python/rlcode/graph_matching_project/graph_matching/data/target.npy")  # 母图, 邻接矩阵[可达为1, 不可达为0]
        self.origin_graph = self.graph.copy()  # 保存母图的复制，在reset和匹配子图时使用
        self.sub_graph = None  # 子图, 邻接矩阵
        self.edge_index = []  # shape:[2, num_edges]
        for i in range(self.graph.shape[0]):
            idx = np.where(self.graph[i] > 0)[0]
            idx = np.array([np.ones_like(idx) * i, idx])
            self.edge_index.append(idx)

        self.num_graph_nodes = self.graph.shape[0]
        self.num_subgraph_nodes = 0
        self.sub_graph_nodes = None
        self.nodes_selected = None
        self.steps = 0
        self.terminated = False

    def reset(self):
        self.nodes_selected = []
        self.steps = 0
        self.terminated = False
        self.graph = self.origin_graph.copy()
        self.num_subgraph_nodes = 40

        if (not self.args.test) or self.args.random_subgraph:
            # generate a random subgraph, number of nodes: 40
            self.sub_graph_nodes = [np.random.randint(0, self.num_graph_nodes)]
            al_sub_graph_nodes = [True] * self.num_graph_nodes
            al_nodes = []
            al_sub_graph_nodes[self.sub_graph_nodes[0]] = False
            for i in range(self.num_subgraph_nodes - 1):
                for j in range(self.num_graph_nodes):
                    if self.graph[j, self.sub_graph_nodes[-1]] and al_sub_graph_nodes[j]:
                        al_nodes.append(j)
                        al_sub_graph_nodes[j] = False
                al_num = np.random.randint(0, len(al_nodes))
                self.sub_graph_nodes.append(al_nodes[al_num])
                del al_nodes[al_num]
            self.sub_graph = np.zeros([self.num_subgraph_nodes, self.num_subgraph_nodes])

            for i_1, i_2 in enumerate(self.sub_graph_nodes):
                for j_1, j_2 in enumerate(self.sub_graph_nodes):
                    self.sub_graph[i_1, j_1] = self.graph[i_2, j_2]
        else:
            self.sub_graph = np.load("/home/josep/code/python/rlcode/graph_matching_project/graph_matching/data/source.npy")
            # for i in range(self.sub_graph.shape[0]):
            #     self.sub_graph[i][i] = 1

        # find the most connected node
        connectivity = np.sum(self.sub_graph, axis=1)
        node_st = np.argmax(connectivity)
        # reconstruct the graph
        self.sub_graph, idx = topological_reconstruct(node_st, self.sub_graph)
        if (not self.args.test) or self.args.random_subgraph:
            self.sub_graph_nodes = np.array(self.sub_graph_nodes)[idx]
        else:
            self.sub_graph_nodes = None

        val_actions = self.get_valid_actions(self.nodes_selected)
        node_feats = self.get_node_features(val_actions)
        edge_indexes = self.get_edge_indexes(val_actions)
        state = {"x": node_feats, "edge_index": edge_indexes, "valid_actions": val_actions, "subgraph": self.sub_graph}
        return state

    def is_match(self, nodes_select) -> bool:
        # if not self.is_terminated():
        #     return False
        g1 = nx.DiGraph()  # g1为子图
        g2 = nx.DiGraph()  # g2为母图生成的子图
        # judge all nodes
        # nodes = range(self.sub_graph.shape[0])
        # only judge selected nodes
        nodes = range(len(nodes_select))
        g1.add_nodes_from(nodes)
        g2.add_nodes_from(nodes)
        for i in nodes:
            for j in nodes:
                if self.sub_graph[i, j] == 1:
                    g1.add_edge(i, j)
                if self.origin_graph[nodes_select[i], nodes_select[j]] == 1:
                    g2.add_edge(i, j)
        DiGM = isomorphism.DiGraphMatcher(g1, g2)
        if DiGM.is_isomorphic():
            return True
        else:
            # plot_digraph(g1)
            # plot_digraph(g2)
            return False

    def is_terminated(self) -> bool:
        return self.terminated

    def get_simple_reward(self) -> Tensor:
        if self.is_match(self.nodes_selected):
            return th.tensor(10)
        else:
            return th.tensor(0)

    def get_valid_actions(self, nodes_select):
        """
            return: [0, 1, 25, ..., 39]
        """
        if len(nodes_select) == 0:
            return [i for i in range(self.num_graph_nodes)]
        actions = set()
        for i in nodes_select:
            idx = self.graph[i].copy()
            idx[i] = 0
            a = np.where(idx > 0)[0]
            for j in a:
                actions.add(j)
        for i in nodes_select:
            if i in actions:
                actions.remove(i)
        return np.array(list(actions))

    def get_random_action(self):
        actions = self.get_valid_actions(self.nodes_selected)
        action_step = []
        for i in range(self.num_graph_nodes):
            if actions[i] == 1:
                action_step.append(i)
        action = action_step[np.random.randint(0, len(action_step))]
        return action

    def get_correct_action(self):
        if self.sub_graph_nodes is None:
            print("No correct actions for validation set.")
            exit(-1)
        return self.sub_graph_nodes[self.steps]

    def get_edge_indexes(self, actions):
        idx = self.edge_index[0]
        for i in range(1, self.num_graph_nodes):
            idx = np.c_[idx, self.edge_index[i]]
        return idx

    def get_node_features(self, actions):
        # feats = self.graph[actions]
        # point = self.sub_graph[self.nodes_sorted[self.steps]]
        # point = np.concatenate([point, np.zeros(40 - len(point))])
        # repeat_points = np.tile(point, (len(actions), 1))
        # feats = np.c_[feats, repeat_points]
        feats = self.graph
        point = self.sub_graph[self.steps]
        point = np.concatenate([point, np.zeros(40 - len(point))])
        repeat_points = np.tile(point, (self.num_graph_nodes, 1))
        feats = np.c_[feats, repeat_points]
        return feats

    def step(self, action):
        """
            action: int, 表示第几个点, [0, self.num_nodes - 1]
        """

        self.steps += 1
        self.nodes_selected.append(action)

        if self.steps == self.sub_graph.shape[0]:
            self.terminated = True
            return None, self.get_simple_reward(), self.is_terminated(), None

        val_actions = self.get_valid_actions(self.nodes_selected)
        node_feats = self.get_node_features(val_actions)
        edge_indexes = self.get_edge_indexes(val_actions)
        next_state = {"x": node_feats, "edge_index": edge_indexes, "valid_actions": val_actions,
                      "subgraph": self.sub_graph}

        reward = self.get_simple_reward()
        # self.graph[action] = [0] * self.num_graph_nodes
        return next_state, reward, self.is_terminated(), val_actions


if __name__ == '__main__':
    GraphMatchingEnv()
