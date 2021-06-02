import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism


class GraphMatchingEnv(object):

    def __init__(self) -> None:
        self.graph = np.load("../data/source.npy")  # 母图, 邻接矩阵[可达为1, 不可达为0]
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
        self.nodes_sorted = None
        self.nodes_selected = None
        self.steps = 0
        self.terminated = False

    def reset(self):
        self.nodes_selected = []
        self.steps = 0
        self.terminated = False
        self.graph = self.origin_graph.copy()

        # generate a random subgraph, number of nodes: [3, 30]
        # self.num_subgraph_nodes = np.random.randint(3, 31)
        self.num_subgraph_nodes = 5
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

        # sort subgraph nodes by connectivity
        connectivity = np.sum(self.sub_graph, axis=1)
        self.nodes_sorted = np.argsort(connectivity)

        val_actions = self.get_valid_actions()
        node_feats = self.get_node_features(val_actions)
        edge_indexes = self.get_edge_indexes(val_actions)
        state = {"x": node_feats, "edge_index": edge_indexes, "valid_actions": val_actions}
        return state

    def is_match(self) -> bool:
        # if not self.is_terminated():
        #     return False
        g1 = nx.DiGraph()  # g1为子图
        g2 = nx.DiGraph()  # g2为母图生成的子图
        # judge all nodes
        # nodes = range(self.sub_graph.shape[0])
        # only judge selected nodes
        nodes = range(len(self.nodes_selected))
        g1.add_nodes_from(nodes)
        g2.add_nodes_from(nodes)
        for i in nodes:
            for j in nodes:
                if self.sub_graph[self.nodes_sorted[i], self.nodes_sorted[j]] == 1:
                    g1.add_edge(i, j)
                if self.origin_graph[self.nodes_selected[i], self.nodes_selected[j]] == 1:
                    g2.add_edge(i, j)
        DiGM = isomorphism.DiGraphMatcher(g1, g2)
        if DiGM.is_isomorphic():
            return True
        else:
            return False

    def is_terminated(self) -> bool:
        return self.terminated

    def get_simple_reward(self) -> float:
        if self.is_match():
            return 1
        else:
            return 0

    def get_valid_actions(self):
        """
            return: [0, 1, 25, ..., 39]
        """
        if len(self.nodes_selected) == 0:
            return [i for i in range(self.num_graph_nodes)]
        actions = []
        for i in self.nodes_selected:
            idx = self.graph[i].copy()
            idx[i] = 0
            actions.extend(np.where(idx > 0)[0])
        actions = np.unique(actions)

        return actions

    def get_random_action(self):
        actions = self.get_valid_actions()
        action_step = []
        for i in range(self.num_graph_nodes):
            if actions[i] == 1:
                action_step.append(i)
        action = action_step[np.random.randint(0, len(action_step))]
        return action

    def get_correct_action(self):
        return self.sub_graph_nodes[self.nodes_sorted[self.steps]]

    def get_edge_indexes(self, actions):
        idx = self.edge_index[0]
        for i in range(1, self.num_graph_nodes):
            idx = np.c_[idx, self.edge_index[i]]
        return idx

    def get_node_features(self, actions):
        # feats = self.graph[actions]
        # point = self.sub_graph[self.nodes_sorted[self.steps]]
        # point = np.concatenate([point, np.zeros(32 - len(point))])
        # repeat_points = np.tile(point, (len(actions), 1))
        # feats = np.c_[feats, repeat_points]
        feats = self.graph
        point = self.sub_graph[self.nodes_sorted[self.steps]]
        point = np.concatenate([point, np.zeros(32 - len(point))])
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

        val_actions = self.get_valid_actions()
        node_feats = self.get_node_features(val_actions)
        edge_indexes = self.get_edge_indexes(val_actions)
        next_state = {"x": node_feats, "edge_index": edge_indexes, "valid_actions": val_actions}

        reward = self.get_simple_reward() * len(self.nodes_selected) / self.num_subgraph_nodes

        self.graph[action] = [0] * self.num_graph_nodes
        return next_state, reward, self.is_terminated(), val_actions


if __name__ == '__main__':
    GraphMatchingEnv()
