import numpy as np


class GraphMatchingEnv(object):

    def __init__(self) -> None:
        self.graph = np.load("./source.npy")  # 母图, 邻接矩阵[可达为1, 不可达为0]
        self.num_nodes = self.graph.shape[0]
        self.orgin_graph = self.graph.copy()  # 保存母图的复制，在reset和匹配子图时使用
        self.sub_graph = None  # 子图, 邻接矩阵
        self.nodes_set = []
        self.steps = 0
        self.terminated = False

    def reset(self):
        self.nodes_set = []
        self.steps = 0
        self.terminated = False
        self.graph = self.orgin_graph.copy()
        num = np.random.randint(3, 31)
        sub_graph_nodes = np.random.choice(np.arange(self.num_nodes), size=num, replace=False)
        self.sub_graph = np.zeros([num, num])  # 随机生成一个可以匹配的新子图,节点数: [3, 30]

        for i_1, i_2 in enumerate(sub_graph_nodes):
            for j_1, j_2 in enumerate(sub_graph_nodes):
                self.sub_graph[i_1, j_1] = self.graph[i_2, j_2]
        state = {"graph": self.graph, "sub_graph": self.sub_graph}
        return state

    def step(self, action):
        """
            action: int, 表示第几个点, [0, self.num_nodes - 1]
        """
        self.steps += 1
        if self.steps == self.sub_graph.shape[0]:
            self.terminated = True
        self.nodes_set.append(action)
        # TODO: 选择了一个点后, 将该点的向量全置为0, 但其他点到该点的路径不变
        # state包含了self.graph和self.sub_graph
        for i in range(self.num_nodes):
            self.graph[action, i] = 0
        next_state = {"graph": self.graph, "sub_graph": self.sub_graph}
        reward = self.get_simple_reward()
        return next_state, reward

    def is_match(self) -> bool:
        # TODO: 判断当前图是否匹配
        return True

    def is_terminated(self) -> bool:
        return self.terminated

    def get_simple_reward(self):
        # TODO: 如果图相匹配则返回1, 否则0
        if self.is_match():
            return 1
        else:
            return 0

    def get_valid_actions(self):
        # TODO: 不能选重复的点, self.nodes_set中储存之前选过的点, actions是长为self.num_nodes的数组, 如果该点可以选择则为1,否则为0
        actions = [1]*self.num_nodes
        for i in self.nodes_set:
            actions[i] = 0
        return actions

    def get_random_action(self):
        # TODO: 直接返回一个可用的动作
        actions = self.get_valid_actions()
        action_step = []
        for i in range(self.num_nodes):
            if actions[i] == 1:
                action_step.append(i)
        action = action_step[np.random.randint(0, len(action_step))]
        return action
