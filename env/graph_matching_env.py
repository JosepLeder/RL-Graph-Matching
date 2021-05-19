import torch as th
import numpy as np


class GraphMatchingEnv(object):

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.graph = None # 母图, 邻接矩阵[可达为1, 不可达为0]
        self.sub_graph = None # 子图, 邻接矩阵
        self.nodes_set = None
        self.step = 0
        self.terminated = False

        

    def reset(self):
        self.sub_graph = None # 随机生成一个可以匹配的新子图,节点数: [3, 30]
        self.nodes_set = None
        self.step = 0
        self.is_terminated = False

        return state

    def step(self, action):
        """
            action: int, 表示第几个点, [0, self.num_nodes - 1]
        """

        self.setp += 1

        # TODO: 选择了一个点后, 将该点的向量全置为0, 但其他点到该点的路径不变
        # state包含了self.graph和self.sub_graph


        return next_state, reward

    def is_match(self) -> bool:
        #TODO: 判断当前图是否匹配
        return False

    def is_terminated(self) -> bool:
        if self.is_match():
            return True
        if self.step >= self.num_nodes

        return False

    def get_simple_reward(self):
        
        # TODO: 如果图相匹配则返回1, 否则0
        return reward

    def get_valid_actions(self):
        # TODO: 不能选重复的点, self.nodes_set中储存之前选过的点, actions是长为self.num_nodes的数组, 如果该点可以选择则为1,否则为0

        return actions

    def get_random_action(self):
        # TODO: 直接返回一个可用的动作
        return action
