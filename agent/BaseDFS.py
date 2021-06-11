from __future__ import absolute_import

import torch
from copy import deepcopy
from utils.utils import save_dict


class BaseDeepFirstSearch(object):

    def __init__(self, env, threshold=int(5e3)):
        self.env = env
        self.device = torch.device("cpu")

        self.env.reset()
        self.subgraph = deepcopy(self.env.sub_graph)

        self.threshold = threshold
        self.steps = 0

        self.best_score = 0
        self.total_steps = 0

        # Save train data for visualizations
        self.x_steps = []
        self.y_rewards = []

    def dfs(self, nodes_selected):

        self.steps += 1
        self.total_steps += 1
        # Limit DFS steps
        if self.steps > self.threshold:
            return False
        if self.env.is_match(nodes_selected):
            ns = len(nodes_selected)
            if ns > self.best_score:
                self.x_steps.append(self.total_steps)
                self.y_rewards.append(ns)
                print(self.total_steps, ns)
                print(nodes_selected)
                self.best_score = ns
            if ns == self.subgraph.shape[0]:
                return True
            else:
                val_actions = self.env.get_valid_actions(nodes_selected)
                for a in val_actions:
                    nodes_selected.append(a)
                    if self.dfs(nodes_selected):
                        return True
                    else:
                        nodes_selected.pop(-1)
        else:
            return False

    def save_result(self):
        dct = {
            "x": self.x_steps,
            "y": self.y_rewards
        }
        save_dict("results/{}.json".format(self.__class__.__name__), dct)


