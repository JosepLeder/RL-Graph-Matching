from __future__ import absolute_import

import torch
import numpy as np

from agent.DDQN import DDQN
from agent.BaseDFS import BaseDeepFirstSearch
from copy import deepcopy
from utils.utils import state2data


class DDQN_DeepFirstSearch(BaseDeepFirstSearch):

    def __init__(self, args, env, threshold=int(5e3)):
        super(DDQN_DeepFirstSearch, self).__init__(env, threshold)
        self.ddqn = DDQN(args)

        # Calculate Q values
        self.q_values = []
        self.q_indexes = []
        state = self.env.reset()
        state = state2data(state, self.device)
        self.subgraph = deepcopy(env.sub_graph)
        done = False
        while not done:
            with torch.no_grad():
                q_vals = self.ddqn.current_net(state).view(-1)
                self.q_values.append(q_vals)
                self.q_indexes.append(np.argsort(q_vals))
                action = torch.argmax(q_vals)
            next_state, reward, done, valid_actions = env.step(action)

            # Move to the next state
            state = state2data(state, self.device)


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
                indexes = self.q_indexes[ns]
                for a in val_actions:
                    if a in indexes:
                        nodes_selected.append(a)
                        if self.dfs(nodes_selected):
                            return True
                        else:
                            nodes_selected.pop(-1)
        else:
            return False
