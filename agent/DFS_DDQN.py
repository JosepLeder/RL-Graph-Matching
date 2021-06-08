from __future__ import absolute_import

import torch
import numpy as np

from agent.DDQN import DDQN
from copy import deepcopy
from utils.utils import state2data


class DeepFirstSearchDDQN(object):

    def __init__(self, args, env):
        self.ddqn = DDQN(args)
        self.ddqn.load(args.load_dir)
        self.env = env
        self.device = torch.device("cpu")

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

        self.neg_set = set()
        self.steps = 0


    def dfs(self, nodes_selected):

        self.steps += 1

        if self.env.is_match(nodes_selected):
            print(nodes_selected)
            ns = len(nodes_selected)
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


