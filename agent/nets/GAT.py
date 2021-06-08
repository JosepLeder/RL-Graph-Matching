import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv



class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_head, dropout, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GATConv(n_feat, n_hid, dropout=dropout, negative_slope=alpha, concat=True)
                           for _ in range(n_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GATConv(n_hid * n_head, 1, dropout=dropout, negative_slope=alpha, concat=False)
        self.l1 = nn.Linear(n_hid * n_head, 600)
        self.l2 = nn.Linear(600, 120)
        self.l3 = nn.Linear(120, 1)

    def forward(self, data):

        x, adj = data.x, data.edge_index
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


if __name__ == '__main__':
    from env.graph_matching_env import GraphMatchingEnv
    from itertools import count
    from utils.utils import ReplayBuffer, Transition
    device = torch.device('cuda')

    env = GraphMatchingEnv()
    memory = ReplayBuffer(150000)
    model = GAT(40, 144, 40, 0.1, 0.999)

    train_step = 0
    while train_step < 10000:
        # Initialize the environment and state
        state = env.reset()
        total_reward = 0
        valid_actions = None
        for t in count():
            action = env.get_correct_action()
            next_state, reward, done, valid_actions = env.step(action)

            total_reward += reward
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = next_state

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            if done:
                break
        if memory.size() > 64:
            transitions = memory.sample(64)
            batch = Transition(*zip(*transitions))
            state = batch.state
            x = torch.tensor([s['x'] for s in state])
            e = torch.tensor([s['edge_index'] for s in state])
            print(x.shape)
            model([x, e])
