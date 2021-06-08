import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from agent.nets.GAT import GAT
from torch_geometric.data import batch
from torch.distributions import Categorical
from utils.utils import ReplayBuffer, Transition

steps_cnt = 0

episode_durations = []
evaluate_performance = []


class DDQN(object):

    def __init__(self, args):
        self.args = args
        # set up device
        self.device = torch.device("cpu")

        # Create network to represent Q-Function
        self.current_net = GAT(args.n_feat, args.n_hid, args.n_head, args.dropout, args.alpha).to(self.device)
        self.target_net = GAT(args.n_feat, args.n_hid, args.n_head, args.dropout, args.alpha).to(self.device)
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.current_net.parameters(), lr=5e-4)
        self.memory = ReplayBuffer(args.replay_size)

        self.steps_cnt = 0

        self.create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def select_action(self, data, valid_actions=None):
        """
            Sample an action from categorical distribution.
        """
        x = self.current_net(data).view(-1)
        q = torch.ones_like(x) * -9e15

        if valid_actions is not None:
            q[valid_actions] = x[valid_actions]
        dist = Categorical(F.softmax(x, dim=0))
        action = dist.sample()
        return action.cpu().item()

    def select_best_action(self, data, valid_actions=None):
        """
            Select the action with most rewards
        """
        with torch.no_grad():
            x = self.current_net(data).view(-1)
            q = torch.ones_like(x) * -9e15
            if valid_actions is not None:
                q[valid_actions] = x[valid_actions]
            return torch.argmax(q)

    def optimize(self):
        if self.memory.size() < self.args.batch_size:
            return 0
        transitions = self.memory.sample(self.args.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = [s for s in batch.next_state if s is not None]
        batch_actions = torch.tensor(batch.action).reshape((len(batch.action), 1))
        batch_rewards = torch.tensor(batch.reward)


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to current_net
        # state_action_values = torch.tensor(
        #     [self.current_net(s).cpu().detach().numpy() for s in batch.state]
        # ).cuda().gather(1, batch.action)
        state_action_values = self.current_net(batch.state[0])[batch.action[0]]
        for s in range(1, len(batch.state)):
            state_action_values = torch.cat([state_action_values, self.current_net(batch.state[s])[batch.action[s]]])

        # Compute argmax(a{t+1})[Q(s_{t+1}, a_{t+1})] for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        next_state_values = torch.zeros(self.args.batch_size, device=self.device)
        next_state_values[non_final_mask] = torch.tensor(
            [self.target_net(s).detach().view(-1).numpy() for s in non_final_next_states]
        ).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.args.gamma) + batch_rewards
        # Compute MSE loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # clip the gradient to avoid gradient gradient explosion
        for param in self.current_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def save(self, filename):
        torch.save(self.current_net.state_dict(), filename + self.create_time + "_current")
        torch.save(self.optimizer.state_dict(), filename + self.create_time + "_current_optimizer")


    def load(self, filename):
        self.current_net.load_state_dict(torch.load(filename + "_current"))
        self.optimizer.load_state_dict(torch.load(filename + "_current_optimizer"))
        self.target_net = deepcopy(self.current_net)
