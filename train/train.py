import argparse
import numpy as np

from itertools import count

import torch

from agent.DDQN import DDQN
from env.graph_matching_env import GraphMatchingEnv
from utils.utils import state2data


def parseargs():
    """
    Parse arguments.
    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, help='LeakyReLU angle of the negative slope', default=0.2)
    parser.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)
    parser.add_argument('--gamma', type=float, help='Discount factor', default=0.999)
    parser.add_argument('--tau', type=float, help='Alter target update rate', default=0.995)
    parser.add_argument('--n_feat', type=int, help='Alter target update rate', default=40+32)
    parser.add_argument('--n_hid', type=int, help='Alter target update rate', default=256)
    parser.add_argument('--n_head', type=int, help='Alter target update rate', default=20)

    parser.add_argument('--replay_size', type=int, help='Size of replay buffer.', default=150000)
    parser.add_argument('--sample_size', type=int, help='Size of replay buffer.', default=1000)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--maxsteps', type=int, help='Max training steps', default=int(1e6))

    parser.add_argument('--eval_freq', type=int, help='Evaluate frequency', default=int(10))

    _args = parser.parse_args()

    return _args


def evaluate_model(agent, env):
    durations = []
    for i_episode in range(1):
        # Initialize the environment and state
        state = env.reset()
        state = state2data(state, agent.device)
        total_reward = 0
        for t in count():
            # Select the action with the most rewards
            action = agent.select_best_action(state)
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            if done:
                next_state = None
            else:
                next_state = state2data(next_state, agent.device)
            state = next_state

            if done or t + 1 >= 2000:
                durations.append(total_reward)
                break
    mean = np.mean(durations)
    print("avg score: {}".format(mean))


def train_model(args, agent, env):

    keys = agent.current_net.state_dict().keys()

    train_step = 0
    eps = 0
    while train_step < args.maxsteps:
        # Initialize the environment and state
        state = env.reset()
        state = state2data(state, agent.device)
        total_reward = 0
        valid_actions = None
        for t in count():
            # Select and perform an action
            if agent.memory.size() < args.sample_size:
                action = env.get_correct_action()
            else:
                action = agent.select_action(state, valid_actions)
            next_state, reward, done, valid_actions = env.step(action)

            total_reward += reward
            reward = torch.tensor([reward], device=agent.device)
            if done:
                next_state = None
            else:
                next_state = state2data(next_state, agent.device)

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if done:
                # Perform one step of the optimization (on the target network)
                agent.optimize()

                # Update the target network using alternative target network method
                # phi = tau * phi + (1 - tau) * phi_updated
                target_state = agent.target_net.state_dict()
                policy_state = agent.current_net.state_dict()
                for key in keys:
                    target_state[key] = args.tau * target_state[key] + (1 - args.tau) * policy_state[key]
                agent.target_net.load_state_dict(target_state)

                train_step += t
                eps += 1

                print(eps)
                # evaluate_model(agent, env)
                break

        if eps % args.eval_freq == 0:
            evaluate_model(agent, env)


if __name__ == '__main__':
    args = parseargs()
    agent = DDQN(args)
    env = GraphMatchingEnv()
    train_model(args, agent, env)

