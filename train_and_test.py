import argparse
import numpy as np

from itertools import count

import torch

from agent.DDQN import DDQN
from agent.DFS_DDQN import DeepFirstSearchDDQN
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
    parser.add_argument('--dropout', type=float, help='Dropout rate', default=0)
    parser.add_argument('--gamma', type=float, help='Discount factor', default=0.999)
    parser.add_argument('--tau', type=float, help='Alter target update rate', default=0.999)

    parser.add_argument('--n_feat', type=int, help='Number of node features', default=200+40)
    parser.add_argument('--n_hid', type=int, help='Number od hidden parameters', default=512)
    parser.add_argument('--n_head', type=int, help='Number of GAT layers', default=4)

    parser.add_argument('--replay_size', type=int, help='Size of replay buffer.', default=150000)
    parser.add_argument('--sample_size', type=int, help='Minimum of samples.', default=2000)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=500)
    parser.add_argument('--maxsteps', type=int, help='Max training steps', default=int(10000))

    parser.add_argument('--eval_freq', type=int, help='Evaluate frequency', default=int(1))

    parser.add_argument('--save_dir', type=str, help='Path to save the model', default='results/')
    parser.add_argument('--load_dir', type=str, help='Path to load the model', default='results/')

    # test configures
    parser.add_argument('--test', type=bool, help='Training or testing.', default=True)
    parser.add_argument('--random_subgraph', type=bool,
                        help='Use random subgraph for testing if true, false otherwise.',
                        default=False)
    parser.add_argument('--use_rl', type=bool,
                        help='Use reinforcement learning method if true, false otherwise.',
                        default=True)
    parser.add_argument('--use_dfs', type=bool,
                        help='Use deep first search method if true, false otherwise.',
                        default=True)
    parser.add_argument('--pre_train', type=bool,
                        help='Use pre-training neural network parameters if true, false otherwise.',
                        default=True)


    _args = parser.parse_args()

    return _args


def evaluate_model(agent, env):
    durations = []
    for i_episode in range(10):
        # Initialize the environment and state
        state = env.reset()
        val_actions = state['valid_actions']
        data = state2data(state, agent.device)
        num_beast_match = 0
        for t in count():
            # Select the action with the most rewards
            action = agent.select_best_action(data, val_actions)
            next_state, reward, done, _ = env.step(action.item())
            if reward > 1e-6:
                num_beast_match = t + 1
            if not done:
                val_actions = next_state['valid_actions']
                data = state2data(next_state, agent.device)

            if done or t + 1 >= 40:
                durations.append(num_beast_match)
                break
    mean = np.mean(durations)
    return mean


def train_model(args, agent, env):

    keys = agent.current_net.state_dict().keys()

    train_step = 0
    eps = 0
    best_score = 0
    while train_step < args.maxsteps:

        if eps % args.eval_freq == 0:
            score = evaluate_model(agent, env)
            if score > best_score:
                best_score = score
                agent.save(args.save_dir)
            print("avg score: {}, best score: {}".format(score, best_score))

        # Initialize the environment and state
        state = env.reset()
        state = state2data(state, agent.device)
        total_reward = 0

        valid_actions = None
        for t in count():
            # Select and perform an action
            if (not args.test) and (agent.memory.size() >= args.sample_size):
                action = agent.select_best_action(state, valid_actions)
            else:
                action = env.get_correct_action()

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
                loss = agent.optimize()

                # Update the target network using alternative target network method
                # phi = tau * phi + (1 - tau) * phi_updated
                target_state = agent.target_net.state_dict()
                policy_state = agent.current_net.state_dict()
                for key in keys:
                    target_state[key] = args.tau * target_state[key] + (1 - args.tau) * policy_state[key]
                agent.target_net.load_state_dict(target_state)

                train_step += t
                eps += 1

                print("Episode: {}, loss: {}".format(eps, loss))
                break



if __name__ == '__main__':
    args = parseargs()

    env = GraphMatchingEnv(args)
    if args.test:
        args.load_dir = args.load_dir + "2021-06-06 22:12:26"
        agent = DeepFirstSearchDDQN(args, env)
        agent.dfs([])
        print(agent.steps)
    else:
        agent = DDQN(args)
        train_model(args, agent, env)

