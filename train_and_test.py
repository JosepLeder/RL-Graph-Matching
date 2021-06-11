import argparse
import numpy as np

from itertools import count

import torch
import torch.nn.functional as F

from agent.DDQN import DDQN
from agent.BaseDFS import BaseDeepFirstSearch
from agent.DFS_DDQN import DDQN_DeepFirstSearch
from env.graph_matching_env import GraphMatchingEnv
from utils.utils import state2data, save_dict


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
    parser.add_argument('--load_dir', type=str, help='Path to load the model', default=None)

    # test configures
    parser.add_argument('--test', type=bool, help='Training or testing.', default=True)
    parser.add_argument('--agent', type=str, help='Agent to use for training.', default="DDQN_DFS")
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


def train_model(args, agent, env, filename=None):

    keys = agent.current_net.state_dict().keys()

    train_step = 0
    eps = 0
    best_score = 0
    x_steps = []
    y_rewards = []
    while train_step < args.maxsteps:

        if train_step % args.eval_freq == 0:
            score = evaluate_model(agent, env)
            if score > best_score:
                best_score = score
                if args.save_dir is not None:
                    agent.save(args.save_dir)
            x_steps.append(train_step)
            y_rewards.append(best_score)
            print("avg score: {}, best score: {}".format(score, best_score))

        # Initialize the environment and state
        state = env.reset()
        state = state2data(state, agent.device)
        total_reward = 0

        valid_actions = None
        for t in count():
            # Select and perform an action
            if agent.memory.size() >= args.sample_size:
                action = agent.select_action(state, valid_actions)
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

            if done or reward == 0:
                train_step += t
                eps += 1
                # Perform one step of the optimization (on the target network)
                loss = agent.optimize()
                print("Trsinstep: {} Episode: {}, loss: {}".format(train_step, eps, loss))

                # Update the target network using alternative target network method
                # phi = tau * phi + (1 - tau) * phi_updated
                target_state = agent.target_net.state_dict()
                policy_state = agent.current_net.state_dict()
                for key in keys:
                    target_state[key] = args.tau * target_state[key] + (1 - args.tau) * policy_state[key]
                agent.target_net.load_state_dict(target_state)
                break

    if filename is not None:
        dct = {
            "x": x_steps,
            "y": y_rewards
        }
        save_dict(filename, dct)



if __name__ == '__main__':
    args = parseargs()

    env = GraphMatchingEnv(args)
    if args.test:
        if args.agent == "DFS":
            agent = BaseDeepFirstSearch(env)
            agent.threshold = int(1e6)
            agent.dfs([])
            agent.save_result()
        elif args.agent == "DDQN_DFS":
            args.load_dir = "results/2021-06-06 22:12:26"
            agent = DDQN_DeepFirstSearch(args, env)
            q = agent.q_values[0]
            prob = F.softmax(q, dim=0)
            for i in range(200):
                agent.steps = 0
                agent.threshold = int(prob[i] * 1e6)
                agent.dfs([i])
            agent.save_result()
        elif args.agent == "DDQN":
            args.save_dir = None
            agent = DDQN(args)
            train_model(args, agent, env, "DDQN.json")
        elif args.agent == "pretrain_DDQN":
            # args.save_dir = None
            args.load_dir = "results/2021-06-06 22:12:26"
            agent = DDQN(args)
            train_model(args, agent, env, "results/pretrain_DDQN.json")

    else:
        agent = DDQN(args)
        train_model(args, agent, env)

