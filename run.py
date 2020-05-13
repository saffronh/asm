import numpy as np
import numpy.random as npr
import time
import argparse
import sys
from collections import Counter

import gym
from gym import wrappers, logger

import pandas as pd

from asm_env import ASMEnv, Government
from QLearningAgentClass import QLearningAgent


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward):
        return self.action_space.sample()

    def reset(self):
        return

    def end_of_episode(self):
        return


if __name__ == '__main__':

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    # make dataframe for storing results
    col_names = ['epoch_num', 'episode_num',
        'episode_length', 'num_agents', 'subsidy_below', 'evict_every',
        'subsidy_prob_amount', 'mining_prob_lower_bound', 'mining_prob_upper_bound',
        'farming_alpha', 'agent_id', 'cumulative_reward', 'mining_amount',
        'farming_amount', 'a0', 'a1', 'a2', 'a3', 'a4']

    epochs = 10 # number of epochs
    episodes = 2000  # max iterations in single epoch
    steps_per_episode = 2000
    num_agents = 3
    mining_prob_bounds = [0.55, 0.75]
    subsidy_below = 0.3

    #agents = [RandomAgent(env.action_space[0]) for _ in range(num_agents)]
    # these agents require strings for actions
    actions = [str(i) for i in range(5)]
    agents = [QLearningAgent(actions=actions) for _ in range(num_agents)]

    # # You provide the directory to write to (can be an existing
    # # directory, including one with existing data -- all monitor files
    # # will be namespaced). You can also dump to a tempdir if you'd
    # # like: tempfile.mkdtemp().
    # outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    # env.seed(0)

    varying_params = [
        {'evict_every': 20, 'subsidy_prob_amount': 0.1, 'alpha': 20},
        {'evict_every': 20, 'subsidy_prob_amount': 0.25, 'alpha': 20},
        {'evict_every': 50, 'subsidy_prob_amount': 0.1, 'alpha': 20},
        {'evict_every': 50, 'subsidy_prob_amount': 0.25, 'alpha': 20},
        {'evict_every': 20, 'subsidy_prob_amount': 0.1, 'alpha': 120},
        {'evict_every': 20, 'subsidy_prob_amount': 0.25, 'alpha': 120},
        {'evict_every': 25, 'subsidy_prob_amount': 0.1, 'alpha': 120},
        {'evict_every': 25, 'subsidy_prob_amount': 0.25, 'alpha': 120},
    ]

    for ep in range(epochs):
        print("Epoch number: %d" % ep)
        # pickle results every epoch - in this case one epoch consists of running
        # the 8 diff params (high rate: LL, HL, LH, HH; low rate: LL, HL, LH, HH)
        results = []

        for params_dict in varying_params:
            print("Params: alpha: %d subsidy: %d eviction rate: %d" % (
                params_dict['alpha'], params_dict['subsidy_prob_amount'],
                params_dict['evict_every']))
            govt = Government(subsidy_below=subsidy_below,
                    evict_every=params_dict['evict_every'],
                    subsidy_prob_amount=params_dict['subsidy_prob_amount'])
            env = ASMEnv(num_agents=num_agents, govt=govt,
                episode_length=steps_per_episode,
                mining_prob_bounds=mining_prob_bounds,
                alpha=params_dict['alpha'], reset_mining_probs_every_ep=False)

            for t in range(episodes):
                print("Episode number: %d" % t)

                # initialize dict to store results - still need agent ID, cum reward, mining amount, farming amount
                results_dict = {
                    'epoch_num': ep,
                    'episode_num': t,
                    'episode_length': steps_per_episode,
                    'num_agents': num_agents,
                    'subsidy_below': subsidy_below,
                    'subsidy_prob_amount': params_dict['subsidy_prob_amount'],
                    'evict_every': params_dict['evict_every'],
                    'mining_prob_lower_bound': mining_prob_bounds[0],
                    'mining_prob_upper_bound': mining_prob_bounds[1],
                    'farming_alpha': params_dict['alpha'],
                }

                done = False
                observation = env.reset()
                cum_reward = [0] * num_agents
                reward = [0] * num_agents
                # one episode
                while not done:
                    actions = []
                    for i in range(num_agents):
                        # observation
                        q_agent_obs = observation[i].tobytes() # make hashable
                        agent_action = agents[i].act(q_agent_obs, reward[i]) # this probably wont converge
                        actions.append(int(agent_action))
                    observation, reward, done, info = env.step(actions)
                    for i in range(num_agents):
                        cum_reward[i] += reward[i]
                    #time.sleep(1) # for testing purposes; remove for faster episodes

                print("Cumulative reward: ", cum_reward)
                print("Mining amount: ", env.get_cumulative_mining_amount())
                print("Farming amount: ", env.get_cumulative_farming_amount())

                for i in range(num_agents):
                    agent_results = results_dict.copy()
                    agent_results['agent_id'] = i
                    agent_results['cumulative_reward'] = cum_reward[i]
                    agent_results['mining_amount'] = env.get_cumulative_mining_amount(i)
                    agent_results['farming_amount'] = env.get_cumulative_farming_amount(i)
                    action_history_count = Counter(agents[i].action_history)
                    for k, v in action_history_count.items():
                        agent_results['a' + k] = v
                    results.append(agent_results)  # append to master list of results
                    agents[i].end_of_episode()

        results_df = pd.DataFrame(data=results, columns=col_names)
        results_df.to_pickle(f'results_{int(time.time())}.pkl')

        for agent in agents:
            agent.reset()


    env.close()
