import numpy as np
import numpy.random as npr
import time
import argparse
import sys

import gym
from gym import wrappers, logger

from asm_env import ASMEnv
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

    epochs = 50 # number of epochs
    episodes = 10000 # max iterations in single epoch
    steps_per_episode = 1000
    num_agents = 3

    env = ASMEnv(num_agents=num_agents, episode_length=steps_per_episode,
        subsidy_timesteps=steps_per_episode//5, evict_every=25)
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

    for ep in range(epochs):
        print("Epoch number: %d" % ep)
        for t in range(episodes):
            print("Episode number: %d" % t)
            done = False
            observation = env.reset()
            cum_reward = [0] * num_agents
            reward = [0] * num_agents
            # one episode
            while not done:
                env.render()
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
            for agent in agents:
                agent.end_of_episode()

        for agent in agents:
            agent.reset()

    env.close()
