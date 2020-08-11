import gym
from gym import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import random

import pdb

PLANNER_ID = "p"
# up, right, down, left in terms of (y, x) coords
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
MAXIMUM_POLLUTION_LEVEL = 25
#subsidy_amount and subsidize_below actions both: no-op or 0%-36%, [0%, 4%, 8%, 12%, 16, ... 36%]
SUBSIDY_AMOUNT_NUM_ACTIONS = 11
SUBSIDIZE_BELOW_NUM_ACTIONS = 11
HEIGHT = 10
WIDTH = 10

DEFAULT_COLOURS = {'W': [180, 180, 180],  # Grey board walls
                   'E': [0, 0, 0],  # Black eviction
                   'F': [0, 255, 0],  # Green farming ground
                   'M': [255, 0, 0],  # Red mining ground

                   # Colours for agents. R value is a unique identifier
                   '0': [159, 67, 255],  # Purple
                   '1': [2, 81, 154],  # Blue
                   '2': [254, 151, 0],  # Orange
                   '3': [204, 0, 204],  # Magenta
                   '4': [100, 255, 255],  # Cyan
                   '5': [99, 99, 255],  # Lavender
                   '6': [250, 204, 255],  # Pink
                   '7': [238, 223, 16]}  # Yellow


DEFAULT_CONFIG = dict(
    episode_length=1000,
    planner_step_every=100,
    mining_prob_bounds=[0.4, 0.75],
    alpha=20,
    budget_per_step=0.5,
    planner_budget_penalty=0.02,
    eviction_cost=1.5,
    mining_recency_limit=10)


def get_pollution_rgb(rgb_val, pollution_level):
    return rgb_val * (1 - pollution_level/float(MAXIMUM_POLLUTION_LEVEL + 1))


class ASM(MultiAgentEnv):

    def __init__(self, env_config):
        config = DEFAULT_CONFIG.copy()
        config.update(env_config)
        self.num_agents = config["num_agents"]
        self.height = HEIGHT
        self.width = WIDTH

        # weight for planner reward (farming vs. mining reward)
        self.farming_reward_weight = 0.8

        self.planner_budget_penalty = config["planner_budget_penalty"]

        self.planner_step_every = config["planner_step_every"]
        self.episode_length = config["episode_length"]
        self.budget_per_episode = config["budget_per_step"] * self.episode_length
        self.alpha = config["alpha"]
        self.eviction_cost = config["eviction_cost"]
        self.mining_recency_limit = config["mining_recency_limit"]

        # initialize mining probabilities
        self.mining_probs = (config["mining_prob_bounds"][1] -
            config["mining_prob_bounds"][0]) * np.random.random(
            size=(self.height, self.width)) + config["mining_prob_bounds"][0]

        self.get_pollution_rgb_vec = np.vectorize(get_pollution_rgb)
        # for incrementing pollution (to a maximum level) if already some pollution
        self.increment_pollution_vec = np.vectorize(
            lambda x: min(x+1, MAXIMUM_POLLUTION_LEVEL) if x > 0 else 0)

        self.planner_action_dims = [
            SUBSIDY_AMOUNT_NUM_ACTIONS, SUBSIDIZE_BELOW_NUM_ACTIONS,
            self.num_agents + 1, self.height, self.width]

        self._planner_observation_space = spaces.Dict({
                "action_mask": spaces.MultiBinary(sum(self.planner_action_dims)),
                "actual_obs": spaces.Box(0, 255, shape=(self.height*4+2, self.width*4+2, 3), dtype=np.uint16)
            })
        self._citizen_observation_space = spaces.Dict({
            "actual_obs": spaces.Box(0, 255, shape=(self.height*4+2, self.width*4+2, 3),
                dtype=np.uint16),
            "citizen_rewards": spaces.Box(low=-50, high=50, shape=(self.num_agents,)),
            # Sees subsidy policy: how much to subsidize and under what threshold
            "planner_subsidy_policy": spaces.Box(low=0.0, high=1.0, shape=(2,)),
            })
        self._planner_action_space = spaces.MultiDiscrete(self.planner_action_dims)
            # ORDER:
            #subsidy_amount: no-op or from 0% to 40%, subsidize_below no-op or from 0% to 40%
            #eviction person or no-op, eviction y location (height), eviction x location (width)
        self._citizen_action_space = spaces.Discrete(7) #WASD, mine, farm, no-op
        self.planner_any_action_mask = np.full(
            (sum(self.planner_action_dims),), 1, dtype=np.uint8)
        # no-ops will be the first
        self.subsidy_amount_noop = 0
        self.subsidize_below_noop = SUBSIDY_AMOUNT_NUM_ACTIONS
        planner_subsidy_mask_indices = list(range(self.planner_action_dims[0] + self.planner_action_dims[1]))
        planner_subsidy_mask_indices.remove(self.subsidy_amount_noop) # subsidy_amount no-op index
        planner_subsidy_mask_indices.remove(self.subsidize_below_noop) # subsidize_below no-op index
        self.planner_subsidy_noop_mask = self.planner_any_action_mask.copy()
        self.planner_subsidy_noop_mask[planner_subsidy_mask_indices] = 0

        self.reset()


    @property
    def all_agent_ids(self):
        return [str(idx) for idx in range(self.num_agents)] + [PLANNER_ID]

    @property
    def citizen_ids(self):
        return [str(idx) for idx in range(self.num_agents)]

    @property
    def planner_observation_space(self):
        return self._planner_observation_space

    @property
    def planner_action_space(self):
        return self._planner_action_space

    @property
    def citizen_observation_space(self):
        return self._citizen_observation_space

    @property
    def citizen_action_space(self):
        return self._citizen_action_space


    def reset(self):
        self.step_count = 0
        self.planner_spent_so_far = 0
        self.last_reward_dict = {}
        self.locs_map = np.full((self.height, self.width), -1, dtype=np.int16)
        # mining recency - how recent was the last miner at each location
        self.mining_recency_map = np.full(
            (self.height, self.width), -1, dtype=np.int16)
        # farming histories for each agent at each location
        self.farming_history_map = np.zeros(
            (self.height, self.width, self.num_agents), dtype=np.uint16)
        self.pollution_map = np.zeros((self.height, self.width), dtype=np.uint16)
        self.evicted_agent = None # for indicating on map
        self.individual_farming_histories = np.zeros(
            (self.num_agents,), dtype=np.uint16)

        # initialize agent locations
        self.citizen_locs = np.zeros((self.num_agents, 2), dtype=np.uint16)
        for ind in range(self.num_agents):
            while True:
                y = np.random.choice(self.height)
                x = np.random.choice(self.width)
                # check to see another agent hasn't been initialized there
                if self.locs_map[y, x] < 0:
                    # update agent position
                    self.locs_map[y, x] = ind
                    self.citizen_locs[ind, :] = y, x
                    break

        # reset govt parameters #TODO: make random (or 0? agents will only see
        # this in the initial, very first observation - the planner will change
        # it on the first step)
        self.subsidize_below = np.random.uniform(low=0.1, high=0.3)
        self.subsidy_amount = np.random.uniform(low=0.1, high=0.3)

        # get and return observations
        return self.get_observations()


    def get_observations(self):
        obs_dict = {idx: {"actual_obs": self.get_rgb_obs()} for idx in self.all_agent_ids}

        # planner can take subsidy action
        if self.step_count % self.planner_step_every == 0:
            obs_dict[PLANNER_ID]["action_mask"] = self.planner_any_action_mask
        else:
            obs_dict[PLANNER_ID]["action_mask"] = self.planner_subsidy_noop_mask

        # add planner policy and citizen rewards to citizen observations
        citizen_rewards = np.array([self.last_reward_dict.get(str(idx), 0)
            for idx in range(self.num_agents)])
        planner_subsidy_policy = np.array(
            [self.subsidy_amount, self.subsidize_below])

        for idx in self.citizen_ids:
            obs_dict[idx]["citizen_rewards"] = citizen_rewards
            obs_dict[idx]["planner_subsidy_policy"] = planner_subsidy_policy

        return obs_dict


    def get_rgb_obs(self):
        """Shows pollution, farmable/mineable status, and agent locations."""
        # each grid is 4px wide, walls on each side are 2px wide each
        #rgb_arr = np.empty((4*self.height+2, 4*self.width+2, 3), dtype=np.uint16)
        # fill in farmable/nonfarmable land
        farmable_mask = self.mining_recency_map < 0
        farmable_rgb_mask = np.stack(
            [farmable_mask, farmable_mask, farmable_mask], axis=2)
        farmable_rgb = np.where(
            farmable_rgb_mask, DEFAULT_COLOURS["F"], DEFAULT_COLOURS["M"])
        rgb_with_pollution = self.get_pollution_rgb_vec(
            farmable_rgb, np.expand_dims(self.pollution_map, axis=2))
        # scale up to 4x4 for each grid
        rgb_arr = np.kron(rgb_with_pollution, np.ones((4,4,1)))
        # truncate floats to ints
        rgb_arr = rgb_arr.astype(np.uint8)

        # add agent positions
        for idx in range(self.num_agents):
            y, x = self.citizen_locs[idx, :]
            colour_locs = np.array(
                np.meshgrid([y*4+1, y*4+2], [x*4+1, x*4+2])).T.reshape(-1, 2)
            for c_y, c_x in colour_locs:
                # colour each of the 4 relevant tiles
                rgb_arr[c_y, c_x, :] = DEFAULT_COLOURS[str(idx)]

        # add walls
        wall_colour = DEFAULT_COLOURS["W"][0]
        first_dim_wall = np.full((1, 4*self.height, 3), wall_colour, dtype=np.uint32)
        rgb_arr = np.vstack([first_dim_wall, rgb_arr, first_dim_wall])
        second_dim_wall = np.full((4*self.height + 2, 1, 3), wall_colour, dtype=np.uint32)
        rgb_arr = np.hstack([second_dim_wall, rgb_arr, second_dim_wall])
        return rgb_arr


    def step(self, action_dict):
        self.step_count += 1
        # update mining recency figures
        self.update_mining_recency_map()
        # update pollution
        self.update_pollution_map()
        # if planner evicts someone or updates subsidy policy, do so here
        evicted_agent = self.planner_step(action_dict[PLANNER_ID])
        # execute citizen actions - move or mine/farm
        mining_reward_dict, farming_reward_dict = self.citizen_step(action_dict, evicted_agent)
        # get rewards from mining/farming and penalize if planner goes over budget
        reward_dict = {
            idx: mining_reward_dict.get(idx, 0) + farming_reward_dict.get(idx, 0) for idx in self.citizen_ids
        }

        if self.planner_spent_so_far > self.budget_per_episode:
            for k in reward_dict.keys():
                reward_dict[k] -= self.planner_budget_penalty * (self.planner_spent_so_far - self.budget_per_episode)

        reward_dict[PLANNER_ID] = (sum(mining_reward_dict.values()) * (1 - self.farming_reward_weight)
            + sum(farming_reward_dict.values()) * self.farming_reward_weight)

        self.last_reward_dict = reward_dict
        obs_dict = self.get_observations()
        done_dict = {k: False for k in self.all_agent_ids}
        done_dict["__all__"] = self.step_count >= self.episode_length
        info_dict = {k: {} for k in self.all_agent_ids}
        # return
        return obs_dict, reward_dict, done_dict, info_dict


    def planner_step(self, actions):
        assert(len(actions) == 5)
        # update subsidy policies
        if actions[0] != self.subsidy_amount_noop:
            self.subsidy_amount = actions[0] * 0.04
        if actions[1] != self.subsidize_below_noop:
            self.subsidize_below = actions[1] * 0.04

        # evict someone, if anyone
        if actions[2] != self.num_agents:
            if self.locs_map[actions[3], actions[4]] < 0:
                # evict directly if someone isn't already there
                self.locs_map[actions[3], actions[4]] = actions[2]
                self.citizen_locs[actions[2], :] = actions[3], actions[4]
                self.planner_spent_so_far += self.eviction_cost
                return actions[2]
        return None


    def citizen_step(self, action_dict, evicted_agent=None):
        """Acts wrt action dict (only uses citizen actions) and any evictions"""
        mining_reward_dict = {idx: 0 for idx in self.citizen_ids}
        farming_reward_dict = {idx: 0 for idx in self.citizen_ids}

        # make list of shuffled citizen agent indices - AS INTEGERS
        citizen_indices = [int(i) for i in self.citizen_ids]
        random.shuffle(citizen_indices)

        # handle actions according to the randomly shuffled citizen indices
        for idx in citizen_indices:
            if evicted_agent is not None and idx == evicted_agent:
                # don't execute their action if they were just evicted
                continue
            agent_action = action_dict[str(idx)]
            curr_loc = self.citizen_locs[idx, :]
            if agent_action == 6: # NO-OP
                continue
            elif agent_action == 4:
                mining_reward_dict[str(idx)] = self.mine_and_get_reward(curr_loc, idx)
            elif agent_action == 5:
                farming_reward_dict[str(idx)] == self.farm_and_get_reward(curr_loc, idx)
            else:
                # Action must be a movement action, so update their positions.
                y_move, x_move = MOVES[agent_action]
                new_pos = curr_loc[0] + y_move, curr_loc[1] + x_move
                # check to see the new position is within bounds, and another
                # agent hasn't moved into that position, else stay put
                if (new_pos[0] < self.height and new_pos[0] >= 0) and (new_pos[1] < self.width and new_pos[1] >= 0):
                    if self.locs_map[new_pos[0], new_pos[1]] < 0:
                        # update agent position
                        self.locs_map[new_pos[0], new_pos[1]] = idx
                        # remove from old pos
                        self.locs_map[curr_loc[0], curr_loc[1]] = -1
                        self.citizen_locs[idx, :] = new_pos[0], new_pos[1]

        return mining_reward_dict, farming_reward_dict



    def update_mining_recency_map(self):
        self.mining_recency_map += 1
        # replace all values equal to or over 10 with -1 (last mining step was 10 ago, so we can farm now)
        np.place(self.mining_recency_map, self.mining_recency_map >= self.mining_recency_limit, -1)


    def update_pollution_map(self):
        """Updates pollution spread in a downstream triangle shape."""
        # increment pollution level of polluted areas
        self.pollution_map = self.increment_pollution_vec(self.pollution_map)
        # get locs of lowest polluted areas to increment pollution around it
        ys, xs = np.where(self.pollution_map == 2)
        for i in range(len(ys)):
            if ys[i] + 1 < self.height:
                if xs[i] + 1 < self.width:
                    self.pollution_map[ys[i] + 1, xs[i] + 1] = min(max(1, self.pollution_map[ys[i] + 1, xs[i] + 1]), MAXIMUM_POLLUTION_LEVEL)
                    #m_new[ys[i], xs[i] + 1] = max(1, m_new[ys[i], xs[i] + 1])
                self.pollution_map[ys[i] + 1, xs[i]] = min(max(1, self.pollution_map[ys[i] + 1, xs[i]]), MAXIMUM_POLLUTION_LEVEL)
            elif xs[i] + 1 < self.width:
                    self.pollution_map[ys[i], xs[i] + 1] = min(max(1, self.pollution_map[ys[i], xs[i] + 1]), MAXIMUM_POLLUTION_LEVEL)


    def mine_and_get_reward(self, coords, agent_id):
        # Mining success is dependent on initialized probs
        prob_of_success = self.mining_probs[coords[0], coords[1]]
        reward = np.random.binomial(n=1.0, p=prob_of_success)
        # keep track of mining history if successfully mined
        if reward != 0:
            # increment mining recency and pollution maps
            self.mining_recency_map[coords[0], coords[1]] = min(
                self.mining_recency_map[coords[0], coords[1]] + 1,
                self.mining_recency_limit)
            self.pollution_map[coords[0], coords[1]] = min(
                self.pollution_map[coords[0], coords[1]] + 1,
                MAXIMUM_POLLUTION_LEVEL)
        return reward


    def farm_and_get_reward(self, coords, agent_id):
        """Farming success is dependent on own farming exp and the farming done at the location"""
        farmable = self.mining_recency_map[coords[0], coords[1]] < 0
        if farmable:
            # calculate farming reward
            individual_farm_history = self.individual_farming_histories[agent_id]
            location_farm_history = np.sum(self.farming_history_map[coords[0], coords[1]])
            exp = individual_farm_history + location_farm_history
            #pdb.set_trace()
            prob_of_success = exp/float(exp+self.alpha)
            if prob_of_success < self.subsidize_below:
                prob_of_success_subsidized = min(prob_of_success + self.subsidy_amount, 1.0)
                # add subsidized prob directly to cost
                self.planner_spent_so_far += self.subsidy_amount
                reward = np.random.binomial(n=1.0, p=prob_of_success_subsidized)
            else:
                reward = np.random.binomial(n=1.0, p=prob_of_success)
            # keep track of farming history if successfully farmed
            if reward != 0:
                self.farming_history_map[coords[0], coords[1], agent_id] += 1
                self.individual_farming_histories[aget_id] += 1
        else:
            reward = 0
        return reward

