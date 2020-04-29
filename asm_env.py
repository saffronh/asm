import gym
from gym import spaces

import numpy as np

# grid type enums
MINING = 1
FARMING = 2
NEITHER = 0

def one_hot_to_index(arr):
    if arr.ndim != 1 and arr.shape[1] > 1:
        raise InputError("Must be 1D array or 2D array with 2nd dim as 1")
    arr.astype(int)
    arr_as_list = list(np.reshape(arr, -1))
    return arr_as_list.index(1)

class ASMEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_agents, episode_length=5000,
            subsidy_timesteps=1000, evict_every=40,
            subsidy_prob_amount=0.2, mining_prob_bounds=[0.4, 0.75], alpha=20,
            is_global_obs=True, reset_mining_probs_every_ep=True):
        super(ASMEnv, self).__init__()

        self._num_agents = num_agents
        self.episode_length = episode_length
        self.subsidy_timesteps = subsidy_timesteps
        self.evict_every = evict_every
        self.subsidy_prob_amount = subsidy_prob_amount
        self.mining_prob_bounds = mining_prob_bounds
        self.is_global_obs = is_global_obs # whether the observation is global
        self.reset_mining_probs_every_ep = reset_mining_probs_every_ep
        self.alpha = alpha # tweaks farming reward curve y=x/x+a

        self.width = 20
        self.height = 20
        self.mining_height = 8
        self.farming_height = 8
        self.obs_width = 7
        self.obs_height = 7

        # action space is a list of gym Spaces, length num_agents
        self._action_space = [spaces.Discrete(5) for _ in range(self.num_agents)]
        # observations are agent coordinates and evictions
        if self.is_global_obs:
            single_agent_obs = spaces.Tuple((spaces.Box(low=0, high=episode_length,
                shape=(self.width, self.height), dtype=np.int32),
                spaces.MultiBinary(self.num_agents)))
        else:
            single_agent_obs = spaces.Tuple((spaces.Box(low=0, high=episode_length,
                shape=(self.obs_width, self.obs_height), dtype=np.int32),
                spaces.MultiBinary(self.num_agents)))

        # observation sspace is a list of gym Spaces, length num_agents
        self._observation_space = [single_agent_obs
            for _ in range(self.num_agents)]

        # up, right, down, left in terms of (x, y) coords
        self.moves = ((0, -1), (1, 0), (0, 1), (-1, 0))

        self.step_count = None
        self.agent_positions = []
        self.world_state = None
        self.evictions = None
        self.agent_farming_history = None

        # initialize mining probabilities
        self.reset_mining_probs()

        self.reset()

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def step(self, actions):
        # execute one timestep
        self.step_count += 1
        assert(len(actions) == self.num_agents)
        reward = [0 for _ in range(self.num_agents)]
        # perform evictions every evict_every timesteps
        if self.step_count % self.evict_every == 0:
            evicted_agent = self.evict()
        else:
            evicted_agent = None

        shuffled_agent_indices = np.arange(len(actions))
        # handle agent actions randommly - create randomly ordered index
        np.random.shuffle(shuffled_agent_indices)
        for ind in shuffled_agent_indices:
            if evicted_agent is not None and ind == evicted_agent:
                # don't execute their action if they were just evicted
                continue
            #agent_action = one_hot_to_index(actions[ind])
            agent_action = actions[ind]
            curr_agent_position = self.agent_positions[ind]
            if agent_action == 4:
                agent_reward = self.mine_or_farm(curr_agent_position, ind)
                reward[ind] = agent_reward
            else:
                x_move, y_move = self.moves[agent_action]
                new_pos = curr_agent_position[0] + x_move, curr_agent_position[1] + y_move
                # check to see the new position is within bounds, and another
                # agent hasn't moved into that position, else stay put
                if (new_pos[0] < self.width and new_pos[0] >= 0) and (new_pos[1] < self.height and new_pos[1] >= 0):
                    if self.world_state[new_pos[0], new_pos[1], 1] < 0:
                        # update agent position
                        self.world_state[new_pos[0], new_pos[1], 1] = ind
                        # remove from old pos
                        self.world_state[curr_agent_position[0], curr_agent_position[1], 1] = -1
                        self.agent_positions[ind] = new_pos[0], new_pos[1]

        obs = self.get_observations()
        done = self.step_count >= self.episode_length
        info = [{} for _ in range(self.num_agents)]

        return obs, reward, done, info

    def mine_or_farm(self, coords, agent_id):
        grid_type = self.get_grid_type(coords)
        if grid_type == MINING:
            reward = self.get_mining_reward(coords)
            # keep track of mining history if successfully mined
            if reward:
                prev_num_miners = self.world_state[coords[0], coords[1], 0]
                self.world_state[coords[0], coords[1], 0] = prev_num_miners + 1
        elif grid_type == FARMING:
            reward = self.get_farming_reward(coords, agent_id)
            # keep track of farming history if successfully farmed
            if reward:
                prev_num_farmers = self.world_state[coords[0], coords[1], 0]
                self.world_state[coords[0], coords[1], 0] = prev_num_farmers + 1
        else:
            reward = 0
        return reward

    def get_grid_type(self, coords):
        x, y = coords
        if y < self.mining_height:
            return MINING
        elif y < self.height - self.farming_height:
            return NEITHER
        elif y < self.height:
            return FARMING

    def get_mining_reward(self, coords):
        prob_of_success = self.mining_probs[coords[0], coords[1]]
        return np.random.binomial(n=1.0, p=prob_of_success)

    def get_farming_reward(self, coords, agent_id):
        x = self.agent_farming_history[agent_id]
        prob_of_success = x/float(x+self.alpha)
        if self.step_count < self.subsidy_timesteps:
            prob_of_success_subsidized = min(prob_of_success + self.subsidy_prob_amount, 1.0)
            return np.random.binomial(n=1.0, p=prob_of_success_subsidized)
        return np.random.binomial(n=1.0, p=prob_of_success)

    def get_observations(self):
        obs = []
        for ind in range(self.num_agents):
            if self.is_global_obs:
                agent_obs = (self.world_state[:, :, 1], self.evictions)
            else:
                x, y = self.agent_positions[ind]
                local_height_rad = self.observation_height//2
                local_width_rad = self.observation_width//2
                local_obs = self.world_state[
                    x-local_width_rad:x+local_width_rad+1,
                    y-local_height_rad:x+local_height_rad+1,
                    1]
                agent_obs = (local_obs, self.evictions)
            obs.append(agent_obs)
        return obs

    def evict(self):
        # randomly evict one agent that is on mining side to farming side
        shuffled_agent_indices = np.arange(self.num_agents)
        np.random.shuffle(shuffled_agent_indices)
        for ind in shuffled_agent_indices:
            coords = self.agent_positions[ind]
            # look for first mining agent
            if self.get_grid_type(coords) == MINING:
                while True:
                    # loop until an open farming spot is found
                    x = np.random.choice(self.width)
                    y_relative = np.random.choice(self.farming_height)
                    y = self.height - y_relative - 1
                    # check that its the farming side
                    assert(self.get_grid_type((x, y)) == FARMING)
                    if self.world_state[x, y, 1] < 0:
                        # successful farming side spot found; update positions
                        self.world_state[x, y, 1] = ind
                        # set previous position to -1
                        self.world_state[coords[0], coords[1], 1] = -1
                        self.agent_positions[ind] = x, y
                        # note the eviction
                        self.evictions = np.zeros(
                            (self.num_agents,), dtype=np.int32)
                        self.evictions[ind] = 1
                        return ind
        return None

    def reset_mining_probs(self):
        """Resets mining probabilities. Can choose when this is done."""
        self.mining_probs = (self.mining_prob_bounds[1] -
            self.mining_prob_bounds[0]) * np.random.random(
            size=(self.width, self.height)) + self.mining_prob_bounds[0]
        self.mining_probs[:, self.mining_height:] = 0

    def reset(self):
        # reset the state to initial state
        self.step_count = 0
        if self.reset_mining_probs_every_ep:
            self.reset_mining_probs()
        # initialize world state - for each grid theres a mining/farming history, and
        # indication of which agent if any is in the grid (0 to
        # NUM_AGENTS-1 to represent each agent, and -1 if no agent)
        self.world_state = np.zeros(
            (self.width, self.height, 2), dtype=np.int32)
        # initialize with "no agents" anywhere
        self.world_state[:, :, 1] = -1
        # evictions in previous timestep
        self.evictions = np.zeros((self.num_agents,), dtype=np.int32)
        # history of farming for each agent
        self.agent_farming_history = np.zeros(
            (self.num_agents,), dtype=np.int32)
        # initialize agent positions
        self.agent_positions = [None for _ in range(self.num_agents)]
        for ind in range(self.num_agents):
            while True:
                x = np.random.choice(self.width)
                y = np.random.choice(self.height)
                # check to see another agent hasn't beenn initialized in that position
                if self.world_state[x, y, 1] < 0:
                    # update agent position
                    self.world_state[x,  y, 1] = ind
                    self.agent_positions[ind] = x, y
                    break
        obs = self.get_observations()
        return obs

    def render(self, mode="human", close=False):
        # render environment to the screen
        return
