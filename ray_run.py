import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, DEFAULT_CONFIG
from ray.tune.registry import register_env

from asm_env_ma import ASMEnv

import tensorflow as tf

"""
TODO:
Define own PolicyOptimizer class for optimizing only agent or government policy
Define own policy class (?) of fixed goverment policies
Define nested training loop - change action spaces to have No-Op subspaces + mask? Does this require a new PolicyOptimizer class too?


"""

def policy_mapping_fn(agent_id):
    if agent_id.startswith('g'):
        return "govt_policy"
    else:
        return "citizen_policy_" + agent_id[-1]

def run(n_agents=3, episode_length=40000, config=None):
    ray.init()
    tf.compat.v1.enable_v2_behavior()
    # initialize trainer
    env = ASMEnv(n_agents=n_agents)
    register_env("asm", lambda _: ASMEnv(n_agents=n_agents, episode_length=episode_length))
    policies = {
        "govt_policy": (PPOTFPolicy, env.observation_space, env.govt_action_space, {}),
    }
    for idx in range(n_agents):
        policies[f"citizen_policy_{idx}"] = (PPOTFPolicy,
            env.observation_space, env.citizen_action_space, {})
    if config is None:
        ppo_config = {
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": list(policies.keys()),
            },
            "simple_optimizer": True,
            "observation_filter": "NoFilter",
            "framework": "tf",
        }
    else:
        ppo_config = config
    ppo_trainer = PPOTrainer(env="asm", config=ppo_config)
    print(ppo_trainer.train())
    print("DONE!")
    ray.shutdown()

def tune_run(n_agents=3, episode_length=4000, config=None):
    ray.init()
    tf.compat.v1.enable_v2_behaviøor()
    # initialize trainer
    env = ASMEnv(n_agents=n_agents)
    register_env("asm", lambda _: ASMEnv(n_agents=n_agents, episode_length=episode_length))
    policies = {
        "govt_policy": (PPOTFPolicy, env.observation_space, env.govt_action_space, {}),
    }
    for idx in range(n_agents):
        policies[f"citizen_policy_{idx}"] = (PPOTFPolicy,
            env.observation_space, env.citizen_action_space, {})
    if config is None:
        ppo_config = DEFAULT_CONFIG.copy()
    else:
        ppo_config = config
    ppo_config["env"] = "asm"
    ppo_config["train_batch_size"] = 400
    ppo_config["timesteps_per_iteration"] = episode_length
    ppo_config["multiagent"] = {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": list(policies.keys()),
        }
    tune.run("PPO", stop={"training_iteration": 100}, config=ppo_config)

if __name__ == '__main__':
    tune_run()
