import argparse

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, DEFAULT_CONFIG
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved

from action_mask_model import ActionMaskModel
from asmenv import ASM

tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--episode_length", type=int, default=1000)
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)
parser.add_argument("--train_separate", type=bool, default=False)
parser.add_argument("--num_agents", type=int, default=3)

#TODO: Add fixed policies
PLANNER_FIXED_POLICIES = {}


def policy_mapping_fn(agent_id):
    if agent_id.startswith('p'):
        return "planner_policy"
    else:
        return "citizen_policy_" + agent_id[-1]


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    env_config = {
        "num_agents": args.num_agents,
        "episode_length": args.episode_length,
    }
    register_env("asm_env", lambda _: ASM(env_config))
    ModelCatalog.register_custom_model(
        "action_mask_model", ActionMaskModel)

    env = ASM({"num_agents": args.num_agents})

    config = DEFAULT_CONFIG.copy()

    config["env"] = "asm_env"
    config["env_config"] = env_config
    config["train_batch_size"] = args.episode_length // 2
    config["timesteps_per_iteration"] = args.episode_length
    config["simple_optimizer"] = True

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.train_separate:

        # initialize citizen policies
        citizen_policies = {
            f"citizen_policy_{idx}": (
                PPOTFPolicy, env.citizen_observation_space, env.citizen_action_space,
                {"model": {"use_lstm": True}}
            )
            for idx in range (args.num_agents)}

        # TRAINING!
        # train citizen agents only on fixed policies (that they observe)
        citizen_config = config.copy()
        citizen_config.update({
            "multiagent": {
                "policies_to_train": list(citizen_policies.keys()),
                "policies": {**citizen_policies, **PLANNER_FIXED_POLICIES},
                "policy_mapping_fn": policy_mapping_fn,
            },
        })
        # fix citizens, train government policy
        learned_planner_policy = {
            "planner_policy": (
                PPOTFPolicy, env.planner_observation_space, env.planner_action_space,
                {"model": {
                    "custom_model": "action_mask_model",
                    "use_lstm": True,
                    "custom_model_config": {"actual_obs_space": env.planner_observation_space["actual_obs"]}}})
        }
        planner_config = config.copy()
        planner_config.update({
            "multiagent": {
                "policies_to_train": list(learned_planner_policy.keys()),
                "policies": {**citizen_policies, **learned_planner_policy},
                "policy_mapping_fn": policy_mapping_fn,
            },
        })

        # training loop
        citizen_results = tune.run(args.run, config=citizen_config, stop=stop)
        # stop after some criteria, then train the government (some convergence criteria of the policies?)
        planner_results = tune.run(args.run, config=planner_config, stop=stop)
        if args.as_test:
            check_learning_achieved(citizen_results, args.stop_reward)
            check_learning_achieved(planner_results, args.stop_reward)
    else:

        # initialize policies
        policies = {
            "planner_policy": (PPOTFPolicy, env.planner_observation_space, env.planner_action_space,
                {"model": {
                    "custom_model": "action_mask_model",
                    "use_lstm": True,
                    "custom_model_config": {"actual_obs_space": env.planner_observation_space["actual_obs"]}}})
        }
        for idx in range(args.num_agents):
            policies[f"citizen_policy_{idx}"] = (PPOTFPolicy,
                env.citizen_observation_space, env.citizen_action_space,
                {"model": {"use_lstm": True}})

        # train with 2 loops
        dual_training_config = config.copy()
        dual_training_config.update({
            "multiagent": {
                "policies_to_train": list(policies.keys()),
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            }
        })

        results = tune.run(args.run, config=dual_training_config, stop=stop)
        if args.as_test:
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()



