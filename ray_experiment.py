from multiprocessing import cpu_count
import gym
import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune import register_env
from ray.tune.logger import pretty_print

from space_lander.envs.lunar_lander import *
from space_lander.envs.spacex_lander import *

NUM_WORKERS = cpu_count()
ray.init(local_mode=True)

# Create environment
env_names = [('SpaceXLander', SpaceXLander), ('LunarLanderv1', LunarLanderv1),
             ('LunarLander', LunarLander)]
env_name, env = env_names[2]
SAVE_DIR = f'~/space-lander/space_lander/cache/{env_name}'


# HACK: sudo vim /usr/local/lib/python3.7/site-packages/ray/rllib/evaluation/rollout_worker.py
# add: from gym import wrappers for monitoring


def dqn_train(config, reporter):
    # Instantiate a trainer
    cfg = {
        # Max num timesteps for annealing schedules. Exploration is annealed from
        # 1.0 to exploration_fraction over this number of timesteps scaled by
        # exploration_fraction
        "schedule_max_timesteps" : 1000000,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": 1000,
        # Fraction of entire training period over which the exploration rate is
        # annealed
        "exploration_fraction"   : 0.1,
        # Final value of random action probability
        "exploration_final_eps"  : 0.02,
        "n_step"                 : 3,
        "buffer_size"            : 500000,
        # "sample_batch_size"         : 32,
        # "train_batch_size"          : 128,
        # "learning_starts"           : 5000,
        # "target_network_update_freq": 5000,
        # "num_workers"               : NUM_WORKERS,
        # "per_worker_exploration"    : True,
        # "worker_side_prioritization": True,
        # "min_iter_time_s"           : 1,
    }
    trainer = DQNTrainer(config={**config, **cfg})
    
    while True:
        result = trainer.train()  # Executes one training step
        print(pretty_print(result))
        reporter(**result)  # notifies TrialRunner


register_env(env_name, lambda config: env())

tune.run(
    dqn_train,
    config={
        "env": env_name,
    },
    # checkpoint_freq=1000,
    local_dir=SAVE_DIR,
    resources_per_trial={"cpu": cpu_count(), "gpu": 0},
    verbose=1,
    # resume=True,
    # restore=SAVE_DIR
    
)
