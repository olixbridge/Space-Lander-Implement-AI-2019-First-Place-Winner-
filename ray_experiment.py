import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune import register_env
from ray.tune.logger import pretty_print

from space_lander.envs.spacex_lander import *

NUM_WORKERS = 4
ray.init()

# Create environment
env_names = ['SpaceXLander', 'LunarLander']
env_name = env_names[1]


def dqn_train(config, reporter):
    # Instantiate a trainer
    cfg = {
        # "n_step"                    : 3,
        # "buffer_size"               : 100000,
        # "sample_batch_size"         : 32,
        # "train_batch_size"          : 128,
        # "learning_starts"           : 5000,
        # "target_network_update_freq": 5000,
        # "timesteps_per_iteration"   : 1000,
        # "num_workers"               : NUM_WORKERS,
        # "per_worker_exploration"    : True,
        # "worker_side_prioritization": True,
        # "min_iter_time_s"           : 1,
    }
    trainer = DQNTrainer(config={**config, **cfg}, env=env_name)
    
    while True:
        result = trainer.train()  # Executes one training step
        print(pretty_print(result))
        reporter(**result)  # notifies TrialRunner


register_env(env_name, lambda config: LunarLander())

tune.run(
    dqn_train,
    num_samples=1,
    # config={
    #     "env": env_name,
    # },
    # local_dir=f'space_lander/cache/{env_name}',
    # resources_per_trial={"cpu": 2, "gpu": 0},
    # resources for an individual trial (launched as ray actors)
    verbose=1,
    # queue_trials=True,
    # reuse_actors=True,
    # resume=True,
)
