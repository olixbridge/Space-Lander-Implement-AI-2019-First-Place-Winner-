import os

from stable_baselines import DQN
from tqdm import tqdm

from space_lander.envs.spacex_lander import *

# Create environment
env_names = ['SpaceXLander-v0', 'LunarLanderv2-v0']
env_name = env_names[0]
env = gym.make(env_name)

# Instantiate the agent
model = DQN(policy='MlpPolicy',
            env=env,
            learning_rate=1e-3,
            prioritized_replay=True,
            verbose=1,
            tensorboard_log=f"./{env_name}")

# Train the agent
obs = env.reset()

def eval_and_show(*args, **kwargs):
    if args[0]['t'] % 1000 == 0:
        print('Evaluating', args[0]['t'])
        done = False
        while not done:
            action, _states = model.predict(args[0]['obs'])
            obs, reward, done, info = env.step(action)
            env.render()
        # env.close()

# Train the agent
model.learn(total_timesteps=1000000,
            callback=eval_and_show,
            log_interval=2)

# Save the agent
filename = f'{env_name}'
model.save(filename)

# model.load(load_path=f'{filename}.pkl')  # Load the agent
