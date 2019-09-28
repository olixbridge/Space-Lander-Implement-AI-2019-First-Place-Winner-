from tqdm import tqdm
from stable_baselines import DQN
from space_lander.envs.lunar_lander import *
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
model.learn(total_timesteps=int(2e2), log_interval=10)

# Save the agent
model.save(env_name)

# Enjoy trained agent
obs = env.reset()
for i in tqdm(range(1000)):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()