from racecar_gym.env import RaceEnv
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = RaceEnv(
    scenario='circle_cw_competition_collisionStop', # e.g., 'austria_competition', 'circle_cw_competition_collisionStop'
    render_mode='rgb_array_birds_eye',
    reset_when_collision=False, # Only work for 'austria_competition' and 'austria_competition_collisionStop'
)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3("MlpPolicy", env, action_noise=action_noise, buffer_size = 10000,verbose=1)
model.learn(total_timesteps=50000000, log_interval=10)
model.save("td3_pendulum")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = TD3.load("td3_pendulum")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
