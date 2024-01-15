import envpool
import numpy as np

# offical atari example
# # make gym env
# env = envpool.make("Pong-v5", env_type="gym", num_envs=100)
# # or use envpool.make_gym(...)
# obs = env.reset()  # should be (100, 4, 84, 84)
# act = np.zeros(100, dtype=int)
# obs, rew, term, trunc, info = env.step(act)

# see all available envs
# print(envpool.list_all_envs())

num_envs = 4
env = envpool.make("Breakout-v5", env_type="gym", num_envs=num_envs, episodic_life=True, reward_clip=True)
obs = env.reset()
num_act = env.action_space.n
act = np.ones(num_envs, dtype=int)
obs, rew, term, trunc, info = env.step(act)
rew_list = [0] * num_envs
while True:
    act = np.random.randint(0, num_act, size=num_envs)
    obs, rew, term, trunc, info = env.step(act)
    rew_list = [rew_list[i] + rew[i] for i in range(num_envs)]
    # env.render()
    for i in range(num_envs):
        if term[i] or trunc[i]:
            print(f"env {i} terminated, reward: {rew_list[i]}")
            rew_list[i] = 0


    


