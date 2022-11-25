import gym
import numpy as np
import time
from tqdm import tqdm

env = gym.make("CartPole-v1",render_mode="rgb_array")


def basic_policy(PoleAngle):

    if PoleAngle < 0 : # falling towards right
        return 0 # move cart to the right side
    return 1 # move the cart to left

total_rewards = list()
N_episodes = 200
N_steps = 200

for episode in range(N_episodes):
    rewards = 0
    # CartPosition,  PoleAngle,= env.reset()# CartVelocity,PoleAngluarVelocity 
    Observations = env.reset()
    print(Observations)
    PoleAngle = Observations[0][2]
    print(PoleAngle)
    for step in tqdm(range(N_steps)):
        env.render()
        print('render done')
        action = basic_policy(PoleAngle)
        Observation, reward, done, info = env.step(action)
        time.sleep(0.001)
        rewards += reward
        if done:
            break
    total_rewards.append(rewards)

stats = {
    "mean": np.mean(total_rewards),
    "standard deviation": np.std(total_rewards),
    "min": np.min(total_rewards),
    "max": np.max(total_rewards),
}

print(f"Final stats: \n{stats}")
