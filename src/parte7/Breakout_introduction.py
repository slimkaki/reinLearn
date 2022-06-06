import gym
import numpy as np

# para visualizar o ambiente render_mode='human'
env = gym.make("ALE/Breakout-v5", render_mode='human')
#env = gym.make("ALE/Breakout-v5")

print(env.observation_space.shape)
print(env.observation_space)
print(env.action_space)

env.reset()
done = False
rewards=0
while done != True:
    #env.render() => nao eh necessario fazer o render para os ambientes ALE
    action = np.random.randint(0, env.action_space.n)
    state2, reward, done, _ = env.step(action) 
    print(action)
    #print(str(state2)+"  "+str(reward))
    rewards += reward

print(f'Rewards = {rewards}')
