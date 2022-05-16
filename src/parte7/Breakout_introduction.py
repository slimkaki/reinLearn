import gym
import numpy as np

# para visualizar o ambiente render_mode='human'
# env = gym.make("ALE/Breakout-v5", render_mode='human')
env = gym.make("ALE/Breakout-v5")
env.reset()

done = False
while done != True:
    #env.render() => nao eh necessario fazer o render para os ambientes ALE
    action = np.random.randint(0, env.action_space.n)
    state2, reward, done, _ = env.step(action) 
    print(str(state2)+"  "+str(reward))
