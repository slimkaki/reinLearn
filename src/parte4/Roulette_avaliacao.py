import gym
import gym_toytext

#
# testando se existe retorno igual a 35 ou nao. 
# parece que em momento algum o ambiente retorna um 
# reward = 35. 
# 

env = gym.make('Roulette-v0').env
env.reset()
reward = 0
while reward < 35:
    state, reward, done, info = env.step(2)
    print(state, reward)
