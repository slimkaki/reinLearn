from IPython.display import clear_output
import gym
import gym_toytext
import numpy as np
from QLearning import QLearning
from numpy import loadtxt
from time import sleep

env = gym.make('Roulette-v0').env

#2600loss - stable
qlearn = QLearning(env, alpha=0.001, gamma=0.001, epsilon=0.9, epsilon_min=0.001, epsilon_dec=0.9999, episodes=1000000)
# real player like?
qlearn = QLearning(env, alpha=0.001, gamma=0.001, epsilon=0.9, epsilon_min=0.1, epsilon_dec=0.7, episodes=1000000)

q_table = qlearn.train('data/q-table-roulette.csv', None)
#q_table = loadtxt('data/q-table-roulette.csv', delimiter=',')

state = env.reset()
done = False
rewards = 0
actions = 0

while not done:
    action = np.argmax(q_table)
    state, reward, done, info = env.step(action)
    actions += 1   

    rewards += reward

    print("\n")    
    print("Action: ", action)
    print("Reward: ", reward)
    print("Done? ", done)

clear_output(wait=True)
print("\n")
print("Finished!")
print("Total Rewards: ", rewards)