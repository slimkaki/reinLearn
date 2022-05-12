import gym
import numpy as np
from QLearning_BlackJack import QLearning


env = gym.make('Blackjack-v1').env
q_table = np.loadtxt('data/q-table-blackjack.csv', delimiter=',')

vitorias = 0
empates = 0
derrotas = 0

for _ in range(100):
    state= env.reset()
    done = False
    while not done:
        n_state = QLearning.stateNumber(state)
        action = np.argmax(q_table[n_state])
        state, reward, done, info = env.step(action)

    if reward == 1:
        vitorias += 1
    elif reward == 0:
        empates += 1
    elif reward == -1:
        derrotas += 1

print(vitorias, empates, derrotas)