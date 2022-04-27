import numpy as np
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt

#
# Esta é uma versão do Q-Learning com adicao de um metodo para tratamento dos estados
# do ambiente BlackJack. 
#

class QLearning:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        n_spaces = env.observation_space[0].n * env.observation_space[1].n * env.observation_space[2].n
        self.q_table = np.zeros([n_spaces, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample() # Explore action space
        return np.argmax(self.q_table[state]) # Exploit learned values

    def train(self, filename, plotFile):
        actions_per_episode = []
        for i in range(1, self.episodes+1):
            state = self.env.reset()
            reward = 0
            done = False
            actions = 0

            while not done:
                n_state = QLearning.stateNumber(state)
                action = self.select_action(n_state)
                next_state, reward, done, _ = self.env.step(action) 
                n_next_state = QLearning.stateNumber(next_state)

                # Adjust Q value for current state
                old_value = self.q_table[n_state, action]
                next_max = np.max(self.q_table[n_next_state])
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[n_state, action] = new_value
                
                state = next_state
                actions += 1

            if i % 100 == 0:
                actions_per_episode.append(actions)
                sys.stdout.write("Episodes: " + str(i) +'\r')
                sys.stdout.flush()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        savetxt(filename, self.q_table, delimiter=',')
        if (plotFile is not None): self.plotactions(plotFile, actions_per_episode)
        return self.q_table

    def plotactions(self, plotFile, actions_per_episode):
        plt.plot(actions_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Actions')
        plt.title('# Actions vs Episodes')
        plt.savefig(plotFile+".jpg")     
        plt.close()
    
    @staticmethod
    def stateNumber(state):
        (x,y,z) = state
        y = y * 32
        z = z * 352
        return x+y+z
