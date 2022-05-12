import numpy as np
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt

class QLearning:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
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
        #return self.env.action_space.sample()

    def train(self, filename, plotFile):
        rewards_per_episode = []
        rewards = 0

        for i in range(1, self.episodes+1):
            state = self.env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action) 
        
                # Adjust Q value for current state
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[state, action] = new_value
                
                state = next_state
                rewards += reward

            if i % 100 == 0:
                rewards_per_episode.append(rewards)
                rewards = 0
                sys.stdout.write("Episodes: " + str(i) +'\r')
                sys.stdout.flush()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        savetxt(filename, self.q_table, delimiter=',')
        if (plotFile is not None): self.plotactions(plotFile, rewards_per_episode)
        return self.q_table

    def plotactions(self, plotFile, actions_per_episode):
        plt.plot(actions_per_episode)
        plt.xlabel('100 Episodes')
        plt.ylabel('# Sum of rewards')
        plt.title('# Episodes vs Rewards')
        plt.savefig(plotFile+".jpg")     
        plt.close()
    