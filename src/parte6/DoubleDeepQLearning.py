import numpy as np
import random
from keras.activations import relu, linear
import keras

class DoubleDeepQLearning:

    #
    # Implementacao do algoritmo proposto em 
    # Deep Reinforcement Learning with Double Q-learning, van Hasselt et al., 2015
    # https://arxiv.org/abs/1509.06461
    #
    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model
        # criando um segundo modelo, que é clone do modelo online
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    #
    # usa o target_model para escolher as acoes
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        action = self.target_model.predict(state, verbose=0)
        return np.argmax(action[0])

    #
    # cria uma memoria longa de experiencias
    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal)) 

    def experience_replay(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size) #escolha aleatoria dos exemplos
            states = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            rewards = np.array([i[2] for i in batch])
            next_states = np.array([i[3] for i in batch])
            terminals = np.array([i[4] for i in batch])

            # np.squeeze(): Remove single-dimensional entries from the shape of an array.
            # Para se adequar ao input
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)

            # usando o model para calcular os valores de q
            next_max = np.amax(self.model.predict_on_batch(next_states), axis=1)
            
            targets = rewards + self.gamma * (next_max) * (1 - terminals)
            targets_full = self.model.predict_on_batch(states)
            indexes = np.array([i for i in range(self.batch_size)])
            
            # usando o modelo alvo para estimar os q-valores
            targets_full[[indexes], [actions]] = targets
            self.model.fit(states, targets_full, epochs=1, verbose=0)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec

    def train(self):
        rewards = []
        for i in range(self.episodes+1):
            state = self.env.reset()
            state = np.reshape(state, (1, self.env.observation_space.shape[0]))
            score = 0
            max_steps = 200 # a condicao de parada para cada env pode mudar. lunarland = 3000
            for _ in range(max_steps):
                action = self.select_action(state)
                self.env.render()
                next_state, reward, terminal, _ = self.env.step(action)
                score += reward
                next_state = np.reshape(next_state, (1, self.env.observation_space.shape[0]))
                self.experience(state, action, reward, next_state, terminal)
                state = next_state
                self.experience_replay()
                if terminal:
                    print(f'Episódio: {i+1}/{self.episodes}. Score: {score}')
                    break
            rewards.append(score)

            # a cada N episodios, atualiza os pesos do target_model copiando os pesos do model
            if i % 50 == 0:
                print('fazendo a copia dos pesos entre as redes')
                self.target_model.set_weights(self.model.get_weights())

        return rewards
