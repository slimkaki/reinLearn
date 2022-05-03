import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

env = gym.make('CartPole-v1').env
input_shape = env.observation_space.shape
n_outputs = env.action_space.n

#
# definicao da DNN
model = keras.models.Sequential([
    keras.layers.Dense(32, activation='elu', input_shape=input_shape),
    keras.layers.Dense(32, activation='elu'),
    keras.layers.Dense(n_outputs)
])

#
# selecao da acao via exploration ou explotation
def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

from collections import deque
replay_buffer = deque(maxlen=2000)

def sample_experience(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

#
# criacao de hiperparametros, do otimizador e da funcao de perda
batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.mean_squared_error


def training_step(batch_size):
    experiences = sample_experience(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, n_outputs)

    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


rewards_per_episode = []
for episode in range(600):
    obs = env.reset()
    rewards = 0
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        rewards += reward
        if done:
            break
    if episode > 50:
        training_step(batch_size)
    
    print(episode, rewards)
    rewards_per_episode.append(rewards)

import matplotlib.pyplot as plt
plt.plot(rewards_per_episode)
plt.xlabel('Episodes')
plt.ylabel('# Rewards')
plt.title('# Rewards vs Episodes')
plt.savefig("results/cartpole_v1.jpg")     
plt.close()

#state = env.reset()
#done = False
#while not done:
#    action = np.argmax(q_table[state])
#    state, reward, done, info = env.step(action)