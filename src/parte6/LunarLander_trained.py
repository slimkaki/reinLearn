import gym
from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def createModel(nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(526))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model

ENV_NAME = 'LunarLander-v2'
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

memory = SequentialMemory(limit=500000, window_length=1)
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(), 
    attr='eps', 
    value_max=1., 
    value_min=.01, 
    value_test=.05, 
    nb_steps=10000)

model = createModel(nb_actions)

dqn = DQNAgent(
    model=model, 
    nb_actions=nb_actions, 
    memory=memory, 
    nb_steps_warmup=50,
    target_model_update=1e-2, 
    policy=policy,
    enable_double_dqn=True)
dqn.compile(Adam(lr=1e-3), metrics=['mse'])

dqn.load_weights('data/dqn_{}_weights.h5f'.format(ENV_NAME))
dqn.test(env, nb_episodes=5, visualize=True)