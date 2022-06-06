import gym
from tensorflow import keras
import numpy as np
import tensorflow as tf

# para visualizar o ambiente render_mode='human'
env = gym.make("ALE/Breakout-v5", render_mode='human')
state = env.reset()
model = keras.models.load_model('data/model_breakout', compile=False)

done = False
rewards = 0
steps = 0

while not done:
    Q_values = model.predict(state[np.newaxis], verbose=0)
    action = np.argmax(Q_values[0])
    state, reward, done, info = env.step(action)
    print(action)
    rewards += reward
    #env.render()
    steps += 1

print(f'Score = {rewards}')
input('press a key...')