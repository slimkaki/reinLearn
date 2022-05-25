from tabnanny import verbose
import gym
from tensorflow import keras
import numpy as np

env = gym.make('CartPole-v0').env
state = env.reset()
model = keras.models.load_model('data/model_cart_pole', compile=False)
done = False
rewards = 0
steps = 0

while not done and steps < 250:
    Q_values = model.predict(state[np.newaxis], verbose=0)
    action = np.argmax(Q_values[0])
    state, reward, done, info = env.step(action)
    rewards += reward
    env.render()
    steps += 1

print(f'Score = {rewards}')
input('press a key...')