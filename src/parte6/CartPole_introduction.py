import gym

#
# Documentacao disponivel em: https://www.gymlibrary.ml/environments/classic_control/cart_pole/
#

env = gym.make('CartPole-v1').env
input_shape = env.observation_space.shape
n_outputs = env.action_space.n

print(f'input shape = {input_shape}')
print(f'n_outputs = {n_outputs}')

env.reset()
for _ in range(1000):
    env.render()
    state, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(state)
env.close()