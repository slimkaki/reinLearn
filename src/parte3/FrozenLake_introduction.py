import gym
env = gym.make("FrozenLake-v1", is_slippery=True).env

#
# is_slippery=False torna o ambiente deterministico
#

print(env.observation_space.n)
print(env.action_space.n)

# inicializando o ambiente
state = env.reset()

# imprimindo o ambiente
print(env.render(mode='human'))

# eh possivel fazer env.render(mode='ansi') para imprimir no terminal 

print('Executando algumas acoes')

print('\n\n')

print('indo para baixo')
state, reward, done, info = env.step(1)
print(env.render(mode='human'))
print(reward, done)

print('indo para baixo')
state, reward, done, _ = env.step(1)
print(env.render(mode='human'))
print(reward, done)

print('indo para direita')
state, reward, done, info = env.step(2)
print(env.render(mode='human'))
print(reward, done)

print('indo para direita')
state, reward, done, info = env.step(2)
print(env.render(mode='human'))
print(reward, done)

print('indo para baixo')
state, reward, done, info = env.step(1)
print(env.render(mode='human'))
print(reward, done)

print('indo para direita')
state, reward, done, info = env.step(2)
print(env.render(mode='human'))
print(reward, done)
