import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
from rl.callbacks import FileLogger

from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

#
# criando uma funcao de callback para armazenar o log e os pesos do modelo
# durante o treinamento do mesmo
def build_callbacks(env_name):
    checkpoint_weights_filename = 'data/dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'results/dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
    callbacks += [FileLogger(log_filename, interval=1000)]
    return callbacks

#
# criando um modelo específico para o problema
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

#
# configurando o environment e seeds
ENV_NAME = 'LunarLander-v2'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# 
# configurando e compilando o agente
memory = SequentialMemory(limit=500000, window_length=1)
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(), 
    attr='eps', 
    value_max=1., 
    value_min=.01, 
    value_test=.05, 
    nb_steps=10000)

model = createModel(nb_actions)
print(model.summary())

dqn = DQNAgent(
    model=model, 
    nb_actions=nb_actions, 
    memory=memory, 
    nb_steps_warmup=50,
    target_model_update=1e-2, 
    policy=policy,
    enable_double_dqn=True)

dqn.compile(Adam(lr=1e-3), metrics=['mse'])
callbacks = build_callbacks(ENV_NAME)
# para interromper o processo basta teclar Ctrl+C
dqn.fit(env, nb_steps=500000, visualize=False, verbose=2, callbacks=callbacks)

# depois de finalizado o treinamento sao salvos os pesos finais
dqn.save_weights('data/dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# avaliacao do modelo considerando 5 episodios
dqn.test(env, nb_episodes=5, visualize=True)
