import numpy as np
import gym

#from keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras import layers, Model
import tensorflow as tf

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
    #Network defined by the Deepmind paper
    inputs = layers.Input(shape=(1, 210, 160, 3))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(nb_actions, activation="linear")(layer5)

    return Model(inputs=inputs, outputs=action)


with tf.device("cpu:0"):

    #
    # configurando o environment e seeds
    ENV_NAME = 'ALE/Breakout-v5'
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # 
    # configurando e compilando o agente
    memory = SequentialMemory(limit=50000, window_length=1)
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
    # considerando que o render nao funciona neste ambiente,
    # eh necessario configurar visualize como False e render_mode como human para ver
    # a visualizacao da execucao
    env = gym.make(ENV_NAME, render_mode='human')
    dqn.test(env, nb_episodes=1, visualize=False)
