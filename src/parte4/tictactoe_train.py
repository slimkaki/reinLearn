from kaggle_environments import make
import random

env = make("tictactoe", debug=True)

# treinamento o agente (na posicao 1) contra o agente random
trainer = env.train([None, "random"])

episodes = 1
obs = trainer.reset()
for _ in range(episodes):
    done = False
    while not done:
        #env.render()
        action = random.randint(0,8)
        print(f'action = {action}')
        obs, reward, done, info = trainer.step(action)
        #
        # TODO usar obs, action e done para treinar o agente
        #
        print(f'obs = {obs}')
        print(f'reward = {reward}')
        print(f'done = {done}')
        print(f'info = {info}')
        print('\n\n')
    