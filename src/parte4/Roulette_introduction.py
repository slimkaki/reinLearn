import gym
import gym_toytext

#
# O ambiente roulette foi descontinuado no projeto Gym. Mas eh mantido
# em https://github.com/Rohan138/gym-legacy-toytext
#
# para usar este ambiente eh necessario instalar o pacote
# gym-legacy-toytext - que jah estah no arquivo de requirements.txt deste projeto
#

env = gym.make('Roulette-v0').env
env.reset()
done = False
rewards = 0
while not done:
    action = int(input("Escolha entre 0 e 36. Digite 37 para sair "))
    state, reward, done, info = env.step(action)
    print(f' Voce ganhou: {reward}')
    rewards += reward

print(f'Total de pontos = {rewards}')