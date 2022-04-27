#
# O objetivo deste arquivo eh mostrar como funciona o
# environment do black jack.
#
# Neste ambiente o jogador joga contra o dealer e soh
# pode realizar duas acoes:
# - pedir por mais cartas (1)
# - parar (0)
#
# O ambiente eh representado por uma 3-tuple:
# (
#   soma das cartas do jogador, 
#   qual a carta o dealer estah mostrando, 
#   se o jogador estah segurando um Ás ou nao)
#

import gym
env = gym.make("Blackjack-v1")
print('Actions: '+ str(env.action_space))
print('Spaces: '+ str(env.observation_space))

print('\n')
print('Inicializando o jogo...')
state = env.reset()
print(state)
done = False

while not done:
    action = int(input("Pedir por mais cartas (1) ou parar (0)? "))
    state, reward, done, info = env.step(action)
    print(f'Soma das cartas do meu jogador: {state[0]}')
    print(f'Carta que o dealer está mostrando: {state[1]}')
    print(f'O meu jogador tem um Ás? {state[2]}')

print('\n')
if reward == 1:
    print('Meu jogador venceu')
elif reward == 0:
    print('Jogo empatou')
elif reward == -1:
    print('Dealer ganhou')
    
print('Mão do dealer' + str(env.dealer))
