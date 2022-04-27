# Atividades parte 4: quebrando a banca

## Jogador de Roleta simplificado (Roulette-v0)

Neste ambiente o agente pode jogar um número entre 0 e 36 em um ambiente modificado de casino.
Para cada rodada da roleta, o agente aposta em um número. O agente recebe uma recompensa de 35 pontos se ele apostar no 0 e o 0 for sorteado. O agente recebe uma recomensa de 1 ponto se ele apostar em um número e a paridade (par ou ímpar) do número sorteado for a mesma que o número escolhido pelo agenda. Em qualquer outra situação o agente recebe uma recompensa de -1. Além de escolher entre 0 e 36, o agente também pode escolher 37 que significa sair do jogo.

O arquivo [Roulette_introduction.py](./Roulette_introduction.py) mostra como utilizar este ambiente. Um agente utilizando Reinforcement Learning está implementado em [Roulette.py](Roulette.py).

* Execute algumas vezes o arquivo [Roulette_introduction.py](./Roulette_introduction.py) para entender como o ambiente funciona. 

* Execute o arquivo [Roulette.py](Roulette.py). Qual foi o resultado encontrado? O resultado apresentado faz sentido? Os hiperparâmetros configurados são os melhores? 

## Jogador de BlackJack

A biblioteca Gym possui um ambiente que simula um jogo de BlackJack (*Blackjack-v1*). Neste link [https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py](https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py) você tem acesso ao código fonte e a documentação sobre as regras da versão de BlackJack implementadas neste ambiente. 

Além da documentação, você tem acesso a duas implementações:

* [BlackJack_Manual.py](./BlackJack_Manual.py): onde você pode jogar várias partidas de BlackJack e entender a representação de estado adotada pelo ambiente, e;
* [BlackJack_Agent.py](./BlackJack_Agent.py): que tem uma implementação de agente que aprende a jogar BlackJack usando aprendizagem por reforço. 

Atividades propostas: 

* Execute diversas vezes o arquivo [BlackJack_Manual.py](./BlackJack_Manual.py) para entender como o ambiente funciona. 

* Execute o arquivo [BlackJack_Agent.py](./BlackJack_Agent.py) com o objetivo de criar uma nova q-table. Qual o desempenho do agente? 

* Como podemos obter um agente com o melhor desempenho possível? É possível criar um agente que ganha ou empata em no mínimo 90% dos jogos? Se sim, quais são os hiperparâmetros para este agente? Se não, qual é o melhor resultado encontrado? 