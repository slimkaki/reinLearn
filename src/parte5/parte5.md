# Atividades parte 5

Até este momento, trabalhamos com diversos ambientes que tem uma quantidade razoavelmente pequena de estados e ações discretas. O objetivo deste exercício é mostrar o uso do Q-Learning em um cenário onde é necessário discretizar o espaço de estados. 

Para isto, vamos utilizar o ambiente [MountainCar-v0](https://www.gymlibrary.ml/environments/classic_control/mountain_car/) da bibliote Gym. Neste ambiente temos que aprender a controlar um carro que precisar sair da base de uma montanha e chegar no topo da mesma.

<img src="mountain_car.gif" alt="Mountain car environment" width="300"/>

As ações são discretas: 

* 0: acelera para a esquerda.
* 1: não acelera.
* 2: acelera para a direita. 

Dado uma ação, o carro se move segundo esta dinâmica: 

$velocity_{t+1} = velocity_{t} + (action - 1) * force - cos(3 * position_{t}) * gravity$

$position_{t+1} = position_{t} + velocity_{t+1}$

onde $force = 0.001$ e $gravity = 0.0025$. 

* Execute o arquivo [MountainCar.py](MountainCar.py). O veículo consegue chegar no topo da montanha? O que está acontecendo? O que precisa ser alterado?