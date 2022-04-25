

### Trabalhe com o arquivo [FrozenLake_introduction.py](src/FrozenLake_introduction.py)

* Leia a descrição do ambiente em [https://github.com/openai/gym/wiki/FrozenLake-v0](https://github.com/openai/gym/wiki/FrozenLake-v0).

* Veja o que está codificado no arquivo e execute o mesmo.

* Quantos estados e quantas ações o ambiente FrozenLake-v1 tem?

* O que aconteceu com a execução das ações? O resultado foi o esperado? Descreva o que aconteceu.

### Trabalhe com o arquivo [FrozenLake.py](src/FrozenLake.py)

* Abra em um editor de texto e descomente as linhas 12 e 13 e comente a linha 14. O código deve ficar como abaixo:

````python
# only execute the following lines if you want to create a new q-table
qlearn = QLearning(env, alpha=0.9, gamma=0.95, epsilon=0.8, epsilon_min=0.0001, epsilon_dec=0.9999, episodes=500000)
q_table = qlearn.train('data/q-table-frozen-lake.csv','results/actions_frozen_lake')
#q_table = loadtxt('data/q-table-frozen-lake.csv', delimiter=',')
````

* Execute o arquivo [FrozenLake.py](src/FrozenLake.py) com o comando:

````bash
python FrozenLake.py
````

* Agora faça o algoritmo [FrozenLake.py](src/FrozenLake.py) ler a Q-table a partir do arquivo gerado anteriormente e veja qual é o comportamento. Execute diversas vezes. Ele consegue chegar ao objetivo sempre? Ele consegue chegar ao objetivo na maioria das vezes? 


## Atividades pós-aula


### MountainCar.py

* Leia a descrição do ambiente em [https://gym.openai.com/envs/MountainCarContinuous-v0/](https://gym.openai.com/envs/MountainCarContinuous-v0/).

Execute o arquivo [MountainCar.py](src/MountainCar.py) com o comando:

````bash
python MountainCar.py
````

* Antes de mostrar o carro em movimento, a solução imprime um log. Cada linha termina com o texto "Actions in this episode NNN". Por que este texto começa com 200 e espera-se que ao longo da execução este número vá diminuindo?

* A movimentação do carro aconteceu? Foi possível ver o carro se movimentando uma interface gráfica? O resultado apresentado mostrou um carro capaz de chegar no objetivo? 

### Implementação de Deep Q-Learning

Ler o capítulo *18. Reinforcement Learning* até a seção *Implementing Deep Q-Learning* do livro [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). 

## Referências

- https://gym.openai.com/
- https://github.com/openai/gym/wiki/FrozenLake-v0
- https://github.com/openai/gym/wiki/MountainCar-v0
- https://gym.openai.com/envs/LunarLander-v2/


