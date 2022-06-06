# Deep Q-Learning

Leia o material disponível [aqui](../../slides/deep_reinforcement_learning.md). 

# Deep Reinforcement Learning with Double Q-learning

Leia o material disponível [aqui](../../slides/double_deep_reinforcement_learning.md).

# Descrição dos scripts encontrados neste diretório

* [DeepQLearning.py](./DeepQLearning.py): implementação do algoritmo Deep Q-Learning.
* [DoubleDeepQLearning.py](./DoubleDeepQLearning.py): implementação do algoritmo Double Deep Q-Learning.
* [CartPole_introduction.py](./CartPole_introduction.py): mostra como o ambiente CartPole funciona.
* [LunarLander_introduction.py](./LunarLander_introduction.py): mostra como o ambiente LunarLander funciona. 
* [CarPole.py](./CartPole.py): treina um agente usando o algoritmo Deep Q-Learning para atuar no ambiente CartPole. 
* [LunarLander.py](./LunarLander.py): treina um agente usando o algoritmo Deep Q-Learning para atuar no ambiente LunarLander.
* [CarPole_DDQ.py](./CartPole_DDQ.py): treina um agente usando o algoritmo Double Deep Q-Learning para atuar no ambiente CartPole. 
* [LunarLander_DDQ.py](./LunarLander_DDQ.py): treina um agente usando o algoritmo Double Deep Q-Learning para atuar no ambiente LunarLander.

* [CartPole_DQNAgent.py](./CartPole_DQNAgent.py): treina um agente usando a implementação do algoritmo Double Deep Q-Learning da biblioteca RL do Keras para atuar no ambiente CartPole.
* [LunarLander_DQNAgent.py](./LunarLander_DQNAgent.py): treina um agente usando a implementação do algoritmo Double Deep Q-Learning da biblioteca RL do Keras para atuar no ambiente LunarLander. Esta versão está um pouco diferente da anterior. Mas está melhor estruturada e documentada. Pode ser utilizada como implementação base que recebe diferentes environments. 

As implementações [CartPole_trained.py](./CartPole_trained.py) e [LunarLander_trained.py](./LunarLander_trained.py) utilizam os modelos treinados. No entanto, a versão do CartPole usa o modelo *from scratch*, enquanto que a versão LunarLander usa o modelo gerado pela biblioteca RL do Keras. Sugiro considerar fortemente o uso desta última versão nos seus projetos. 

Os diretórios *data* e *results* são utilizados para armazenar os modelos e dados sobre o processo de treinamento, respectivamente. 