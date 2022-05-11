# Atividades parte 3

O ambiente Frozen Lake é um ambiente não determinístico onde um agente deve encontrar um caminho do lugar onde ele está para outro lugar passando por buracos. Se ele chegar no objetivo sem cair no buraco então ele termina a tarefa e tem 1 ponto de reward. Se ele cair em um dos buracos então ele termina a tarefa com 0 pontos de reward. Cada ação que não leva para um estado terminal tem reward igual a 0.  

Neste ambiente o agente consegue executar 4 ações: ir para cima, ir para baixo, ir para esquerda e ir para direita. Como o chão é de gelo, não necessariamente a ação de ir para baixo vai levar o agente para baixo, por exemplo. Isto acontece com todas as quatro ações. Por isso que este ambiente é não determinístico.

Atividades sugeridas:

## Trabalhe com o arquivo [FrozenLake_introduction.py](./FrozenLake_introduction.py)

1. Leia a documentação do código fonte disponível em [https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)

2. Veja o que está codificado no arquivo e execute o mesmo.

3. Quantos estados e quantas ações o ambiente FrozenLake-v1 tem?

4. O que aconteceu com a execução das ações? O resultado foi o esperado? Descreva o que aconteceu.

## Trabalhe com o arquivo [FrozenLake.py](./FrozenLake.py)

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

* Agora faça o algoritmo [FrozenLake.py](./FrozenLake.py) ler a Q-table a partir do arquivo gerado anteriormente e veja qual é o comportamento. Execute diversas vezes. Ele consegue chegar ao objetivo sempre? Ele consegue chegar ao objetivo na maioria das vezes? 

* E se executarmos 100 vezes? Quantas vezes o agente consegue atingir o objetivo? Execute o comando abaixo:

````bash
python FrozenLake100times.py
````

* Como podemos melhorar o desempenho deste agente?

* Teste diferentes configurações de hiperparâmetros. Qual é o comportamento visto no gráfico de episódios versus rewards? 

## Outro mapa

Existem dois mapas pré-configurados em [https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py). O mapa 4x4 e um mapa 8x8. E se mudarmos o mapa para 8x8? 

````python
import gym
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True).env
````

* O que muda? O problema se torna mais complexo? É necessário mudar algum dos hiperparâmetros? 

