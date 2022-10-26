# Atividades parte 4: quebrando a banca

## Jogador de BlackJack

A biblioteca Gym possui um ambiente que simula um jogo de [BlackJack (*Blackjack-v1*)](https://www.gymlibrary.dev/environments/toy_text/blackjack/). A documentação deste ambiente está disponível neste link [https://www.gymlibrary.dev/environments/toy_text/blackjack/](https://www.gymlibrary.dev/environments/toy_text/blackjack/). 

Além da documentação, você tem acesso a duas implementações:

* [BlackJack_Manual.py](./BlackJack_Manual.py): onde você pode jogar várias partidas de BlackJack e entender a representação de estado adotada pelo ambiente, e;
* [BlackJack_Agent.py](./BlackJack_Agent.py): que tem uma implementação de agente que aprende a jogar BlackJack usando aprendizagem por reforço. 

Atividades propostas: 

* Execute diversas vezes o arquivo [BlackJack_Manual.py](./BlackJack_Manual.py) para entender como o ambiente funciona. Principalmente como a representação do espaço de estados funciona. 

* Execute o arquivo [BlackJack_Agent.py](./BlackJack_Agent.py) com o objetivo de criar uma nova q-table. Qual o desempenho do agente? 

* Como podemos obter um agente com o melhor desempenho possível? É possível criar um agente que ganha ou empata em no mínimo 90% dos jogos? Se sim, quais são os hiperparâmetros para este agente? Se não, qual é o melhor resultado encontrado? 

## Jogador de jogo da velha

Para esta atividade vamos utilizar a biblioteca `kaggle_environments`. Sendo assim, temos que fazer um update no nosso `virtualenv`. Para fazer este update execute o seguinte comando no diretório raiz do projeto: 

```bash
pip install -r requirements_parte4.txt
```

* Depois de instalada a biblioteca `kaggle_environments`, execute o notebook jupyter `tictactoe_env.ipynb` e entenda o que está acontecendo. 

* O arquivo `tictactoe_train.py` apresenta uma estrutura básica para o loop de aprendizado de um agente. Que modificações são necessárias neste arquivo para fazer com que um agente aprenda a jogar *tictactoe* usando aprendizagem por reforço? 

### Referência sobre a biblioteca do kaggle para reinforcement learning

1. Link para o repositório no github: [https://github.com/Kaggle/kaggle-environments](https://github.com/Kaggle/kaggle-environments).
1. Tutorial no Kaggle: [https://www.kaggle.com/code/tarunbisht11/get-started-with-kaggle-environment](https://www.kaggle.com/code/tarunbisht11/get-started-with-kaggle-environment).