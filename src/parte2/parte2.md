# Atividades parte 2

Ainda considerando o exemplo a implementação do `TaxiDriver`, responda as perguntas abaixo.

## Manipulando $\alpha$ e $\gamma$

* Se $\alpha$ for um valor muito próximo de zero? Explique o comportamento encontrado.

* Se $\gamma$ for zero? Explique o comportamento encontrado. 

## Considerando uma escolha de ação sempre aletatória

O que acontece se a escolha das ações em cada estado for sempre aleatória? Ou seja, se a função `select_action` ao invés de ser definida como abaixo:

````python
def select_action(self, state):
    rv = random.uniform(0, 1)
    if rv < self.epsilon:
        return self.env.action_space.sample() # Explore action space
    return np.argmax(self.q_table[state]) # Exploit learned values
````

É definida assim:

````python
def select_action(self, state):
    return self.env.action_space.sample() # Explore action space
````

Qual o comportamento do agente? 

## Considerando um agente que nunca explora novas ações

O que acontece se a escolha das ações em cada estado for sempre buscando a melhor ação? Ou seja:

````python
def select_action(self, state):
    return np.argmax(self.q_table[state]) # Exploit learned values
```` 

Para responder as questões acima utilize as implementações do `TaxiDriverGym.py` e `QLearning.py` que estão neste diretório. 

## Sumarizando os resultados através de imagens

Como podemos sumarizar os diferentes resultados através de imagens?

Implemente gráficos que mostrem o aprendizado (ou não) do agente ao longo dos episódios. 