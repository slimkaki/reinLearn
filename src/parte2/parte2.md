# Atividades parte 2

* Se $\alpha$ for um valor muito próximo de zero? Explique o comportamento encontrado.

* Se $\gamma$ for zero? Explique o comportamento encontrado. 

* O que acontece se a escolha das ações em cada estado for sempre aleatória? Ou seja, se a função `select_action` ao invés de ser definida como abaixo:

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

Outra forma seria configurar `epsilon=1` e `epsilon_min=1`. Explique o comportamento encontrado. 

* O que acontece se a escolha das ações em cada estado for sempre buscando a melhor ação? Ou seja:

````python
def select_action(self, state):
    return np.argmax(self.q_table[state]) # Exploit learned values
````

Explique o comportamento encontrado em ambos os casos quando executamos `TaxiDriverGym.py`. 
