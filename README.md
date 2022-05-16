# Aprendizagem por Reforço 

Este projeto possui diversos materiais sobre aprendizagem por reforço (*reinforcement learning*). 

Você deve fazer o download do projeto em uma máquina que tenha interpretador python. É possível 
executar os mesmos códigos em ambientes como Colab. No entanto, ao executar em ambientes como Colab algumas funcionalidades de animação ficam comprometidas. 

## Configuração do ambiente

Recomenda-se criar um ambiente virtual (*virtualenv*) usando python 3.7:

````bash
python3.7 -m virtualenv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
````

Para sair do ambiente virtual, digite `deactivate`. Se você já possui o ambiente virtual configurado basta digitar `source venv/bin/activate`. 

## Pontos de atenção

Se você tiver problemas para instalar o pacote Box2D então, provavelmente, a solução será instalar o software swig na sua máquina. Para máquina Linux basta instalar via apt-get:

````bash
sudo apt-get install swig
````

Para máquinas com Mac via `brew` e para máquinas com Windows eu não tenho a mínima ideia. 

## Instalando o Arcade Learning Environment

Para instalar o Arcade Learning Environment (ALE) é necessário instalar o autorom da seguinte forma: 

````bash
pip install autorom[accept-rom-license]
````

**Este formato de instalação via pip só funciona no bash!** 

Mais informações podem ser obtidas [aqui](https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management)

