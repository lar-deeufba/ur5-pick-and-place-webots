# UR5 Pick and Place

<b>Autor:</b> <a href="https://orcid.org/my-orcid?orcid=0009-0006-2253-4195" target="_blank">Allan Almeida</a>

## Visão Geral

Este projeto implementa uma aplicação _pick and place_ com o manipulador robótico UR5e no Webots.

A cinemática direta e a inversa, bem como todas as funções de controle, são implementadas em Python. O robô é controlado enviando velocidades para as juntas usando 
uma trajetória polinomial de quinta ordem. O robô é capaz de pegar uma garrafa de uma mesa e colocá-la em outra mesa (e também dar um gole para o manequim :grin: :beer:).

A parte de visão computacional do projeto é implementada utilizando uma CNN (rede neural convolucional). A CNN é treinada para detectar a posição da garrafa na imagem. A rede é implementada em Tensorflow e Keras, e utiliza um modelo pré-treinado VGG16 como base. A rede é modificada para prever a posição da garrafa em relação à imagem e convertê-la para as coordenadas reais XYZ 
usando interpolação bilinear. Ela é treinada e avaliada em um conjunto de dados de 5000 imagens.

<a href="https://youtu.be/XgvYlSNmqiI">Vídeo de demonstração</a>

## Dependências

- Webots
- Python >= 3.6
- Jupyter Notebook (Anaconda ou pip)

## Utilização

1. Abra o Webots e carregue o mundo `my_first_simulation.wbt`. Alternativamente, você pode abrir o Webots e carregar o mundo apenas executando o _shell script_:
   ```
   ./launch.sh
   ```
2. Crie um ambiente virtual python e instale as dependências do arquivo `requirements.txt`
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Abra o _notebook_ Jupyter `Trabalho.ipynb` e execute a primeira célula para importar as dependências e iniciar a simulação
4. Execute as demais células, uma a uma, para ver o robô em ação
5. Você pode usar as funções do arquivo `ur5.py` para controlar o robô e executar outras tarefas que desejar

Se divirta! :sparkles: :robot:
