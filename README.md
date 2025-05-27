## Descrição do Problema
- Basicamente o problema, era divido em duas partes, pré processamento das imagens  do dataset e treinamento de um modelo para classificar minhas imagens selecionadas.


## Técnicas Utilizadas
  - **carregar_imagens**: Função que ira carregar as imagens de gato e cachorros que escolhi aleatoriamente, sendo 6 de cada.  Essa função, possui três parâmetros, pasta das imagens, extensões permitidas e tamanho das imagens. Ela ira encontrar a pasta com as imagens, iterar sobre cada imagem aplicando a função de pre_processar_imagem que ira redimensionar a imagem no tamanho especificado na prova (128x128) e aplicar um filtro gaussiano de blur para reduzir os ruídos da imagens.
  
  - **pre_processar_imagem**: Função que ira aplicar o redimensionamento nas imagens e o filtro gaussiano no momento do carregamento das minhas imagens.
  
  - **ajustar_tamanho_cnn**: Função usada para redimensionar as imagens do dataset carregado na cnn. Recebe a imagem e o tamanho que sera redimensioando.
  
  - **preparar_imagem_cnn**: Nessa função as imagens são transformadas em RGB e redimensionadas para ficarem no intervalo de 0 e 1, sendo melhor para o treinamento nas CNN's.

 - **carregar_dados**: Nessa função, serão carregadas as imagens do dataset onde já ocorrerá a divisão da base entre treino e teste.

- **construir_modelo**: Essa função será responsável por construir as camadas da minha rede, onde foram aplicados três conversões para 2d para filtrar a minha imagem em 32 pixels, a função de maxpooling para extrair as melhores características, a camada de Flatten para achatar minhas saídas antes de passar para minha rede, uma camada de Dense(64, activation='relu'), no qual sera definido os 64 neurônios da minha rede e por fim a ultima camada de dense que de fato sera a minha camada de saída, possuindo 2 neurônios (gato e cachorro).

- **treinar_modelo**: Camada onde fato serão definidas alguns parâmetros do meu modelo, como o numero de épocas, o otimizador dos pesos na rede (Adam) e a função de perda.

- **avaliar_modelo**: Irá realizar o predict após o modelo treinado para que seja possível verificar as estatiticas do modelo pelo método do próprio scikit learning classification_report.

## Etapas realizadas 
- Realizei o pré processamento das imagens, depois criei uma rede convolucional para a partir dos dados do dataset informado na prova, treinar um modelo capaz de classificar as imagens que selecionei e processei anteriomente.

## Resultados Obtidos
- Obtive um média de 75% para classificar os cachorros e de 77% para os gatos
  ![[Pasted image 20250526211343.png]]
## Tempo Total Gasto
- 2:14 horas.

## Dificuldades Encontradas
- As dificuldades maiores, foram na parte de redimensionar a minha base e treinar meu modelo.