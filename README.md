# Classificação da maturação de frutos de café com MLP

Trabalho desenvolvido para a disciplina **Tendências em Inteligência Computacional — 2026.2**, ministrada pelo professor **Rodrigo Gabriel Ferreira Soares**.

## Integrantes

- Lucas Medrado Santos
- Álvaro Brandão Neto
- João Pedro Mafaldo de Paula

## Objetivo

Implementar uma rede neural Multilayer Perceptron (MLP) do zero com NumPy para classificar frutos de café em cinco categorias de maturação:

- Seco
- Passado do ponto
- Maduro
- Semimaduro
- Verde

A classificação utiliza características de cor extraídas das regiões dos frutos delimitadas pelas anotações da base.

## Tecnologias utilizadas

- **Python:** linguagem de programação.
- **NumPy:** implementação da MLP e cálculos do treinamento.
- **Pillow:** leitura das imagens e criação das máscaras dos frutos.
- **scikit-learn:** padronização dos dados e cálculo das métricas.
- **Matplotlib:** geração dos gráficos.
- **Google Colab:** execução do notebook.

A rede neural foi implementada com NumPy, sem utilizar modelos prontos do scikit-learn, PyTorch ou TensorFlow.

## Base de dados

Foi utilizada a base [Coffee Fruit Maturity](https://www.kaggle.com/datasets/cienciacafeto/coffee-fruit-maturity), contendo **859 fotografias** e anotações dos contornos dos frutos. Uma fotografia pode conter vários frutos.

O arquivo de configuração da versão utilizada identifica o projeto [Ciencia Cafeto no Roboflow — versão 1](https://universe.roboflow.com/ciencia-cafeto/coffee-fruit-maturity-befkg/dataset/1) e informa a licença **CC BY 4.0**.

Foi preservada a divisão original da base:

| Conjunto | Quantidade de frutos processados |
|---|---:|
| Treino | 7.483 |
| Validação | 2.411 |
| Teste | 1.149 |
| **Total** | **11.043** |

### Preparação dos dados

Para cada fruto, foram calculadas seis características, considerando apenas os pixels dentro do contorno anotado:

1. Média do canal vermelho.
2. Média do canal verde.
3. Média do canal azul.
4. Desvio-padrão do canal vermelho.
5. Desvio-padrão do canal verde.
6. Desvio-padrão do canal azul.

As características foram padronizadas com `StandardScaler`. Os parâmetros da padronização foram calculados exclusivamente no conjunto de treino e aplicados à validação e ao teste.

## Implementação da MLP

A implementação utiliza:

- Inicialização aleatória dos pesos com semente 42.
- ReLU nas camadas ocultas.
- Softmax na camada de saída.
- Entropia cruzada como função de perda.
- Backpropagation para calcular os gradientes.
- Gradiente descendente com momento para atualizar pesos e vieses.

O treinamento utiliza todo o conjunto de treino a cada época.

Os parâmetros configuráveis são:

| Parâmetro | Função |
|---|---|
| `camadas_ocultas` | Define a quantidade de camadas ocultas e seus neurônios |
| `taxa_aprendizado` | Controla o tamanho das atualizações dos pesos |
| `momento` | Incorpora parte das atualizações anteriores |
| `epocas` | Define a quantidade de ciclos de treinamento |

Por exemplo, `camadas_ocultas = [16, 8]` define duas camadas ocultas, com 16 e 8 neurônios.

## Como executar

1. Baixe a base [Coffee Fruit Maturity](https://www.kaggle.com/datasets/cienciacafeto/coffee-fruit-maturity) em formato ZIP.
2. Renomeie o arquivo para `coffee-fruit-maturity.zip`.
3. No Google Drive, crie a pasta `projeto_mlp_cafe` dentro de **Meu Drive**.
4. Envie o ZIP para essa pasta.
5. Acesse o [Google Colab](https://colab.research.google.com/).
6. Abra o notebook `mlp_maturacao_cafe.ipynb` deste repositório pela opção **GitHub**, ou baixe o notebook e faça seu upload no Colab.
7. Execute todas as células na ordem, autorizando a conexão com o Google Drive quando solicitado.

O caminho esperado pelo notebook é:

```text
/content/drive/MyDrive/projeto_mlp_cafe/coffee-fruit-maturity.zip
```

Se o arquivo estiver em outro local, ajuste a variável `arquivo_zip`.

Caso alguma dependência esteja ausente, execute uma célula com:

```python
%pip install numpy matplotlib pillow scikit-learn
```

Os dados extraídos em `/content/dados_cafe` são temporários. Ao iniciar uma nova sessão do Colab, execute novamente as células de carregamento.

## Experimentos

Foram comparadas três configurações, mantendo **momento de 0,9** e **500 épocas**.

| Configuração | Camadas ocultas | Taxa de aprendizado | Acurácia de validação | F1 macro de validação |
|---|---|---:|---:|---:|
| A | 16 e 8 | 0,01 | 90,5% | 0,543 |
| B | 8 | 0,01 | 90,4% | 0,529 |
| C | 16 e 8 | 0,05 | **91,7%** | **0,657** |

A configuração **C** foi selecionada pelo maior **F1 macro na validação**, métrica que atribui o mesmo peso a cada classe.

A arquitetura escolhida possui:

- **Entrada:** 6 características.
- **Primeira camada oculta:** 16 neurônios.
- **Segunda camada oculta:** 8 neurônios.
- **Saída:** 5 classes.

O conjunto de teste foi reservado para a avaliação final.

## Resultados no teste

O modelo selecionado obteve:

- **Acurácia:** 89,3%.
- **F1 macro:** 0,649.

| Classe | Precisão | Recall | F1-score | Quantidade |
|---|---:|---:|---:|---:|
| Seco | 0,500 | 0,704 | 0,585 | 27 |
| Passado do ponto | 0,400 | 0,421 | 0,410 | 19 |
| Maduro | 0,658 | 0,803 | 0,723 | 122 |
| Semimaduro | 0,696 | 0,447 | 0,545 | 123 |
| Verde | 0,980 | 0,986 | 0,983 | 858 |

Os gráficos apresentados no notebook incluem:

- Comparação do F1 macro das configurações.
- Perda de treino e validação por época.
- Acurácia de treino e validação por época.
- Matriz de confusão no conjunto de teste.

## Conclusões e limitações

A MLP apresentou o melhor desempenho na classe Verde, com F1-score de 0,983. As maiores dificuldades foram observadas na classe Passado do ponto, com F1-score de 0,410, e no reconhecimento de frutos Semimaduros, cujo recall foi de 44,7%.

A base apresenta desequilíbrio entre as classes. Os frutos verdes representam aproximadamente 74,7% do conjunto de teste. Uma previsão constante de Verde alcançaria aproximadamente 74,7% de acurácia, o que reforça a importância de analisar também o F1 macro e as métricas por classe.

As classes Seco e Passado do ponto possuem poucos exemplos no teste, limitando a estabilidade das estimativas de desempenho.

O sistema depende das anotações para delimitar os frutos e não realiza sua localização automática em fotografias novas. Como utiliza características de cor, mudanças de iluminação e de variedade do café podem afetar as previsões.

Os resultados representam o desempenho na divisão de dados utilizada, sem garantir o mesmo desempenho em outras condições de captura.
