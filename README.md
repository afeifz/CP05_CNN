# Checkpoint 1 — CNN + Transfer Learning

## Projeto

Aplicação de Redes Neurais Convolucionais (CNN) com Transfer Learning para classificação de imagens utilizando o **PneumoniaMNIST**, pertencente ao conjunto **MedMNIST**.

### Integrantes

- Nome: Mohamed Afif — RM: 554445
- Nome: Lucca Cardinale — RM: 556668


## Dataset

Foi utilizado o **PneumoniaMNIST**, um dataset do MedMNIST para classificação binária de imagens de raio-X do tórax.

As duas classes utilizadas pelo projeto são:

- `0` — Normal
- `1` — Pneumonia

O dataset é dividido em conjuntos de treinamento, validação e teste.

> Observação: este projeto tem finalidade acadêmica. Os resultados do modelo não devem ser utilizados para diagnóstico médico.

## Backbone escolhido

O backbone utilizado foi o **MobileNetV2**, com pesos pré-treinados no **ImageNet**.

A escolha foi feita porque o MobileNetV2 possui uma arquitetura convolucional eficiente e relativamente leve, permitindo executar Transfer Learning em ambiente como o Google Colab sem exigir uma quantidade excessiva de recursos computacionais.

## Estratégia de Transfer Learning

O treinamento foi realizado em duas etapas.

### Etapa 1 — Backbone congelado

Inicialmente, todas as camadas do MobileNetV2 foram congeladas. Foi adicionada uma nova cabeça de classificação composta por:

- Global Average Pooling;
- Dropout;
- camada Dense com 64 neurônios e ReLU;
- Dropout;
- camada de saída com 1 neurônio e ativação sigmoid.

Nesta etapa foi utilizado Adam com learning rate `1e-3`.

### Etapa 2 — Fine-Tuning

Após o treinamento inicial, o backbone foi parcialmente descongelado.

Foram liberadas para treinamento as últimas 30 camadas do MobileNetV2, enquanto as camadas anteriores permaneceram congeladas. As camadas BatchNormalization foram mantidas congeladas para reduzir instabilidade durante o ajuste.

Nesta etapa foi utilizado Adam com learning rate `1e-5`.

## Pré-processamento

As imagens originais do PneumoniaMNIST possuem resolução de 28×28 pixels e são monocromáticas.

Para utilização com o MobileNetV2:

1. As imagens foram convertidas para `float32`;
2. O canal de escala de cinza foi replicado para formar uma imagem RGB;
3. As imagens foram redimensionadas para 96×96 pixels;
4. Foi aplicado o `preprocess_input` do MobileNetV2;
5. Durante o treinamento foram aplicadas técnicas simples de data augmentation.

## Métricas

O treinamento acompanha:

- Accuracy;
- Loss;
- AUC.

No conjunto de teste também é apresentado:

- Classification Report;
- Matriz de Confusão.

## Resultados


**Melhor Validation Accuracy:** 0.9370

**Test Accuracy:** 0.1576

**Test AUC:** 0.8606


## Estrutura da pasta

```text
CP05_CNN/
│
├── codigo/
│   └── checkpoint1_pneumoniamnist_mobilenetv2.ipynb
│
├── resultados/
│   ├── exemplos_dataset.png
│   ├── curva_accuracy.png
│   ├── curva_loss.png
│   ├── curva_auc.png
│   └── matriz_confusao.png
│
└── README.md
```

## Como executar

1. Abrir o arquivo `.ipynb` no Google Colab.
2. Ativar o ambiente com GPU em:
   `Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU` (se disponível).
3. Executar as células em ordem.
4. O dataset será baixado automaticamente pelo MedMNIST.
5. O modelo será treinado em duas etapas.
6. Os gráficos serão salvos na pasta `results`.
7. O modelo será salvo no formato `.keras` dentro da pasta `models`.

## Link do vídeo


**Vídeo:** https://youtu.be/yiCCB9F4icc

## Dificuldades encontradas

• Adaptar imagens monocromáticas de 28×28 para a entrada RGB exigida pelo MobileNetV2.

• Definir uma estratégia de Fine-Tuning sem descongelar todo o backbone.

• Controlar o treinamento com learning rate menor na etapa de ajuste fino.

## Conclusão

O projeto demonstra a utilização de Transfer Learning em uma tarefa de classificação de imagens, utilizando o MobileNetV2 como extrator de características pré-treinado no ImageNet. O treinamento em duas etapas permite inicialmente adaptar a nova cabeça de classificação e, posteriormente, realizar um ajuste fino das camadas finais do backbone para o problema específico do PneumoniaMNIST.
