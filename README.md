# Análise de Preços de Casas — Ames Housing Dataset

Projeto desenvolvido para o desafio **C3 — Data Analysis and Machine Learning Hackathon**
(FAESA Centro Universitário, Prof. M.Sc. Howard Roatti).

## Objetivo

Explorar o conjunto de dados de preços de casas em Ames, Iowa (EUA), aplicando todo o
fluxo de um projeto de ciência de dados: análise exploratória, engenharia de
características, modelos de aprendizagem supervisionada (regressão e classificação)
e técnicas de aprendizagem não supervisionada (clusterização, redução de
dimensionalidade, regras de associação e detecção de outliers).

## Dataset

[House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

- `train.csv` — 1460 imóveis com 79 variáveis explicativas + `SalePrice` (preço de venda)
- `test.csv` — 1459 imóveis sem `SalePrice`, usado apenas para gerar features de forma consistente

## Estrutura do repositório

```
.
├── HousePrices_Trabalho/
│   ├── House_Prices.ipynb  # notebook principal com toda a análise
│   ├── train.csv
│   └── test.csv
├── requirements.txt
└── README.md
```

## Metodologia

| Etapa | Técnica utilizada |
|---|---|
| Análise Exploratória | distribuição de tipos de variável, mapa de valores faltantes, distribuição e assimetria do `SalePrice`, correlações, identificação visual de outliers |
| Feature Engineering | tratamento de NAs (regra específica por variável, conforme o dicionário de dados), remoção de outliers extremos, criação de features (`TotalSF`, `TotalBath`, `HouseAge`, `RemodAge` etc.), encoding ordinal para variáveis de qualidade, one-hot encoding para nominais, transformação logarítmica (log1p) das variáveis assimétricas |
| Regressão | Regressão Linear e Random Forest Regressor para prever `SalePrice` |
| Classificação | `SalePrice` binarizado pela mediana (Alto/Baixo); Regressão Logística e Random Forest Classifier |
| Clusterização | K-Means (k=4, escolhido pelo método do cotovelo) sobre características do imóvel |
| Redução de Dimensionalidade | PCA (scree plot + projeção 2D) |
| Análise de Associação | algoritmo **Apriori implementado manualmente** (geração de itemsets por nível, com poda) sobre variáveis discretizadas |
| Detecção de Outliers | Local Outlier Factor (LOF) |

## Resultados principais

**Regressão** (previsão do valor de `SalePrice`)

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| Regressão Linear | R$ 22.817 | R$ 15.475 | 0,91 |
| Random Forest Regressor | R$ 23.520 | R$ 16.413 | 0,90 |

**Classificação** (previsão da faixa Alta/Baixa de preço)

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Regressão Logística | 0,901 | 0,903 | 0,897 | 0,900 | 0,971 |
| Random Forest Classifier | 0,911 | 0,941 | 0,877 | 0,908 | 0,982 |

**Não supervisionado**

- **K-Means:** 4 clusters, com preço médio variando de R$ 121.014 (imóveis menores/mais antigos) a R$ 323.243 (imóveis grandes e de alta qualidade)
- **PCA:** são necessárias 119 componentes para explicar 90% da variância (alta dimensionalidade após o one-hot encoding); as 2 primeiras componentes explicam apenas 11,3%
- **Apriori:** 145 itemsets frequentes e 170 regras com confiança ≥ 60%. Destaque: imóveis de baixa qualidade têm forte associação com preço baixo (confiança até 95,7%), enquanto imóveis novos e grandes se associam fortemente a preço alto (confiança 92,5%)
- **LOF:** 44 imóveis (3,0% da base) identificados como outliers, em geral por combinações atípicas de área, lote ou idade em relação ao preço

## Conclusões

Os modelos de regressão e classificação atingiram desempenho elevado (R² ≈ 0,90 e
AUC ≈ 0,97–0,98), confirmando que variáveis como qualidade geral, área habitável e
número de garagens são fortes preditoras do preço de venda. As técnicas não
supervisionadas reforçaram esses achados: o K-Means segmentou os imóveis em grupos
com perfis de preço bem distintos, e o Apriori confirmou em forma de regras as
relações entre qualidade/tamanho/idade e faixa de preço — uma validação cruzada
interessante entre abordagens supervisionadas e não supervisionadas.

## Como executar

```bash
python -m pip install -r requirements.txt
cd HousePrices_Trabalho
jupyter notebook House_Prices.ipynb
```

## Tecnologias

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · SciPy

## Autores

- Gabriela Gave Gavi, Jeronymo Francisco Moreira Neto, José Luiz dos Santos Azeredo Mendes, Pedro Henrique Bispo Sarmento, Pedro Henrique Ferreira Bonela 
