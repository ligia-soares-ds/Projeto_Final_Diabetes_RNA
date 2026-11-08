# Predição de Diabetes com Redes Neurais Artificiais

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ligia-soares-ds/Projeto_Final_Diabetes_RNA/blob/main/Projeto_Final_Diabetes_RNA.ipynb)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-2.21-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-f7931e)

Rede neural multicamadas (MLP) em TensorFlow/Keras para **estimar o risco de diabetes a partir de indicadores de saúde autorrelatados** do BRFSS 2015 (CDC, EUA). O projeto compara a rede com uma regressão logística, mede o efeito de cada técnica de regularização (ablação), compara otimizadores, escolhe um limiar de decisão pensado para rastreio e interpreta o modelo com importância por permutação e SHAP.

Projeto final da disciplina **BIZ 5802 – Modelagem de Dados com Redes Neurais Artificiais** (PPG em Zoologia, IB-USP, 2026/2), ministrada pelo Dr. Welton Dionisio da Silva e pelo Dr. Rodrigo Hirata Willemart.

**Equipe (Grupo 4):** Aline Muniz, Dandara Cidade, Fernando Pablo, Gabriela Santos e Ligia Assis.

---

## Pergunta

> Uma rede neural consegue identificar quem tem diabetes usando apenas respostas de um questionário de saúde, a ponto de servir como **instrumento de rastreio** (indicar quem deve fazer exames confirmatórios)?

## Dados

[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) (Kaggle), versão limpa das respostas de 2015 do *Behavioral Risk Factor Surveillance System*.

- **253.680 respostas**, **21 variáveis preditoras** (14 binárias, 4 ordinais e 3 numéricas) e nenhum valor ausente.
- **Alvo:** `Diabetes_binary` (1 = diabetes diagnosticado; 0 = sem diabetes ou pré-diabetes).
- **Prevalência:** 13,9%. Uma regra que sempre responde "sem diabetes" já acerta cerca de 86% dos casos, por isso a acurácia **não** é a métrica principal.

Foi usado o arquivo com a prevalência real, e não a versão 50/50. Testar numa população artificial em que metade das pessoas tem diabetes inflaria a precisão (valor preditivo positivo). O desbalanceamento foi tratado só no treino, com `class_weight`.

## Metodologia

| Etapa | O que foi feito |
|---|---|
| Divisão | Treino, validação e teste estratificados (70/15/15), com `random_state = 42` |
| Pré-processamento | `StandardScaler` nas 7 variáveis não binárias, ajustado **só no treino** (sem vazamento) |
| Desbalanceamento | Pesos de classe (0: 0,58; 1: 3,59) na perda de treino e pesos equivalentes na validação, para que as duas curvas fiquem na mesma escala |
| Baselines | Classificador "sempre negativo" e regressão logística, que equivale a uma rede de um único neurônio sigmoide |
| Rede neural | 21 entradas → Dense 256 → 128 → 64 (ReLU) → 1 (sigmoide), com inicialização Glorot, entropia cruzada binária e Adam (lr = 0,001) em lotes de 1024. São **46.849 parâmetros** |
| Ablação | A: sem regularização · B: Early Stopping · C: B + Dropout 30% · D: B + L2 (λ = 10⁻⁴) · E: B + BatchNorm · F: todas juntas |
| Otimizadores | SGD, SGD + Momentum, RMSProp e Adam, com a mesma arquitetura |
| Limiar | Índice de Youden escolhido **na validação** e aplicado uma única vez no teste |
| Avaliação | ROC-AUC, PR-AUC, sensibilidade, especificidade, precisão, F1, Brier, pseudo-R² de McFadden e R² de Tjur |
| Análises extras | Perfil dos erros, desempenho por subgrupo, calibração por decis (com correção de prior) |
| Interpretabilidade | Importância por permutação e valores SHAP |

## Principais resultados

### Regularização (validação)

| Experimento | ROC-AUC | PR-AUC | Melhor época |
|---|---|---|---|
| A: sem regularização | 0,774 | 0,340 | 5 de 60 |
| B: Early Stopping | 0,827 | 0,424 | 5 |
| **C: ES + Dropout** | **0,829** | **0,429** | 13 |
| D: ES + L2 | 0,828 | 0,426 | 15 |
| E: ES + BatchNorm | 0,826 | 0,422 | 3 |
| F: ES + Dropout + L2 + BN | 0,829 | 0,429 | 28 |

Sem regularização, a rede começa a sobreajustar já na 5ª época. O **Early Stopping** é o que mais ajuda, porque sozinho recupera quase todo o desempenho. Dropout, L2 e BatchNorm acrescentam diferenças pequenas. Entre os otimizadores, Adam e RMSProp chegam ao mesmo desempenho (ROC-AUC ≈ 0,829) com menos da metade das épocas do SGD.

### Modelo final no teste (MLP com ES + Dropout, Adam, limiar de Youden = 0,495)

| Modelo | ROC-AUC | PR-AUC | Sensibilidade | Especificidade | Precisão |
|---|---|---|---|---|---|
| Regressão logística | 0,822 | 0,396 | 0,795 | 0,703 | 0,302 |
| **MLP** | **0,830** | **0,428** | **0,815** | 0,697 | 0,304 |

- A MLP supera a regressão logística, mas por pouco. Isso indica que o teto de desempenho está mais na informação disponível nos dados autorrelatados do que na arquitetura do modelo.
- A rede **detecta cerca de 8 em cada 10 pessoas com diabetes**. Como a prevalência é baixa, cerca de 70% dos alertas são falsos positivos, o que é aceitável para triagem, mas não para diagnóstico.
- Depois da correção de prior, as probabilidades ficam **bem calibradas**. O Brier cai de 0,120 (referência que só usa a prevalência) para 0,097, e os 30% de pessoas com maior risco previsto concentram **73% dos casos** de diabetes.
- O desempenho varia entre grupos: o ROC-AUC é menor em idosos (0,76) e em quem relata saúde regular ou ruim (0,72).

### O que o modelo mais usa

Pela importância por permutação (queda no ROC-AUC), as variáveis mais usadas são **saúde geral autorrelatada**, **IMC**, **idade**, **pressão alta** e **colesterol alto**. A análise SHAP confirma essa ordem e a direção dos efeitos: valores altos aumentam o risco, e renda maior o diminui.

## Limitações

- Os dados são **transversais e autorrelatados**, então as associações não são causais. Variáveis como "saúde geral ruim" podem ser consequência do diabetes, e não causa.
- O alvo reflete o diagnóstico informado pelo entrevistado e está sujeito a subdiagnóstico.
- A amostra é dos EUA em 2015. Para aplicar o modelo a outras populações, como a brasileira, seria preciso validá-lo com dados locais.
- Os experimentos usam uma única semente e um único valor para cada hiperparâmetro, então diferenças pequenas entre as configurações regularizadas podem ser ruído.

## Como executar

1. Clique em **Abrir no Colab**, no topo desta página.
2. Execute as células em ordem (**Ambiente de execução → Executar tudo**). Os dados são baixados automaticamente do Kaggle com `kagglehub`.
3. Em CPU, o notebook completo leva cerca de 20 a 30 minutos. As etapas de ablação e de comparação de otimizadores são as mais demoradas.

Para rodar localmente:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn shap kagglehub
jupyter notebook Projeto_Final_Diabetes_RNA.ipynb
```

O último bloco do notebook gera as figuras usadas no relatório e as salva em `figuras_relatorio/`.

## Estrutura do repositório

```
├── Projeto_Final_Diabetes_RNA.ipynb   # notebook completo: dados, modelos, resultados e figuras
└── README.md
```

## Principais referências

- Teboul, A. *Diabetes Health Indicators Dataset*. Kaggle, 2021.
- CDC. *Behavioral Risk Factor Surveillance System: overview BRFSS 2015*. 2016.
- Xie, Z. et al. Building risk prediction models for type 2 diabetes using machine learning techniques. *Preventing Chronic Disease*, 16, e130, 2019.
- Srivastava, N. et al. Dropout: a simple way to prevent neural networks from overfitting. *JMLR*, 15, 2014.
- Kingma, D. P.; Ba, J. Adam: a method for stochastic optimization. *ICLR*, 2015.
- King, G.; Zeng, L. Logistic regression in rare events data. *Political Analysis*, 9(2), 2001.
- Lundberg, S. M.; Lee, S.-I. A unified approach to interpreting model predictions. *NIPS*, 2017.

A lista completa de referências está no relatório técnico do projeto.
