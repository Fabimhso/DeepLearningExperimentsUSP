# 🧠 Deep Learning Experiments — USP Specialization

Experimentos em Aprendizado Profundo e Métodos Clássicos
Especialização – Universidade de São Paulo (USP)

Este repositório reúne uma série de cinco experimentos práticos desenvolvidos no contexto da disciplina Experimentos com Aprendizado Profundo e Métodos Clássicos, pertencente a uma especialização na Universidade de São Paulo (USP).

O objetivo principal foi explorar, comparar e integrar técnicas modernas de Deep Learning com métodos clássicos de Machine Learning, avaliando desempenho, generalização, compressão de dados e reutilização de representações profundas.

# 📌 Visão Geral dos Experimentos
## 🔹 Experimento 1 — CNNs Básicas para Classificação

#Objetivo:
## Avaliar o desempenho de uma CNN simples em um problema de classificação supervisionada.

##O que foi feito:

Treinamento de CNN em dados reais

Avaliação de acurácia e matriz de confusão

Estabelecimento de baseline para os experimentos seguintes

## 🔹 Experimento 2 — Geração de Dados com GANs Condicionais

## Objetivo:
Gerar dados sintéticos condicionais e avaliar seu impacto no treinamento de classificadores.

O que foi feito:

Treinamento de GAN condicional

Geração de amostras sintéticas

Treinamento de CNN com dados reais e sintéticos

Comparação com treinamento apenas em dados reais

## 🔹 Experimento 3 — Autoencoders Adversariais (CAAE)

## Objetivo:
Explorar Autoencoders Adversariais como alternativa conceitual às GANs para geração de dados.

O que foi feito:

Treinamento de um Conditional Adversarial Autoencoder (CAAE)

Variação da dimensão do espaço latente

Mistura de dados reais e sintéticos no treinamento

Avaliação da métrica FID

Comparação de desempenho da CNN:

com dados reais

com dados sintéticos

com dados mistos

Determinação da proporção ideal de dados sintéticos

# 📌 Conclusão:
O uso controlado de dados sintéticos melhora o desempenho, desde que não sejam utilizados no conjunto de teste.

## 🔹 Experimento 4 — Features Profundas + Métodos Clássicos

## Objetivo:
Reutilizar representações profundas como atributos de alto nível para métodos clássicos.

O que foi feito:

Treinamento de CNN para extração de features

Treinamento de ensembles globais (MLP e SVM)

Aplicação de Mapas Auto-Organizáveis (SOM)

Análise da pureza dos clusters

Treinamento de classificadores locais (especialistas) por cluster

Comparação entre:

Ensembles globais

Especialistas locais via SOM

# 📌 Conclusão:
Especialistas locais podem superar modelos globais em regiões específicas do espaço de características.

##🔹 Experimento 5 — Compressão e Redução Dimensional com Autoencoders

Objetivo:
Analisar compressão de dados e preservação de informação relevante.

O que foi feito:

Treinamento de Autoencoder não supervisionado

Variação da dimensão latente (16, 32, 64, 128)

Extração do espaço latente

Treinamento de MLP e SVM sobre o espaço comprimido

Comparação com features profundas extraídas de CNN

Análise do trade-off compressão × desempenho

# 📌 Conclusão:
Dimensões latentes intermediárias oferecem o melhor equilíbrio entre compressão e desempenho preditivo.

# 📊 Conclusões Gerais

Representações profundas podem ser reutilizadas com sucesso por métodos clássicos

Dados sintéticos podem melhorar generalização, quando usados corretamente

Autoencoders permitem redução dimensional eficiente, preservando informação relevante

A combinação de Deep Learning + métodos clássicos resulta em modelos mais flexíveis e interpretáveis

Técnicas de especialização local (SOM + especialistas) são eficazes em cenários complexos

## 🛠️ Tecnologias e Stack Utilizadas
## 🔧 Linguagem

- Python 3

## 🧠 Deep Learning

- PyTorch

- Torchvision

## 📊 Machine Learning Clássico

- Scikit-learn

- SVM

- MLP

- Métricas de avaliação

## 📈 Visualização

- Matplotlib

## 🗂️ Outros

- NumPy

- Jupyter / Scripts Python

- Git & GitHub

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte da Especialização na Universidade de São Paulo (USP), na disciplina:

Experimentos com Aprendizado Profundo e Métodos Clássicos

## 👨‍💻 Autor

Fabiano Henrique
Computer Science Student & AI/ML Student
Especialização em andamento — USP (Universidade de Sao Paulo)
