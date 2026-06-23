# IA901 — Segmentação de Tumores Cerebrais com ACU-Net 2.5D

## Descrição do projeto

Este projeto propõe a segmentação automática de tumores cerebrais em imagens de ressonância magnética multimodal. A solução utiliza uma arquitetura **ACU-Net 2.5D** implementada em PyTorch.

O modelo recebe imagens das modalidades:

* T1;
* T1CE;
* T2;
* FLAIR.

Para prever uma fatia central, o modelo utiliza também a fatia anterior e a posterior. Dessa forma, cada entrada possui:

```text
3 fatias consecutivas × 4 modalidades = 12 canais
```

A saída da rede corresponde a três regiões tumorais:

* **WT — Whole Tumor:** tumor inteiro;
* **TC — Tumor Core:** núcleo tumoral;
* **ET — Enhancing Tumor:** região realçada por contraste.

---

## Estrutura do projeto

```text
segmentacao-tumores-2.5D/
├── README.md
│
├── assets/
│   └── artigo-base.pdf
│
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
│       └── datasheets_for_datasets_207726_298997.pdf
│
├── notebooks/
│   └── ACU_Net_2.5D-Parcial.ipynb
│
└── src/
```

---

## Organização das pastas

### `README.md`

Arquivo principal de documentação do projeto. Contém a descrição da proposta, estrutura do repositório, bases utilizadas, instruções de execução e informações sobre reprodutibilidade.

### `assets/`

Contém materiais de apoio utilizados no projeto.

Atualmente, esta pasta contém:

```text
artigo-base.pdf
```

Esse arquivo corresponde ao artigo utilizado como referência conceitual para a arquitetura ACU-Net e para a segmentação de tumores cerebrais em MRI multimodal.

### `data/`

Armazena informações e arquivos relacionados aos datasets utilizados.

#### `data/raw/`

Diretório destinado aos dados originais, sem modificações.

Os volumes completos das bases BraTS não são armazenados diretamente no repositório devido ao grande volume dos arquivos e às regras de distribuição dos datasets. O download dos dados é realizado pelo notebook por meio da Kaggle API.

Atualmente, a pasta contém:

```text
datasheets_for_datasets_207726_298997.pdf
```

Esse documento descreve características, origem, composição e limitações dos datasets utilizados.

#### `data/interim/`

Diretório destinado a dados intermediários gerados durante o pré-processamento, tais como:

* imagens normalizadas com Brain-Only Z-Score;
* máscaras convertidas para WT, TC e ET;
* imagens recortadas;
* fatias utilizadas na estratégia 2.5D;
* resultados temporários de processamento.

Os dados intermediários completos não são versionados no GitHub devido ao tamanho. Eles podem ser gerados novamente pela execução do notebook.

#### `data/processed/`

Diretório destinado aos dados finais utilizados pelo modelo.

No projeto, esses dados correspondem principalmente a tensores contendo:

```text
Entrada:
3 fatias consecutivas × 4 modalidades = 12 canais

Saída:
WT, TC e ET = 3 canais
```

Os tensores são produzidos dinamicamente durante o treinamento e não são mantidos no repositório devido ao volume de armazenamento necessário.

### `notebooks/`

Contém o notebook principal do projeto:

```text
ACU_Net_2.5D-Parcial.ipynb
```

O notebook realiza as seguintes etapas:

1. configuração das credenciais da Kaggle API;
2. download das bases BraTS;
3. leitura dos arquivos NIfTI;
4. visualização das modalidades T1, T1CE, T2 e FLAIR;
5. normalização Brain-Only Z-Score;
6. conversão da máscara `SEG` para WT, TC e ET;
7. criação das entradas 2.5D;
8. definição da arquitetura ACU-Net em PyTorch;
9. treinamento com Dice Loss e otimizador Adam;
10. avaliação por Dice Score e matrizes de confusão;
11. visualização de máscaras reais e previstas.

### `src/`

Diretório reservado para códigos-fonte organizados em módulos Python.

No estágio atual do projeto, a implementação principal está concentrada no notebook do diretório `notebooks/`. Futuramente, funções de pré-processamento, geração de lotes, treinamento e avaliação podem ser separadas em arquivos Python dentro desta pasta.

---

## Bases de dados

O projeto utiliza as bases BraTS 2018 e BraTS 2020.

| Base de Dados | Características                                                                                                                                     | Uso no projeto                                                                           |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| BraTS 2018    | Imagens de ressonância magnética multimodal T1, T1CE, T2 e FLAIR, além da máscara clínica `SEG`. Inclui casos de gliomas de alto e baixo grau.      | Utilizada como parte do conjunto de treinamento e validação da ACU-Net 2.5D.             |
| BraTS 2020    | Base posterior com maior diversidade de pacientes e condições de aquisição. Mantém as modalidades T1, T1CE, T2, FLAIR e as máscaras de segmentação. | Complementa a BraTS 2018, aumentando a diversidade dos pacientes utilizados pelo modelo. |

Os pacientes são combinados e separados por paciente em:

```text
80% → treinamento
20% → validação
```

---

## Tecnologias utilizadas

* Python;
* Google Colab;
* Kaggle API;
* PyTorch;
* CUDA, quando disponível;
* NiBabel;
* SimpleITK;
* NumPy;
* Matplotlib;
* Pandas;
* Seaborn;
* Scikit-learn.

---

# Como executar o projeto no Google Colab

## 1. Abrir o notebook

No GitHub, acesse o arquivo:

```text
projetos/segmentacao-tumores-2.5D/notebooks/ACU_Net_2.5D-Parcial.ipynb
```

Em seguida, utilize uma das opções:

* clicar em **Open in Colab**, caso o botão esteja disponível;
* baixar o notebook e enviá-lo manualmente para o Google Colab;
* abrir o Google Colab e selecionar **Arquivo → Fazer upload do notebook**.

---

## 2. Ativar GPU

No Google Colab, acesse:

```text
Ambiente de execução
→ Alterar tipo de ambiente de execução
→ Acelerador de hardware
→ GPU
```

O notebook utiliza GPU automaticamente quando o CUDA estiver disponível.

---

## 3. Configurar as credenciais da Kaggle API

O notebook utiliza a Kaggle API para baixar os datasets BraTS.

Na sua conta Kaggle, obtenha as credenciais:

```text
KAGGLE_USERNAME
KAGGLE_KEY
```

No Google Colab:

1. clique no ícone de chave na barra lateral;
2. crie o segredo `KAGGLE_USERNAME`;
3. crie o segredo `KAGGLE_KEY`;
4. permita que o notebook acesse esses segredos.

O notebook recupera essas informações utilizando o mecanismo de segredos do Colab.

---

## 4. Executar o notebook

Execute as células na ordem em que aparecem.

O fluxo de execução é:

```text
Configuração de bibliotecas
↓
Configuração da Kaggle API
↓
Download dos datasets
↓
Pré-processamento das imagens
↓
Geração das entradas 2.5D
↓
Treinamento da ACU-Net
↓
Avaliação e visualização dos resultados
```

Os datasets são baixados temporariamente no ambiente do Google Colab. Após a extração, os arquivos compactados podem ser removidos para economizar espaço.

---

## Parâmetros principais do treinamento

| Parâmetro             |                Valor |
| --------------------- | -------------------: |
| Estratégia de entrada |                 2.5D |
| Modalidades           | T1, T1CE, T2 e FLAIR |
| Canais de entrada     |                   12 |
| Classes de saída      |                    3 |
| Classes previstas     |          WT, TC e ET |
| Tamanho do lote       |                    8 |
| Lotes por paciente    |                    5 |
| Passos por época      |                  100 |
| Número de épocas      |                  100 |
| Otimizador            |                 Adam |
| Taxa de aprendizado   |               `5e-5` |
| Função de perda       |            Dice Loss |
| Tamanho após recorte  |          `160 × 192` |

---

## Reprodutibilidade

Para favorecer a reprodução dos experimentos, o projeto documenta:

* utilização das bases BraTS 2018 e BraTS 2020;
* divisão dos pacientes entre treino e validação;
* normalização Brain-Only Z-Score;
* estratégia de entrada 2.5D;
* arquitetura ACU-Net;
* hiperparâmetros de treinamento;
* versões das bibliotecas;
* informações de GPU e CUDA;
* uso de semente aleatória fixa.

Os resultados podem variar entre execuções devido a diferenças de ambiente, GPU, CUDA, versões de bibliotecas, aumento de dados e operações não determinísticas.

---

## Observações importantes

* O modelo possui finalidade acadêmica e experimental.
* Os resultados não substituem a avaliação de profissionais da área da saúde.
* A correção N4 Bias Field Correction foi demonstrada no notebook, mas não foi integrada ao pipeline final de treinamento.
* A arquitetura é uma adaptação 2.5D da ACU-Net, não uma reprodução literal de uma arquitetura completamente 3D.
* Os volumes completos do BraTS não são incluídos no repositório devido ao tamanho e às regras de distribuição dos datasets.

---

## Uso dos dados

Os datasets BraTS devem ser utilizados de acordo com os termos estabelecidos pelos seus responsáveis. Os dados são empregados neste projeto exclusivamente para fins acadêmicos e experimentais.
