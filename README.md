# Validador de Imagens de Raio-X (Radiology Image Validator)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg) ![Keras](https://img.shields.io/badge/Keras-Powered-red.svg)

## Visão Geral

Este projeto implementa uma Rede Neural Convolucional (CNN) robusta, projetada para funcionar como um validador de imagens de raios-X. O objetivo principal é classificar uma imagem de entrada em uma de duas categorias: `Raio-X` ou `Outros`.

O modelo foi treinado com um conjunto de dados altamente diversificado para identificar com precisão se uma imagem é uma radiografia de qualquer parte do corpo ou se é uma imagem aleatória (animais, objetos, paisagens, etc.). Ele serve como um excelente filtro de entrada para qualquer sistema que precise garantir que está processando apenas imagens radiológicas.

## Funcionalidades Principais

* **Classificação Binária**: Distingue com precisão entre imagens da classe `Raio-X` e da classe `Outros`.
* **Modelo Pré-Treinado Incluso**: Um modelo `validador_raiox.keras` já treinado está disponível no repositório para uso imediato. 
* **Script de Treinamento**: Inclui o script `treinar_validador_raiox.py` para que qualquer pessoa possa treinar o modelo do zero com seu próprio conjunto de dados. 
* **Script de Validação**: Oferece o script `validar_imagem.py` para classificar facilmente uma nova imagem. 
* **Suporte a Múltiplos Formatos**: O sistema foi projetado para carregar e processar imagens nos formatos `.png`, `.jpg` e `.jpeg`. 

## Modelo Pré-Treinado

Para facilitar o uso, este repositório inclui o arquivo `validador_raiox.keras`. Este modelo foi treinado com o extenso conjunto de dados descrito abaixo e pode ser usado para inferência imediata sem a necessidade de treinamento.

## Dataset Utilizado para Treinamento

O modelo foi treinado com um conjunto de dados balanceado e diversificado, contendo aproximadamente 4400 imagens no total (2200 por classe).

### Classe `Raio-X` (2200 imagens)

Para esta classe, foram utilizadas imagens de múltiplos datasets públicos para cobrir diversas partes do corpo e tipos de exames:

* [NIH ChestX-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data) - Raio-X de Tórax
* [Knee Osteoarthritis Dataset](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) - Raio-X de Joelhos
* [The Vertebrae X-Ray Images](https://www.kaggle.com/datasets/yasserhessein/the-vertebrae-xray-images) - Raio-X de Vértebras
* [Spine Fracture Prediction from X-Rays](https://www.kaggle.com/datasets/vuppalaadithyasairam/spine-fracture-prediction-from-xrays/data) - Raio-X da Coluna
* [Computed Tomography (CT) of the Brain](https://www.kaggle.com/datasets/trainingdatapro/computed-tomography-ct-of-the-brain) - Tomografias do Cérebro
* [Bone Fracture Dataset](https://www.kaggle.com/datasets/orvile/bone-fracture-dataset) - Raios-X da Tíbia e Fíbula

### Classe `Outros` (Imagens Não-Raio-X, ~2200 imagens)

Para ensinar ao modelo o que **não** é um raio-X, foram utilizadas imagens dos datasets **Imagenette** e **Imagewoof** (versões de 320 pixels). Estes datasets contêm fotos de animais, objetos e cenas do cotidiano.

* **Fonte:** [fastai/imagenette](https://github.com/fastai/imagenette)

## Estrutura do Projeto

```
seu-repositorio/

├── dataset-validador-main/         <-- pasta para os dados de treinamento (se for treinar)
│   ├── raio-X/
│   │   ├── imagem_raiox_1.jpg
│   │   └── ...
│   └── outros/
│       ├── imagem_aleatoria_1.jpg
│       └── ...

├── treinar_validador_raiox.py   <-- script para treinar o modelo
├── validar_imagem.py            <-- script para validar uma nova imagem
├── validador_raiox.keras        <-- o modelo pré-treinado
├── requirements.txt             <-- lista de dependências
└── README.md                    <-- este arquivo
```

## Pré-requisitos

* Python 3.8 ou superior
* Bibliotecas listadas no arquivo `requirements.txt`:
    * `tensorflow==2.15.0` 
    * `numpy==1.24.3` 
    * `pillow==10.2.0` 
    * `scikit-learn==1.3.0` 
    * `kaggle==1.6.12` 

## Instalação

1.  **Clone este repositório:**
    ```bash
    git clone https://seu-link-do-github/seu-repositorio.git
    cd seu-repositorio
    ```

2.  **Crie um ambiente virtual (altamente recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # no windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Como Usar

Existem duas formas principais de utilizar este projeto.

### Uso 1: Validar uma Imagem (Com o Modelo Pré-Treinado)

Esta é a forma mais comum, para quem deseja apenas usar o validador.

1.  **Abra o arquivo `validar_imagem.py`** em um editor de texto. 
2.  **⚠️ IMPORTANTE:** Altere a variável `caminho_sua_imagem` para o caminho completo da imagem que você deseja analisar. 
    ```python
    # altere essa linha com o caminho da sua imagem
    caminho_sua_imagem = "C:/Users/seu_nome/Desktop/imagem_teste.png" 
    ```
3.  **Execute o script** no seu terminal:
    ```bash
    python validar_imagem.py
    ```
    O resultado (Imagem Válida ou Inválida) será exibido no console. 

### Uso 2: Treinar um Novo Modelo

Para usuários avançados que desejam treinar o modelo com seu próprio conjunto de dados.

1.  **Prepare seu dataset** seguindo a estrutura de pastas descrita acima (`dataset_validador/Raio-X/` e `dataset_validador/Outros/`).
2.  **Abra o arquivo `treinar_validador_raiox.py`**. 
3.  **⚠️ IMPORTANTE:** Altere a variável `pasta_base_dataset` para o caminho da sua pasta de dataset. 
    ```python
    # altere esta linha com o caminho do seu dataset
    pasta_base_dataset = "dataset_validador" 
    ```
4.  **Execute o treinamento** no seu terminal:
    ```bash
    python treinar_validador_raiox.py
    ```
    Ao final do processo, um novo arquivo `validador_raiox.keras` será gerado com o modelo recém-treinado. 

## Detalhes Técnicos

* **Arquitetura da CNN**: O modelo utiliza três camadas de convolução (`Conv2D`) com 32, 64 e 128 filtros, seguidas por camadas de `MaxPooling2D`. A classificação é feita por uma camada `Dense` de 128 neurônios com uma camada de `Dropout` de 0.5 para regularização. 
* **Parâmetros de Treinamento**: O modelo foi treinado por 50 épocas com um tamanho de lote (batch size) de 32. 
* **Pré-processamento**: As imagens são redimensionadas para 64x64 pixels, convertidas para escala de cinza e normalizadas para o intervalo [0, 1]. 

## Aviso Legal

Este projeto foi desenvolvido para fins acadêmicos e educacionais. O modelo gerado, embora treinado com dados públicos, não possui certificação clínica e **não deve ser utilizado para diagnósticos médicos reais** ou para tomada de decisões clínicas.
