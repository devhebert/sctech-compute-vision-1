# Mini-Projeto: Inspeção de Qualidade de Peças Fundidas (Indústria 4.0)

Este projeto implementa um pipeline completo para inspeção visual industrial, combinando:

1. **Visão Computacional Clássica (OpenCV)** para análise exploratória e destaque de defeitos.
2. **Deep Learning (TensorFlow/Keras)** para classificação automática em larga escala.

## Objetivo

Classificar imagens de peças metálicas do dataset **Casting Product Image Data for Quality Inspection** em duas classes:
- **OK** (sem defeito)
- **Defeituosa** (com trinca/ranhura)

Além da classificação, o projeto demonstra visualmente como técnicas clássicas (blur, canny, limiarização e morfologia) evidenciam defeitos antes do treinamento da CNN.

## Estrutura do projeto

```text
.
├── main.py
├── requirements.txt
├── mypy.ini
├── README.md
├── data/
│   └── casting_data/                 # dataset extraído (não versionado)
│       ├── train/
│       └── test/
├── outputs/
│   ├── exploratory/                  # imagens intermediárias do OpenCV
│   └── plots/                        # curvas de loss/acurácia
├── scripts/
│   └── download_dataset_instructions.txt
└── src/
    └── compute_vision_pipeline/
        ├── __init__.py
        ├── config.py                 # dataclasses tipadas de configuração
        ├── exploratory.py            # pipeline clássico OpenCV
        ├── training.py               # ingestão, augmentation, CNN e gráficos
        └── pipeline.py               # orquestração fim-a-fim
```

## O que cada módulo faz

- `config.py`: define configurações tipadas do projeto (paths, OpenCV e treino).
- `exploratory.py`: aplica grayscale, blur, threshold, Canny e morfologia.
- `training.py`: carrega dados com `image_dataset_from_directory`, aplica Data Augmentation dinâmico, treina CNN e gera gráfico final.
- `pipeline.py`: coordena a execução das sprints técnicas.
- `main.py`: ponto de entrada para rodar tudo com um comando.

## Requisitos

- Python 3.11+
- Pacotes em `requirements.txt`

## Como executar

### 1. Clonar repositório

```bash
git clone <url-do-seu-repo>
cd sctech-compute-vision-1
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar e extrair dataset

Siga `scripts/download_dataset_instructions.txt` e garanta a estrutura:

```text
data/casting_data/train/ok_front
data/casting_data/train/def_front
data/casting_data/test/ok_front
data/casting_data/test/def_front
```

### 5. Rodar pipeline completo

```bash
python main.py
```

## Saídas geradas

1. **Análise exploratória OpenCV** em `outputs/exploratory/`:
   - `*_gray.png`
   - `*_blurred.png`
   - `*_thresholded.png`
   - `*_edges.png`
   - `*_dilated.png`
   - `*_eroded.png`

2. **Auditoria do treinamento** em `outputs/plots/training_curves.png`:
   - Curvas de **Loss** (train vs val)
   - Curvas de **Accuracy** (train vs val)

## Tipagem estática (estilo forte)

O projeto usa:
- `type hints` em funções, classes e retornos
- `@dataclass` para configurações
- `mypy` em modo estrito (`mypy.ini`)

Para checar tipagem:

```bash
mypy src main.py
```

## Mapeamento para as Sprints da atividade

- **Sprint 1**: estrutura do projeto, versionamento e instruções do dataset.
- **Sprint 2**: grayscale + blur (`exploratory.py`).
- **Sprint 3**: threshold + Canny + morfologia (`exploratory.py`).
- **Sprint 4**: ingestão em lote + augmentation (`training.py`).
- **Sprint 5**: arquitetura CNN com Conv2D/MaxPooling2D + Flatten + Dense (`training.py`).
- **Sprint 6**: gráfico de loss/acurácia + documentação final (`training.py` + `README.md`).

## Roteiro sugerido para o vídeo (até 5 min)

1. Mostrar objetivo do sistema e execução do `python main.py`.
2. Explicar o que Canny/Blur/Threshold/Morfologia evidenciaram nas saídas de `outputs/exploratory/`.
3. Mostrar arquitetura CNN e bloco de Data Augmentation.
4. Interpretar `training_curves.png` e comentar possível overfitting/treino saudável.
