
# Persian ASR Repository

This repository provides resources and tools for Automatic Speech Recognition (ASR) in Persian. It includes datasets, pretrained models, and a speech segmentation tool, all aimed at advancing Persian ASR technology.

## Table of Contents
1. [Persian ASR Dataset and Preparation Scripts](#persian-asr-dataset-and-preparation-scripts)
2. [Pretrained Persian ASR Models](#pretrained-persian-asr-models)
3. [CTC-Based Speech Segmentation Tool](#ctc-based-speech-segmentation-tool)


## Persian ASR Dataset and Preparation Scripts
This section contains information and scripts to prepare and use the Persian ASR dataset. Detailed instructions can be found in the [README](./data/README.md) located in the data directory.


## Pretrained Persian ASR Models
We provide several pretrained ASR models for Persian and will release more in the future.

### Available Models:
| Model | Hugging Face Repository | WER (Greedy Decoding) | WER (Beam=5) | WER (Beam=5 + LM) |
|-|-|-|-|-|
| Wav2Vec2 XLS-R 300M | [wav2vec2-xls-r-300m-fa](https://huggingface.co/alifarokh/wav2vec2-xls-r-300m-fa) | 27.92% | 27.89% | 22.63% |
| Conformer Medium | [nemo-conformer-medium-fa](https://huggingface.co/alifarokh/nemo-conformer-medium-fa) | 32.08% | 31.94% | 27.47% |


### Usage:
Before running the model scripts, the dataset must be prepared in the `.jsonl` format. The preparation scripts for common datasets can be found in the `data` directory. To learn more about how to generate the necessary `.jsonl` files, refer to the [data README](./data/README.md).

To use a pretrained model, navigate to the corresponding model folder under `models`. Each model folder contains two scripts: `train.py` and `inference.py`.
You can get detailed usage instructions by running:
```bash
python train.py --help
python inference.py --help
```


## CTC-Based Speech Segmentation Tool
This tool performs speech segmentation based on Connectionist Temporal Classification (CTC). For detailed usage instructions, refer to the [README](./speech_segmentation/README.md) inside the segmentation tool folder.


## Installation
To use the tools and models in this repository, clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/alifarrokh/persian-asr.git
cd persian-asr
pip install -r requirements.txt
```


## Contribution
We welcome contributions! Please submit a pull request or open an issue to suggest improvements or report bugs.
