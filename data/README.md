# Persian ASR Datasets
This file provides information about Persian ASR datasets. Each dataset has a corresponding folder that includes scripts to download and prepare the data for use in training ASR models. All dataset folders follow the same structure for consistency.


## Directory Structure
Each dataset folder contains the following:

- `download.sh`: Shell script to download the dataset to its corresponding folder.
- `prepare.py`: Python script to prepare the dataset by creating subset files in JSON Lines (`.jsonl`) format.


## Dataset Representation
This project uses the [NeMo](https://github.com/NVIDIA/NeMo) toolkit's convention for representing dataset subsets. Each dataset subset (such as `train` and `test`) is represented by a corresponding `[subset].jsonl` file. Each line in a `.jsonl` file is a JSON object with the following fields:

- `id`: Unique identifier for the audio sample.
- `text`: The transcription of the audio.
- `duration`: Duration of the audio in seconds.
- `audio_filepath`: Path to the audio file.

The `prepare.py` script in each dataset folder creates these `.jsonl` files.


## Usage
1. Navigate to the desired dataset folder.
2. Run `download.sh` to download the dataset.
   ```sh
   ./download.sh
   ```
   **Note**: For the Common Voice Fa dataset, there is no direct link for automatic download. Please manually download the dataset from the [Mozilla Common Voice website](https://commonvoice.mozilla.org/en/datasets) before running `download.sh`.
3. Run `prepare.py` to prepare the data and generate the required `.jsonl` files.
   ```sh
   python prepare.py
   ```


## Datasets
The following datasets are included in this directory:

| Name | Sample Rate | Duration | Samples | Speakers |
|-|-|-|-|-|
| [Shenasa AI<sup>1</sup>](https://github.com/shenasa-ai/speech2text) | 16 KHz | 200 GB | Crawled | - |
| [Common Voice Fa](https://commonvoice.mozilla.org/en/datasets) | - | 300-400 hrs | - | - |
| [ArmanAV](https://www.sciencedirect.com/science/article/abs/pii/S0957417423021504) | - KHz | 220 hrs | - | 1700 |
| [Deepmine](https://data.deepmine.ir/) | - | - | 370K | >1400 |
| [ASR Farsi Youtube<sup>1</sup>](https://huggingface.co/datasets/pourmand1376/asr-farsi-youtube-chunked-30-seconds) | - KHz | - | >140K | Crawled |
| [Farsdat<sup>2</sup>](https://catalog.elra.info/en-us/repository/browse/ELRA-S0112/) | 22.5 KHz | - | - | 300 |
| [ShEmo](https://github.com/mansourehk/ShEMO)                   | 44.1 KHz | 3.5 hrs | 3000 | 87 |
| [Persian Speech Corpus](https://fa.persianspeechcorpus.com/) | - | 2.5 hrs | 399 | 1 |
| SFAVD | - | - | - | - | |

<sup>1</sup> These datasets were crawled from the internet and do not have exact labels.

<sup>2</sup> These datasets are not free.
