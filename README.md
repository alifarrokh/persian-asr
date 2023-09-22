# Persian CTC-Segmentation
This repository facilitates constructing a Persian ASR (Automatic Speech Recognition) dataset by finding utterance alignments within large audio files. It uses CTC-segmentation algorithm ([Ludwig Kürzinger et al., 2020](https://arxiv.org/abs/2007.09127)) and a Persian ASR model trained with Connectionist Temporal Classification (CTC) to find the most probable alignment between a pair of text and speech. The ASR model includes a XLS-R representation learning model ([Alexis Conneau et al., 2020](https://arxiv.org/abs/2006.13979)) pre-trained on 53 languages using a contrastive self-supervised objective, and a linear layer which was trained on labeled Persian speech data. Since the XLS-R model has $O(n^2)$ space/time complexity, a chunking mechanism with overlapping chunks is used to reduce the spece/time complexity of the inference while keeping a proper context in the center of windows. This mechanism is specially essential when dealing with large audio files (few hours of speech).

# Installation & Usage
### 1. Install The Requirements
```
pip install -r requirements.txt
```

### 2. Download The Model
```
pip install gdown
gdown 1JO_UmvZC-yDWxOfZl3TThpqGI1IQAfDW
```

### 3. Prepare Data
1. Create a corrosponding transcript (one sentence per line) for each audio file.
2. Create a `csv` file that contains relative paths to audio files and their transcripts. Sample `metadata.csv`:
```
audio_path,transcript_path
audios/1.mp3,transcripts/1.txt
audios/2.mp3,transcripts/2.txt
```
There is a `sample_input` directory inside the repository that contains an example.

### 4. Run The Algorithm
```
python segment.py \
  --metadata metadata.csv \
  --output_dir output \
  --device cuda
```
Run `python segment.py -h` for more information about the arguments.
