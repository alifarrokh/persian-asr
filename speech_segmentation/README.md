# CTC-Segmentation
This tool simplifies the creation of a Persian ASR dataset by aligning utterances within large audio files. It employs the CTC-segmentation algorithm (Ludwig Kürzinger et al., 2020) to determine the most probable alignment between corresponding text and speech. To address the $O(n^2)$ space and time complexity of transformer models, the tool implements a chunking mechanism with overlapping segments. This approach reduces computational overhead while maintaining contextual integrity in the center of the windows, making it particularly effective for processing lengthy audio files containing hours of speech. We plan to publish a Persian ASR dataset created with this tool in the near future.


## Usage
### 1. Prepare Data
1. Create a corrosponding transcript (one sentence per line) for each audio file.
2. Create a `csv` file that contains relative paths to audio files and their transcripts. Sample `metadata.csv`:
```
audio_path,transcript_path
audios/1.mp3,transcripts/1.txt
audios/2.mp3,transcripts/2.txt
```
There is a `sample_input` directory inside the repository that contains an example.

### 2. Run The Script
```
python segment.py \
  --metadata metadata.csv \
  --output_dir output \
  --device cuda
```
Run `python segment.py -h` for more information about the arguments.
