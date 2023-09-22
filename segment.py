"""
Persian CTC-Segmentation of speech
"""

import argparse
import os
from math import ceil
import warnings

import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import librosa
from librosa import get_duration
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets.utils.logging import disable_progress_bar
import matplotlib.pyplot as plt
import ctc_segmentation as cs

from utils import get_vocab, read_csv, read_lines, normalize_text, format_duration
from utils import write_output, determine_utterance_segments


# Disable warnings
warnings.filterwarnings('ignore')
disable_progress_bar()

# Argparse
parser = argparse.ArgumentParser(description='Persian CTC Segmentation')
parser.add_argument('--output_dir', default='output', type=str, help='Path to output directory')
parser.add_argument('--metadata', required=True, type=str, help='Path to metadata (csv) file')
parser.add_argument(
    '--window_len',
    default=8000,
    type=int,
    help='Window size for CTC Segmentation algorithm'
)
parser.add_argument('--device', default='cpu', type=str, help='Device (cpu/cuda)')
parser.add_argument(
    '--threshold',
    default=-2,
    type=float,
    help='Segment validation threshold (Sum of log probabilities)'
)
parser.add_argument(
    '--max_chunk_seconds',
    default=20,
    type=float,
    help='Maximum duration of audio chunks for inference.'
)

# Parse & validate args
args = parser.parse_args()
assert args.device in ["cuda", "cpu"], "Invalid value for device"
assert not os.path.exists(args.output_dir), "Output directory already exists"
assert os.path.isfile(args.metadata), "Input metadata does not exist"

# load model
vocabulary = get_vocab()
processor = Wav2Vec2Processor.from_pretrained("./")
model = Wav2Vec2ForCTC.from_pretrained("./").to(args.device)

# Model-related helper functions
def output_frames(n_features):
    """
    Calculates the numbber of output frames based on input signal length
    """
    return model._get_feat_extract_output_lengths(n_features)


# Load metadata
data = read_csv(args.metadata)
base_input_dir = os.path.split(os.path.abspath(args.metadata))[0]

# Create output directory
os.mkdir(args.output_dir)
res_transcript = os.path.join(args.output_dir, 'result.csv')
res_report = os.path.join(args.output_dir, 'report.txt')
duration_hist_path = os.path.join(args.output_dir, 'duration_hist.png')

# Process each audio
print('Segmenting audio files ...')
next_audio_number = 1
for item in tqdm(data):

    # Load the speech file
    audio_path = os.path.join(base_input_dir, item['audio_path'])
    original_speech_array, original_sample_rate = torchaudio.load(audio_path)
    item['index_duration'] = original_speech_array.shape[1] / original_sample_rate
    original_speech_array = original_speech_array.squeeze().numpy()

    # Preprocess the speech file
    sample_rate = processor.feature_extractor.sampling_rate
    speech_array = librosa.resample(
        np.asarray(original_speech_array),
        orig_sr=original_sample_rate,
        target_sr=sample_rate
    )

    # Feature extraction
    features = processor(
        speech_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    input_values = features.input_values.to(args.device)

    # Create speech chunks
    n_all_samples = input_values.shape[1]
    samples_per_chunk = int(args.max_chunk_seconds * sample_rate)
    border_samples = int(samples_per_chunk / 6)
    middle_samples = samples_per_chunk - (2 * border_samples)
    n_chunks = ceil(n_all_samples / middle_samples)

    # ASR inference
    final_logits = None
    for i in range(n_chunks):
        chunk_start = i * middle_samples
        chunk_end = chunk_start + samples_per_chunk
        last_chunk = False
        if i == n_chunks - 1:
            last_chunk = True
        elif i == n_chunks - 2:
            if (input_values.shape[1] - 1) - chunk_end <= 2 * border_samples:
                chunk_end = input_values.shape[1]
                last_chunk = True
        chunk = input_values[:, chunk_start:chunk_end]

        with torch.no_grad():
            logits = model(chunk, attention_mask=torch.ones_like(chunk)).logits
            if args.device == 'cuda':
                logits = logits.cpu()
            logits = logits.numpy().reshape((-1, len(vocabulary)))

        left_border = 0 if i == 0 else border_samples
        right_border = 0 if last_chunk else border_samples
        start_index =  output_frames(left_border) if i > 0 else 0
        end_index = logits.shape[0] - (output_frames(right_border) if not last_chunk else 0)
        logits = logits[start_index:end_index, :]

        if final_logits is None:
            final_logits = logits
        else:
            final_logits = np.concatenate([final_logits, logits], axis=0)

        if last_chunk:
            break

    # Compute log probabilities
    log_probs = np.log(softmax(final_logits, axis=1))
    item['log_probs'] = log_probs
    item['index_duration'] = item['index_duration'] / log_probs.shape[0]

    # Load transcripts
    transcript_path = os.path.join(base_input_dir, item['transcript_path'])
    item['text'] = read_lines(transcript_path)
    item['text_normal'] = normalize_text(item['text'])

    # CTC Segmentation config
    config = cs.CtcSegmentationParameters()
    config.char_list = vocabulary
    config.min_window_size = args.window_len
    config.index_duration = item['index_duration']
    config.excluded_characters = ".,-?!:»«;'›‹()"
    config.blank = vocabulary.index("|")
    ground_truth_mat, utt_begin_indices = cs.prepare_text(config, item['text_normal'])
    config.blank = 0

    # Run CTC Segmentation
    ctc_seg_result = cs.ctc_segmentation(config, item['log_probs'], ground_truth_mat)
    timings, char_probs, char_list = ctc_seg_result
    segments = determine_utterance_segments(
        config,
        utt_begin_indices,
        char_probs,
        timings,
        item['text_normal'],
        char_list
    )

    # Write outputs
    next_audio_number = write_output(
        res_transcript,
        next_audio_number,
        args.output_dir,
        item['audio_path'],
        original_speech_array,
        original_sample_rate,
        segments,
        item['text'],
        args.threshold,
    )

# Evaluate the result
result_csv = read_csv(res_transcript, delimiter=' $ ')
n_total = len(result_csv)
n_success = sum(map(lambda l: l['audio_path'].endswith('.wav'), result_csv))
success_rate = round(n_success / n_total * 100, 2)
wavs = filter(lambda f: f.endswith('.wav'), os.listdir(args.output_dir))
wavs = map(lambda f: os.path.join(args.output_dir, f), wavs)
wavs = map(lambda p: get_duration(path=p), wavs)
durations = list(wavs)
total_duration = sum(durations)
duration = format_duration(total_duration)

# Write the report
with open(res_report, 'w', encoding='utf-8') as report:
    report.write(f'All Sentences: {n_total}\n')
    report.write(f'Number of Audios: {n_success}\n')
    report.write(f'Success Rate: {success_rate}%\n')
    report.write(f'Total Duration: {duration}\n')

# Plot duration histogram
durations = np.array(durations)
plt.figure(figsize=(14, 8))
plt.hist(durations, bins=30)
plt.title('Duration Histogram')
plt.xlabel('Audio Duration (seconds)')
plt.ylabel('Number of Audios')
plt.savefig(duration_hist_path)
