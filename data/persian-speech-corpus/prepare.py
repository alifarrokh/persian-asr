"""Perpare Persian Speech Corpus."""
from os.path import dirname, abspath
import sys

# Add the parent directory to sys.path
sys.path.append(dirname(dirname(abspath(__file__))))

import os
import pathlib
import librosa
from utils import create_jsonl, plot_duration_histogram, save_dataset_info


DATASET_DIR = pathlib.Path(__file__).parent.resolve()


def clean_text(t: str):
    """Preprocess the given sentence"""
    # TODO: Implement this function!
    return t


def prepare_dataset():
    """Prepare Persian Speech Corpus"""
    with open('dataset/orthographic-transcript.txt', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]
    data = []
    for line in lines:
        wav_name, text = line.split(' ', maxsplit=1)
        wav_name = wav_name.replace('"', '').strip()
        audio_path = os.path.join(DATASET_DIR, 'dataset/wav', wav_name)

        # A few wav files do not exist
        if os.path.exists(audio_path):
            data.append({
                'id': wav_name.replace('.wav', ''),
                'text': clean_text(text.strip().lstrip('"').rstrip('"').strip()),
                'audio_filepath': audio_path,
                'duration': librosa.get_duration(path=audio_path),
            })
    create_jsonl(data, 'all.jsonl')


if __name__ == '__main__':
    prepare_dataset()
    plot_duration_histogram(DATASET_DIR)
    save_dataset_info(DATASET_DIR)
