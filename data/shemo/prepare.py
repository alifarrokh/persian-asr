"""Perpare Shemo Dataset."""
from os.path import dirname, abspath
import sys

# Add the parent directory to sys.path
sys.path.append(dirname(dirname(abspath(__file__))))

import os
import pathlib
import librosa
from utils import (
    create_jsonl,
    plot_duration_histogram,
    save_dataset_info,
    load_json,
)


DATASET_DIR = pathlib.Path(__file__).parent.resolve()


def clean_text(t: str):
    """Preprocess the given sentence"""
    # TODO: Implement this function!
    return t


def prepare_dataset():
    """Prepare Shemo Dataset"""
    wav_names = [f for f in os.listdir('dataset/wavs') if f.endswith('.wav')]
    assert len(wav_names) == 3000
    data = []
    for wav_name in wav_names:
        file_id = wav_name.replace('.wav', '')
        audio_path = os.path.join(DATASET_DIR, 'dataset/wavs', wav_name)
        text_path = os.path.join(DATASET_DIR, 'dataset/final text', file_id + '.ort')
        # with open(text_path, encoding='utf-8') as f:
        #     text = f.read().strip()
        # print(text)
        if not os.path.exists(text_path):
            print(text_path)
        data.append({
            'id': file_id,
            'text': clean_text(''),
            'audio_filepath': audio_path,
            'duration': librosa.get_duration(path=audio_path),
        })
    exit()
    create_jsonl(data, 'all.jsonl')


if __name__ == '__main__':
    prepare_dataset()
    plot_duration_histogram(DATASET_DIR)
    save_dataset_info(DATASET_DIR)
