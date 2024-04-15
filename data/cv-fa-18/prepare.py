"""Perpare Persian Common Voice Dataset."""
from os.path import dirname, abspath
import sys

# Add the parent directory to sys.path
sys.path.append(dirname(dirname(abspath(__file__))))

import os
import re
import pathlib
import string
import random
import pandas as pd
from utils import create_jsonl, plot_duration_histogram, save_dataset_info


DATASET_DIR = pathlib.Path(__file__).parent.resolve()
MIN_DURATION = 0.5 # TODO: do not filter samples here
MAX_DURATION = 11
SEED = 42


def clean_text(t: str):
    """Preprocess the given sentence"""
    all_punct = list(set('!"#&\'(),-.:;=_»«،…”“–؛؟٬' + string.punctuation))
    for c in all_punct:
        t = t.replace(c, ' ')

    optional_persian_replace = 'ء'
    for c in optional_persian_replace:
        t = t.replace(c, ' ')

    persian_remove = '\u0640\u064b\u064c\u064e\u064f\u0650\u0651\u0652\u0654'
    for c in persian_remove:
        t = t.replace(c, '')

    persian_replace = {
        'ة': 'ت',
        'ﺘ': 'ت',
        'ﺖ': 'ت',
        'ك': 'ک',
        'ﮐ': 'ک',
        'ﮔ': 'گ',
        'ي': 'ی',
        'ﯿ': 'ی',
        'ﯾ': 'ی',
        'ﯽ': 'ی',
        'ے': 'ی',
        'ى': 'ی',
        'ۀ': 'ه',
        'ﻮ': 'و',
        'ﻭ': 'و',
        'ؤ': 'و',
        'ﻪ': 'ه',
        'ە': 'ه',
        'ﻧ': 'ن',
        'ﻥ': 'ن',
        'ﻤ': 'م',
        'ﻢ': 'م',
        'ﻡ': 'م',
        'ﻟ': 'ل',
        'ﻌ': 'ع',
        'ﻋ': 'ع',
        'ﻀ': 'ض',
        'ﺸ': 'ش',
        'ﺷ': 'ش',
        'ﺴ': 'س',
        'ﺱ': 'س',
        'ﺮ': 'ر',
        'ﺭ': 'ر',
        'ﺪ': 'د',
        'ﺩ': 'د',
        'ﺧ': 'خ',
        'ﺒ': 'ب',
        'ﺑ': 'ب',
        'ﺎ': 'ا',
        'ﺍ': 'ا',
        'ﭘ': 'پ',
    }
    for c_old, c_new in persian_replace.items():
        t = t.replace(c_old, c_new)

    t = re.sub(r'[\r\n\s]+', ' ', t).strip()

    charset = set('آأابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئ ')
    assert set(t).issubset(charset)

    return t


def load_subset_ids(subset_name: str) -> list:
    """Return the list of ids available in the given Common Voice subset"""
    if subset_name not in ['train', 'dev', 'test']:
        raise ValueError(f'{subset_name} is not a valid common voice split.')
    df = pd.read_csv(f'dataset/{subset_name}.tsv', sep='\t', low_memory=False, quoting=3, quotechar='')
    file_names = df['path'].to_list()
    ids = [fname.replace('.mp3', '') for fname in file_names]
    return ids


def prepare_dataset():
    """Prepare Common Voice Fa 18"""
    durations = pd.read_csv('dataset/clip_durations.tsv', delimiter='\t', low_memory=False)
    durations = {item['clip']:item['duration[ms]']/1000 for item in durations.to_dict('records')}

    validated = pd.read_csv('dataset/validated.tsv', sep='\t', low_memory=False, quoting=3, quotechar='')
    validated = validated[validated['sentence'].apply(lambda s: not bool(re.search(r'[a-zA-Z]+', s)))]

    validated['id'] = validated['path'].apply(lambda p: p.replace('.mp3', ''))
    validated['audio_filepath'] = validated['path'].apply(lambda p: os.path.join(DATASET_DIR, 'dataset/clips', p))
    validated['duration'] = validated['path'].apply(lambda p: durations[p])
    validated['text'] = validated['sentence'].apply(clean_text)
    validated = validated[['id', 'sentence_id', 'audio_filepath', 'text', 'duration']]

    # Filter too short/long samples
    validated = validated[(MIN_DURATION < validated['duration']) & (validated['duration'] < MAX_DURATION)]

    # Split data
    validated = validated.to_dict('records')
    dev_ids = load_subset_ids('dev')
    test_ids = load_subset_ids('test')
    eval_ids = set(dev_ids + test_ids)
    eval_sentence_ids = [sample['sentence_id'] for sample in validated if sample['id'] in eval_ids]
    
    train_data = [sample for sample in validated if (sample['id'] not in eval_ids) and (sample['sentence_id'] not in eval_sentence_ids)]
    dev_data = [sample for sample in validated if sample['id'] in dev_ids]
    test_data = [sample for sample in validated if sample['id'] in test_ids]

    random.seed(SEED)
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    # Create jsonl files
    create_jsonl(train_data, 'train.jsonl')
    create_jsonl(dev_data, 'dev.jsonl')
    create_jsonl(test_data, 'test.jsonl')


if __name__ == '__main__':
    prepare_dataset()
    plot_duration_histogram(DATASET_DIR)
    save_dataset_info(DATASET_DIR)
