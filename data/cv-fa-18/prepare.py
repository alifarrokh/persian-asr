"""Perpare Persian Common Voice Dataset."""
import os
import re
import pathlib
import hashlib
import string
import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MIN_DURATION = 0.5
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


def create_jsonl(data: list[dict], filename: str):
    """Save `data` as a jsonl file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f'{json.dumps(item)}\n')


def load_jsonl(file_path) -> list[dict]:
    """Load a jsonl file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [json.loads(l) for l in lines]
    return lines


def load_json(json_path: str):
    """Load a json file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def md5(s: str):
    """Calculate the MD5 hash of the given string"""
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def prepare_dataset():
    """ Prepare the Common Voice """
    durations = pd.read_csv('dataset/clip_durations.tsv', delimiter='\t', low_memory=False)
    durations = {item['clip']:item['duration[ms]']/1000 for item in durations.to_dict('records')}

    validated = pd.read_csv('dataset/validated.tsv', sep='\t', low_memory=False, quoting=3, quotechar='')
    validated = validated[validated['sentence'].apply(lambda s: not bool(re.search(r'[a-zA-Z]+', s)))]

    current_file_path = pathlib.Path(__file__).parent.resolve()

    validated['id'] = validated['path'].apply(lambda p: p.replace('.mp3', ''))
    validated['audio_filepath'] = validated['path'].apply(lambda p: os.path.join(current_file_path, 'dataset/clips', p))
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


def proper_n_bins(x):
    """ Find the proper number of bins for plotting the histogram of x """
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins


def plot_duration_histogram():
    """ Plot the duration histogram of dataset splits """
    train_data = load_jsonl('train.jsonl')
    dev_data = load_jsonl('dev.jsonl')
    test_data = load_jsonl('test.jsonl')

    train_durations = np.array([item['duration'] for item in train_data])
    dev_durations = np.array([item['duration'] for item in dev_data])
    test_durations = np.array([item['duration'] for item in test_data])

    train_bins = proper_n_bins(train_durations)
    dev_bins = proper_n_bins(dev_durations)
    test_bins = proper_n_bins(test_durations)

    plt.hist(train_durations, density=True, bins=train_bins, color='blue', alpha=0.4)
    plt.hist(dev_durations, density=True, bins=dev_bins, color='red', alpha=0.4)
    plt.hist(test_durations, density=True, bins=test_bins, color='yellow', alpha=0.4)

    plt.legend(['Train', 'Dev', 'Test'])
    plt.xlabel('Duration (second)')
    plt.ylabel('Density (%)')
    plt.title('Audio Duration Histogram')
    plt.savefig('duration_histogram.png')


def save_metadata():
    """ Extract dataset metadata """
    format_name = lambda name: name.replace('.jsonl', '')
    subsets = [f for f in os.listdir('.') if f.endswith('.jsonl')]
    subsets = {format_name(name): load_jsonl(os.path.join('', name)) for name in subsets}

    metadata = {}
    for split_name, data in subsets.items():
        durations = [item['duration'] for item in data]
        metadata[split_name] = {
            'samples': len(data),
            'min(s)': f'{np.min(durations):.1f}',
            'max(s)': f'{np.max(durations):.1f}',
            'mean(s)': f'{np.mean(durations):.1f}',
            'total(h)': f'{sum(durations) / 3600:.1f}',
        }

    vocab = set(''.join([item['text'] for split in subsets.values() for item in split]))
    vocab = sorted(list(vocab), key=ord)
    vocab_dict = {c: idx for idx,c in enumerate(vocab)}
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f)

    with open('info.txt', 'w', encoding='utf-8') as f:
        f.write(str(pd.DataFrame(list(metadata.values()), index=list(metadata.keys()))) + '\n\n')
        f.write(f'Vocabulary: {len(vocab)}\n')
        f.write(str(vocab) + '\n')


if __name__ == '__main__':
    prepare_dataset()
    plot_duration_histogram()
    save_metadata()
