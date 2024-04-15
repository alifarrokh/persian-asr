import os
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_jsonl(data: list[dict], filename: str):
    """Save data as a jsonl file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f'{json.dumps(item)}\n')


def load_jsonl(file_path) -> list[dict]:
    """Load a jsonl file as a list of dictionaries"""
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


def proper_n_bins(x):
    """Find the proper number of bins for plotting the histogram of x"""
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins



def plot_duration_histogram(ds_root: str):
    """Plot the duration histogram of dataset splits"""
    subsets = [f for f in os.listdir(ds_root) if f.endswith('.jsonl')]
    subsets = {name: load_jsonl(os.path.join(ds_root, name)) for name in subsets}

    names = []
    for subset_name, data in subsets.items():
        names.append(subset_name.replace('.jsonl', '').capitalize())
        durations = np.array([item['duration'] for item in data])
        bins = proper_n_bins(durations)
        plt.hist(durations, density=True, bins=bins, alpha=0.4)

    plt.legend(names)
    plt.xlabel('Duration (second)')
    plt.ylabel('Density (%)')
    plt.title('Audio Duration Histogram')
    plt.savefig('duration_histogram.png')


def save_dataset_info(ds_root: str):
    """Extract dataset info"""
    subsets = [f for f in os.listdir(ds_root) if f.endswith('.jsonl')]
    subsets = {name.replace('.jsonl', ''): load_jsonl(os.path.join(ds_root, name)) for name in subsets}

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

    # Save the vocabulary
    vocab = set(''.join([item['text'] for split in subsets.values() for item in split]))
    vocab = sorted(list(vocab), key=ord)
    vocab_dict = {c: idx for idx,c in enumerate(vocab)}
    with open(os.path.join(ds_root, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f)

    with open(os.path.join(ds_root, 'info.txt'), 'w', encoding='utf-8') as f:
        f.write(str(pd.DataFrame(list(metadata.values()), index=list(metadata.keys()))) + '\n\n')
        f.write(f'Vocabulary: {len(vocab)}\n')
        f.write(str(vocab) + '\n')
