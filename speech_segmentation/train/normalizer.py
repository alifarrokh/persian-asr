"""
Clean the text in the 'sentence' column of a given CSV file

Some of the preprocessing methods were adapted from:
https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v3
"""
import os
import re
import string
import argparse
import pickle

import pandas as pd
import num2fawords
from parsivar import Normalizer

from dictionary import dictionary_mapping, fixator_dictionary


_normalizer = Normalizer(half_space_char="\u200c", statistical_space_correction=True)
chars_to_ignore = [
    ",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
    "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬", 'ٔ', ",", "?",
    ".", "!", "-", ";", ":", '"', "“", "%", "‘", "”", "�", "–", "…", "_", "”", '“', '„',
    'ā', 'š', 'ّ', 'ْ',
]
chars_to_ignore = chars_to_ignore + list(string.ascii_lowercase + string.digits)
chars_to_ignore = f"""[{"".join(chars_to_ignore)}]"""
ZWNJ = "\u200c"
silent_chars = ["ا", "د", "ذ", "ر", "ز", "و", "آ"] + [ZWNJ] + [" "]


def multiple_replace(text, chars_to_mapping):
    """Perform multiple replaces in the text."""
    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))


def remove_special_characters(text, chars_to_ignore_regex):
    """Remove a list of characters from the given text"""
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
    return text


def convert_word_nums_to_text(word):
    """Convert numbers to Persian words"""
    try:
        word = int(word)
        word = num2fawords.words(word)
    except ValueError as _e:
        pass
    return word


def normalizer_at_word_level(text):
    """Perform world-level normalization"""
    words = text.split()
    _text = []

    for word in words:
        word = convert_word_nums_to_text(word)
        word = fixator_dictionary.get(word, word)

        _text.append(word)

    return " ".join(_text) + " "


def finder(ss, s, starter=False):
    """Return the list of occurrences of ss in s"""
    found = []
    for m in re.finditer(ss, s):
        if starter:
            found.append(m.start())
        else:
            found.append((m.start(), m.end()))
    return found


def substring_replace(ss, s, start, end, stripped=True):
    """Replace a part of string s with ss given indices of boundaries"""
    s_start = s[:start]
    s_end = s[end:]

    counter = 0
    if stripped:
        counter = 1 if s_start.endswith(" ") else counter
        s_start = s_start.rstrip()

    return s_start + ss + s_end, counter


def fix_connected_specials(text, special_list):
    """Fix zero width non joiner connection for a list of special strings"""
    for special in special_list:
        pointer = 0
        for f in sorted(finder(special, text, False)):
            start, end = f[0] + pointer - 1, f[1] + pointer - 1
            if len(text) >= (end + 1):
                if len(text) == (end + 1):
                    new_text, extra_pointer = substring_replace(
                        f"{ZWNJ}{special}",
                        text,
                        start + 1,
                        end + 1,
                        stripped=True
                    )
                    text = new_text
                    pointer += 1 + 1 - 1 - extra_pointer
                else:
                    if text[end + 1] == " ":
                        new_text, extra_pointer = substring_replace(
                            f"{ZWNJ}{special}",
                            text,
                            start + 1,
                            end + 1,
                            stripped=True
                        )
                        text = new_text
                        pointer += 1 + 1 - 1 - extra_pointer
    return text


def clean(text, is_normalize=True, filter_trivials=False):
    """Clean the text"""

    # Filter empty/non-str inputs
    if not isinstance(text, str):
        return None
    if len(text.strip()) == 0:
        return None
    text = text.lower().strip()

    # Parsivar normalizer
    if is_normalize:
        text = _normalizer.normalize(text)

    # Dictionary mapping
    text = multiple_replace(text, dictionary_mapping)
    text = re.sub(" +", " ", text)

    # Remove specials
    text = remove_special_characters(text, chars_to_ignore)
    text = re.sub(" +", " ", text)

    # Replace connected آ
    pointer = 0
    special = "آ"
    for f in sorted(finder(special, text, True)):
        index = f + pointer - 1
        if len(text) >= index:
            if text[index] not in silent_chars:
                new_text, extra_pointer = substring_replace(
                    f"{text[index]}{ZWNJ}",
                    text,
                    index,
                    index + 1,
                    stripped=True
                )
                text = new_text
                pointer += 1 + 1 - 1 - extra_pointer

    # Replace connected ها
    special_list = [
        # "ام", "ای", "است", "ایم", "اید", "اند",
        "هایمان", "هایم", "هایت", "هایش",
        "هایتان", "هایشان", "هام", "هات",
        "هاتان", "هامون", "هامان", "هاش",
        "هاتون", "هاشان", "هاشون",
        "هایی", "های", "هاس", "ها"
    ]
    text = fix_connected_specials(text, special_list)

    # Fix connected افزار
    pointer = 0
    special = "افزار"
    for f in sorted(finder(special, text, False)):
        start, end = f[0] + pointer - 1, f[1] + pointer - 1

        if len(text) >= (end + 1):
            new_text, extra_pointer = substring_replace(
                f"{ZWNJ}{special}",
                text,
                start + 1,
                end + 1,
                stripped=True
            )
            text = new_text
            pointer += 1 + 1 - 1 - extra_pointer

    # Replace connected تر/ترین
    special_list = [
        "ترین", "تر"
    ]
    text = fix_connected_specials(text, special_list)

    # Normalizer at word level
    text = normalizer_at_word_level(text)
    text = re.sub(" +", " ", text)
    text = text.strip()

    # Filter short text
    if filter_trivials:
        if not len(text) > 2:
            text = None

    # Final replacements (not required for common-voice)
    non_cv_replace_dict = {
        '٪': ' درصد ',
        '٥': ' پنج ',
    }
    text = multiple_replace(text, non_cv_replace_dict)

    # Fix spaces
    text = re.sub(' +', ' ', text)
    text = text.strip()

    return text


def is_valid(text):
    """Check whether given text is valid, i.e., non-empty with non-english characters"""
    invalid_chars = string.ascii_letters
    for c in invalid_chars:
        if c in text:
            return False
    if len(text.strip()) == 0:
        return False
    return True


def clean_csv(csv_path, sep=','):
    """Clean the all sentences in a given CSV file"""
    # Load the CSV file
    csv_dir, csv_name = os.path.split(csv_path)
    csv_name_no_ext = ''.join(csv_name.split('.')[:-1])
    df = pd.read_csv(csv_path, sep=sep)
    assert 'sentence' in df.columns, "The CSV file does not contain 'sentence' column."
    len_all = len(df)
    print(f'Processing CSV: {csv_name}')
    print(f'All Samples: {len_all}')

    # Clean sentences
    valid_rows = df.apply(lambda row: is_valid(row['sentence']), axis=1)
    df = df[valid_rows]
    df['sentence'] = df['sentence'].apply(clean)
    df = df[df['sentence'].notnull()]
    print(f'Clean Samples: {len(df)} ({len(df)*100/len_all:0.3f}%)')

    # Create & save the vocabulary
    sentences = df['sentence'].tolist()
    vocab = list(set(list(' '.join(sentences))))
    vocab = sorted(vocab, key=ord)
    print(f'Vocabulary Size: {len(vocab)}')
    vocab_path = os.path.join(csv_dir, f'{csv_name_no_ext}_vocab.pickle')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    # Save the clean CSV
    clean_csv_path = os.path.join(csv_dir, f'{csv_name_no_ext}_clean.csv')
    df.to_csv(clean_csv_path, index=False, encoding='utf-8')


if __name__ == '__main__':

    # Argparse
    parser = argparse.ArgumentParser(description='Persian Text Preprocessing')
    parser.add_argument('--csv_path', required=True, type=str, help='Path to the input CSV file')
    parser.add_argument(
        '--delimiter',
        default=',',
        type=str,
        help='CSV delimiter (Enter $\'\\t\' for tab)'
    )

    # Parse & validate args
    args = parser.parse_args()
    assert os.path.exists(args.csv_path), "The input CSV file does not exist."

    clean_csv(args.csv_path, sep=args.delimiter)
