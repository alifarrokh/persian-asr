import json
import math
import os
import numpy as np
import soundfile as sf


def read_lines(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines if l.strip()]
    return lines


def read_csv(csv_path, delimiter=','):
    lines = read_lines(csv_path)
    lines = list(map(lambda l: list(map(str.strip, l.split(delimiter))), lines))
    keys = lines[0]
    lines = lines[1:]
    res = []
    for line in lines:
        res.append({keys[i]:line[i] for i in range(len(keys))})
    return res


def get_vocab():
    vocab = json.load(open('vocab.json'))
    vocab = sorted(list(vocab.items()), key=lambda x: x[1])
    vocab = list(map(lambda x: x[0], vocab))
    return vocab


def normalize_text(transcript):
    vocab = get_vocab()
    lines = map(lambda l: l.replace(' ', '|'), transcript)
    lines = map(lambda l: ''.join([c for c in l if c in vocab]).strip(), lines)
    lines = list(lines)
    return lines


def _compute_time(index, align_type, timings):
    """Compute start and end time of utterance.
    Adapted from https://github.com/lumaku/ctc-segmentation

    Args:
        index:  frame index value
        align_type:  one of ["begin", "end"]

    Return:
        start/end time of utterance in seconds
    """
    middle = (timings[index] + timings[index - 1]) / 2
    if align_type == "begin":
        return max(timings[index + 1] - 0.5, middle)
    elif align_type == "end":
        return min(timings[index - 1] + 0.5, middle)


def determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text, char_list):
    """Utterance-wise alignments from char-wise alignments.
    Adapted from https://github.com/lumaku/ctc-segmentation

    Args:
        config: an instance of CtcSegmentationParameters
        utt_begin_indices: list of time indices of utterance start
        char_probs:  character positioned probabilities obtained from backtracking
        timings: mapping of time indices to seconds
        text: list of utterances
    Return:
        segments, a list of: utterance start and end [s], and its confidence score
    """
    segments = []
    min_prob = np.float64(-10000000000.0)
    for i in range(len(text)):
        start = _compute_time(utt_begin_indices[i], "begin", timings)
        end = _compute_time(utt_begin_indices[i + 1], "end", timings)

        start_t = start / config.index_duration_in_seconds
        start_t_floor = math.floor(start_t)

        # look for the left most blank symbol and split in the middle to fix start utterance segmentation
        if char_list[start_t_floor] == config.char_list[config.blank]:
            start_blank = None
            j = start_t_floor - 1
            while char_list[j] == config.char_list[config.blank] and j > start_t_floor - 20:
                start_blank = j
                j -= 1
            if start_blank:
                start_t = int(round(start_blank + (start_t_floor - start_blank) / 2))
            else:
                start_t = start_t_floor
            start = start_t * config.index_duration_in_seconds

        else:
            start_t = int(round(start_t))

        end_t = int(round(end / config.index_duration_in_seconds))

        # Compute confidence score by using the min mean probability after splitting into segments of L frames
        n = config.score_min_mean_over_L
        if end_t <= start_t:
            min_avg = min_prob
        elif end_t - start_t <= n:
            min_avg = char_probs[start_t:end_t].mean()
        else:
            min_avg = np.float64(0.0)
            for t in range(start_t, end_t - n):
                min_avg = min(min_avg, char_probs[t : t + n].mean())
        segments.append((start, end, min_avg))
    return segments


def write_output(output, next_audio_number, output_dir, audio_path, speech_signal, sr, segments, text, th, delimiter='$'):
    d = f' {delimiter} '
    
    if not os.path.exists(output):
        with open(output, 'w', encoding='utf-8') as outfile:
            outfile.write(f'original_audio_path{d}start{d}end{d}score{d}audio_path{d}text\n')
    
    with open(output, 'a', encoding='utf-8') as outfile:
        for i, segment in enumerate(segments):
            start, end, score = segment
            audio_name = f'{next_audio_number}.wav'
            segment_audio_path = os.path.join(output_dir, audio_name) if score > th else 'INVALID'
            outfile.write(f'{audio_path}{d}{start}{d}{end}{d}{score}{d}{segment_audio_path}{d}{text[i]}\n')
            if score > th:
                start_index, end_index = int(start * sr), int(end * sr)
                segment_signal = speech_signal[start_index:end_index]
                sf.write(segment_audio_path, segment_signal, sr, subtype='PCM_24')
                next_audio_number += 1
    
    return next_audio_number


def format_duration(total_duration):
    unit = 'seconds'
    if total_duration > 60:
        total_duration /= 60
        unit = 'minutes'
    if total_duration > 60:
        total_duration /= 60
        unit = 'hours'
    total_duration = round(total_duration, 2)
    return f'{total_duration} {unit}'
