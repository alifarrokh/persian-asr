"""
Fine-tune XLSR-53 on Persian
"""
import os
import json
import argparse
import warnings

import numpy as np
import torchaudio

from datasets import load_metric
from transformers import TrainingArguments, Trainer
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from utils import load_ds, DataCollatorCTCWithPadding


# Disable warnings & WANDB
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

# Argparse
parser = argparse.ArgumentParser(description='Fine-tune XLSR-53')
parser.add_argument(
    '--train_csv',
    required=True,
    type=str,
    help='Path to the train CSV dataset containing "path" and "sentence" columns'
)
parser.add_argument(
    '--valid_csv',
    required=True,
    type=str,
    help='Path to the validation CSV dataset containing "path" and "sentence" columns'
)
parser.add_argument(
    '--wav_dir',
    required=True,
    type=str,
    help='Path to the directory where all .wav files are stored'
)

# Parse & validate args
args = parser.parse_args()
assert os.path.exists(args.train_csv), "The train CSV dataset does not exist."
assert os.path.exists(args.valid_csv), "The validation CSV dataset does not exist."
assert os.path.isdir(args.wav_dir), "The Wavs directory does not exist."

# Load datasets
train_ds = load_ds(args.train_csv, args.wav_dir)
valid_ds = load_ds(args.valid_csv, args.wav_dir)
print('==> Datasets loaded.')
print(f'Train samples: {len(train_ds)}')
print(f'Validation samples: {len(valid_ds)}\n')

# Create vocab
all_sentences = train_ds['sentence'] + valid_ds['sentence']
all_chars = sorted(list(set(' '.join(all_sentences))), key=ord)
vocab_dict = {v: k for k, v in enumerate(all_chars)}
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(f'Vocabulary Size: {len(vocab_dict)}')

# Save the vocab
with open('vocab.json', 'w', encoding='utf-8') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# Create tokenizer, feature extractor and processor
tokenizer = Wav2Vec2CTCTokenizer(
    "./vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token=" "
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(item):
    """Prepare a dataset by loading and processing its audio files"""
    a, sr = torchaudio.load(item['path'])
    a = torchaudio.functional.resample(a, orig_freq=sr, new_freq=16000).numpy().flatten()

    # batched output is "un-batched"
    item["input_values"] = processor(a, sampling_rate=16000).input_values[0]

    with processor.as_target_processor():
        item["labels"] = processor(item["sentence"]).input_ids

    del a
    return item


train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names)
valid_ds = valid_ds.map(prepare_dataset, remove_columns=valid_ds.column_names)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")


def compute_metrics(pred):
    """Compute WER for the given predictions"""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.2,
    hidden_dropout=0.2,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.2,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
model.freeze_feature_encoder()
model.gradient_checkpointing_enable()


training_args = TrainingArguments(
    output_dir="./xlsr-fa",
    group_by_length=True,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=100,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=3e-4,
    warmup_steps=1500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds.with_format("torch"),
    eval_dataset=valid_ds.with_format("torch"),
    tokenizer=processor.feature_extractor,
)
trainer.train()
