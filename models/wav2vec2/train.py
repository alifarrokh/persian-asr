"""Train a CTC-based Wav2Vec2 ASR model using Huggingface Transformers."""
import os
import warnings
import argparse
import numpy as np
import pandas as pd
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from evaluate import load as load_metric
from datasets import Dataset
from omegaconf import OmegaConf
from data_utils import WaveformDataCollator, SampleLoader


# Disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Arguments
parser = argparse.ArgumentParser(
    description="train a CTC-based Wav2Vec2 ASR model."
)
parser.add_argument(
    '--config', '-c',
    type=str,
    default="conf/xls-r-300m.yaml",
    help="path to the config (.yaml) file",
)
args = parser.parse_args()

# Validate the args
if not args.config.endswith('yaml') or not os.path.exists(args.config):
    raise ValueError(f"Config file {args.config} is either invalid or does not exist.")

# Load the config
config = OmegaConf.load(args.config)
base_model_args = OmegaConf.to_container(config.base_model)
model_sample_rate = base_model_args.pop('sample_rate')

output_dir = os.path.join('exps', config.name)
if os.path.exists(output_dir):
    raise ValueError(f"Experiment directory {output_dir} already exists.")

# Create the tokenizer, feature extractor, and wav2vec2 processor
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    config.dataset.root_path,
    unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token=' '
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=model_sample_rate,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
data_collator = WaveformDataCollator(processor=processor, padding=True)
sample_loader = SampleLoader(processor, model_sample_rate)

# Load and prepare the dataset
train_dataset = pd.read_json(config.dataset.train_manifest, lines=True)
dev_dataset = pd.read_json(config.dataset.dev_manifest, lines=True)

train_dataset = Dataset.from_pandas(train_dataset)
dev_dataset = Dataset.from_pandas(dev_dataset)

train_map_args = {'keep_in_memory': not config.dataset.cache_train_samples}
if not train_map_args['keep_in_memory']:
    train_map_args['cache_file_name'] = config.dataset.train_cache_path
dev_map_args = {'keep_in_memory': not config.dataset.cache_dev_samples}
if not dev_map_args['keep_in_memory']:
    dev_map_args['cache_file_name'] = config.dataset.dev_cache_path

train_dataset = train_dataset.map(
    sample_loader,
    remove_columns=train_dataset.column_names,
    **train_map_args
)
dev_dataset = dev_dataset.map(
    sample_loader,
    remove_columns=dev_dataset.column_names,
    **dev_map_args
)

# Define the evaluation metric
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")
def compute_metrics(pred):
    """Compute evaluation metrics"""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


# Load the model
freeze_feature_extractor = base_model_args.pop('freeze_feature_extractor')
model = Wav2Vec2ForCTC.from_pretrained(
    base_model_args.pop('checkpoint'),
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    **base_model_args
)

# Freeze the CNN feature extractor
if freeze_feature_extractor:
    model.freeze_feature_extractor()

# Define the training args
eval_ratio = config.eval_ratio
effective_batch_size = config.train.per_device_train_batch_size * config.train.gradient_accumulation_steps
steps_per_epoch = int(len(train_dataset) / effective_batch_size)
log_steps = max(int(steps_per_epoch * eval_ratio), 1)

training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=os.path.join(output_dir, 'logs'),

    save_strategy="steps",
    logging_strategy="steps",
    evaluation_strategy="steps",

    save_steps=log_steps,
    logging_steps=log_steps,
    eval_steps=log_steps,

    **OmegaConf.to_container(config.train)
)

# Fine-tune the model
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=processor.feature_extractor,
)
trainer.train()
