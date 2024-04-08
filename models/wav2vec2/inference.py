"""Run inference on a pretrained CTC-based encoder-decoder ASR model."""
import os
import argparse
import warnings
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
from evaluate import load as load_metric
import pyctcdecode
from data_utils import WaveformDataCollator, SampleLoader


# Disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Arguments
parser = argparse.ArgumentParser(
    description="Run inference on a pretrained CTC-based encoder-decoder ASR model."
)
parser.add_argument(
    '--manifest', '-m',
    type=str,
    required=True,
    help="path to inference manifest (.jsonl)",
)
parser.add_argument(
    '--checkpoint', '-c',
    type=str,
    default="alifarokh/wav2vec2-xls-r-300m-fa",
    help="path to a checkpoint or a huggingface repository",
)
parser.add_argument(
    '--preds', '-p',
    type=str,
    default="preds.txt",
    help="path to the file where predicted results will be saved",
)
parser.add_argument('--batch-size', '-b', type=int, default=64, help="inference batch size")
parser.add_argument('--beam', type=int, default=5, help="decoding beam width")
parser.add_argument('--lm', type=str, help="path to language model (.bin/.arpa)")
parser.add_argument('--alpha', type=float, default=0.5, help="weight for language model")
parser.add_argument('--beta', type=float, default=1.5, help="weight for length score adjustment")
args = parser.parse_args()

# Validate the args
if not args.manifest.endswith('.jsonl') or not os.path.exists(args.manifest):
    raise ValueError(f"Manifest file {args.manifest} is either invalid or does not exist.")
if args.lm is not None and not os.path.exists(args.lm):
    raise ValueError(f"Language model {args.lm} does not exist.")

# Create the tokenizer, feature extractor, and wav2vec2 processor
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.checkpoint)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.checkpoint)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
data_collator = WaveformDataCollator(processor=processor, padding=True)
sample_loader = SampleLoader(processor, feature_extractor.sampling_rate)

# Load the dataset
dataset = pd.read_json(args.manifest, lines=True)
dataset = Dataset.from_pandas(dataset)

do_eval = 'text' in dataset.column_names
if do_eval:
    labels = dataset['text']
    dataset = dataset.remove_columns(['text'])

dataset = dataset.map(sample_loader, remove_columns=dataset.column_names)

# Load the model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint).to(DEVICE)

# Prepare the CTC decoder
vocab_list = list(tokenizer.get_vocab().keys())
decoder_args = {}
if args.lm != '':
    decoder_args['kenlm_model_path'] = args.lm
    decoder_args['alpha'] = args.alpha
    decoder_args['beta'] = args.beta
ctc_decoder = pyctcdecode.build_ctcdecoder(vocab_list, **decoder_args)

# Inference
preds = []
test_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)
with open(args.preds, 'w', encoding='utf-8') as out:
    for batch in tqdm(test_dataloader, desc='Decoding'):
        batch = {k:v.to(DEVICE) for k,v in batch.items()}

        with torch.no_grad():
            logits = model(**batch).logits.cpu().numpy()

        for i in range(logits.shape[0]):
            log_probs = logits[i]
            pred_str = ctc_decoder.decode(log_probs, beam_width=args.beam)
            out.write(pred_str + '\n')
            preds.append(pred_str)

# Evaluation
if do_eval:
    wer = load_metric('wer').compute(predictions=preds, references=labels)
    cer = load_metric('cer').compute(predictions=preds, references=labels)
    print(f'WER(%): {wer*100:.2f}')
    print(f'CER(%): {cer*100:.2f}')
