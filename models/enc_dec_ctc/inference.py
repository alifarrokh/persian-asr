"""Run inference on a pretrained CTC-based encoder-decoder ASR model."""
import os
import json
import logging
import argparse
import warnings
from tqdm import tqdm
from nemo.collections.asr.models import EncDecCTCModel
from evaluate import load as load_metric
import pyctcdecode


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
    default="alifarokh/nemo-conformer-medium-fa",
    help="path to a nemo checkpoint (.nemo) or a huggingface repository",
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

# Load the model
logging.getLogger('nemo_logger').setLevel(logging.ERROR)
model = EncDecCTCModel.from_pretrained(args.checkpoint)

# Load the test dataset
with open(args.manifest, encoding='utf-8') as f:
    data = [json.loads(l) for l in f.readlines()]
audio_paths = [item['audio_filepath'] for item in data]

do_eval = 'text' in data[0]
labels = [item['text'] for item in data] if do_eval else None

# Prepare the CTC decoder
alphabet = model.cfg.labels
vocab_list = alphabet + ['[PAD]'] # Add CTC blank (and padding) token

decoder_args = {}
if args.lm != '':
    decoder_args['kenlm_model_path'] = args.lm
    decoder_args['alpha'] = args.alpha
    decoder_args['beta'] = args.beta
ctc_decoder = pyctcdecode.build_ctcdecoder(vocab_list, **decoder_args)

# Inference
preds = []
hyps = model.transcribe(audio_paths, batch_size=args.batch_size, return_hypotheses=True)
with open(args.preds, 'w', encoding='utf-8') as out:
    for hyp in tqdm(hyps, desc='Decoding'):
        log_probs = hyp.y_sequence.numpy()
        pred_str = ctc_decoder.decode(log_probs, beam_width=args.beam)
        out.write(pred_str + '\n')
        preds.append(pred_str)

# Evaluation
if do_eval:
    wer = load_metric('wer').compute(predictions=preds, references=labels)
    cer = load_metric('cer').compute(predictions=preds, references=labels)
    print(f'WER(%): {wer*100:.2f}')
    print(f'CER(%): {cer*100:.2f}')
