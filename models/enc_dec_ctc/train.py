"""Train a CTC-based encoder-decoder ASR model using NeMo toolkit."""
import os
import argparse
import nemo
import nemo.collections.asr as nemo_asr
import nemo.utils
from ruamel.yaml import YAML
from omegaconf import DictConfig
import pytorch_lightning as pl


# Arguments
parser = argparse.ArgumentParser(
    description="train a CTC-based encoder-decoder ASR model."
)
parser.add_argument(
    '--config', '-c',
    type=str,
    default="conf/conformer-medium.yaml",
    help="path to the config (.yaml) file",
)
args = parser.parse_args()

# Validate the args
if not args.config.endswith('yaml') or not os.path.exists(args.config):
    raise ValueError(f"Config file {args.config} is either invalid or does not exist.")

# Load the config
yaml = YAML(typ='safe')
with open(args.config, encoding='utf-8') as f:
    params = yaml.load(f)
config = DictConfig(params)

# Update the training args
with open(config.model.train_ds.manifest_filepath, encoding='utf-8') as f:
    n_train_samples = sum(1 for _ in f)
effective_batch_size = config.model.train_ds.batch_size * config.trainer.accumulate_grad_batches
steps_per_epoch = round(n_train_samples / effective_batch_size)

config.model.optim.sched.warmup_steps = config.vars.warmup_epochs * steps_per_epoch
config.trainer.val_check_interval = config.vars.log_interval
config.trainer.log_every_n_steps = max(1, int(config.vars.log_interval * steps_per_epoch))

# Create the model and trainer
trainer = pl.Trainer(**config.trainer)
exp_manager = nemo.utils.exp_manager.exp_manager(trainer, config.exp_manager)
model = nemo_asr.models.EncDecCTCModel(cfg=config.model, trainer=trainer)
trainer.fit(model)
