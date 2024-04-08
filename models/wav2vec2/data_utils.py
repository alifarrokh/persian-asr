"""Data Processing Utils for Wav2Vec2."""
from dataclasses import dataclass
import torch
import torchaudio
from torchaudio.functional import resample
from transformers import Wav2Vec2Processor


@dataclass
class WaveformDataCollator:
    """Waveform Data Collator for Wav2Vec2"""

    processor: Wav2Vec2Processor
    padding: bool | str = True

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": item["input_values"]} for item in items]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        if "labels" in items[0]:
            label_features = [{"input_ids": item["labels"]} for item in items]
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

            # Replace paddings with -100
            mask = labels_batch.attention_mask.ne(1)
            batch["labels"] = labels_batch["input_ids"].masked_fill(mask, -100)

        return batch


@dataclass
class SampleLoader:
    """Load and resample a wav file"""

    processor: Wav2Vec2Processor
    sample_rate: int

    def __call__(self, item: dict):
        waveform, orig_sr = torchaudio.load(item['audio_filepath'])
        waveform = waveform[:1, :] # Convert to mono

        if orig_sr != self.sample_rate:
            waveform = resample(waveform, orig_freq=orig_sr, new_freq=self.sample_rate)

        item['input_values'] = self.processor(waveform.squeeze(), sampling_rate=self.sample_rate).input_values[0]
        item['input_length'] = len(item['input_values'])

        if 'text' in item:
            item['labels'] = self.processor(text=item['text']).input_ids
        return item
