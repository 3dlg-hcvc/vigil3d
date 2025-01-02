from typing import Optional

from omegaconf import DictConfig

import clip
import torch


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, **kwargs):
        self.text = kwargs["text"]
        self.prompt = cfg.get("prompt", kwargs.get("prompt"))  # expects a string with a single {} for formatting
        self.delimiter = cfg.get("delimiter")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx: int):
        raw_text = self.text[idx]

        # preprocessing
        if isinstance(raw_text, str):
            if self.delimiter is not None:
                parsed_text = raw_text.split(self.delimiter)
            else:
                parsed_text = raw_text

            if self.prompt:
                parsed_text = [self.prompt.format(t) for t in parsed_text]
        elif isinstance(raw_text, list):
            parsed_text = list(raw_text)  # copy
            if self.prompt:
                parsed_text = [self.prompt.format(t) for t in parsed_text]

        out = clip.tokenize(parsed_text, truncate=True)
        return out


def clip_collation_fn(batch):
    return torch.cat(batch, dim=0)
