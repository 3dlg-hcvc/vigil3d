import clip
import torch.nn as nn
from omegaconf import DictConfig

def get_clip(cfg: DictConfig) -> nn.Module:
    model, _ = clip.load(cfg.get("model_type", "ViT-B/32"), device=cfg.get("device", "cuda"), jit=cfg.get("jit", False))
    return model
