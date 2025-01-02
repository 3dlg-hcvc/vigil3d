import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from .env import log_slurm


def prepare_trainer(cfg: DictConfig):
    args = OmegaConf.to_container(cfg, resolve=True)
    if cfg.get("callbacks"):
        args["callbacks"] = [hydra.utils.instantiate(callback) for callback in cfg.callbacks]

    if cfg.get("logger"):
        args["logger"] = hydra.utils.instantiate(cfg.logger, _convert_="all")

        if cfg.logger.name == "NeptuneLogger":
            log_slurm(args["logger"].experiment)

    if cfg.get("profiler"):
        args["profiler"] = getattr(L.profilers, cfg.profiler.name)(**cfg.profiler.params)

    return args


def predict(
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    trainer_args: DictConfig,
    save_dir: str,
    return_predictions: bool = True,
):
    trainer_args = prepare_trainer(trainer_args)
    trainer = L.Trainer(**trainer_args, default_root_dir=save_dir)
    return trainer.predict(model=model, datamodule=datamodule, return_predictions=return_predictions)
