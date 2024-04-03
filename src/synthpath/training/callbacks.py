import os
import shutil
import time
from os.path import join
from typing import Any, Dict, Literal

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


class ModelCheckpointEdit(ModelCheckpoint):

    """
    The default ModelCheckpoint tries to set the last.ckpt checkpoint as a symbolic link to another checkpoint when possible.
    This is problematic when the logging directory is on a filesystem for which symlink permission is not granted.
    """

    @staticmethod
    def _link_checkpoint(trainer: "pl.Trainer", filepath: str, linkpath: str) -> None:
        if trainer.is_global_zero:
            if os.path.lexists(linkpath):
                os.remove(linkpath)
            try:
                os.symlink(filepath, linkpath)
            except OSError:
                shutil.copy(filepath, linkpath)
        trainer.strategy.barrier()


class TimeIteration(Callback):

    """Average time to process a batch"""

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.train_dataloader_len = len(trainer.train_dataloader)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch_time = time.perf_counter() - pl_module.epoch_start_time
        mean_iter_time = epoch_time / pl_module.train_dataloader_len
        pl_module.log("mean_iter_time", mean_iter_time, on_step=False, on_epoch=True)


class SimpleWandbCheckpointing(Callback):
    def __init__(
        self, save_checkpoints: Literal["on_end", "real_time"] = "on_end"
    ) -> None:
        super().__init__()
        assert save_checkpoints.lower() in [
            "on_end",
            "real_time",
        ], "save_checkpoints must be 'on_end' or 'real_time'"
        self._save_checkpoints = save_checkpoints.lower()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._just_saved_locally = False
        if pl_module.logger._log_model != False:
            print(
                f"\nlog_model must be set to False when using WandbCheckpointing callback but was \
{pl_module.logger._log_model}. Setting to False...\n"
            )
            pl_module.logger._log_model = False
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if callback._save_on_train_epoch_end != False:
                    print(
                        f"\nFor {callback.__class__.__name__} save_on_train_epoch_end must be set to \
False when using WandbCheckpointing callback but was {callback._save_on_train_epoch_end}. \
Setting to False...\n"
                    )
                    callback._save_on_train_epoch_end = False

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._just_saved_locally = False

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        self._just_saved_locally = True
        if not hasattr(self, "_checkpoints_dir"):
            for key, val in checkpoint["callbacks"].items():
                if "ModelCheckpoint" in key:
                    dir = val["dirpath"]
                    break
            self._checkpoints_dir = dir

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (self._save_checkpoints == "real_time" and self._just_saved_locally) or (
            self._save_checkpoints == "on_end"
            and trainer.current_epoch + 1 == trainer.max_epochs
        ):
            entity = pl_module.logger.experiment.entity
            project = pl_module.logger.experiment.project
            id = pl_module.logger.experiment.id
            dir = self._checkpoints_dir

            name = f"model-{id}"

            api = wandb.Api()
            run = api.run(join(entity, project, id))

            for old_artifact in run.logged_artifacts():
                if name in old_artifact.name:
                    old_artifact.delete(delete_aliases=True)

            artifact = wandb.Artifact(
                name=name,
                type="model",
            )
            artifact.add_dir(local_path=dir, name="all_checkpoints")

            pl_module.logger.experiment.log_artifact(artifact)