import json
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks import one_hot

from synthpath.training.utils import format_loss, format_val, format_val_dice_only, format_test, inverse_sigmoid#
from synthpath.training.losses import DiceLoss, CELoss, DiceCELoss
from synthpath.evaluation.non_monai_metrics import NonMONAIMetrics
import copy

from typing import Any


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        criterion_anat: nn.Module,
        criterion_path: nn.Module,
        inference_class: Any,
        threshold_probs: float = 0.5,
        test_pred_dir: str = None,
        test_metrics_path: str = None,
        test_save_anat: bool = False,
        target_label: int = 19, # not used anymore
        watch_log_freq: int = 50,
        dice_weight: float | None = None,
        path_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["network", "criterion"])
        self.network = network
        self.criterion_anat = criterion_anat
        self.criterion_path = criterion_path
        self.inference_class = inference_class
        self.threhold_logits = inverse_sigmoid(threshold_probs)
        self.test_pred_dir = Path(test_pred_dir) if test_pred_dir else None
        self.test_metrics_path = Path(test_metrics_path) if test_metrics_path else None
        self.test_save_anat = test_save_anat
        self.target_label = target_label
        self.watch_log_freq = watch_log_freq
        self.dice_weight = dice_weight
        self.path_weight = path_weight
        self.DiceMetric_anat = DiceMetric(include_background=False, reduction="none")
        self.DiceMetric_path = DiceMetric(include_background=True, reduction="none")
        self.HD95Metric_anat = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
        self.HD95Metric_path = HausdorffDistanceMetric(include_background=True, reduction="none", percentile=95)
        self.NonMONAIMetrics_path = NonMONAIMetrics(include_background=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def on_train_start(self):
        self.logger.watch(self.network, log="all", log_freq=self.watch_log_freq)

    def training_step(self, batch: dict):
        image_key = "image_from_labels" if "image_from_labels" in batch else "image"
        image, anatomy, pathology = (
            batch[image_key]["data"],
            batch["anatomy"]["data"],
            batch["pathology"]["data"],
        )
        pred = self(image)
        pred_anat, pred_path = torch.split(pred, split_size_or_sections=[pred.shape[1]-1, 1], dim=1)
        label_anat = one_hot(anatomy, num_classes=pred_anat.shape[1], dim=1)
        label_path = pathology
        loss_anat = self.criterion_anat(pred_anat, label_anat)
        loss_path = self.criterion_path(pred_path, label_path) 
        loss, loss_dict = format_loss((loss_anat, loss_path), self.dice_weight, self.path_weight)
        self.log_dict(loss_dict, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict):
        image, anatomy, pathology = (
            batch["image"]["data"],
            batch["anatomy"]["data"],
            batch["pathology"]["data"],
        )
        pred = self(image.to(dtype=torch.float))
        pred_anat, pred_path = torch.split(pred, split_size_or_sections=[pred.shape[1]-1, 1], dim=1)
        label_anat = one_hot(anatomy, num_classes=pred_anat.shape[1], dim=1)
        label_path = pathology
        pred_anat = one_hot(torch.argmax(pred_anat, dim=1, keepdim=True), num_classes=pred_anat.shape[1])
        pred_path = torch.where(pred_path > self.threhold_logits, 1, 0)
        self.DiceMetric_anat(pred_anat, label_anat)
        self.DiceMetric_path(pred_path, label_path)
        
        if (self.current_epoch + 1) % (5 * self.trainer.check_val_every_n_epoch) == 0:
            self.HD95Metric_anat(pred_anat.cpu(), label_anat.cpu())
            self.HD95Metric_path(pred_path.cpu(), label_path.cpu())

    def on_validation_epoch_end(self):
        dice_anat = self.DiceMetric_anat.aggregate()
        dice_path = self.DiceMetric_path.aggregate()
        self.DiceMetric_anat.reset()
        self.DiceMetric_path.reset()
        if (self.current_epoch + 1) % (5 * self.trainer.check_val_every_n_epoch) == 0:
            hd95_anat = self.HD95Metric_anat.aggregate()
            hd95_path = self.HD95Metric_path.aggregate()
            self.HD95Metric_anat.reset()
            self.HD95Metric_path.reset()
            val_dict = format_val(dice_anat, dice_path, hd95_anat, hd95_path)
        else:
            val_dict = format_val_dice_only(dice_anat, dice_path)
        self.log_dict(val_dict, on_step=False, on_epoch=True)

    def on_test_start(self):
        if self.test_metrics_path:
            self.test_metrics_json = {}

    def test_step(self, batch: dict):
        image, anatomy, pathology = (
            batch["image"]["data"],
            batch["anatomy"]["data"],
            batch["pathology"]["data"],
        )
        pred = self.inference_class(copy.deepcopy(self.network), image.to(dtype=torch.float), anatomy)
        pred_anat, pred_path = torch.split(pred, split_size_or_sections=[pred.shape[1]-1, 1], dim=1)
        label_anat = one_hot(anatomy, num_classes=pred_anat.shape[1], dim=1).to(dtype=torch.uint8) # dtype allowed?
        label_path = pathology.to(dtype=torch.uint8) # dtype allowed
        pred_anat = one_hot(torch.argmax(pred_anat, dim=1, keepdim=True), num_classes=pred_anat.shape[1])
        
        pred_path_soft = pred_path.clone()
        
        pred_path = torch.where(pred_path > self.threhold_logits, 1, 0)
        dice_anat = self.DiceMetric_anat(pred_anat, label_anat)
        dice_path = self.DiceMetric_path(pred_path, label_path)
        hd95_anat = self.HD95Metric_anat(pred_anat.cpu(), label_anat.cpu())
        hd95_path = self.HD95Metric_path(pred_path.cpu(), label_path.cpu())
        
        pre_path, rec_path, lf1_path, lpre_path, lrec_path, ap_path = self.NonMONAIMetrics_path(pred_path, label_path, pred_path_soft)
        
        test_dict = format_test(dice_anat, dice_path, hd95_anat, hd95_path, pre_path, rec_path, lf1_path, lpre_path, lrec_path, ap_path)
        if self.test_pred_dir:
            self.save_pred(pred_anat[0,0,...], pred_path[0,0,...], batch)
        if self.test_metrics_path:
            self.add_test_metrics(
                test_dict, final=False, batch=batch
            )

    def on_test_epoch_end(self):
        dice_anat = self.DiceMetric_anat.aggregate()
        dice_path = self.DiceMetric_path.aggregate()
        hd95_anat = self.HD95Metric_anat.aggregate()
        hd95_path = self.HD95Metric_path.aggregate()
        
        pre_path, rec_path, lf1_path, lpre_path, lrec_path, ap_path = self.NonMONAIMetrics_path.aggregate()
        
        test_dict = format_test(dice_anat, dice_path, hd95_anat, hd95_path, pre_path, rec_path, lf1_path, lpre_path, lrec_path, ap_path)
        self.DiceMetric_anat.reset()
        self.DiceMetric_path.reset()
        self.HD95Metric_anat.reset()
        self.HD95Metric_path.reset()
        self.NonMONAIMetrics_path.reset()
        
        print("\n\nDice:")
        for key, val in test_dict["dice"].items():
            print(key, val)
        print("\nHD95")
        for key, val in test_dict["hd95"].items():
            print(key, val)
            
        print("\nPre")  
        for key, val in test_dict["pre"].items():
            print(key, val)
        print("\nRec")
        for key, val in test_dict["rec"].items():
            print(key, val)
        print("\nLF1")
        for key, val in test_dict["lf1"].items():
            print(key, val)
        print("\nLPre")
        for key, val in test_dict["lpre"].items():
            print(key, val)
        print("\nLRec")
        for key, val in test_dict["lrec"].items():
            print(key, val)
        print("\nAP")
        for key, val in test_dict["ap"].items():
            print(key, val) 
            
        print("\n\n")
        
        if self.test_metrics_path:
            self.add_test_metrics(test_dict, final=True)
            with open(self.test_metrics_path, "w") as f:
                json.dump(self.test_metrics_json, f, indent=4)
        config_file = Path.cwd() / "config.yaml"
        config_file.unlink()
        print("Deleted test config")
        
    def save_pred(self, pred_anat: torch.Tensor, pred_path: torch.tensor, batch: dict):
        path_save_path = self.test_pred_dir / (str(batch["image"]["stem"][0]) + "_pred_path.nii.gz")
        original_path = batch["image"]["path"][0]
        image = nib.load(original_path)
        header, affine = image.header, image.affine
        pred_path = pred_path.cpu().numpy().astype(np.uint8)
        pred_path = nib.Nifti1Image(pred_path, affine=affine, header=header)
        nib.save(pred_path, path_save_path)
        if self.test_save_anat:
            anat_save_path = self.test_pred_dir / (str(batch["image"]["stem"][0]) + "_pred_anat.nii.gz")
            pred_anat = pred_anat.cpu().numpy().astype(np.uint8)
            pred_anat = nib.Nifti1Image(pred_anat, affine=affine, header=header)
            nib.save(pred_anat, anat_save_path)

    def add_test_metrics(self, metrics: dict, final: bool, batch: dict = None):
        if not final:
            assert batch, "Batch must be provided when adding non-final test metrics"
            self.test_metrics_json[batch["image"]["stem"][0]] = metrics
        else:
            self.test_metrics_json["mean"] = metrics
