from collections.abc import Sequence
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torchio as tio
from torch.utils.data import DataLoader

import nibabel as nib
import torch


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        modality: str,
        batch_size: int = 1,
        num_workers: int = 2,
        transforms_train: tio.Transform = None,
        transforms_val: tio.Transform = None,
        transforms_test: tio.Transform = None,
        val_data_for_test: bool = False,
        pre_load_data: bool = False,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        assert modality in ["flair", "dwi"]
        self.modality = modality
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms_train = transforms_train
        self.transforms_val = transforms_val
        self.transforms_test = transforms_test
        self.val_data_for_test = val_data_for_test
        self.pre_load_data = pre_load_data

    def build_dataset(
        self, name: str, transforms: tio.Transform
    ) -> tio.SubjectsDataset:
        image_paths = sorted(self.data_dir.glob(f"{name}/{self.modality if name in ['test', 'val'] else 'images'}/*.nii.gz"))
        anatomy_paths = sorted(self.data_dir.glob(f"{name}/labels_anatomy/*.nii.gz"))
        pathology_paths = sorted(
            self.data_dir.glob(f"{name}/labels_pathology/*.nii.gz")
        )

        subjects = []

        for image, anatomy, pathology in zip(
            image_paths, anatomy_paths, pathology_paths
        ):
            
            if not self.pre_load_data:
                subject = tio.Subject(
                    {
                        "image": tio.ScalarImage(image),
                        "anatomy": tio.LabelMap(anatomy),
                        "pathology": tio.LabelMap(pathology),
                    }
                )
            else:
                spacing = nib.load(image).header.get_zooms()
                stem = image.stem.split(".")[0]
                subject = tio.Subject(
                    {
                        "image": tio.ScalarImage(tensor=torch.tensor(nib.load(image).get_fdata()).unsqueeze(0)),
                        "anatomy": tio.LabelMap(tensor=torch.tensor(nib.load(anatomy).get_fdata()).unsqueeze(0)),
                        "pathology": tio.LabelMap(tensor=torch.tensor(nib.load(pathology).get_fdata()).unsqueeze(0)),
                        "spacing": spacing
                    }
                )
                subject["image"]["stem"] = stem
                subject["image"]["path"] = image
            subjects.append(subject)

        return tio.SubjectsDataset(subjects, transform=transforms)

    def setup(self, stage=None) -> None:
        assert stage.lower() in [
            "fit",
            "test",
        ], "stage must be one of fit, test. Predict not implemented"

        if stage == "fit":
            self.train_dataset = self.build_dataset(
                name="train", transforms=self.transforms_train
            )
            self.val_dataset = self.build_dataset(
                name="val", transforms=self.transforms_val
            )

        if stage == "test":
            # To avoid unfairly tuning test time training on the test dataset
            if self.val_data_for_test: 
                self.test_dataset = self.build_dataset(
                    name="val", transforms=self.transforms_test
                )
            else:
                self.test_dataset = self.build_dataset(
                    name="test", transforms=self.transforms_test
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )


class RealDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        modality: str,
        batch_size: int = 1,
        num_workers: int = 2,
        transforms_train: tio.Transform = None,
        transforms_val: tio.Transform = None,
        pre_load_data: bool = False
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        assert modality in ["flair", "dwi"]
        self.modality = modality
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms_train = transforms_train
        self.transforms_val = transforms_val
        self.pre_load_data = pre_load_data

    def build_dataset(
        self, name: str, transforms: tio.Transform
    ) -> tio.SubjectsDataset:
        image_paths = sorted(self.data_dir.glob(f"{name}/{self.modality if name in ['test', 'val'] else 'images'}/*.nii.gz"))
        anatomy_paths = sorted(self.data_dir.glob(f"{name}/labels_anatomy/*.nii.gz"))
        pathology_paths = sorted(
            self.data_dir.glob(f"{name}/labels_pathology/*.nii.gz")
        )

        subjects = []

        for image, anatomy, pathology in zip(
            image_paths, anatomy_paths, pathology_paths
        ):
            
            if not self.pre_load_data:
                subject = tio.Subject(
                    {
                        "image": tio.ScalarImage(image),
                        "anatomy": tio.LabelMap(anatomy),
                        "pathology": tio.LabelMap(pathology),
                    }
                )
            else:
                spacing = nib.load(image).header.get_zooms()
                stem = image.stem.split(".")[0]
                subject = tio.Subject(
                    {
                        "image": tio.ScalarImage(tensor=torch.tensor(nib.load(image).get_fdata()).unsqueeze(0)),
                        "anatomy": tio.LabelMap(tensor=torch.tensor(nib.load(anatomy).get_fdata()).unsqueeze(0)),
                        "pathology": tio.LabelMap(tensor=torch.tensor(nib.load(pathology).get_fdata()).unsqueeze(0)),
                        "spacing": spacing
                    }
                )
                subject["image"]["stem"] = stem
                subject["image"]["path"] = image
            subjects.append(subject)

        return tio.SubjectsDataset(subjects, transform=transforms)

    def setup(self, stage=None) -> None:
        assert stage.lower() in [
            "fit",
            "test",
        ], "stage must be one of fit, test. Predict not implemented"

        if stage == "fit":
            self.train_dataset = self.build_dataset(
                name="test", transforms=self.transforms_train
            )
            self.val_dataset = self.build_dataset(
                name="val", transforms=self.transforms_val
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )
        
        
        
        
class RealDataModuleWithTest(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        modality: str,
        batch_size: int = 1,
        num_workers: int = 2,
        transforms_train: tio.Transform = None,
        transforms_val: tio.Transform = None,
        transforms_test: tio.Transform = None,
        pre_load_data: bool = False
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        assert modality in ["flair", "dwi"]
        self.modality = modality
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms_train = transforms_train
        self.transforms_val = transforms_val
        self.transforms_test = transforms_test
        self.pre_load_data = pre_load_data

    def build_dataset(
        self, name: str, transforms: tio.Transform
    ) -> tio.SubjectsDataset:
        isTrainVal = False
        if name in ["val_train", "val_val"]:
            split = "train" if name == "val_train" else "val"
            name = "val"
            isTrainVal = True
        
        image_paths = sorted(self.data_dir.glob(f"{name}/{self.modality if name in ['test', 'val'] else 'images'}/*.nii.gz"))
        anatomy_paths = sorted(self.data_dir.glob(f"{name}/labels_anatomy/*.nii.gz"))
        pathology_paths = sorted(
            self.data_dir.glob(f"{name}/labels_pathology/*.nii.gz")
        )
        
        if isTrainVal:
            length = len(image_paths)
            split_index = int(0.8 * length)
            
            if split == "train":
                slicer = slice(None, split_index)
            if split == "val":
                slicer = slice(split_index, None)
                
            image_paths = image_paths[slicer]
            anatomy_paths = anatomy_paths[slicer]
            pathology_paths = pathology_paths[slicer]

        subjects = []

        for image, anatomy, pathology in zip(
            image_paths, anatomy_paths, pathology_paths
        ):
            
            if not self.pre_load_data:
                subject = tio.Subject(
                    {
                        "image": tio.ScalarImage(image),
                        "anatomy": tio.LabelMap(anatomy),
                        "pathology": tio.LabelMap(pathology),
                    }
                )
            else:
                spacing = nib.load(image).header.get_zooms()
                stem = image.stem.split(".")[0]
                subject = tio.Subject(
                    {
                        "image": tio.ScalarImage(tensor=torch.tensor(nib.load(image).get_fdata()).unsqueeze(0)),
                        "anatomy": tio.LabelMap(tensor=torch.tensor(nib.load(anatomy).get_fdata()).unsqueeze(0)),
                        "pathology": tio.LabelMap(tensor=torch.tensor(nib.load(pathology).get_fdata()).unsqueeze(0)),
                        "spacing": spacing
                    }
                )
                subject["image"]["stem"] = stem
                subject["image"]["path"] = image
            subjects.append(subject)

        return tio.SubjectsDataset(subjects, transform=transforms)

    def setup(self, stage=None) -> None:
        assert stage.lower() in [
            "fit",
            "test",
        ], "stage must be one of fit, test. Predict not implemented"

        if stage == "fit":
            self.train_dataset = self.build_dataset(
                name="val_train", transforms=self.transforms_train
            )
            self.val_dataset = self.build_dataset(
                name="val_val", transforms=self.transforms_val
            )
            
        if stage == "test":
            self.test_dataset = self.build_dataset(
                name="test", transforms=self.transforms_test
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )