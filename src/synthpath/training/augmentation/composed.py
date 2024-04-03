import copy
from bisect import bisect
from collections.abc import Iterable, Sequence
from typing import Literal
import numpy as np
import torchio as tio

import synthpath.training.augmentation.transforms as trans
import synthpath.training.constants as C


class SyntheticDataTransforms(tio.Transform):

    """Transforms for synthetic data. Based on SynthSeg transforms."""

    def __init__(
        self,
        anat_key: str = "anatomy",
        path_key: str = "pathology",
        aff_on: bool = True,
        aff_scales: float = 0.2,
        aff_degrees: float = 180,
        aff_isotropic: bool = False,
        aff_interp: str = "linear",
        nlin_on: bool = True,
        nlin_control_points: int = 7,
        nlin_max_disp: float = 10,
        nlin_locked_borders: int = 2,
        nlin_interp: str = "linear",
        crop_on: bool = False,
        crop_size: Sequence[int, int, int] = (96, 96, 96),
        sampling_mean: dict[str, Sequence[float] | float] | str | None = None,
        sampling_std: dict[str, Sequence[float] | float] | str | None = None,
        sampling_default_mean: Sequence[float] | float = (0, 1),
        sampling_default_std: Sequence[float] | float = (0, 0.137),
        sampling_ignore_background: bool = True,
        sampling_mean_shift: float = 0.02,
        sampling_std_shift: float = 0.01,
        all_possible_labels: list[int] | None = None,
        labels_to_regions_map: dict | None = None,
        default_intensities_mean_dict: dict | None = None,
        default_intensities_std_dict: dict | None = None,
        confound_prob: float = 1.0,
        image_key: str = "image",
        confound_target_label: int = 19,  # That target to find confounds for
        confound_new_label: int = 20,  # The label given to the confounds
        confound_distance: float = 0.02,  # How close to target label to be considered a confound
        confound_method: str = "relative",
        sublabelling_prob: float = 0.0,
        sublabelling_method: str = "intervals",
        sublabelling_minsize: int = 100,
        sublabelling_max_intervals:int = 10,
        sublabelling_shift_prob: float = 0.2,
        sublabelling_mean_shift_mean: float = 0.1,
        sublabelling_mean_shift_std: float = 0.2,
        sublabelling_std_shift_mean: float = 0.05,
        sublabelling_std_shift_std: float = 0.1,
        bias_on: bool = True,
        bias_coefficients: float = 0.5,
        bias_order: int = 3,
        gamma_on: bool = True,
        gamma_log_gamma_std: float = 0.3,
        res_on: bool = True,
        res_max_spacing: float = 2,
        res_upsample_interp: str = "linear",
        ########################
    ):
        super().__init__()

        # Set mutable defaults
        if all_possible_labels is None:
            all_possible_labels = copy.deepcopy(C.all_possible_labels)
        if labels_to_regions_map is None:
            labels_to_regions_map = copy.deepcopy(C.labels_to_regions_map)
        #if default_intensities_mean_dict is None:
        #    default_intensities_mean_dict = copy.deepcopy(C.default_intensities_mean)
        #if default_intensities_std_dict is None:
        #    default_intensities_std_dict = copy.deepcopy(C.default_intensities_std)
        
        
        if isinstance(sampling_mean, str):
            sampling_mean = getattr(C, sampling_mean) 
        if isinstance(sampling_std, str):
            sampling_std = getattr(C, sampling_std)
        
        if sampling_mean:
            sampling_mean = self.sampling_dict_to_list(
                d=sampling_mean,
                #d_default=default_intensities_mean_dict,
                labels_to_regions_map=labels_to_regions_map,
                shift=sampling_mean_shift,
            )
        if sampling_std:
            sampling_std = self.sampling_dict_to_list(
                d=sampling_std,
                #d_default=default_intensities_std_dict,
                labels_to_regions_map=labels_to_regions_map,
                shift=sampling_std_shift,
            )


        # Confounds transform
        T_conf = trans.GenConfounds(
            p=confound_prob,
            label_key_anat=anat_key,
            label_key_path=path_key,
            image_key=image_key,
            target_label=confound_target_label,
            new_label=confound_new_label,
            dist=confound_distance,
            method=confound_method
        )
        (
            sampling_mean,
            sampling_std,
            all_possible_labels,
        ) = self.add_values_to_sampling_list(
            sampling_mean=sampling_mean,
            sampling_std=sampling_std,
            all_possible_labels=all_possible_labels,
            target_label=confound_target_label,
            new_label=confound_new_label,
        )
       
        # Remove image transform
        T_remove_image = trans.RemoveImage(image_key=image_key)


        # Affine transform
        if aff_on:
            T_aff = tio.RandomAffine(
                scales=aff_scales,
                degrees=aff_degrees,
                isotropic=aff_isotropic,
                image_interpolation=aff_interp,
            )
        else:
            T_aff = None

        # Nonlinear transform
        if nlin_on:
            T_nlin = tio.RandomElasticDeformation(
                num_control_points=nlin_control_points,
                max_displacement=nlin_max_disp,
                locked_borders=nlin_locked_borders,
                image_interpolation=nlin_interp,
            )
        else:
            T_nlin = None

        # Crop transform
        if crop_on:
            T_crop = trans.RandomCrop(tuple(crop_size))
        else:
            T_crop = None

        # Sampling Intensity
        T_gen = trans.RandomVarLabelsToImage(
            label_key_anat=anat_key,
            label_key_path=path_key,
            image_key=image_key,
            target_label=confound_target_label,
            mean=sampling_mean,
            std=sampling_std,
            default_mean=sampling_default_mean,
            default_std=sampling_default_std,
            ignore_background=sampling_ignore_background,
            all_possible_labels=all_possible_labels,
            sublabelling_prob = sublabelling_prob,
            sublabelling_method = sublabelling_method,
            sublabelling_max_intervals = sublabelling_max_intervals,
            sublabelling_minsize = sublabelling_minsize,
            sublabelling_shift_prob = sublabelling_shift_prob,
            sublabelling_mean_shift_mean = sublabelling_mean_shift_mean,
            sublabelling_mean_shift_std = sublabelling_mean_shift_std,
            sublabelling_std_shift_mean = sublabelling_std_shift_mean,
            sublabelling_std_shift_std = sublabelling_std_shift_std,
            
        )

        # Bias transform
        if bias_on:
            T_bias = tio.RandomBiasField(
                coefficients=bias_coefficients, order=bias_order
            )
        else:
            T_bias = None

        # Gamma transform
        if gamma_on:
            T_gamma = trans.RandomGammaNorm(log_gamma_std=gamma_log_gamma_std)
        else:
            T_gamma = None

        # Resolution transform
        if res_on:
            T_res = trans.RandomResolution(
                max_spacing=res_max_spacing, upsample_interp=res_upsample_interp
            )
        else:
            T_res = None

        # Composed transform
        transform_list = [
            T_conf,
            T_remove_image if not (sublabelling_prob > 0 and sublabelling_method == "intervals") else None,
            T_aff,
            T_nlin,
            T_crop,
            T_gen,
            T_remove_image if (sublabelling_prob > 0 and sublabelling_method == "intervals") else None,
            T_bias,
            T_gamma,
            T_res,
        ]

        self.transform = tio.Compose([T for T in transform_list if T is not None])

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        return self.transform(subject)

    @staticmethod
    def sampling_dict_to_list(
        d: dict,
        #d_default: dict,
        labels_to_regions_map: dict,
        shift: float,
    ) -> list[Sequence[float] | float]:
        # Fill d with default values where keys are not provided
        #if not d == d_default:
        #    for key, value in d_default.items():
        #        if key not in d:
        #            d[key] = value

        l = []
        for label in sorted(labels_to_regions_map):
            region = labels_to_regions_map[label]
            value = d[region]
            l.append(value)

        if shift:
            l = [
                (e - shift, e + shift) if not isinstance(e, Iterable) else e for e in l
            ]

        return l

    @staticmethod
    def add_values_to_sampling_list(
        sampling_mean: list,
        sampling_std: list,
        all_possible_labels: list,
        target_label: int,
        new_label: int,
    ) -> list[Sequence[float] | float]:
        index_to_copy = all_possible_labels.index(target_label)
        index_to_insert = bisect(all_possible_labels, new_label)

        if sampling_mean:
            mean_value_to_copy = sampling_mean[index_to_copy]
            sampling_mean.insert(index_to_insert, mean_value_to_copy)

        if sampling_std:
            std_value_to_copy = sampling_std[index_to_copy]
            sampling_std.insert(index_to_insert, std_value_to_copy)

        all_possible_labels.insert(index_to_insert, new_label)

        return sampling_mean, sampling_std, all_possible_labels


# Just wraps Rescale for no reason except clarity
class ValTransforms(tio.Transform):
    
    def __init__(
        self,
        percentiles: tuple[float, float] | list[float] = (0, 100)
    ):
        super().__init__()
        
        self.T = trans.RescaleIntensity(percentiles=tuple(percentiles))
        
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        return self.T(subject)
    
    
# Just wraps Rescale for no reason except clarity - same as above. Different class to simplify future changes.
class TestTransforms(tio.Transform):
    
    def __init__(
        self,
        percentiles: tuple[float, float] | list[float] = (0, 100)
    ):
        super().__init__()

        self.T = trans.RescaleIntensity(percentiles=percentiles)
        
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        return self.T(subject)
    
    
    
class RealDataTransforms(tio.Transform):

    """Transforms for synthetic data. Based on SynthSeg transforms."""

    def __init__(
        self,
        anat_key: str = "anatomy",
        path_key: str = "pathology",
        aff_on: bool = True,
        aff_scales: float = 0.2,
        aff_degrees: float = 20,
        aff_isotropic: bool = False,
        aff_interp: str = "linear",
        nlin_on: bool = True,
        nlin_control_points: int = 7,
        nlin_max_disp: float = 10,
        nlin_locked_borders: int = 2,
        nlin_interp: str = "linear",
        crop_on: bool = False,
        crop_size: Sequence[int, int, int] = (96, 96, 96),
        image_key: str = "image",
        confound_target_label: int = 19,
        bias_on: bool = True,
        bias_coefficients: float = 0.5,
        bias_order: int = 3,
        rescale_on: bool = True,
        rescale_percentiles: bool = False,
        gamma_on: bool = True,
        gamma_log_gamma_std: float = 0.3,
        res_on: bool = True,
        res_max_spacing: float = 2,
        res_upsample_interp: str = "linear",
    ):
        super().__init__()

        # Affine transform
        if aff_on:
            T_aff = tio.RandomAffine(
                scales=aff_scales,
                degrees=aff_degrees,
                isotropic=aff_isotropic,
                image_interpolation=aff_interp,
            )
        else:
            T_aff = None

        # Nonlinear transform
        if nlin_on:
            T_nlin = tio.RandomElasticDeformation(
                num_control_points=nlin_control_points,
                max_displacement=nlin_max_disp,
                locked_borders=nlin_locked_borders,
                image_interpolation=nlin_interp,
            )
        else:
            T_nlin = None

        # Crop transform
        if crop_on:
            T_crop = trans.RandomCrop(tuple(crop_size))
        else:
            T_crop = None

        # Bias transform
        if bias_on:
            T_bias = tio.RandomBiasField(
                coefficients=bias_coefficients, order=bias_order
            )
        else:
            T_bias = None

        # Gamma transform
        if gamma_on:
            T_gamma = trans.RandomGammaNorm(log_gamma_std=gamma_log_gamma_std)
        else:
            T_gamma = None

        # Resolution transform
        if res_on:
            T_res = trans.RandomResolution(
                max_spacing=res_max_spacing, upsample_interp=res_upsample_interp
            )
        else:
            T_res = None
            
        # Rescale intensities
        if rescale_on:
            if rescale_percentiles:
                T_rescale = trans.RescaleIntensity(percentiles=(0.5,99.5))
            else:
                T_rescale = trans.RescaleIntensity()
        else:
            T_rescale = None

        # Composed transform
        transform_list = [
            T_aff,
            T_nlin,
            T_crop,
            T_bias,
            T_gamma,
            T_res,
            T_rescale
        ]

        self.transform = tio.Compose([T for T in transform_list if T is not None])

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        return self.transform(subject)
