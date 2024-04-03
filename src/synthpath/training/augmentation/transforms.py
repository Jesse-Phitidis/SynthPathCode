import logging
import random
from collections.abc import Sequence
from numbers import Number
import copy
import time
import random

from scipy import ndimage
from scipy.ndimage import binary_dilation
import numpy as np
import torch
import torchio as tio

import cc3d

logger = logging.getLogger(__name__)
         

class RescaleIntensity(tio.Transform):
    def __init__(
        self,
        percentiles: tuple = (0, 100)
    ):
        super().__init__()
        self.percentiles = tuple([x/100 for x in percentiles])
        
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        
        for image in subject.get_images():
            data = image.data 
            
            if not self.percentiles == (0, 1):
                lower = torch.quantile(data, self.percentiles[0])
                upper = torch.quantile(data, self.percentiles[1])
                data = torch.clamp(data, lower, upper)
            
            mini = data.min()
            maxi = data.max()
            rng = maxi-mini
            
            data = (data - mini) / rng
            
            image.set_data(data)
            
            return subject
        


class CombineLabels(tio.Transform):
    def __init__(
        self,
        anat_key: str,
        path_key: str,
        new_label_value: int,
    ):
        super().__init__()
        self.anat_key = anat_key
        self.path_key = path_key
        self.new_label_value = new_label_value

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        anat_data = subject[self.anat_key]["data"]
        path_data = subject[self.path_key]["data"]
        anat_data[path_data == 1] = self.new_label_value
        subject.add_image(subject[self.anat_key], image_name="label")
        subject.remove_image(self.anat_key)
        subject.remove_image(self.path_key)
        return subject


class RemoveImage(tio.Transform):
    def __init__(self, image_key: str):
        super().__init__()
        self.image_key = image_key

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.remove_image(self.image_key)
        return subject


class GenConfounds(tio.Transform):
    def __init__(
        self,
        p: Number,
        label_key_anat: str,
        label_key_path: str,
        image_key: str,
        target_label: int,
        new_label: int,
        dist: float,
        method: str = "relative",
    ):
        super().__init__(p=p)
        self.label_key_anat = label_key_anat
        self.label_key_path = label_key_path
        self.image_key = image_key
        self.target_label = target_label
        self.new_label = new_label
        self.dist = dist
        self.method = method

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        label_confounds = copy.deepcopy(subject[self.label_key_anat])
        subject.add_image(label_confounds, image_name="sampling_label")
        
        target_mask = subject[self.label_key_path]["data"] > 0
        
        if self.method == "relative":
        
            image_data = RescaleIntensity(percentiles=(0, 100))(
                subject[self.image_key]["data"]
            )

            is_present = target_mask.sum() > 0
            if is_present:
                mean_value = image_data[target_mask].mean()

            if not is_present:
                conf_mask = torch.zeros_like(image_data, dtype=torch.bool)
            else:
                # This line is task specific
                not_background_and_ventricle_mask = (
                    (label_confounds["data"] != 0)
                    & (label_confounds["data"] != 3)
                    & (label_confounds["data"] != 4)
                    & (label_confounds["data"] != 11)
                    & (label_confounds["data"] != 12)
                )
                conf_mask = (
                    (image_data > mean_value - self.dist)
                    & (image_data < mean_value + self.dist)
                    & (not_background_and_ventricle_mask)
                )
                
        elif self.method == "random":
            
            # This line is task specific (again...more lines of code but less run time :D)
            not_background_and_ventricle_mask = (
                (label_confounds["data"] != 0)
                & (label_confounds["data"] != 3)
                & (label_confounds["data"] != 4)
                & (label_confounds["data"] != 11)
                & (label_confounds["data"] != 12)
            )
            
            conf_mask = self.get_random_conf_mask(not_background_and_ventricle_mask)
        
        else:
            raise ValueError(f"self.method must be 'relative' or 'random' but {self.method} was given")
            
            

        label_confounds["data"][target_mask] = self.target_label
        label_confounds["data"][conf_mask & ~target_mask] = self.new_label

        return subject

    def get_random_conf_mask(self, mask: torch.tensor):
    
        seed_coords = self.select_n_coords(mask, (0,20)) # Number of confounder seeds
        conf_mask = torch.zeros_like(mask)
        
        for coord in seed_coords:
            conf = torch.zeros_like(mask)
            conf[*coord] = 1
            n_growths = random.randint(0,20) # Number of random dilations per seed
            for _ in range(n_growths):
                voxel_to_grow = self.select_n_coords(conf, (1,1))
                conf_voxel = torch.zeros_like(conf)
                conf_voxel[*voxel_to_grow[0]] = 1
                conf_voxel_grown = torch.from_numpy(binary_dilation(conf_voxel, iterations=1, structure=self.get_random_struct(((1,1),(1,5),(1,5),(1,5))))) # assumes single channel
                conf[conf_voxel_grown==1]=1
            conf_mask[conf==1]=1
            
        return conf_mask
            
    def select_n_coords(self, mask: torch.tensor, n: tuple):
        
        idx_tuple = torch.where(mask)
        idx = torch.stack(idx_tuple, dim=1)
        n_voxels = idx.shape[0]
        n_selected = random.randint(*n)
        assert n_selected <= n_voxels, f"more coords requested than voxels in mask"
        indices = [random.randint(0, n_voxels-1) for _ in range(n_selected)]
        coords = idx[indices]
        
        return coords

    def get_random_struct(self, size: tuple):
        size = [random.randint(*s) for s in size]
        return np.random.randint(0,2,size)
            
            

class RandomCrop(tio.Transform):
    def __init__(self, size: tuple):
        super().__init__()
        self.size = size

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        shape = subject["label"].shape[1:]
        crop_list = []
        for i in range(3):
            num_vox_to_crop = shape[i] - self.size[i]
            ini = random.randint(0, num_vox_to_crop)
            fin = num_vox_to_crop - ini
            crop_list.extend([ini, fin])
        crop_tuple = tuple(crop_list)
        T = tio.Crop(crop_tuple)
        subject = T(subject)
        return subject


class RandomVarLabelsToImage(tio.Transform):

    """Uses RandomLabelsToImage but is able to deal with missing labels (used_label does not seem to work how I expect)."""

    def __init__(
        self,
        label_key_anat: str,
        label_key_path: str,
        image_key: str,
        target_label: int,
        mean: Sequence[Sequence[Number] | Number] | Number | None = None,
        std: Sequence[Sequence[Number] | Number] | Number | None = None,
        default_mean: Sequence[Number] | Number = (0.1, 0.9),
        default_std: Sequence[Number] | Number = (0.01, 0.1),
        ignore_background: bool = False,
        all_possible_labels: list | None = None,
        sublabelling_prob: float = 0.2,
        sublabelling_method: str = "intervals",
        sublabelling_max_intervals: int = 10,
        sublabelling_minsize: int = 100,
        sublabelling_shift_prob: float = 0.2,
        sublabelling_mean_shift_mean: float = 0.1,
        sublabelling_mean_shift_std: float = 0.2,
        sublabelling_std_shift_mean: float = 0.05,
        sublabelling_std_shift_std: float = 0.1,
    ):
        """
        Args:
            label_key: Key to access the label map with subject[label_key].
            mean: Sequence of means where all potential classes must be present, even if they may not appear in every subject.
                Can be list of numbers, or tuples/lists of length 2 (if tuple or list it is the range from which the uniform distrubtion is sampled)
            std: Same idea as mean.
            default_mean: A number, or tuple/list of length 2. This is the default value for labels where mean was not provied. Should not be
                the case unless no mean provided.
            default_std: Same idea as default_mean.
            ignore_background: If True, the given parameters for the 0 label will not be used and 0 will be set instead.
            all_possible_labels: List of all possilbe labels even if they do not appear in every subject.

        """
        super().__init__()

        self.label_key_anat = label_key_anat
        self.label_key_path = label_key_path
        self.image_key = image_key
        self.target_label = target_label
        self.mean = mean
        self.std = std
        self.default_mean = default_mean
        self.default_std = default_std
        self.ignore_background = ignore_background

        self.all_possible_labels = sorted(all_possible_labels)
        
        self.sublabelling_prob = sublabelling_prob
        self.sublabelling_method = sublabelling_method
        self.sublabelling_max_intervals = sublabelling_max_intervals
        self.sublabelling_minsize = sublabelling_minsize
        self.sublabelling_shift_prob = sublabelling_shift_prob
        self.sublabelling_mean_shift_mean = sublabelling_mean_shift_mean
        self.sublabelling_mean_shift_std = sublabelling_mean_shift_std
        self.sublabelling_std_shift_mean = sublabelling_std_shift_mean
        self.sublabelling_std_shift_std = sublabelling_std_shift_std

        self.T_seq = tio.SequentialLabels(include=["sampling_label"])

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        
        if "sampling_label" not in subject:
            combined_image = copy.deepcopy(subject[self.label_key_anat])
            path_data = subject[self.label_key_path]["data"]
            combined_image["data"][path_data > 0] = self.target_label
            subject.add_image(combined_image, image_name="sampling_label", )

        labels_present = (
            torch.unique(subject["sampling_label"]["data"]).to(dtype=int).tolist()
        )

        target_present_or_new_value = False
        new_mean, new_std = [], []
        for i, label in enumerate(self.all_possible_labels):
            if label in labels_present:
                new_mean.append(self.mean[i] if self.mean else None)
                new_std.append(self.std[i] if self.std else None)
                if label == self.target_label:
                    target_present_or_new_value = i

        subject = self.T_seq(subject)
        
        if np.random.random() < self.sublabelling_prob and target_present_or_new_value is not False:
            subject, new_mean, new_std = self.do_sublabelling(subject, target_present_or_new_value, new_mean, new_std, self.sublabelling_method)

        T_gen = tio.RandomLabelsToImage(
            label_key="sampling_label",
            mean=new_mean if new_mean[0] is not None else None,
            std=new_std if new_std[0] is not None else None,
            default_mean=self.default_mean,
            default_std=self.default_std,
            ignore_background=self.ignore_background,
        )
        
        subject= T_gen(subject)

        subject.remove_image("sampling_label")  # We don't need the sampling label map anymore

        return self.clip_intensities(subject)


    def clip_intensities(self, subject):
        data = subject["image_from_labels"]["data"]
        clipped = torch.clip(data, min=0, max=1)
        subject["image_from_labels"].set_data(clipped.to(dtype=data.dtype))
        return subject

    
    def do_sublabelling(self, subject: tio.Subject, target: int, mean: list, std: list, method: str):
        
        ori_mean_target = mean[target]
        ori_std_target = std[target]
        assert ori_mean_target is not None and ori_std_target is not None, "None value for mean or std cannot be used with sublabelling"
        
        ori_mean_target_centre = np.mean([ori_mean_target[0], ori_mean_target[1]]) if isinstance(ori_mean_target, tuple) else ori_mean_target
        ori_std_target_centre = np.mean([ori_std_target[0], ori_std_target[1]]) if isinstance(ori_std_target, tuple) else ori_std_target
            
        label_map = subject["sampling_label"]["data"][0,...] # Assuming not multichannel...never is
        dtype = label_map.dtype
        device = label_map.device
        
        label_map = label_map.cpu().numpy()
        target_map = label_map == target    
        
        if method == "intervals":
            masks_list = self.get_masks_by_thresholding(subject, target_map)
        elif method == "regional":
            masks_list = self.get_masks_by_region_growing(target_map)
        else:
            raise ValueError
            
        next_label = np.max(label_map) + 1

        assert np.max(np.sum(np.stack(masks_list, axis=0), axis=0)), "some masks overlap"
        
        # Add each new mask to label map except first one for original target index
        for mask in masks_list[1:]:
            label_map[mask] = next_label
            next_label += 1
        
        # For each new label add to means and stds and either copy or update
        for i in range(len(masks_list)):
            
            if np.random.random() < self.sublabelling_shift_prob:
                mean_shift = np.random.normal(self.sublabelling_mean_shift_mean, self.sublabelling_mean_shift_std)
                std_shift = np.random.normal(self.sublabelling_std_shift_mean, self.sublabelling_std_shift_std)
                mean_op, std_op = np.random.choice(["+", "-"], size=2)
                mean_out = ori_mean_target_centre + (mean_shift if mean_op == "+" else - mean_shift)
                std_out = ori_std_target_centre + (std_shift if std_op == "+" else - std_shift)
            else:
                mean_out = ori_mean_target
                std_out = ori_std_target
                
            if i == 0:
                mean[target] = mean_out
                std[target] = std_out
            else:
                mean.append(mean_out)
                std.append(std_out)
        
        # Return subject with tensor image
        label_map = torch.tensor(label_map, dtype=dtype, device=device).unsqueeze(0)
        subject["sampling_label"]["data"] = label_map
        
        return subject, mean, std
    
    
    def get_masks_by_thresholding(self, subject: tio.Subject, target_map: np.array):
        
        image_data = subject[self.image_key]["data"][0, ...].cpu().numpy()
        
        n_thresholds = np.random.randint(2, self.sublabelling_max_intervals + 2)
        
        if n_thresholds == 2:
            return [target_map]
        
        intervals = self.get_intervals(image_data[target_map], n_thresholds)
        
        masks_list = []
        for i in range(1, n_thresholds):
            if i == 1:
                mask = (image_data <= intervals[i]) & target_map
            else:
                mask = ((image_data > intervals[i-1]) & (image_data <= intervals[i])) & target_map
            
            if np.sum(mask) != 0:
                masks_list.append(mask)
            
        return masks_list
            
    
    def get_intervals(self, array: np.array, n: int):
        quantiles = np.linspace(0, 1, n)
        intervals = np.quantile(array, quantiles)
        return intervals
    
    
    def get_masks_by_region_growing(self, target_map: np.array):
        
        component_map, component_num = cc3d.connected_components(target_map, return_N=True, connectivity=26)

        masks_list = []

        for i in range(1, component_num + 1):
            
            component_mask_i = component_map == i
            
            size = np.sum(component_mask_i)
            max_num = max(1, size // self.sublabelling_minsize)
            num = np.random.randint(1, max_num + 1)
            
            masks = self.split_mask(component_mask_i, num)
            
            masks_list.extend(masks)
            
        return masks_list
    
    
    def split_mask(self, binary_mask: np.array, n_components: int):
        
        if n_components == 1:
            return [binary_mask]
            
        # Find the coordinates of the original mask
        coords = np.column_stack(np.where(binary_mask == 1))

        # Shuffle the coordinates
        np.random.shuffle(coords)

        # Select 'n' random seeds
        seeds = coords[:n_components]

        # Initialize masks with the seed points
        masks = [np.zeros_like(binary_mask, dtype=bool) for _ in range(n_components)]

        for i in range(n_components):
            masks[i][seeds[i][0], seeds[i][1], seeds[i][2]] = 1

        # Dilate seeds iteratively until the entire original mask area is covered
        while True:
            prev_masks = [mask.copy() for mask in masks]

            for i in range(n_components):
                masks[i] = ndimage.binary_dilation(masks[i], structure=np.ones((3,3,3))) & binary_mask
                masks[i] = masks[i] & ~np.logical_or.reduce([masks[j] for j in range(n_components) if j != i])

            # Check if the masks have stabilized
            if all(np.array_equal(masks[i], prev_masks[i]) for i in range(n_components)):
                break

        return masks
        
        
        
        

class RandomGammaNorm(tio.Transform):
    def __init__(self, log_gamma_std: Number = 0.63):
        super().__init__()
        self.log_gamma_std = log_gamma_std

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        log_gamma = random.normalvariate(
            0, self.log_gamma_std
        )  # In SynthSeg they use sigma squared = 0.4, so sigma = 0.63 to 2dp
        T = tio.RandomGamma(log_gamma=(log_gamma, log_gamma))
        subject = T(subject)
        return subject


class RandomResolution(tio.Transform):
    def __init__(self, max_spacing: Number = 9, upsample_interp: str = "linear"):
        super().__init__()
        self.max_spacing = max_spacing
        self.upsample_interp = upsample_interp

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Randomly select an axis in which to sample low resolution
        axis = random.randint(0, 2)
        # Get the current spacing along that axis
        if len(str(subject.get_first_image().path)) != 0: # The datasets contained paths
            current_spacing = subject.get_first_image().spacing[axis]
        else: # The dataset is preloaded tensors and the subject has a key for the spacing
            current_spacing = subject["spacing"][axis]
        # Randomly sample a new spacing between the current spacing and 9mm
        r_spac = random.uniform(current_spacing, max(self.max_spacing, current_spacing))
        if current_spacing > self.max_spacing:
            logger.warning(
                f"Max spacing is {self.max_spacing} but current spacing is {current_spacing}. Spacing will be unchaged."
            )
        # Randomly sample a slice thickness between current spacing and new spacing
        r_thick = random.uniform(current_spacing, r_spac)
        # Randomly sample alpha parameter
        alpha = random.uniform(0.95, 1.05)
        # Calculate slice thickness sigma
        sigma_thick = (
            2 * alpha * np.log(10) * (1 / (2 * np.pi)) * (r_thick / r_spac)
        )  # I think they us log base e but not sure

        # Create blur transform with sigma_thick along axis
        # Create std tuple
        std = tuple([sigma_thick if i == axis else 0 for i in [0, 0, 1, 1, 2, 2]])
        # Create transform
        T_blur = tio.RandomBlur(std=std)

        # Create downsample transform to and from r_spac along axis
        # Calculate downsampling factor
        f = r_spac / current_spacing
        # Create transform - this uses nearest nighbour downsampling whereas synthseg uses bspline. Both use linear upsampling.
        # I could implement this easily by using tio.Resample twice instead of tio.RandomAnisotropy.
        T_down_up = tio.RandomAnisotropy(
            axes=axis,
            downsampling=f,
            image_interpolation=self.upsample_interp,
            scalars_only=True,
        )

        # Apply transforms
        subject = T_blur(subject)
        subject = T_down_up(subject)

        return subject
