import io
import os
import random
from enum import Enum
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from src.util.depth_transform import DepthNormalizerBase

from . import DatasetMode


class BaseReflectionDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        depth_transform: Union[DepthNormalizerBase, None] = None,
        augmentation_args: dict = None,
        resize_to_hw=None,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        assert os.path.exists(self.dataset_dir), (
            f"Dataset does not exist at: {self.dataset_dir}"
        )
        self.disp_name = disp_name

        # training arguments
        # self.depth_transform: DepthNormalizerBase = depth_transform
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]  # [['rgb.png', 'depth.tif'], [], ...]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, reflection_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            reflection_data = self._load_binary_data(
                reflection_rel_path=reflection_rel_path
            )
            # rasters.update(reflection_data)
            # valid mask
            rasters["reflection"] = reflection_data.clone()
            reflection_data[reflection_data > 0.5] = 1.0
            reflection_data[reflection_data <= 0.5] = 0.0

            # Convert to uint8 for consistency
            rasters["gt_mask"] = reflection_data.bool()

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _load_binary_data(self, reflection_rel_path):
        # Read reflection data
        reflection_mask = self._read_reflection_file(reflection_rel_path)
        reflection_mask = (
            torch.from_numpy(reflection_mask).float().unsqueeze(0)
        )  # [1, H, W]
        reflection_mask = reflection_mask / 255  # Normalize to [0, 1]
        return reflection_mask

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path, reflection_rel_path = filename_line

        return rgb_rel_path, reflection_rel_path

    def _read_image(self, img_rel_path) -> np.ndarray:
        image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = Image.open(image_to_read)  # [H, W, rgb]
        image = np.asarray(image)
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb

    def _read_reflection_file(self, rel_path):
        reflection_mask = self._read_image(rel_path)
        return reflection_mask.astype(np.uint8)  # Convert to uint8 for consistency

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # Normalization
        # rasters["depth_raw_norm"] = self.depth_transform(
        #     rasters["depth_raw_linear"], rasters["valid_mask_raw"]
        # ).clone()
        # rasters["depth_filled_norm"] = self.depth_transform(
        #     rasters["depth_filled_linear"], rasters["valid_mask_filled"]
        # ).clone()

        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters

    def _augment_data(self, rasters_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}

        return rasters_dict

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None
