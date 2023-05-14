import copy
import os
import cv2
import logging

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["MaskFormerSemanticCopyDatasetMapper"]

def crop_partial_img(img_chw, mask_hw, cls_id=1):
    if isinstance(mask_hw, np.ndarray):
        mask_hw = torch.tensor(mask_hw)
    binary_mask_hw = (mask_hw == cls_id)
    binary_mask_hw_np = binary_mask_hw.numpy().astype(np.uint8)
    # RETR_EXTERNAL to keep online the outer contour
    contours, _ = cv2.findContours(binary_mask_hw_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop annotated objects off the image
    # Compute a minimum rectangle containing the object
    if len(contours) == 0:
        return (img_chw, binary_mask_hw)
    assert len(contours) != 0
    cnt = contours[0]
    x_min = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
    x_max = tuple(cnt[cnt[:,:,0].argmax()][0])[0]
    y_min = tuple(cnt[cnt[:,:,1].argmin()][0])[1]
    y_max = tuple(cnt[cnt[:,:,1].argmax()][0])[1]
    for cnt in contours:
        x_min = min(x_min, tuple(cnt[cnt[:,:,0].argmin()][0])[0])
        x_max = max(x_max, tuple(cnt[cnt[:,:,0].argmax()][0])[0])
        y_min = min(y_min, tuple(cnt[cnt[:,:,1].argmin()][0])[1])
        y_max = max(y_max, tuple(cnt[cnt[:,:,1].argmax()][0])[1])
    # Index of max bounding rect are inclusive so need 1 offset
    x_max += 1
    y_max += 1
    # mask_roi is a boolean arrays
    mask_roi = binary_mask_hw[y_min:y_max,x_min:x_max]
    img_roi = img_chw[y_min:y_max,x_min:x_max,:]
    return (img_roi, mask_roi)

def copy_and_paste(novel_img_hwc, novel_mask_hw, base_img_hwc, base_mask_hw, mask_id):
    base_img_hwc = base_img_hwc.copy()
    base_mask_hw = base_mask_hw.copy()
    # Horizontal Flipping
    if torch.rand(1) < 0.5:
        novel_img_hwc = np.flip(novel_img_hwc, axis=1)
        novel_mask_hw = np.flip(novel_mask_hw, axis=1)
    
    # Parameters for random resizing
    scale = np.random.uniform(0.1, 1.0)
    src_h, src_w = novel_mask_hw.shape
    if src_h * scale > base_mask_hw.shape[0]:
        scale = base_mask_hw.shape[0] / src_h
    if src_w * scale > base_mask_hw.shape[1]:
        scale = base_mask_hw.shape[1] / src_w
    target_H = int(src_h * scale)
    target_W = int(src_w * scale)
    if target_H == 0: target_H = 1
    if target_W == 0: target_W = 1
    # apply
    novel_img_hwc = np.array(Image.fromarray(novel_img_hwc).resize((target_W, target_H)))
    novel_mask_hw = np.array(Image.fromarray(novel_mask_hw).resize((target_W, target_H)))

    # Randomly rotate novel_img and novel_mask
    # angle = np.random.uniform(-10, 10)
    # M = cv2.getRotationMatrix2D((target_W/2, target_H/2), angle, 1)
    # novel_img_hwc = cv2.warpAffine(novel_img_hwc, M, (target_W, target_H))
    # novel_mask_hw = cv2.warpAffine(novel_mask_hw, M, (target_W, target_H))

    # Random Translation
    h, w = novel_mask_hw.shape
    if base_mask_hw.shape[0] > h and base_mask_hw.shape[1] > w:
        paste_x = torch.randint(low=0, high=base_mask_hw.shape[1] - w, size=(1,))
        paste_y = torch.randint(low=0, high=base_mask_hw.shape[0] - h, size=(1,))
        paste_x = int(paste_x)
        paste_y = int(paste_y)
    else:
        paste_x = 0
        paste_y = 0
    
    if False:
        # Paste under base objects
        base_occupied_idx = base_mask_hw[paste_y:paste_y+h,paste_x:paste_x+w] > 0
        novel_mask_hw[base_occupied_idx] = 0
    
    base_img_hwc[paste_y:paste_y+h,paste_x:paste_x+w,:][novel_mask_hw > 0,:] = novel_img_hwc[novel_mask_hw > 0,:]
    base_mask_hw[paste_y:paste_y+h,paste_x:paste_x+w][novel_mask_hw > 0] = mask_id

    return (base_img_hwc, base_mask_hw)

class MaskFormerSemanticCopyDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticCopyDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )
        
        # Check image and sem_seg_gt are numpy array
        assert isinstance(image, np.ndarray), "image should be a numpy array"
        assert isinstance(sem_seg_gt, np.ndarray), "sem_seg_gt should be a numpy array"
        assert image.dtype == np.uint8, image.dtype
        assert sem_seg_gt.dtype == np.float64, sem_seg_gt.dtype

        # Mask out class 16-20
        for i in range(16, 21):
            sem_seg_gt[sem_seg_gt == i] = self.ignore_label
        
        # Convert to int for process
        sem_seg_gt = sem_seg_gt.astype(int)

        # Apply copy-paste
        base_dir = os.getenv("DETECTRON2_DATASETS", "datasets")
        base_dir = os.path.join(base_dir, 'neurips2023', 'final_masks', 'semi_supervised', 'pascal5i_3', 'synthetic_third_stage')

        cls_idx = np.random.randint(16, 21)
        cls_idx = int(cls_idx)

        # Copy-paste
        obj_dir = os.path.join(base_dir, str(cls_idx))
        # Randomly select a jpg
        if torch.rand(1) < 0.5:
            img_fn_list = ["{}.jpg".format(str(i).zfill(6)) for i in range(5)]
        else:
            img_fn_list = [i for i in os.listdir(obj_dir) if i.endswith("jpg")]
        obj_img_path = os.path.join(obj_dir, np.random.choice(img_fn_list))
        obj_mask_path = obj_img_path.replace("jpg", "png")

        obj_img = Image.open(obj_img_path).convert('RGB')
        obj_img = np.array(obj_img)
        obj_mask = np.array(Image.open(obj_mask_path).convert('L'))

        obj_mask[obj_mask > 0] = 1

        if torch.rand(1) < 0.5:
            # Do copy paste
            img_roi, mask_roi = crop_partial_img(obj_img, obj_mask)
            mask_roi = mask_roi.cpu().numpy()
            mask_roi = mask_roi.astype(np.uint8)

            image, sem_seg_gt = copy_and_paste(img_roi, mask_roi, image, sem_seg_gt, cls_idx)
        else:
            image = obj_img
            sem_seg_gt = obj_mask
            sem_seg_gt[sem_seg_gt > 0] = cls_idx
            sem_seg_gt = sem_seg_gt.astype(int)
        
        # Convert back to float64
        sem_seg_gt = sem_seg_gt.astype(np.float64)

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
