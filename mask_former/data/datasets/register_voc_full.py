# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

VOC_FULL_CATEGORIES = [
              {"name": "aeroplane", "id": 1, "trainId": 0},
              {"name": "bicycle", "id": 2, "trainId": 1},
              {"name": "bird", "id": 3, "trainId": 2},
              {"name": "boat", "id": 4, "trainId": 3},
              {"name": "bottle", "id": 5, "trainId": 4},
              {"name": "bus", "id": 6, "trainId": 5},
              {"name": "car", "id": 7, "trainId": 6},
              {"name": "cat", "id": 8, "trainId": 7},
              {"name": "chair", "id": 9, "trainId": 8},
              {"name": "cow", "id": 10, "trainId": 9},
              {"name": "diningtable", "id": 11, "trainId": 10},
              {"name": "dog", "id": 12, "trainId": 11},
              {"name": "horse", "id": 13, "trainId": 12},
              {"name": "motorbike", "id": 14, "trainId": 13},
              {"name": "person", "id": 15, "trainId": 14},
              {"name": "potted plant", "id": 16, "trainId": 15},
              {"name": "sheep", "id": 17, "trainId": 16},
              {"name": "sofa", "id": 18, "trainId": 17},
              {"name": "train", "id": 19, "trainId": 18},
              {"name": "tvmonitor", "id": 20, "trainId": 19}]


def _get_voc_full_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in VOC_FULL_CATEGORIES]
    assert len(stuff_ids) == 20, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in VOC_FULL_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_voc_full(root):
    root = os.path.join(root, "VOCdevkit", "VOC2012_detectron2")
    meta = _get_voc_full_meta()
    for split_name in ["trainaug", "val"]:
        image_dir = os.path.join(root, split_name, "images")
        gt_dir = os.path.join(root, split_name, "annotations")
        name = f"voc_full_sem_seg_{split_name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 8-bit PNG images
        )

    # for name, dirname in [("train", "training"), ("val", "validation")]:
    #     image_dir = os.path.join(root, "images_detectron2", dirname)
    #     gt_dir = os.path.join(root, "annotations_detectron2", dirname)
    #     name = f"ade20k_full_sem_seg_{name}"
    #     DatasetCatalog.register(
    #         name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
    #     )
    #     MetadataCatalog.get(name).set(
    #         stuff_classes=meta["stuff_classes"][:],
    #         image_root=image_dir,
    #         sem_seg_root=gt_dir,
    #         evaluator_type="sem_seg",
    #         ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
    #     )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_voc_full(_root)
