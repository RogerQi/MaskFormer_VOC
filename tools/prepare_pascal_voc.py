import os
import shutil
import numpy as np
from tqdm import tqdm

base_dir = os.getenv("DETECTRON2_DATASETS", "datasets")
base_dir = os.path.join(base_dir, "VOCdevkit")
raw_voc_dir = os.path.join(base_dir, "VOC2012")
target_voc_dir = os.path.join(base_dir, "VOC2012_detectron2")

os.makedirs(target_voc_dir, exist_ok=True)

for split in ["trainaug", "val"]:
    # Read file names splits
    file_list = os.path.join(raw_voc_dir, "ImageSets/Segmentation/{}.txt".format(split))
    file_names = np.loadtxt(file_list, dtype=str)

    split_dir = os.path.join(target_voc_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    image_dir = os.path.join(split_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    annotation_dir = os.path.join(split_dir, "annotations")
    os.makedirs(annotation_dir, exist_ok=True)

    
    for name in tqdm(file_names):
        # Copy images
        src = os.path.join(raw_voc_dir, "JPEGImages", name + ".jpg")
        dst = os.path.join(image_dir, name + ".jpg")
        shutil.copy(src, dst)

        # Copy annotations
        src = os.path.join(raw_voc_dir, "SegmentationClassAug", name + ".png")
        dst = os.path.join(annotation_dir, name + ".png")
        shutil.copy(src, dst)

