# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from .utils import load_binary_mask

CLASS_NAMES = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
)

# novel_classes = [15, 16, 17, 18, 19]
# novel_classes = [0, 1, 2, 3, 4]
novel_classes = list(range(60, 80))

BASE_CLASS_NAMES = [
    c for i, c in enumerate(CLASS_NAMES) if i not in novel_classes
]
NOVEL_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if i in novel_classes]


def _get_coco14_meta(cat_list):
    ret = {
        "stuff_classes": cat_list,
    }
    return ret


def register_all_coco14(root):
    root = os.path.join(root)
    meta = _get_coco14_meta(CLASS_NAMES)
    base_meta = _get_coco14_meta(BASE_CLASS_NAMES)

    novel_meta = _get_coco14_meta(NOVEL_CLASS_NAMES)
    img_dir_root = "./datasets/coco"

    for name, image_dirname, sem_seg_dirname in [
        ("train", "JPEGImages", "coco/stuffthingmaps_detectron2/train2014"),
        ("test", "JPEGImages", "coco/stuffthingmaps_detectron2/val2014"),
    ]:
        image_dir = os.path.join(img_dir_root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"coco14_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )
        MetadataCatalog.get(all_name).set(
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
        )
        # classification
        DatasetCatalog.register(
            all_name + "_classification",
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name + "_classification").set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
            **meta,
        )
        

        # weak zero shot
        image_dir = os.path.join(img_dir_root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_weak_base")
        base_name = f"coco14_weak_base_sem_seg_{name}"

        DatasetCatalog.register(
            base_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(base_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **base_meta,
        )

        # zero shot
        image_dir = os.path.join(img_dir_root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_base")
        base_name = f"coco14_base_sem_seg_{name}"

        DatasetCatalog.register(
            base_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        print(base_name)
        MetadataCatalog.get(base_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **base_meta,
        )

        # classification
        DatasetCatalog.register(
            base_name + "_classification",
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(base_name + "_classification").set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            **base_meta,
        )
        # zero shot
        image_dir = os.path.join(img_dir_root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_novel")
        novel_name = f"coco14_novel_sem_seg_{name}"
        DatasetCatalog.register(
            novel_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(novel_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **novel_meta,
        )

    # fss

    for cls_idx, cur_class_name in enumerate(CLASS_NAMES):
        image_dir = os.path.join(img_dir_root, image_dirname)
        gt_dir = os.path.join(root, "coco/stuffthingmaps_detectron2/val_fss/" + str(cls_idx))
        dataset_name = "coco14_fss_sem_seg_cls" + str(cls_idx)

        class_meta = {
            "stuff_classes": ["nature", cur_class_name],
        }
        DatasetCatalog.register(
            dataset_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **class_meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
# _root = "./datasets/COCO14"
register_all_coco14(_root)