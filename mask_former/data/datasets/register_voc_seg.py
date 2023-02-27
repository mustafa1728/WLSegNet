# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from .utils import load_binary_mask

CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

# novel_classes = [15, 16, 17, 18, 19]
novel_classes = [0, 1, 2, 3, 4]

fold_novel_classes = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14],
    3: [15, 16, 17, 18, 19],
}

BASE_CLASS_NAMES_FOLD = {
    fol: [c for i, c in enumerate(CLASS_NAMES) if i not in fold_novel_classes[fol]] for fol in range(4)
}

NOVEL_CLASS_NAMES_FOLD = {
    fol: [c for i, c in enumerate(CLASS_NAMES) if i in fold_novel_classes[fol]] for fol in range(4)
}

BASE_CLASS_NAMES = [
    c for i, c in enumerate(CLASS_NAMES) if i not in novel_classes
]
NOVEL_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if i in novel_classes]


def _get_voc_meta(cat_list):
    ret = {
        "stuff_classes": cat_list,
    }
    return ret


def register_all_voc_11k(root):
    root = os.path.join(root, "VOC2012")
    meta = _get_voc_meta(CLASS_NAMES)
    base_meta = _get_voc_meta(BASE_CLASS_NAMES)

    novel_meta = _get_voc_meta(NOVEL_CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "JPEGImages", "annotations_detectron2/train"),
        ("test", "JPEGImages", "annotations_detectron2/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"voc_sem_seg_{name}"
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
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_weak_base")
        base_name = f"voc_weak_base_sem_seg_{name}"

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
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_base")
        base_name = f"voc_base_sem_seg_{name}"

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
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_novel")
        novel_name = f"voc_novel_sem_seg_{name}"
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

    # all folds zss

    for fold in range(4):
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, "annotations_detectron2/folds/{}/train_weak_base".format(fold))
        dataset_fold_name = "voc_weak_base_sem_seg_train_fold{}".format(fold)
        DatasetCatalog.register(
            dataset_fold_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        fold_meta = _get_voc_meta(BASE_CLASS_NAMES_FOLD[fold])
        MetadataCatalog.get(dataset_fold_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **fold_meta,
        )

        # classification
        DatasetCatalog.register(
            "fold{}_classification".format(fold),
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg", label_count_file="./datasets/VOC2012/annotations_detectron2/train_base_label_count_fold{}.json".format(fold)
            ),
        )
        fold_meta = _get_voc_meta(BASE_CLASS_NAMES_FOLD[fold])
        MetadataCatalog.get("fold{}_classification".format(fold)).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            **fold_meta,
        )


        gt_dir = os.path.join(root, "annotations_detectron2/val")
        all_name = "voc_sem_seg_val_fold{}".format(fold)
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
                    meta["stuff_classes"].index(n) for n in fold_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in fold_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
        )

    # fss

    for idx, cur_class_name in enumerate(CLASS_NAMES):
        cls_idx = idx + 1
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, "annotations_detectron2/val_fss/" + str(cls_idx))
        dataset_name = "voc_fss_sem_seg_cls" + str(cls_idx)
        # print(dataset_name, gt_dir)
        class_meta = {
            "stuff_classes": ["alien", cur_class_name],
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

    # fss 2 way
    class_pairs = [(i, j) for i in range(1, 21) for j in range(1, 21)] 

    for cls_1, cls_2 in class_pairs:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, "annotations_detectron2/val_fss/2way/{}_{}".format(cls_1, cls_2))
        dataset_name = "voc_fss_sem_seg_2way_cls{}_{}".format(cls_1, cls_2)
        # print(dataset_name, gt_dir)
        class_meta = {
            "stuff_classes": ["alien", CLASS_NAMES[cls_1-1], CLASS_NAMES[cls_2-1]],
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



def register_all_voc_pseudo(root, pseudo_sem_dir):
    root = os.path.join(root, "VOC2012")
    meta = _get_voc_meta(CLASS_NAMES)
    base_meta = _get_voc_meta(BASE_CLASS_NAMES)
    novel_meta = _get_voc_meta(NOVEL_CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "JPEGImages", "annotations_detectron2/train"),
    ]:
        image_dir = os.path.join(root, image_dirname)

        all_name = f"voc_sem_seg_{name}_pseudo"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=pseudo_sem_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=pseudo_sem_dir,
            evaluator_type="sem_seg",
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


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_voc_11k(_root)
_pseudo_dir = os.getenv("DETECTRON2_SEM_PSEUDO", "output/inference")
register_all_voc_pseudo(_root, _pseudo_dir)
