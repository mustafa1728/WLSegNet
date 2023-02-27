import os
import glob
import functools
from mmcv.utils import track_parallel_progress
import numpy as np
from PIL import Image
import json
import warnings
import fire
from itertools import chain

# novel_classes = [0, 1, 2, 3, 4]

novel_class_folds = {
    0: list(range(20)),
    1: list(range(20, 40)),
    2: list(range(40, 60)),
    3: list(range(60, 80)),
}

def process_novel(cls_idx, fold=0, num_classes=80):
    novel_classes = novel_class_folds[fold]
    all_classes = list(range(num_classes))
    base_classes = [c for c in all_classes if c not in novel_classes]
    old_to_new_classes = {c: i for i, c in enumerate(base_classes)}
    return old_to_new_classes[cls_idx]

def count_cls(file_path, ignore_index=[255], depth=1, rem_novel=True, fold=0):
    novel_classes = novel_class_folds[fold]
    cls_label = np.unique(np.asarray(Image.open(file_path))).tolist()
    if rem_novel:
        cls_label = [process_novel(l, fold) for l in cls_label if l not in ignore_index and l not in novel_classes]
    else:
        cls_label = [l for l in cls_label if l not in ignore_index]
    return [os.path.join(*file_path.split(os.sep)[-depth:]), cls_label]


def main(gt_dir, map_file_save_path, rem_novel=False, ignore_index=[255], ext=".png", recursive=False):
    if not os.path.isdir(gt_dir):
        warnings.warn(f"{gt_dir} is not a valid directory")
        return
    gt_file_list = glob.glob(os.path.join(gt_dir, "*" + ext), recursive=recursive)
    print(f"Find {len(gt_file_list)}")
    _func = functools.partial(count_cls, ignore_index=ignore_index, rem_novel=rem_novel, fold=0)
    results = track_parallel_progress(_func, gt_file_list, nproc=16)
    results = {r[0]: r[1] for r in results}
    with open(map_file_save_path, "w") as f:
        json.dump(results, f)

    for fold in range(4):
        _func = functools.partial(count_cls, ignore_index=ignore_index, rem_novel=rem_novel, fold=fold)
        results = track_parallel_progress(_func, gt_file_list, nproc=16)
        results = {r[0]: r[1] for r in results}
        map_file_save_path_fold = map_file_save_path.split(".")[0] + "_fold" + str(fold) + "." + map_file_save_path.split(".")[-1]
        with open(map_file_save_path_fold, "w") as f:
            json.dump(results, f)


def main_ctyscapes(
    gt_dir, map_file_save_path, ignore_index=[255], ext=".png", recursive=False
):
    if not os.path.isdir(gt_dir):
        warnings.warn(f"{gt_dir} is not a valid directory")
        return
    cities = os.listdir(gt_dir)
    gt_file_list = list(
        chain.from_iterable(
            [
                glob.glob(
                    os.path.join(gt_dir, city, "*" + ext),
                )
                for city in cities
            ]
        )
    )
    print(gt_file_list[0])
    print(f"Find {len(gt_file_list)}")
    _func = functools.partial(count_cls, ignore_index=ignore_index, depth=2)
    results = track_parallel_progress(_func, gt_file_list, nproc=16)
    results = {r[0]: r[1] for r in results}
    with open(map_file_save_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    fire.Fire(main)
