import cv2
import os
import  tqdm
import numpy as np

all_classes =  list(range(80))
fold_novel_classes = {
    0: list(range(20)),
    1: list(range(20, 40)),
    2: list(range(40, 60)),
    3: list(range(60, 80)),
}

full_val_dir  =  "./datasets/coco/stuffthingmaps_detectron2/val2014"
fold_val_dir =  "./datasets/coco/stuffthingmaps_detectron2/folds/{}/val_novel"

new_all_labels = []

for fold in [3]:
    
    novel_classes = fold_novel_classes[fold]
    class_mapping = {cl: i for i, cl in enumerate(novel_classes)}

    base_classes = [c for c  in all_classes if  c not in novel_classes]
    save_dir = fold_val_dir.format(fold)
    os.makedirs(save_dir, exist_ok=True)

    for img_name in tqdm.tqdm(os.listdir(full_val_dir)):
        if ".DS_Store" in  img_name:
            continue
        full_lab = cv2.imread(os.path.join(full_val_dir, img_name), 0)

        for c in base_classes:
            full_lab[full_lab ==  c] =  255
        for c in novel_classes:
            full_lab[full_lab ==  c] =  class_mapping[c]


        new_all_labels += np.unique(full_lab).tolist()
        # print(np.unique(new_all_labels), end="\r")

        cv2.imwrite(os.path.join(save_dir, img_name), full_lab)



