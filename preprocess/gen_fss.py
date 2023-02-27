import cv2
import os
import numpy as np
import tqdm

root_ann_dir = "./datasets/VOCdevkit/VOC2012/SegmentationClassAug"
val_txt = "./datasets/VOCdevkit/VOC2012/val.txt"
save_root_dir = "./datasets/VOC2012/annotations_detectron2/val_fss"
vis_save_root_dir = "./datasets/VOC2012/annotations_detectron2/vis_val_fss"



with open(val_txt) as f:
    val_list  = f.readlines()
val_list = [v.replace("\n", "") for v in val_list]

for val_name in tqdm.tqdm(val_list):
    read_path = os.path.join(root_ann_dir, val_name+".png")

    label = cv2.imread(read_path)

    for cls_idx in np.unique(label):
        if cls_idx ==  0:
            continue
        label_cur = label == cls_idx
        label_cur = np.uint8(label_cur)

        save_dir = os.path.join(save_root_dir, str(cls_idx))
        os.makedirs(save_dir, exist_ok=True)

        vis_save_dir = os.path.join(vis_save_root_dir, str(cls_idx))
        os.makedirs(vis_save_dir, exist_ok=True)

        label_cur = label_cur[:, :, 0]

        cv2.imwrite(os.path.join(save_dir, val_name+".png"), label_cur)
        cv2.imwrite(os.path.join(vis_save_dir, val_name+".png"), 255*label_cur)