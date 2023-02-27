import cv2
import os
import numpy as np
import tqdm

root_ann_dir = "./datasets/VOCdevkit/VOC2012/SegmentationClassAug"
val_txt = "./datasets/VOCdevkit/VOC2012/val.txt"
save_root_dir = "./datasets/VOC2012/annotations_detectron2/val_fss/2way"
vis_save_root_dir = "./datasets/VOC2012/annotations_detectron2/vis_val_fss/2way"

fold_novel_classes = {
    0: list(range(1, 6)),
    1: list(range(6, 11)),
    2: list(range(11, 16)),
    3: list(range(16, 21)),
}
all_classes = list(range(20))

with open(val_txt) as f:
    val_list  = f.readlines()
val_list = [v.replace("\n", "") for v in val_list]

for val_name in tqdm.tqdm(val_list):
    read_path = os.path.join(root_ann_dir, val_name+".png")

    label = cv2.imread(read_path)

    for fold in range(4):
        cur_novel_classes = fold_novel_classes[fold]

        for cls_idx1 in cur_novel_classes:
            for cls_idx2 in cur_novel_classes:
                if cls_idx2 <= cls_idx1:
                    continue

                if np.sum((label == cls_idx1).sum() + (label == cls_idx2).sum()) == 0:
                    continue
                
                mask = (label == cls_idx1) + 2 * (label == cls_idx2)
                mask = np.uint8(mask)

                

                save_dir = os.path.join(save_root_dir, "{}_{}".format(cls_idx1, cls_idx2))
                os.makedirs(save_dir, exist_ok=True)

                vis_save_dir = os.path.join(vis_save_root_dir, "{}_{}".format(cls_idx1, cls_idx2))
                os.makedirs(vis_save_dir, exist_ok=True)

                mask = mask[:, :, 0]

                cv2.imwrite(os.path.join(save_dir, val_name+".png"), mask)
                cv2.imwrite(os.path.join(vis_save_dir, val_name+".png"), 127*mask)

                
