import numpy as np
import cv2
import os
import tqdm

colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
                (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25, 0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
                (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
                (0.5, 0.75, 0), (0, 0.25, 0.5), (1, 1, 1)]


def get_color(c):
	if c >= len(colormap):
		return (1, 1, 1)
	else:
		return colormap[c]

dir_name = "./{}"
vis_dir_name = "./vis/{}"

for dir_name_short in ["labels"]:

	for img_name in tqdm.tqdm(os.listdir(dir_name.format(dir_name_short))):
		if "DS_Store" in img_name:
			continue

		img = cv2.imread(os.path.join(dir_name.format(dir_name_short), img_name), 0)
		img[img == 255] = 2550
		if not "labels" in dir_name_short:
			img = img//10

		pseudo_lab_vis = 225*np.array([[get_color(p) for p in prow] for prow in img])
		pseudo_lab_vis = np.uint8(pseudo_lab_vis)
		os.makedirs(vis_dir_name.format(dir_name_short), exist_ok=True)
		cv2.imwrite(os.path.join(vis_dir_name.format(dir_name_short), img_name.replace(".", "_vis.")), pseudo_lab_vis)
