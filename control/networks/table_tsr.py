"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_pub.py
# Abstract       :    Script for inference

# Current Version:    1.0.1
# Date           :    2021-09-23
##################################################################################################
"""

import cv2
import json
import jsonlines
import numpy as np
from tqdm import tqdm
# from davarocr.davar_table.utils import TEDS, format_html
from davarocr.davar_common.apis import inference_model, init_model
import os
import argparse

arg = argparse.ArgumentParser("Table structure recognition")

arg.add_argument("--img_dir", type=str,
                 default="/workspace/warehouse/result/rgb/2.png")

arg.add_argument("--out_dir", type=str,
                 default="/workspace/warehouse/result")

arg.add_argument("--visualize", action="store_true")


arg = arg.parse_args()

# visualization setting
do_visualize = arg.visualize # whether to visualize
out_path = arg.out_dir
vis_dir = os.path.join(out_path, "vis")
savepath = os.path.join(out_path, "pred")

for path in [out_path, vis_dir, savepath]:
    if not os.path.exists(path):
        os.makedirs(path)
# "/home/xuan/Project/OCR/code/git_code/DAVAR-Lab-OCR/demo/table_recognition/lgpma/result/vis/" # path to save visualization results

# path setting
# savepath = "/home/xuan/Project/OCR/code/git_code/DAVAR-Lab-OCR/demo/table_recognition/lgpma/result/pred" # path to save prediction
config_file = '../configs/model/table_tsr/lgpma_pub.py' # config path
checkpoint_file = '../checkpoints/table_tsr/maskrcnn-lgpma-pub-e12-pub.pth' # model path

# loading model from config file and pth file
model = init_model(config_file, checkpoint_file)

# getting image prefix and test dataset from config file
img_prefix = arg.img_dir
# test_dataset = model.cfg["data"]["test"]["ann_file"]
# with jsonlines.open(test_dataset, 'r') as fp:
#     test_file = list(fp)

# generate prediction of html and save result to savepath
pred_dict = dict()

if os.path.isfile:
    imgs = [os.path.basename(img_prefix)]
    img_prefix = os.path.dirname(img_prefix)
else:
    imgs = os.listdir(img_prefix)

for sample in tqdm(imgs):
    # predict html of table
    img_path = os.path.join(img_prefix, sample)
    im = cv2.imread(img_path)
    # print(i.shape)
    h,w,c = im.shape
    max_h = 1024
    if h > max_h:
        ra = h/max_h
        im = cv2.resize(im, (int(w/ra), int(max_h)))
    # print(i.shape)
    result = inference_model(model, im)[0]

    # detection results visualization
    if do_visualize:
        img = cv2.imread(img_path)
        img_name = img_path.split("/")[-1]
        b_res = result['content_ann']['bboxes']
        if h > max_h:
            # b_res = np.asarray(b_res)
            # b_res*=int(ra)
            bboxes = [list(map(lambda x: int(x*ra), [b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]])) for b in b_res if len(b) > 0]
            # bboxes = [[ra*b[0], ra*b[1], ra*b[2], ra*b[1], ra*b[2], ra*b[3], ra*b[0], ra*b[3]] for b in b_res if len(b) > 0]
        else:    
            bboxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in b_res if len(b) > 0]
            # bboxes = [[2*b[0], 2*b[1], 2*b[2], 2*b[1], 2*b[2], 2*b[3], 2*b[0], 2*b[3]] for b in b_res if len(b) > 0]

        for box in bboxes:
            for j in range(0, len(box), 2):
                cv2.line(img, (box[j], box[j + 1]), (box[(j + 2) % len(box)], box[(j + 3) % len(box)]), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(vis_dir, img_name), img)

    result.update({"ratio":ra})
    pred_dict[sample] = result
    with open(os.path.join(savepath, f"res_{img_name.split('.')[0]}_TSR.txt"), "w", encoding="utf-8") as writer:
        json.dump(pred_dict[sample], writer, ensure_ascii=False)

# generate ground-truth of html from pubtabnet annotation of test dataset.
# gt_dict = dict()
# for data in test_file:
#     if data['filename'] in pred_dict.keys():
#         str_true = data['html']['structure']['tokens']
#         gt_dict[data['filename']] = {'html': format_html(data)}

# evaluation using script from PubTabNet
# teds = TEDS(structure_only=True, n_jobs=16)
# scores = teds.batch_evaluate(pred_dict, gt_dict)
# print(np.array(list(scores.values())).mean())
