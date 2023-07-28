import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import argparse
from utils.processing import vis_img, crop_img
from tqdm import tqdm
import shutil
import math as m
import cv2
from typing import Any
import numpy as np
from PIL import Image

# class TableDetection():
#     def __init__(self,
#                  opts) -> None:
#         self.opts = opts
#         self.config_file = opts.config
#         self.checkpoint_file = opts.model
#         self.device = self.opts.device
#         self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)

#     def load_image(self, image):

#         img = Image.open(image)


#         return [img]

#     def forward(self, x):
#         self.input = self.load_image(x)
#         self.output = []
#         for item in self.input:
#             self.ouput.append(inference_detector(self.model, item))

#     def __call__(self, image) -> Any:
#             return self.forward(image)
    
# if __name__ == "__main__":
#     t = TableDetection('a')
#     t(input)

#import to check if the lib is installed
import mmcv

def parse_arg():
    parser = argparse.ArgumentParser("Options for cascadeTabnet table detection inference.")
    parser.add_argument("--img_dir", default="/workspace/warehouse/2.png",
                        help="Single input image or image list to predict.")
    
    parser.add_argument("--save_dir", default="/workspace/warehouse/result",
                        help="Directory to save inference result.")
    
    parser.add_argument("--config", default='./configs/model/table_det/cascade_mask_rcnn_hrnetv2p_w32_20e.py',
                        help="Path to config file.")
    
    parser.add_argument("--model", default='./checkpoints/table_det/epoch_14.pth',
                        help="Path to pretrained model file.")
    
    parser.add_argument("--visualize", action="store_true",
                        help="Call it when we need to visualize the bounding box for table.")
    
    parser.add_argument("--crop", action="store_true",
                        help="Call it when we need to visualize the bounding box for table.")
    # parser.add_argument("--batch_size", default='/workspace/source/pretrained/epoch_14.pth',
    #                     help="Path to pretrained model file.")
    
    return parser.parse_args()


if __name__ == '__main__':
    # Load model

    opts = parse_arg()

    config_file = opts.config
    checkpoint_file = opts.model
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # Test a single image 
    # img = "/workspace/warehouse/data/6.png"
    img_dir = opts.img_dir
    if os.path.isfile:
        imgs = [os.path.basename(img_dir)]
        img_dir = os.path.dirname(img_dir)
    else:
        imgs = os.listdir(img_dir)

    # if not os.path.exists(opts.save_dir):
    #     os.makedirs(opts.save_dir)

    res_dir = os.path.join(opts.save_dir, "res")
    rgb_dir = os.path.join(opts.save_dir, "rgb")

    for directory in [opts.save_dir, res_dir, rgb_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    iter = tqdm(imgs)

    for img in iter:
        # Run Inference
        img_path = os.path.join(img_dir, img)
        result = inference_detector(model, img_path)
        if len(result[0][0]) == 0 or result[0][0][0][4] < 0.97:
            # print("No table found!")
            # exit()
            # iter.set_description(f"No table in {img}")
            # shutil.copy(img_path, os.path.join(non_dir, img))
            continue
        # print(result)

        # print(result[0][0])

        bbox=list(result[0][0][0][0:4])

        table_sq = m.sqrt(((bbox[1]-bbox[3])**2)*(bbox[0]-bbox[2])**2)
        load_img = cv2.imread(img_path)
        h,w,_ = load_img.shape
        img_eq = h*w
        if table_sq < 0.3*h*w:
            # shutil.copy(img_path, os.path.join(tiny_dir, img))
            continue

        f = open(os.path.join(res_dir, f"res_{img.split('.')[0]}.txt"), 'w')

        for e in result[0][0][0]: 
            
            f.write(str(e) + ',')

        f.close()

        # print(f"-> Tabel found in {img}. Result saved to res_{img.split('.')[0]}.txt")
        iter.set_description(f"-> Tabel found in {img}.")

        # print(result[0][0][0][0:3])
        if opts.visualize:
            # vis_img(img_path, rgb_dir, bbox=bbox)
            res_img = vis_img(load_img, bbox=bbox)
            cv2.imwrite(os.path.join(rgb_dir, img), res_img)

        if opts.crop:
            # vis_img(img_path, rgb_dir, bbox=bbox)
            load_img = crop_img(load_img, bbox=bbox)
            cv2.imwrite(os.path.join(rgb_dir, img), load_img)

        # else:
        #     shutil.copy(img_path, os.path.join(rgb_dir, img))
            


    # Visualization results
    # show_result_pyplot(img, result,('Bordered', 'cell', 'Borderless'), score_thr=0.85)