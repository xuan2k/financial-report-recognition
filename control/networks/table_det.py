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
import yaml

class TableDetection():
    def __init__(self,
                 cfg) -> None:
        
        self.cfg = {}

        if isinstance(cfg, str):
            with open(cfg, encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.cfg.update(config)

        elif isinstance(cfg, dict):    
            self.cfg.update(cfg)
        else:
            raise TypeError("Unsupport config input type!")

        self.config_file = self.cfg["config"]
        self.checkpoint_file = self.cfg["checkpoint"]
        self.device = self.cfg["device"]
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.input = None
        self.ouput = None

    def reload(self):
        self.input = None
        self.ouput = None

    def forward(self, x):
        self.reload()
        if isinstance(x, Image.Image):
            # Convert the PIL image to a numpy array
            numpy_array = np.array(x)

            # Convert the numpy array to a cv2 image
            x = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)

        self.input = x
        self.ouput = inference_detector(self.model, self.input)
        self.ouput = self.processing(self.ouput)
        return self.ouput
    
    def processing(self, x):
        if self.input is not None:
            result = {}
            bbox = list(x[0][0][0][0:4].astype(int))
            # new_img = crop_img(self.input, bbox=bbox)
            # result["img"] = new_img
            result["bbox"] = bbox
            return result
        else:
            raise TypeError("The input is in NoneType.")

    def __call__(self, image) -> Any:
            return self.forward(image)
    
# if __name__ == "__main__":
#     cfg = r"/workspace/source/configs/model/table_det/model.yml"
#     t = TableDetection(cfg)
#     print("--> load ok")
#     # t(input)
#     img_path = r"/workspace/warehouse/sample/test2/1.png"
#     img = cv2.imread(img_path)

#     output = t(img)

#     print(f"Check {output}")

#     cv2.imwrite(os.path.join(os.path.dirname(img_path), "res.png"), output["img"])


