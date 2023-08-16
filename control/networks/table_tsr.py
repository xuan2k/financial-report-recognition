"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_pub.py
# Abstract       :    Script for inference

# Current Version:    1.0.1
# Date           :    2021-09-23
##################################################################################################
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import cv2
import json
import jsonlines
import numpy as np
from tqdm import tqdm
from PIL import Image
# from davarocr.davar_table.utils import TEDS, format_html
from davarocr.davar_common.apis import inference_model, init_model
import argparse
import yaml
from typing import Any
import time

class TableStructureRecognition():
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
        self.model = init_model(self.config_file, self.checkpoint_file)
        self.max_height = self.cfg["max_height"]
        self.resize = (False,1.0)
        self.input = None
        self.ouput = None

    def reload(self):
        self.resize = (False,1.0)
        self.input = None
        self.ouput = None

    def forward(self, x):
        self.reload()
        if isinstance(x, Image.Image):
            # Convert the PIL image to a numpy array
            numpy_array = np.array(x)

            # Convert the numpy array to a cv2 image
            x = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    
        h,w,c = x.shape
        ra = 1
        if h > self.max_height:
            ra = h/self.max_height
            x = cv2.resize(x, (int(w/ra), int(self.max_height)))
            self.resize = (True, ra)

        self.input = x
        self.ouput = inference_model(self.model, self.input)[0]
        self.ouput = self.processing(self.ouput)
        return self.ouput
    
    def processing(self, x):
        if self.input is not None:
            result = {}
            bbox = x['content_ann']['bboxes']

            if self.resize[0]:
                new_bbox = [[int(value*self.resize[1]) for value in sublist] for sublist in bbox]
                x['content_ann']['bboxes'] = new_bbox
            result.update(x)
            return result
        else:
            raise TypeError("The input is in NoneType.")

    def __call__(self, image) -> Any:
            return self.forward(image)
    
if __name__ == "__main__":
    cfg = r"/workspace/source/configs/model/table_tsr/model.yml"
    t = TableStructureRecognition(cfg)
    print("--> load ok")
    # t(input)
    ver = 6
    for i in range(6):
        ver = i+1
        try:
            img_path = f"/workspace/warehouse/demo/rgb/res_{ver}.png"
            out = f"/workspace/warehouse/demo/pred_1/res_{ver}_TSR.txt"

            if not os.path.exists(os.path.dirname(out)):
                os.makedirs(os.path.dirname(out))

            img = cv2.imread(img_path)

            output = t(img)
            output["ratio"] = 1.0
            with open(out, 'w', encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False)

            print(f"Check {output}")
        except:
            print(f"Cannot run {ver}")

    # cv2.imwrite(os.path.join(os.path.dirname(img_path), "res.png"), output["img"])