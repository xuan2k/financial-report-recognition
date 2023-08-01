from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
from tqdm import tqdm
import argparse
import os
import json
import yaml
from typing import Any
# import torch

class TextDetection():
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

        self.model = PaddleOCR(use_angle_cls=False, 
                               lang="en", 
                               type="ocr", 
                               recovery=False,
                               table=False,
                               layout=False,
                               rec=False)
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
        self.ouput = self.model.ocr(self.input, cls=False, rec=False)[0]
        self.ouput = self.processing(self.ouput)
        return self.ouput
    
    def processing(self, x):
        if self.input is not None:
            result = {}
            bbox = [line for line in x]
            bboxes = []
            
            for box in bbox:
                x = [int(i[0]) for i in box]
                y = [int(i[1]) for i in box]
                tmp = [min(x),min(y), max(x), max(y)]
                bboxes.append(tmp)
            
            bboxes.reverse()
            result["bbox"] = bboxes
            return result
        else:
            raise TypeError("The input is in NoneType.")

    def __call__(self, image) -> Any:
            return self.forward(image)
    
# if __name__ == "__main__":
#     cfg = r"../configs/model/text_det/model.yml"
#     t = TextDetection(cfg)
#     print("--> load ok")
#     # t(input)
#     img_path = r"/home/xuan/Project/OCR/demo/rgb/1.png"
#     img = cv2.imread(img_path)

#     output = t(img)

#     print(f"Check {output}")

    # cv2.imwrite(os.path.join(os.path.dirname(img_path), "res.png"), output["img"])