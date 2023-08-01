import os
import sys
import json
import random
import cv2
import numpy as np
from typing import Any
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from PIL import Image

class TextRecognition():
    def __init__(self,
                 cfg) -> None:
        

        if isinstance(cfg, str):
            self.cfg = Cfg.load_config_from_file(cfg)

        else:
            raise TypeError("Unsupport config input type!")

        self.model = Predictor(self.cfg)
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

        self.input = Image.fromarray(x)
        self.ouput = self.model.predict(self.input)
        # self.ouput = self.processing(self.ouput)
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
    # cfg = r"../configs/model/text_rec/model.yml"
    # t = TextRecognition(cfg)
    # print("--> load ok")
    # # t(input)
    # img_path = r"/home/xuan/Project/OCR/demo/rgb/1.png"
    # img = cv2.imread(img_path)

    # bb_file = f"/home/xuan/Project/OCR/demo/text/1/res.json"
    # with open(bb_file, 'r') as f:
    #     res = json.load(f)

    # bboxes = res["bboxes"]
    
    # for box in bboxes:

    #     inp = crop_img(img, box)

    #     output = t(inp)

    #     print(f"Check {output}")

    # cv2.imwrite(os.path.join(os.path.dirname(img_path), "res.png"), output["img"])