import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import json
import random
import cv2
import numpy as np
from typing import Any
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from utils.processing import crop_img
from PIL import Image

class TextRecognition():
    def __init__(self,
                 cfg) -> None:
        

        # if isinstance(cfg, str):
        #     self.cfg = Cfg.load_config_from_file(cfg)

        # else:
        #     raise TypeError("Unsupport config input type!")
        self.cfg = Cfg.load_config_from_name("vgg_transformer")

        self.model = Predictor(self.cfg)
        # self.max_height = 16
        # self.max_width =512
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
        
        h,w,c = x.shape
        ra = 1
        # if w > self.max_width:
        #     ra = w/self.max_width
        #     h = h/ra
        #     x = cv2.resize(x, (int(self.max_width), int(h)))
        # if h > self.max_height:
        #     ra = h/self.max_height
        #     w = w/ra
        #     x = cv2.resize(x, (int(w), int(self.max_height)))

        self.input = Image.fromarray(x)
        self.ouput = self.model.predict(self.input)
        # name = self.ouput
        # if '/' in name:
            # name = name.replace('/','_')
        # self.input.save(f"frac/{name}.png")
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
    
if __name__ == "__main__":
    cfg = r"../configs/model/text_rec/model.yml"
    t = TextRecognition(cfg)
    print("--> load ok")
    # t(input)
    ver = 6
    for i in range(6):
        ver = i+1
        img_path = f"/home/xuan/Project/OCR/demo/rgb/res_{ver}.png"
        img = cv2.imread(img_path)

        bb_file = f"/home/xuan/Project/OCR/demo/text_1/{ver}/res.json"
        with open(bb_file, 'r') as f:
            res = json.load(f)

        f.close()

        bboxes = res["bboxes"]
        ress = {"bboxes":bboxes, "text":[]}
        
        for box in bboxes:

            inp = crop_img(img, box)

            output = t(inp)
            ress["text"].append(output)
            # print(f"Check {output}")
        
        print(ress)
        # ress["ratio"] = 1.0
        with open(bb_file, 'w', encoding="utf-8") as f:
            json.dump(ress, f, ensure_ascii=False)


    # cv2.imwrite(os.path.join(os.path.dirname(img_path), "res.png"), output["img"])