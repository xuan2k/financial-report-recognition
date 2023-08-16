import sys
import os
import cv2
import shutil
import math as m
from PIL import Image
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.processing import vis_img, crop_img

from networks.text_det import TextDetection

class TextDetectionTest(TextDetection):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

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
            bbox.reverse()
            result["bboxes"] = bboxes
            result["vis"] = bbox
            return result
        else:
            raise TypeError("The input is in NoneType.")
        
if __name__ == "__main__":
        code = f"/workspace/warehouse/info/vn30.txt"
        f = open(code, 'r')
        index = []
        for x in f:
            index.append(x.strip('\n'))

        cfg = r"/workspace/source/configs/model/table_det/model.yml"

        tdt = TextDetectionTest(cfg)
        
        try:

            for idx in index[1:]:
                root_directory = f"/workspace/warehouse/data/{idx}/png"
                terminal_width = shutil.get_terminal_size().columns 
                print(f"-> Extracting table for {idx}")
                print('-' * terminal_width)
                for fol in tqdm(os.listdir(root_directory)):
                    img_fol = os.path.join(root_directory, fol)
                    save_path = img_fol.replace("data", "result/table")
                    
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    for img in os.listdir(img_fol):
                        img_path = os.path.join(img_fol, img)
                        load_img = Image.open(img_path).convert('RGB')
                        res = tdt(load_img)
                        img_save_path = os.path.join(save_path, img)
                        if res is not None:
                            cv2.imwrite(img_save_path, res)

            print("load OK")
        except KeyboardInterrupt:
            del tdt
            print("Stop by keyboard interupt!")