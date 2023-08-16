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

from networks.text_rec import TextRecognition
from utils.processing import vis_img, crop_img

class TextRecognitionTest(TextRecognition):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def processing(self, x):
        if self.input is not None:
            if len(x[0][0]) == 0 or x[0][0][0][4] < 0.97:
                return None

            table_sq = m.sqrt(((bbox[1]-bbox[3])**2)*(bbox[0]-bbox[2])**2)
            load_img = self.input
            h,w,_ = load_img.shape

            if table_sq < 0.1*h*w:
                return None
            bbox = list(x[0][0][0][0:4].astype(int))
            
            padding = 80
            xmin = max(bbox[0] - padding, 0)
            xmax = bbox[2] + padding
            ymin = bbox[1] + int(0.25*padding)
            ymax = bbox[3] + int(0.5*padding)
            bbox = [xmin, ymin, xmax, ymax]

            new_img = crop_img(self.input, bbox=bbox)
            # new_img = Image.fromarray(new_img).convert("RGB")
            return new_img
        else:
            raise TypeError("The input is in NoneType.")
        
if __name__ == "__main__":
        code = f"/workspace/warehouse/info/vn30.txt"
        f = open(code, 'r')
        index = []
        for x in f:
            index.append(x.strip('\n'))

        cfg = r"/workspace/source/configs/model/table_det/model.yml"

        trt = TextRecognitionTest(cfg)
        
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
                        res = trt(load_img)
                        img_save_path = os.path.join(save_path, img)
                        if res is not None:
                            cv2.imwrite(img_save_path, res)

            print("load OK")
        except KeyboardInterrupt:
            del trt
            print("Stop by keyboard interupt!")