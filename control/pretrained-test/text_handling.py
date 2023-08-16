import os
import sys
import cv2
import json
import shutil
from tqdm import tqdm
from PIL import Image
from text_rec import TextRecognitionTest
from text_det import TextDetectionTest

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.processing import vis_img, crop_img, draw_ocr_box_txt


code = f"/home/xuan/Project/OCR/info/vn30.txt"
f = open(code, 'r')
index = []
for x in f:
    index.append(x.strip('\n'))

cfg = r"../configs/model/text_det/model.yml"
# cfg = ""
t1 = TextDetectionTest(cfg)
t2 = TextRecognitionTest(cfg)
isVisulaize = False

def isVerticalBox(bbox):
    if abs(bbox[0]-bbox[2]) < abs(bbox[1] - bbox[3]):
        return True

def validBbox(bbox):
    n = len(bbox)
    count = 0
    for box in bbox:
        if isVerticalBox(box):
            count += 1
    if count > 0.6*n:
        return False
    else:
        return True

for idx in index[21:29]:
    directory = f"/home/xuan/Project/OCR/result/table/{idx}/png"
    for fol in os.listdir(directory):
        png_path = os.path.join(directory, fol)
        print(f"--> Predicting for {fol} in {idx} ... ")
        for img in tqdm(os.listdir(png_path)):
            img_path = os.path.join(png_path, img)
            save_path = img_path.replace("table", "text")
            save_path = save_path.replace("/png", '')
            save_path = os.path.dirname(save_path)
            read_img = cv2.imread(img_path)
            bbox = t1(read_img)
            res= []
            if not validBbox(bbox["bboxes"]):
                continue
            for box in bbox["bboxes"]:
                inp = crop_img(read_img, box)
                s = t2(inp)
                res.append(s)
            bbox["texts"] = res


            save_ann = os.path.join(save_path,'txt')
            save_viz = os.path.join(save_path,'png')

            for path in [save_ann]:
                if not os.path.exists(path):
                    os.makedirs(path)

            if isVisulaize:
                if not os.path.exists(save_viz):
                    os.makedirs(save_viz)

                font_path='/home/xuan/Project/OCR/IDCard/font/Trixi_Pro_Regular.ttf'

                im2 = draw_ocr_box_txt(Image.fromarray(read_img), bbox["vis"], bbox["texts"], None, font_path=font_path, drop_score=0)
                im2 = Image.fromarray(im2)

                if not os.path.exists(save_viz):
                    os.makedirs(save_viz)

                im2.save(os.path.join(save_viz,img))

            save_ann = os.path.join(save_ann,f"{img.split('.')[0]}.txt")

            with open(save_ann, 'w', encoding='utf-8') as f:
                json.dump(bbox, f, ensure_ascii=False)

            

        print("--> DONE")
    print(f"--> DONE for {idx}")
    terminal_width = shutil.get_terminal_size().columns 
    print('-' * terminal_width,'\n\n')



                
