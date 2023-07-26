from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
from tqdm import tqdm
import argparse
import os
# import torch

def parse_arg():

    parser = argparse.ArgumentParser("Parser for OCR pipeline")
    
    parser.add_argument("--image_dir", type=str, 
                        default='/home/xuan/Project/OCR/code/git_code/PaddleOCR/implement/data')
    
    parser.add_argument("--save_dir", type=str, 
                        default='/home/xuan/Project/OCR/code/git_code/PaddleOCR/implement/result')
    
    parser.add_argument("--binarize", action='store_true')

    parser.add_argument("--rec", action='store_true')

    parser.add_argument("--visualize", action='store_true')



    return parser.parse_args()

def create_font(txt, sz, font_path="./doc/fonts/simfang.ttf"):
    font_size = int(sz[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    length = font.getsize(txt)[0]
    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font

def draw_box_txt_fine(img_size, box, txt, font_path="./doc/fonts/simfang.ttf"):
    box_height = int(
        math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2))
    box_width = int(
        math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2))

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new('RGB', (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new('RGB', (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255))
    return img_right_text

def draw_ocr_box_txt(image,
                     boxes,
                     txts=None,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        box = [tuple(coor) for coor in box]
        # print(box)
        # print(color)
        draw_left.polygon(box, fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)

def binarize(image, thresh = 0.7):
    thresh *= 255
    image = image.point( lambda p: 255 if p > thresh else 0 )
    return image


def run(img_path, save_path, ocr):

    # img_path = '/home/xuan/Project/OCR/code/git_code/PaddleOCR/implement/data/4.png'
    result = ocr.ocr(img_path, cls=False, rec=False)
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
    #         print(line)

    # show result
    # from PIL import Image
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line for line in result]
    font_path='/home/xuan/Downloads/clear-sans/ClearSans-Regular.ttf'
    # font_path='/home/xuan/Project/OCR/code/git_code/PaddleOCR/doc/fonts/simfang.ttf'
    # im_show = draw_ocr(image, boxes, None, None, font_path= font_path)
    # im_show = Image.fromarray(im_show)
    # im_show.save('./result/result.jpg')
    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]

    im2 = draw_ocr_box_txt(image, boxes, txts, None, font_path=font_path, drop_score=0)
    im2 = Image.fromarray(im2)


    im2.save(os.path.join(save_path,'result_visualize.jpg'))

if __name__ == '__main__':
    args = parse_arg()
    ocr = PaddleOCR(use_angle_cls=False, lang="en", type="structure", recovery=True) # need to run only once to download and load model into memory
    
    if os.path.isdir(args.image_dir):
        t = tqdm(os.listdir(args.image_dir))
    else:
        t = tqdm([os.path.basename(args.image_dir)])
        args.image_dir = os.path.dirname(args.image_dir)


    for image in t:
        img_path = os.path.join(args.image_dir, image)
        save_path = os.path.join(args.save_dir, image.split('.')[0])

        run(img_path, save_path, ocr)
        t.set_description(f'-> Done, save result of {image} to {save_path}')

