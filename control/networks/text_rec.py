import os
import numpy as np
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import json


def viet_ocr(detector, img, count = 0):
    img = Image.fromarray(img)
    # img.save(f"./frag/{count}.png")
    s = detector.predict(img)
    return s

def recognize(detector, image, result, save_path = None):
    boxes = result["bboxes"]
    txts = []
    viet = np.array(image)
    # count = 0
    # torch.save(detector.model.state_dict(), "ocr.pth")
    for box in boxes:
        # print(box)
        # x = [int(i[0]) for i in box]
        # y = [int(i[1]) for i in box]
        txt = viet_ocr(detector, viet[box[1]:box[3], box[0]:box[2]])
        # txt = viet_ocr(detector, viet[box[1]:box[3], box[0]:box[2]], count)
        # print(f'-> Detected: {txt}')
        # count += 1
        txts.append(txt)
    if save_path:

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if os.path.isdir(save_path):
            f = open(os.path.join(save_path, "res.json"), 'w', encoding="utf-8")
        else:
            f = open(save_path, 'w', encoding="utf-8")

        # for txt in txts:
        #     f.write(f'{txt}\n')
        result["text"] = txts
        json.dump(result, f, ensure_ascii=False)

        f.close()
    return txts

if __name__=="__main__":
    
    # transformer = r"/home/xuan/Project/OCR/code/git_code/PaddleOCR/pretrained/vgg_transformer.pth"
    recognizer = r'/home/xuan/Project/OCR/code/baseline/control/configs/model/textrec/model.yml'
    # print(recognizer[64])
    # config = Cfg.load_config_from_name('vgg_seq2seq')
    config = Cfg.load_config_from_file(recognizer)
    # config = Cfg.load_config_from_file("/tmp/vgg_seq2seq.pth")
    # config = Cfg.load_config_from_file(transformer)

    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'

    detector = Predictor(config)

    img_path = "/home/xuan/Project/OCR/sample/result/rgb/2.png"
    image = Image.open(img_path).convert("RGB")

    res_path = "/home/xuan/Project/OCR/sample/result/text/2/res.json"

    f = open(res_path, 'r', encoding="utf-8")

    result = json.load(f)

    recognize(detector, image, result, res_path)

    print("check")
    