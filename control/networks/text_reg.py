import os
import numpy as np
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


def viet_ocr(detector, img, count = 0):
    img = Image.fromarray(img)
    # img.save(f"./frag/{count}.png")
    s = detector.predict(img)
    return s

def recognize(image, save_path, boxes):
    txts = []
    viet = np.array(image)

    if not os.path.exists(save_path):
        os.makedirs(save_path)



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
    # torch.save(detector.model.state_dict(), "ocr.pth")
    f = open(os.path.join(save_path, "res.txt"), 'w')
    for box in boxes:
        # print(box)
        x = [int(i[0]) for i in box]
        y = [int(i[1]) for i in box]
        txt = viet_ocr(detector, viet[min(y):max(y), min(x):max(x)])
        # print(f'-> Detected: {txt}')
        # count += 1
        txts.append(txt)
        f.write(f'{txt}\n')

    # print(txts)
    f.close()
    return txts

if __name__=="__main__":
    recognizer = r'/home/xuan/Project/OCR/code/baseline/control/configs/model/textrec/model.yml'
    config = Cfg.load_config_from_file(recognizer)
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'

    detector = Predictor(config)
    print("check")
    