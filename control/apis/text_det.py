import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import cv2
import base64
from networks.text_det import TextDetection
import numpy as np
from PIL import Image
import io


from fastapi import FastAPI, File, UploadFile, Query, Depends, Body, Form, HTTPException

app = FastAPI()
config = r"../configs/model/text_det/model.yml"
tabdet = TextDetection(config)


@app.post("/api/text_det")
async def root(img: bytes = File(...)):
    # img = file.file.read()
    # # buff = base64.b64decode(img)
    new_img = Image.open(io.BytesIO(img))

    img_np = np.array(new_img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # print(img_np)
    # print(img)
    # buff1 = np.frombuffer(base64.b64decode(img),np.uint8)
    # img_np1 = cv2.imdecode(buff1, cv2.IMREAD_COLOR)
    result = tabdet(img_np)

    print(result)
    # cv2.imwrite("test.png", img_np)
    # buff1 = np.frombuffer(base64.b64decode(img),np.uint8)
    # print(buff1)
    # img_np1 = cv2.imdecode(buff1, cv2.IMREAD_COLOR)
    # print(img_np1)
    return result