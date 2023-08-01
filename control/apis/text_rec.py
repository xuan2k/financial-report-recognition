import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import cv2
import base64
from networks.text_rec import TextRecognition
import numpy as np
from PIL import Image
import io
from typing import List

from fastapi import FastAPI, File, UploadFile, Query, Depends, Body, Form, HTTPException

app = FastAPI()
config = r"../configs/model/text_rec/model.yml"
textrec = TextRecognition(config)


@app.post("/api/text_det")
async def root( bbox: List[List[int]], img: bytes = File(...) ):
    # img = file.file.read()
    # # buff = base64.b64decode(img)
    new_img = Image.open(io.BytesIO(img))

    img_np = np.array(new_img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # print(img_np)
    # print(img)
    # buff1 = np.frombuffer(base64.b64decode(img),np.uint8)
    # img_np1 = cv2.imdecode(buff1, cv2.IMREAD_COLOR)
    print(bbox)
    result = textrec(img_np)

    print(result)
    # cv2.imwrite("test.png", img_np)
    # buff1 = np.frombuffer(base64.b64decode(img),np.uint8)
    # print(buff1)
    # img_np1 = cv2.imdecode(buff1, cv2.IMREAD_COLOR)
    # print(img_np1)
    return result