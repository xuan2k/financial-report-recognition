import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import cv2
import base64
from networks.table_tsr import TableStructureRecognition
import numpy as np
from PIL import Image
import io


from fastapi import FastAPI, File, UploadFile, Query, Depends, Body, Form, HTTPException

app = FastAPI()
config = r"../configs/model/table_tsr/model.yml"
tabdet = TableStructureRecognition(config)


@app.post("/api/table_tsr")
async def root(img: bytes = File(...)):
    # img = file.file.read()
    img = base64.b64decode(img)
    # result ={}
    new_img = Image.open(io.BytesIO(img))

    img_np = np.array(new_img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # print(img_np)
    # print(img)
    # buff1 = np.frombuffer(base64.b64decode(img),np.uint8)
    # img_np1 = cv2.imdecode(buff1, cv2.IMREAD_COLOR)
    result = tabdet(img_np)

    print(result)
    return {"bbox" : [int(x) for x in result["bbox"]]}