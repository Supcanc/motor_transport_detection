from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

INFERENCE_IMAGE_SIZE = 1280

@app.post('/prediction/')
async def make_prediction(model_version: str = Form(), files: List[UploadFile] = File(...)):
    try:
        if model_version == 'Standard':
            model = YOLO('../training_results/yolo26n_1280/best_weights.pt')
        else:
            model = YOLO('../training_results/yolo26s_1280/best_weights.pt')
    except Exception as e:
        return f'Model didn\'t load because of this error:\n{e}'
    
    preds = dict()

    for file in files:
        image = Image.open(file.file)
        results = model(image, imgsz=INFERENCE_IMAGE_SIZE)
        preds[file.filename] = [result.boxes for result in results]

    return preds