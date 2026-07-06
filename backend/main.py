from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import shutil

app = FastAPI()

INFERENCE_IMAGE_SIZE = 1280

@app.post('/prediction/')
def make_prediction(model_version: str = Form(), files: List[UploadFile] = File(...)):
    try:
        if model_version == 'Standard':
            model = YOLO('../training_results/yolo26n_1280/best_weights.pt')
        else:
            model = YOLO('../training_results/yolo26s_1280/best_weights.pt')
    except Exception as e:
        return f'Model didn\'t load because of this error:\n{e}'
    
    make_tmp_dir()
    results = dict()

    for file in files:
        image = Image.open(file.filename)
        result = model(image, imgsz=INFERENCE_IMAGE_SIZE)[0]
        image_path = os.path.join(tmp_dir, file.filename)
        result.save(filename=image_path)
        results[image_path] = result.boxes.data.tolist()

    return {'results': results}

def make_tmp_dir():
    global tmp_dir
    tmp_dir = tempfile.mkdtemp()

@app.get('/rm_tmp_dir/')
def rm_tmp_dir():
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        return {
            'is_tmp_dir_deleted': False,
            'details': e
        }
    
    return {'is_tmp_dir_deleted': True}