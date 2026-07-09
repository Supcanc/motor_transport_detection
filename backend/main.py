from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import shutil

INFERENCE_IMAGE_SIZE = 1280

def get_prediction(model_version, files):
    try:
        if model_version == 'Standard':
            model = YOLO('../training_results/yolo26n_1280/best_weights.pt')
        else:
            model = YOLO('../training_results/yolo26s_1280/best_weights.pt')
    except Exception as e:
        return {
            'status': 'failed',
            'details': f'Model didn\'t load because of this error:\n{e}'
        }
    
    mk_tmp_dir_status = make_tmp_dir()
    if mk_tmp_dir_status['status'] == 'failed':
        return {
            'status': 'failed',
            'details': mk_tmp_dir_status['details']
        }

    results = dict()

    try:
        for file in files:
            image = Image.open(file['filename'])
            result = model(image, imgsz=INFERENCE_IMAGE_SIZE)[0]
            image_path = os.path.join(tmp_dir, os.path.basename(file['filename']))
            result.save(filename=image_path)
            results[image_path] = result.boxes.data.tolist()
    except Exception as e:
        return {
            'status': 'failed',
            'details': f'An error occurred during making prediction. Details:\n{e}'
        }

    return {
        'status': 'ok',
        'results': results
    }

def make_tmp_dir():
    global tmp_dir

    try:
        tmp_dir = tempfile.mkdtemp()
    except Exception as e:
        return {
            'status': 'failed',
            'details': f'Didn\'t make temporal directory, because of this error:\n{e}'
        }
    
    return {'status': 'ok'}

def rm_tmp_dir():
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        return {
            'status': 'failed',
            'details': f'Didn\'t remove temporal directory because of this error:\n{e}'
        }
    
    return {'status': 'ok'}