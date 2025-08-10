import base64
import os
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException

from ultralytics import YOLO


class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.initialized = False
        self.manifest = None

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.predictor = YOLO(model_dir, serialized_file)
        self.initialized = True

    def inference(self, model_input):
        predictions = self.predictor.predict(model_input, save =False)
        return predictions

    def preprocess(self, data):
        data = data[0].get("body") or data[0].get("data")
        img = data.get("img")
        split = img.strip().split(',')
        if len(split) < 2:
            raise PredictionException("Invalid image", 513)
        img = Image.open(BytesIO(base64.b64decode(split[1]))).convert("RGB")

        return img

    def postprocess(self, predictions):

        all_polygons = predictions[0].bboxes

        for idx, det_pred in enumerate(
                predictions):
            polygons = [p.polygon for p in det_pred.bboxes]
            all_polygons.extend(polygons)
        # Convert polygons to a NumPy array
        polygons_np = np.array(all_polygons)

        # Calculate bounding boxes using NumPy
        x_min = np.min(polygons_np[:, :, 0], axis=1)
        y_min = np.min(polygons_np[:, :, 1], axis=1)
        x_max = np.max(polygons_np[:, :, 0], axis=1)
        y_max = np.max(polygons_np[:, :, 1], axis=1)

        # Combine results into bounding boxes
        bboxes = np.stack([x_min, x_max, y_min, y_max], axis=1)
        return [[{"horizontal_list": bboxes.tolist()}]]