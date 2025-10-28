import base64
import os
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageDraw

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
        self.predictor = YOLO(os.path.join(model_dir, serialized_file))
        self.initialized = True

    def handle(self, data, context):
        img, conf = self.preprocess(data)
        result = self.inference(img, conf)
        return self.postprocess(result)

    def inference(self, model_input, conf):
        self.image = model_input
        predictions = self.predictor.predict(model_input, conf = conf, save =False, device=self.device)
        return predictions

    def preprocess(self, data):
        data = data[0].get("body") or data[0].get("data")
        img = data.get("img")
        conf = data.get("conf", 0.5)
        split = img.strip().split(',')
        if len(split) < 2:
            raise PredictionException("Invalid image", 513)
        img = Image.open(BytesIO(base64.b64decode(split[1]))).convert("RGB")

        return img, conf
    
    def merge_horizontally_aligned_boxes(self, bboxes_xywh, y_thresh=10, x_gap_thresh=30):
        """
        Merge only horizontally aligned and close boxes into one line box.
        No vertical merging is performed.
        
        Args:
            bboxes_xywh (Tensor): Tensor [N, 4] in xywh format.
            y_thresh (float): max vertical distance between centers to be considered in same line.
            x_gap_thresh (float): max horizontal gap between boxes to merge.
        
        Returns:
            merged_boxes_xywh (Tensor): merged boxes in xywh format.
        """
        if len(bboxes_xywh) == 0:
            return torch.tensor([], dtype=bboxes_xywh.dtype, device=bboxes_xywh.device)

        # Convert to numpy for processing
        boxes = bboxes_xywh.cpu().numpy()
        
        # Calculate centers and corners
        x_centers = boxes[:, 0]
        y_centers = boxes[:, 1]
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        
        x1 = x_centers - widths / 2
        x2 = x_centers + widths / 2
        y1 = y_centers - heights / 2
        y2 = y_centers + heights / 2

        # Sort boxes by y_center to process line by line
        sort_idx = np.argsort(y_centers)
        x1 = x1[sort_idx]
        x2 = x2[sort_idx]
        y1 = y1[sort_idx]
        y2 = y2[sort_idx]
        y_centers = y_centers[sort_idx]

        merged = []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue

            # Start new line
            current_line = [i]
            used.add(i)
            current_y = y_centers[i]

            # Find all boxes in same line
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                    
                # Check if box is in same line using center distance
                if abs(y_centers[j] - current_y) <= y_thresh:
                    # Check horizontal gap with closest box in current line
                    gaps = [x1[j] - x2[k] for k in current_line]
                    min_gap = min(gaps)
                    
                    if min_gap <= x_gap_thresh:
                        current_line.append(j)
                        used.add(j)

            # Create merged box for the line
            x_min = min(x1[k] for k in current_line)
            x_max = max(x2[k] for k in current_line)
            # Use the average y coordinates of boxes in line
            y_min = np.mean([y1[k] for k in current_line])
            y_max = np.mean([y2[k] for k in current_line])

            merged.append([x_min, y_min, x_max, y_max])

        # Convert merged xyxy -> xywh
        merged_xywh = []
        for x1, y1, x2, y2 in merged:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            merged_xywh.append([cx, cy, w, h])

        return torch.tensor(merged_xywh, dtype=bboxes_xywh.dtype, device=bboxes_xywh.device)


    def postprocess(self, predictions):
        # Get xywh boxes and ensure they're on the right device
        xywh = predictions[0].boxes.xywh.to(self.device)
        if len(xywh.shape) == 3:
            xywh = xywh[0]

        # Skip merging if no boxes detected
        if len(xywh) == 0:
            return [[{"horizontal_list": []}]]
        
        # Debug image save
        # img_copy = self.image.copy()
        # draw = ImageDraw.Draw(img_copy)
        # xywh_np = xywh.cpu().numpy()
        # for idx, box in enumerate(xywh_np):
        #     x_center, y_center, w, h = box
        #     xmin = int(x_center - w / 2)
        #     ymin = int(y_center - h / 2)
        #     xmax = int(x_center + w / 2)
        #     ymax = int(y_center + h / 2)
        #     draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="blue", width=1)
        # # img_copy.save("debug_before_merge.png")
        
        # Merge horizontally aligned boxes
        grouped_bboxes = self.merge_horizontally_aligned_boxes(xywh.cpu())

        # Move to CPU and convert to numpy for final processing
        grouped_bboxes = grouped_bboxes.cpu().numpy()

        print(grouped_bboxes)

        # Convert grouped xywh to xmin, xmax, ymin, ymax
        x_center = grouped_bboxes[:, 0]
        y_center = grouped_bboxes[:, 1]
        w = grouped_bboxes[:, 2]
        h = grouped_bboxes[:, 3]
        
        # Calculate coordinates and convert to integers
        xmin = np.floor(x_center - w / 2).astype(np.int32)
        xmax = np.ceil(x_center + w / 2).astype(np.int32)
        ymin = np.ceil(y_center - h / 2).astype(np.int32)
        ymax = np.floor(y_center + h / 2).astype(np.int32)
        
        # Stack in the required format
        bboxes = np.stack([xmin, xmax, ymin, ymax], axis=1)

        # img_copy = self.image.copy()
        # draw = ImageDraw.Draw(img_copy)
        # for box in bboxes:
        #     xmin, xmax, ymin, ymax = box
        #     draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)
        # img_copy.save("debug_after_merge.png")
        
        return [[{"horizontal_list": bboxes.tolist()}]]
