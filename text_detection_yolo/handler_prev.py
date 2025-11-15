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
    
    def merge_horizontally_aligned_boxes(self, bboxes_xywh, y_thresh_pct=0.3, horizontal_overlap_thresh=0.05, max_gap_pct=0.5):
        """
        Merge horizontally adjacent or overlapping boxes that are on the same line.
        Each line of text should remain as a separate box.
        Vertical merging is strictly prohibited.
        
        Args:
            bboxes_xywh (Tensor): Tensor [N, 4] in xywh format.
            y_thresh_pct (float): max vertical distance between centers as percentage of average height (e.g., 0.3 = 30%).
            horizontal_overlap_thresh (float): minimum horizontal overlap ratio to merge (0-1).
            max_gap_pct (float): maximum horizontal gap as percentage of average width (e.g., 0.5 = 50%).
        
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
        indices = np.arange(len(boxes))
        indices = indices[sort_idx]
        
        merged = []
        used = set()

        def calculate_horizontal_overlap(box1_idx, box2_idx):
            """Calculate horizontal overlap ratio between two boxes"""
            x1_start, x1_end = x1[box1_idx], x2[box1_idx]
            x2_start, x2_end = x1[box2_idx], x2[box2_idx]
            
            # Calculate overlap
            overlap_start = max(x1_start, x2_start)
            overlap_end = min(x1_end, x2_end)
            overlap = max(0, overlap_end - overlap_start)
            
            # Calculate overlap ratio relative to smaller box width
            width1 = x1_end - x1_start
            width2 = x2_end - x2_start
            min_width = min(width1, width2)
            
            return overlap / min_width if min_width > 0 else 0

        def calculate_horizontal_gap(box1_idx, box2_idx):
            """Calculate horizontal gap between two boxes"""
            x1_start, x1_end = x1[box1_idx], x2[box1_idx]
            x2_start, x2_end = x1[box2_idx], x2[box2_idx]
            
            # Calculate gap (negative if overlapping)
            if x1_end < x2_start:  # box1 is to the left of box2
                return x2_start - x1_end
            elif x2_end < x1_start:  # box2 is to the left of box1
                return x1_start - x2_end
            else:  # boxes overlap
                return 0

        def boxes_on_same_line(idx1, idx2):
            """Check if two boxes are on the same horizontal line"""
            # Calculate y_threshold based on average height of the two boxes
            avg_height = (heights[idx1] + heights[idx2]) / 2
            y_thresh_pixels = y_thresh_pct * avg_height
            
            # Check y-center alignment
            y_diff = abs(y_centers[idx1] - y_centers[idx2])
            if y_diff > y_thresh_pixels:
                return False
            
            # Check vertical overlap (boxes should overlap vertically to be on same line)
            y1_start, y1_end = y1[idx1], y2[idx1]
            y2_start, y2_end = y1[idx2], y2[idx2]
            
            vertical_overlap_start = max(y1_start, y2_start)
            vertical_overlap_end = min(y1_end, y2_end)
            vertical_overlap = max(0, vertical_overlap_end - vertical_overlap_start)
            
            # Require at least 30% vertical overlap
            min_height = min(heights[idx1], heights[idx2])
            if vertical_overlap < 0.3 * min_height:
                return False
            
            return True

        def should_merge_boxes(box1_idx, box2_idx):
            """Check if two boxes should be merged based on overlap or proximity"""
            overlap_ratio = calculate_horizontal_overlap(box1_idx, box2_idx)
            
            # Merge if there's any overlap
            if overlap_ratio >= horizontal_overlap_thresh:
                return True
            
            # Calculate max_gap based on average width of the two boxes
            avg_width = (widths[box1_idx] + widths[box2_idx]) / 2
            max_gap_pixels = max_gap_pct * avg_width
            
            # Merge if gap is small enough
            gap = calculate_horizontal_gap(box1_idx, box2_idx)
            if gap <= max_gap_pixels:
                return True
            
            return False

        # Process each box
        for i in range(len(boxes)):
            idx_i = indices[i]
            if idx_i in used:
                continue

            # Start new line group
            current_line = [idx_i]
            used.add(idx_i)

            # Iteratively expand the line by finding boxes that should merge
            changed = True
            while changed:
                changed = False
                candidates = []
                
                # Find all unused boxes that are on the same line as any box in current_line
                for j in range(len(boxes)):
                    idx_j = indices[j]
                    if idx_j in used:
                        continue
                    
                    # Check if on same line with any box in current line
                    for existing_idx in current_line:
                        if boxes_on_same_line(existing_idx, idx_j):
                            candidates.append((x1[idx_j], idx_j))
                            break
                
                # Sort candidates by x-coordinate
                candidates.sort()
                
                # Try to add candidates that should merge
                for _, idx_j in candidates:
                    if idx_j in used:
                        continue
                    
                    # Check if this box should merge with ANY box in current line
                    should_merge = False
                    for existing_idx in current_line:
                        if should_merge_boxes(existing_idx, idx_j):
                            should_merge = True
                            break
                    
                    if should_merge:
                        current_line.append(idx_j)
                        used.add(idx_j)
                        changed = True  # Keep iterating to catch boxes that connect through this new box

            # Create merged box for the line - preserve original height
            x_min = min(x1[k] for k in current_line)
            x_max = max(x2[k] for k in current_line)
            
            # For y-coordinates, use the min/max to preserve line height
            y_min_val = min(y1[k] for k in current_line)
            y_max_val = max(y2[k] for k in current_line)

            merged.append([x_min, y_min_val, x_max, y_max_val])

        # Convert merged xyxy -> xywh
        merged_xywh = []
        for x1_m, y1_m, x2_m, y2_m in merged:
            cx = (x1_m + x2_m) / 2
            cy = (y1_m + y2_m) / 2
            w = x2_m - x1_m
            h = y2_m - y1_m
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
        
        # # Debug image save
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
        # img_copy.save("debug_before_merge.png")
        
        # Merge horizontally aligned boxes
        grouped_bboxes = self.merge_horizontally_aligned_boxes(xywh.cpu())

        # Move to CPU and convert to numpy for final processing
        grouped_bboxes = grouped_bboxes.cpu().numpy()

        # print(grouped_bboxes)

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
        
        # Sort bboxes: first by y-coordinate (top to bottom), then by x-coordinate (left to right) within each row
        # Sort by y first
        bboxes = bboxes[np.argsort(bboxes[:, 2])]
        
        # Group by y-coordinate and sort each group by x-coordinate
        y_thresh = 5  # threshold to consider boxes in the same row
        sorted_bboxes = []
        current_row = []
        current_y = None
        
        for box in bboxes:
            if current_y is None:
                current_y = box[2]
                current_row.append(box)
            elif abs(box[2] - current_y) <= y_thresh:
                current_row.append(box)
            else:
                # Sort current row by x-coordinate and add to result
                current_row = sorted(current_row, key=lambda b: b[0])
                sorted_bboxes.extend(current_row)
                current_row = [box]
                current_y = box[2]
        
        # Don't forget the last row
        if current_row:
            current_row = sorted(current_row, key=lambda b: b[0])
            sorted_bboxes.extend(current_row)
        
        bboxes = np.array(sorted_bboxes)

        # img_copy = self.image.copy()
        # draw = ImageDraw.Draw(img_copy)
        # for seq_num, box in enumerate(bboxes):
        #     xmin, xmax, ymin, ymax = box
        #     draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)
        #     # Draw sequence number at the top-left of the box
        #     draw.text((xmin + 5, ymin + 5), str(seq_num), fill="yellow")
        # img_copy.save("debug_after_merge.png")
        
        return [[{"horizontal_list": bboxes.tolist()}]]
