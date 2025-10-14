import cv2
import numpy as np
from doc_aligner import Inference
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException
import os
import base64

class DocAlignerHandler(BaseHandler):
    def __init__(self):
        super(DocAlignerHandler, self).__init__()
        self.model = None
    
    def initialize(self, ctx):
        """Load the DocAligner model."""
        properties = ctx.system_properties
        model_dir = properties.get('model_dir')
        model_path = os.path.join(model_dir, "fastvit_sa24_h_e_bifpn_256_fp32.onnx")
        if not os.path.exists(model_path):
            raise PredictionException(f"Model file not found at {model_path}", 500)
        
        self.model = Inference()
    
    def preprocess(self, data):
        data = data[0].get("body") or data[0].get("data")
        img = data.get("img")

        if not img:
            raise PredictionException("Invalid data provided for inference", 513)
        
        split = img.strip().split(',')
        if len(split) > 2:
            raise PredictionException("Invalid image", 513)
        if len(split) == 2:
            split = split[1]
        img_data = base64.b64decode(split)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise PredictionException("Failed to decode image", 513)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    
    def inference(self, img):
        """Run inference on the preprocessed image."""
        if self.model is None:
            raise PredictionException("Model is not initialized", 500)
        
        try:
            results = self.model(img)
            return self.postprocess(
                {
                    "aligned_corners": results.tolist()
                }
            )
        except Exception as e:
            raise PredictionException(f"Inference failed: {str(e)}", 500)
    
    def postprocess(self, inference_output):
        return [inference_output]