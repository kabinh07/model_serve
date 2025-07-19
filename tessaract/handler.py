import io
from PIL import Image
import pytesseract
import torch
import base64

from ts.torch_handler.base_handler import BaseHandler

class TesseractOCRHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' 

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.initialized = True

    def preprocess(self, data):
        data = data[0].get("data") or data[0].get("body")
        base_64_images = data.get("images")
        images = []
        for image_data in base_64_images:
            if len(image_data.split(',')) > 1:
                image_data = image_data.split(',')[-1]
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
        return images

    def inference(self, data):
        results = []
        for image in data:
            text = pytesseract.image_to_string(image, lang="ben+eng")
            results.append(text)
        return  [{"results": results}]

    def postprocess(self, data):
        return data
