import io
from PIL import Image
import pytesseract
import torch
import base64

from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException

class TesseractOCRHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' 

    def initialize(self, context):
        self.manifest = context.manifest
        self.initialized = True

    def preprocess(self, data):
        data = data[0].get("data") or data[0].get("body")
        base_64_images = data.get("images")
        if not base_64_images:
            raise PredictionException("No images found in request")
        if not isinstance(base_64_images, list):
            base_64_images = [base_64_images]
        self.language = data.get("language", "ben+eng")
        self.output_type = data.get("output_type", "txt")
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
            try:
                if isinstance(self.output_type, str) and self.output_type == "txt":
                    output = pytesseract.image_to_string(image, lang=self.language)
                    results.append({"txt": output})
                elif isinstance(self.output_type, str) and self.output_type == "box":
                    output = pytesseract.image_to_boxes(image, lang=self.language)
                    results.append({"box": output})
                elif isinstance(self.output_type, str) and self.output_type == "data":
                    output = pytesseract.image_to_data(image, lang=self.language)
                    results.append({"data": output})
                elif isinstance(self.output_type, str) and self.output_type == "osd":
                    output = pytesseract.image_to_osd(image, lang=self.language)
                    results.append({"osd": output})
                elif isinstance(self.output_type, list):
                    output = pytesseract.run_and_get_multiple_output(image, lang=self.language, extensions=self.output_type)
                    out_dict = {}
                    for idx, item in enumerate(self.output_type):
                        out_dict[item] = output[idx]
                    results.append(out_dict)
                else:
                    results.append({"error": "Invalid output type"})
            except Exception as e:
                results.append({"error": str(e)})
        return self.postprocess(results)

    def postprocess(self, data):
        if isinstance(data, list):
            return data
        return [data]
