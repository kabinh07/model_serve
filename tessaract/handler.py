import io
from PIL import Image
import pytesseract
import torch
import base64

from ts.torch_handler.base_handler import BaseHandler

class TesseractOCRHandler(BaseHandler):
    """
    TorchServe handler for Tesseract OCR.
    This handler takes an image (as bytes or base64 encoded string) and
    returns the extracted text using Tesseract.
    """
    def __init__(self):
        super().__init__()
        self.initialized = False
        # Optional: Set Tesseract command path if not in system PATH
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # Example for Linux

    def initialize(self, context):
        """
        Initializes the handler.
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # You might want to load any specific Tesseract language data or configurations here
        # For this simple example, we assume Tesseract is configured and available.

        self.initialized = True
        print(f"TesseractOCRHandler initialized. Model directory: {model_dir}")

    def preprocess(self, data):
        """
        Preprocesses the input data.
        Expected input: a list of dictionaries, where each dictionary
        contains 'body' with image bytes or a base64 encoded string.
        """
        images = []
        for row in data:
            image_data = row.get("data") or row.get("body")
            if isinstance(image_data, str):
                # Assume base64 encoded string
                image_data = base64.b64decode(image_data)
            
            # Read image using PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Optional: Add image preprocessing steps here (e.g., grayscale, denoising, deskewing)
            # For example:
            # image = image.convert("L") # Convert to grayscale

            images.append(image)
        return images

    def inference(self, data):
        """
        Performs OCR on the preprocessed images.
        'data' here is the list of PIL Image objects from preprocess.
        """
        results = []
        for image in data:
            # Perform OCR using pytesseract
            # You can customize OCR configuration here, e.g., language, page segmentation mode (PSM)
            # Example: text = pytesseract.image_to_string(image, lang='eng', config='--psm 3')
            text = pytesseract.image_to_string(image)
            results.append(text)
        return results

    def postprocess(self, data):
        """
        Postprocesses the inference results.
        'data' here is the list of extracted texts from inference.
        """
        # In this simple case, the postprocessing is just returning the text.
        # You could format it differently, add metadata, etc.
        return data