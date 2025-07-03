from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException
from PIL import Image
from io import BytesIO
import base64
import tempfile
import json
import os

class QwenHandler(BaseHandler):
    def __init__(self):
        super(QwenHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties

        # Load the model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("qwen2_5_vl_7b_instruct_q4")
        self.model = AutoModelForImageTextToText.from_pretrained("qwen2_5_vl_7b_instruct_q4")
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, requests):
        preprocessed_data = []
        for req in requests:
            data = req.get("body") or req.get("data")
            image_bytes = None
            prompt = ""
            sys_prompt = "You are a helpful assistant."

            if isinstance(data, dict):
                image_bytes = data.get("image")
                prompt = data.get("prompt", "")
                sys_prompt = data.get("sys_prompt", sys_prompt)
            elif isinstance(data, (bytes, bytearray)):
                try:
                    parsed_data = json.loads(data.decode('utf-8'))
                    image_bytes = parsed_data.get("image_bytes")
                    prompt = parsed_data.get("prompt", "")
                    sys_prompt = parsed_data.get("sys_prompt", sys_prompt)
                    if image_bytes:
                        image_bytes = base64.b64decode(image_bytes)
                except json.JSONDecodeError:
                    image_bytes = data
                    prompt = req.get("headers", {}).get("X-Prompt", "")
                    sys_prompt = req.get("headers", {}).get("X-Sys-Prompt", sys_prompt)
            else:
                raise ValueError("Unsupported input data format in preprocess.")

            image = None
            if image_bytes:
                try:
                    if isinstance(image_bytes, bytes):
                        image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    elif isinstance(image_bytes, str):
                        if ',' in image_bytes:
                            image_bytes = image_bytes.split(',')[1]
                        image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert("RGB")
                except Exception as e:
                    raise RuntimeError(f"Error opening image: {e}")

            if image is None:
                raise ValueError("Image data is missing or invalid.")

            # Create a temporary file that will NOT be automatically deleted
            # when the temp_file object is closed. We'll manage deletion manually.
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            image.save(temp_file.name) # Save the PIL image to the temporary file
            temp_file.close() # Close the file handle (important for Windows where files can't be deleted while open)
            
            # Store the path to the temporary file and the cleanup function
            # This is a common pattern for handlers to track resources.
            preprocessed_data.append({
                "image_path": temp_file.name,
                "prompt": prompt,
                "sys_prompt": sys_prompt,
                "cleanup_path": temp_file.name # Store path for later deletion
            })
        return preprocessed_data
    
    def _run_inference(self, image_path, prompt, sys_prompt, max_new_tokens=1024):
        # Assuming processor and model are loaded in self.initialize
        image = Image.open(image_path)
        image_local_path = "file://" + image_path
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"image": image_local_path},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu') # Use self.map_location here if you set it

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]
    
    def handle(self, data, context):
        """
        Main entry point for the handler.
        """
        results = []
        cleanup_paths = [] # To store paths for cleanup

        try:
            preprocessed_requests = self.preprocess(data)
            for req_data in preprocessed_requests:
                cleanup_paths.append(req_data["cleanup_path"]) # Add to cleanup list

                output_text = self._run_inference(
                    image_path=req_data["image_path"],
                    prompt=req_data["prompt"],
                    sys_prompt=req_data["sys_prompt"]
                )
                results.append({"output": output_text})
        finally:
            # Ensure cleanup happens even if an error occurs during inference
            for path in cleanup_paths:
                try:
                    os.remove(path)
                    print(f"Cleaned up temporary file: {path}")
                except OSError as e:
                    print(f"Error cleaning up file {path}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results
