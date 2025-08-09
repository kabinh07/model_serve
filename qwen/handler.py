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
            image_bytes_list = []
            prompt = ""
            sys_prompt = "You are a helpful assistant."
            max_new_tokens = 1024 # Default value
            max_pixels = 1210000 # Default to None, meaning processor's default
            min_pixels = None # Default to None, meaning processor's default
            
            # Flag to indicate if images were found in this request
            has_images = False 

            if isinstance(data, dict):
                image_bytes_list = data.get("images", [])
                prompt = data.get("prompt", "")
                sys_prompt = data.get("sys_prompt", sys_prompt)
                max_new_tokens = data.get("max_new_tokens", 1024)
                # Retrieve max_pixels and min_pixels from input data
                max_pixels = data.get("max_pixels", max_pixels)
                min_pixels = data.get("min_pixels", min_pixels)
                
            elif isinstance(data, (bytes, bytearray)):
                try:
                    parsed_data = json.loads(data.decode('utf-8'))
                    image_bytes_list = parsed_data.get("images", [])
                    prompt = parsed_data.get("prompt", "")
                    sys_prompt = parsed_data.get("sys_prompt", sys_prompt)
                    max_new_tokens = parsed_data.get("max_new_tokens", 1024)
                    max_pixels = parsed_data.get("max_pixels", max_pixels)
                    min_pixels = parsed_data.get("min_pixels", min_pixels)

                    if image_bytes_list:
                        image_bytes_list = [base64.b64decode(img_b) for img_b in image_bytes_list if isinstance(img_b, str)]
                except json.JSONDecodeError:
                    # If it's raw bytes and not JSON, treat it as a single image
                    image_bytes_list = [data]
                    prompt = req.get("headers", {}).get("X-Prompt", "")
                    sys_prompt = req.get("headers", {}).get("X-Sys-Prompt", sys_prompt)
                    # Cannot get max_pixels/min_pixels from headers easily in this fallback
                    # So they will remain their default None values
            else:
                raise ValueError("Unsupported input data format in preprocess.")

            image_paths = []
            cleanup_paths = []

            # Only process images if image_bytes_list is not empty
            if image_bytes_list:
                for img_bytes in image_bytes_list:
                    image = None
                    try:
                        if isinstance(img_bytes, bytes):
                            image = Image.open(BytesIO(img_bytes)).convert("RGB")
                        elif isinstance(img_bytes, str):
                            if ',' in img_bytes:
                                img_bytes = img_bytes.split(',')[1]
                            image = Image.open(BytesIO(base64.b64decode(img_bytes))).convert("RGB")
                    except Exception as e:
                        print(f"Warning: Error opening one of the images. Skipping this image. Error: {e}")
                        continue

                    if image is not None:
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        image.save(temp_file.name)
                        temp_file.close()
                        image_paths.append(temp_file.name)
                        cleanup_paths.append(temp_file.name)
                
                if image_paths: # If at least one image was successfully processed
                    has_images = True
            
            # If no prompt is provided, we can't do anything
            if not prompt:
                raise ValueError("Prompt is missing. Please provide a 'prompt' for the model.")

            preprocessed_data.append({
                "image_paths": image_paths, # Will be empty list if no images
                "prompt": prompt,
                "sys_prompt": sys_prompt,
                "cleanup_paths": cleanup_paths,
                "has_images": has_images, # Pass this flag to _run_inference
                "max_new_tokens": max_new_tokens, # Pass max_new_tokens
                "max_pixels": max_pixels,         # Pass max_pixels
                "min_pixels": min_pixels          # Pass min_pixels
            })
        return preprocessed_data
    
    # Add max_pixels and min_pixels as parameters to _run_inference
    def _run_inference(self, image_paths, prompt, sys_prompt, has_images, max_new_tokens=1024, max_pixels=None, min_pixels=None):
        # Construct messages based on whether images are present
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": []}, # Initialize content as a list
        ]

        # Add text prompt
        messages[1]["content"].append({"type": "text", "text": prompt})

        # Add images only if has_images is True and image_paths is not empty
        images_for_processor = []
        if has_images and image_paths:
            for path in image_paths:
                try:
                    pil_image = Image.open(path).convert("RGB")
                    images_for_processor.append(pil_image)
                    messages[1]["content"].append({"image": "file://" + path})
                except Exception as e:
                    print(f"Warning: Error loading image from path {path} for inference. Skipping this image. Error: {e}")
                    continue
            
            if not images_for_processor and has_images: # If images were expected but none loaded successfully
                print("Warning: No valid images could be loaded for inference, proceeding as text-only.")
                has_images = False # Override the flag if image loading failed

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Prepare kwargs for the processor
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt"
        }
        
        if has_images:
            processor_kwargs["images"] = images_for_processor
            # Conditionally add max_pixels and min_pixels if they are not None
            if max_pixels is not None:
                processor_kwargs["max_pixels"] = max_pixels
            if min_pixels is not None:
                processor_kwargs["min_pixels"] = min_pixels
            inputs = self.processor(**processor_kwargs)
        else:
            # When no images, ensure 'images' argument is not passed or is an empty list
            inputs = self.processor(**processor_kwargs)
            
        inputs = inputs.to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]
    
    def handle(self, data, context):
        """
        Main entry point for the handler.
        """
        results = []
        all_cleanup_paths = []

        try:
            preprocessed_requests = self.preprocess(data)
            for req_data in preprocessed_requests:
                all_cleanup_paths.extend(req_data["cleanup_paths"])

                output_text = self._run_inference(
                    image_paths=req_data["image_paths"],
                    prompt=req_data["prompt"],
                    sys_prompt=req_data["sys_prompt"],
                    has_images=req_data["has_images"], # Pass the flag
                    max_new_tokens=req_data["max_new_tokens"], # Pass max_new_tokens
                    max_pixels=req_data["max_pixels"],         # Pass max_pixels
                    min_pixels=req_data["min_pixels"]          # Pass min_pixels
                )
                results.append({"output": output_text})
        finally:
            for path in all_cleanup_paths:
                try:
                    os.remove(path)
                    print(f"Cleaned up temporary file: {path}")
                except OSError as e:
                    print(f"Error cleaning up file {path}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results
