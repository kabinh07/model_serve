import os
import base64
import tempfile
import torch
import easyocr
import numpy as np
import shutil
from io import BytesIO
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException
from typing import List

class EasyOCRHandler(BaseHandler):
    def __init__(self):
        super(EasyOCRHandler, self).__init__()
        self.readers = {}
        self.initialized = False
    
    def initialize(self, context):
        """
        Initialize the EasyOCR readers for English and Bangla.
        """
        try:
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            gpu_available = torch.cuda.is_available()
            
            if not model_dir:
                raise PredictionException("model_dir not found in system properties", 500)
            
            print(f"Loading EasyOCR models from {model_dir}...")
            
            # Copy custom user_network files to EasyOCR cache directory
            # Note: torch-model-archiver flattens extra-files to root of model_dir
            easyocr_cache = os.path.expanduser('~/.EasyOCR')
            user_network_dst = os.path.join(easyocr_cache, 'user_network')
            os.makedirs(user_network_dst, exist_ok=True)
            
            # Copy the custom network files from flattened model_dir
            for ext in ['.py', '.yaml']:
                src_file = os.path.join(model_dir, f'bengali-fintune{ext}')
                dst_file = os.path.join(user_network_dst, f'bengali-fintune{ext}')
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"📁 Copied bengali-fintune{ext} to EasyOCR cache")
                else:
                    print(f"⚠️ File not found: {src_file}")
            
            # Initialize readers
            # optimization: load into memory once
            self.readers = {
                'en': easyocr.Reader(
                    ['en'], 
                    gpu=gpu_available, 
                    model_storage_directory=model_dir,
                    download_enabled=False
                ),
                'bn': easyocr.Reader(
                    ['bn'],
                    recog_network='bengali-fintune',
                    gpu=gpu_available, 
                    model_storage_directory=model_dir,
                    download_enabled=False
                )
            }
            
            self.initialized = True
            
            print(f"✅ EasyOCR Model initialized successfully!")
            print(f"   Model Dir: {model_dir}")
            print(f"   GPU Available: {gpu_available}")
            print(f"   Loaded Languages: {list(self.readers.keys())}")
            
        except Exception as e:
            error_msg = f"Failed to initialize EasyOCR models: {str(e)}"
            print(f"❌ {error_msg}")
            raise PredictionException(error_msg, 500)

    def preprocess(self, data):
        """
        Preprocess input.
        Expects: {"img": "base64...", "bboxes": [[x1,y1,x2,y2], ...], "lang": "bn" or "en"}
        Returns: (pil_image, bboxes, lang)
        """
        if not self.initialized:
            raise PredictionException("Model is not initialized", 500)
        
        # Handle TorchServe input wrapper
        row = data[0].get("body") or data[0].get("data")
        
        img_data = row.get("img")
        bboxes: List = row.get("bboxes", [])
        lang = self._extract_language(row)
        
        if not img_data or bboxes is None:
            raise PredictionException("Invalid data: missing img or bboxes", 513)
        
        # Decode base64 image
        # We explicitly handle the "data:image/..." header if present
        if ',' in img_data:
            img_data = img_data.split(',')[1]
            
        try:
            img_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_bytes))
            
            # Standardize to RGB. EasyOCR works best with 3 channels.
            print(f"🔍 Image mode before conversion: {img.mode}")
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"🔍 Converted image to RGB mode")
                
        except Exception as e:
            raise PredictionException(f"Failed to decode image: {e}", 513)
        
        print(f"🔍 PREPROCESS - Language='{lang}', Bboxes={len(bboxes)}, Image={img.size}")
        
        return img, bboxes, lang

    def inference(self, model_input):
        """
        Run OCR inference using the 'Temporary File' strategy to ensure
        maximum accuracy by forcing OpenCV's native imread behavior.
        """
        img, bboxes, lang = model_input
        
        reader = self.readers.get(lang)
        if not reader:
            print(f"Warning: Language '{lang}' not supported, falling back to 'en'")
            reader = self.readers['en']
        
        output = []
        
        for bbox_idx, bbox in enumerate(bboxes):
            temp_file_path = None
            try:
                # 1. Parse Coordinates
                # Ensure they are integers and within bounds
                width, height = img.size
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(width, int(bbox[2]))
                y2 = min(height, int(bbox[3]))
                
                print(f"🔍 Bbox {bbox_idx}: Original bbox={bbox}, Image size={img.size}")
                print(f"🔍 Bbox {bbox_idx}: Parsed coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}, Crop dims: {x2-x1}x{y2-y1}")
                
                # Check for invalid crops
                if x2 <= x1 or y2 <= y1:
                    print(f"⚠️ Bbox {bbox_idx} invalid dimensions: {bbox}, skipping.")
                    output.append({"text": "", "confidence": 0.0})
                    continue

                # 2. Crop using PIL (Cleaner than Numpy conversion for this step)
                crop_img = img.crop((x1, y1, x2, y2))
                
                print(f"🔍 Bbox {bbox_idx}: Crop size={crop_img.size}, mode={crop_img.mode}")
                
                # 3. Save to Temporary File
                # We use .png to ensure lossless compression so pixels don't degrade
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img:
                    crop_img.save(temp_img.name)
                    temp_file_path = temp_img.name
                    print(f"🔍 Saved temp file: {temp_file_path}")
                
                # 4. Pass FILE PATH to EasyOCR
                # Use recognize() instead of readtext() since we already have cropped text regions
                # This skips the detection step and directly runs recognition
                print(f"🔍 Bbox {bbox_idx}: Starting recognize() for {lang} language...")
                try:
                    # recognize() expects a list of bboxes covering the whole image
                    # We pass the full crop dimensions as a single bbox
                    crop_w, crop_h = crop_img.size
                    horizontal_list = [[0, crop_w, 0, crop_h]]  # x_min, x_max, y_min, y_max
                    ocr_results = reader.recognize(temp_file_path, horizontal_list=horizontal_list, free_list=[])
                    print(f"🔍 Bbox {bbox_idx}: recognize() completed, got {len(ocr_results) if ocr_results else 0} results")
                except Exception as e:
                    print(f"❌ Bbox {bbox_idx}: recognize() failed with exception: {type(e).__name__}: {str(e)}")
                    raise
                
                # 5. Aggregate Results
                if ocr_results:
                    # Sometimes a single crop contains multiple words/lines
                    combined_text_list = []
                    confidences = []
                    
                    for _, text, confidence in ocr_results:
                        combined_text_list.append(text)
                        confidences.append(confidence)
                    
                    final_text = " ".join(combined_text_list)
                    avg_conf = sum(confidences) / len(confidences)
                    
                    # Log snippet
                    log_text = (final_text[:30] + '..') if len(final_text) > 30 else final_text
                    print(f"✅ Bbox {bbox_idx}: '{log_text}' (Conf: {avg_conf:.4f})")
                    
                    output.append({
                        "text": final_text,
                        "confidence": float(avg_conf)
                    })
                else:
                    # No text found in this box
                    print(f"⚠️ Bbox {bbox_idx}: No text detected (empty result from readtext())")
                    output.append({
                        "text": "",
                        "confidence": 0.0
                    })
                    
            except Exception as e:
                print(f"❌ INFERENCE - Error processing bbox {bbox_idx}: {e}")
                output.append({"text": "", "confidence": 0.0})
                
            finally:
                # 6. Cleanup: Extremely important to remove temp files
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass # Setup for next iteration regardless

        return output

    def postprocess(self, inference_output):
        """
        Wrap result in a list format.
        """
        return [inference_output]

    def _extract_language(self, row):
        """Helper to find language key in input dictionary"""
        lang = None
        # Check various casing
        for key in ['lang', 'language', 'Language', 'Lang']:
            if key in row:
                lang = row.get(key)
                break
        
        if lang:
            # Handle potential bytes from TorchServe
            if isinstance(lang, (bytes, bytearray)):
                lang = lang.decode('utf-8')
            lang = str(lang).strip().lower().strip('\'"')
        
        if lang not in ['en', 'bn']:
            lang = 'en'
        
        return lang