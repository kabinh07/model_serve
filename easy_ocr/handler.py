import os
import sys
import base64
import math
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException
from typing import List

# Add dependencies directory to path
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_DIR, "dependencies"))

from model import Model
from utils import CTCLabelConverter, AttnLabelConverter

# Bangla text normalization
try:
    from normalizer import normalize
    BANGLA_NORMALIZER_AVAILABLE = True
except ImportError:
    BANGLA_NORMALIZER_AVAILABLE = False
    def normalize(*args, **kwargs):
        return args[0] if args else ""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EasyOCRHandler(BaseHandler):
    def __init__(self):
        super(EasyOCRHandler, self).__init__()
        self.model = None
        self.converter = None
        self.opt = None
        self.bangla_normalize = False
        self.initialized = False
    
    def initialize(self, ctx):
        """Load the Easy OCR model."""
        properties = ctx.system_properties
        model_dir = properties.get('model_dir')
        
        # Load model options
        opt_path = os.path.join(model_dir, "opt.txt")
        if not os.path.exists(opt_path):
            raise PredictionException(f"Options file not found at {opt_path}", 500)
        
        # Parse options from opt.txt
        self.opt = self._parse_opt(opt_path)
        
        # Check if Bangla normalization was enabled during training
        self.bangla_normalize = self.opt.get('bangla_normalize', False)
        if self.bangla_normalize and not BANGLA_NORMALIZER_AVAILABLE:
            raise PredictionException(
                "Model was trained with bangla_normalize=True but normalizer library is not available. "
                "Add 'normalizer' to requirements.txt", 500)
        
        # Initialize character converter based on prediction type
        prediction_type = self.opt.get('Prediction', 'Attn')
        charset = self.opt.get('character', '')
        
        if prediction_type == 'CTC':
            self.converter = CTCLabelConverter(charset)
        else:
            self.converter = AttnLabelConverter(charset)
        
        # Load model
        model_path = os.path.join(model_dir, "5.imgW_200_best_accuracy.pth")
        if not os.path.exists(model_path):
            raise PredictionException(f"Model file not found at {model_path}", 500)
        
        # Create model instance
        self.model = Model(type('Opt', (), self.opt)())
        self.model = torch.nn.DataParallel(self.model).to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle module prefix mismatch
        new_state_dict = {}
        state_keys = list(state_dict.keys())
        model_keys = list(self.model.state_dict().keys())
        
        state_has_module = any(key.startswith('module.') for key in state_keys)
        model_has_module = any(key.startswith('module.') for key in model_keys)
        
        if state_has_module and not model_has_module:
            for key, value in state_dict.items():
                new_state_dict[key.replace('module.', '')] = value
        elif not state_has_module and model_has_module:
            for key, value in state_dict.items():
                new_state_dict[f'module.{key}'] = value
        else:
            new_state_dict = state_dict
        
        try:
            self.model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            print(f'Warning: strict load failed: {e}; attempting non-strict load.')
            self.model.load_state_dict(new_state_dict, strict=False)
        
        self.model.eval()
        
        self.initialized = True
        print(f"✅ EasyOCR Model initialized successfully!")
        
    def _parse_opt(self, opt_path):
        """Parse options from opt.txt file."""
        opt = {}
        with open(opt_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip() or line.strip().startswith('---'):
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    
                    if key == 'character':
                        opt[key] = value.rstrip('\n\r')
                        if opt[key].startswith(' '):
                            opt[key] = opt[key][1:]
                        continue
                    
                    value = value.strip()
                    
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                    
                    opt[key] = value
        return opt
    
    def _preprocess_image_with_pad(self, pil_image):
        """Preprocess image with aspect ratio preservation and padding."""
        imgH = self.opt.get('imgH', 32)
        imgW = self.opt.get('imgW', 200)
        keep_ratio_with_pad = self.opt.get('PAD', False)
        rgb = self.opt.get('rgb', False)
        
        if keep_ratio_with_pad:
            w, h = pil_image.size
            ratio = w / float(h)
            
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = math.ceil(imgH * ratio)
            
            resized_image = pil_image.resize((resized_w, imgH), Image.BICUBIC)
            
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(resized_image)
            img_tensor.sub_(0.5).div_(0.5)
            
            c, h, w = img_tensor.size()
            input_channel = 3 if rgb else 1
            padded = torch.FloatTensor(input_channel, imgH, imgW).fill_(0)
            padded[:, :, :w] = img_tensor
            
            if imgW != w:
                padded[:, :, w:] = img_tensor[:, :, w-1].unsqueeze(2).expand(c, h, imgW - w)
            
            return padded
        else:
            resized_image = pil_image.resize((imgW, imgH), Image.BICUBIC)
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(resized_image)
            img_tensor.sub_(0.5).div_(0.5)
            return img_tensor
    
    def preprocess(self, data):
        """
        Preprocess input - MATCHES text_recognizer/handler.py interface.
        Expects: {"img": "base64...", "bboxes": [[x1,y1,x2,y2], ...]}
        """
        if not self.initialized:
            raise PredictionException("Model is not initialized", 500)
        
        data = data[0].get("body") or data[0].get("data")
        
        # Get full image and bboxes (SAME AS SURYA HANDLER)
        img_data = data.get("img")
        bboxes: List = data.get("bboxes")
        
        print(f"🔍 PREPROCESS - Received {len(bboxes) if bboxes else 0} bboxes")
        
        if not img_data or not bboxes:
            raise PredictionException("Invalid data provided for inference", 513)
        
        # Decode base64 image
        split = img_data.strip().split(',')
        if len(split) < 2:
            raise PredictionException("Invalid image", 513)
        
        img = Image.open(BytesIO(base64.b64decode(split[1])))
        
        # Convert to grayscale or RGB based on model config
        if self.opt.get('rgb', False):
            img = img.convert("RGB")
        else:
            img = img.convert("L")
        
        print(f"🔍 PREPROCESS - Image size: {img.size}, Mode: {img.mode}")
        
        return img, bboxes
    
    def inference(self, model_input):
        """
        Run OCR inference - MATCHES recognizer_surya_cy format EXACTLY
        Returns: list of dicts [{"text": ..., "confidence": ...}, ...]
        """
        if self.model is None:
            raise PredictionException("Model is not initialized", 500)
        
        img, bboxes = model_input
        
        print(f"🔍 INFERENCE - Processing {len(bboxes)} bounding boxes")
        
        # Convert PIL to numpy
        img_np = np.array(img)
        img_h, img_w = img_np.shape[0], img_np.shape[1]
        
        # Crop all bounding boxes and preprocess
        cropped_tensors = []
        for idx, bbox in enumerate(bboxes):
            # Ensure coordinates are ints and clamp to image bounds
            try:
                x1, y1, x2, y2 = map(int, bbox)
            except Exception:
                print(f"Warning: invalid bbox format at index {idx}: {bbox} - skipping")
                continue

            x1 = max(0, min(x1, img_w))
            x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h))
            y2 = max(0, min(y2, img_h))

            # Skip empty or inverted boxes
            if x2 <= x1 or y2 <= y1:
                print(f"Warning: bbox {idx} has zero area after clamping: [{x1},{y1},{x2},{y2}] - skipping")
                continue

            cropped = img_np[y1:y2, x1:x2]

            print(f"🔍 INFERENCE - Bbox {idx}: {[x1,y1,x2,y2]}, Crop shape: {cropped.shape}")
            
            # Convert back to PIL
            if len(cropped.shape) == 2:  # Grayscale
                pil_crop = Image.fromarray(cropped).convert('L')
            else:  # RGB
                pil_crop = Image.fromarray(cropped)
            
            # Preprocess with padding
            tensor = self._preprocess_image_with_pad(pil_crop)
            cropped_tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(cropped_tensors).to(device)
        
        print(f"🔍 INFERENCE - Batch tensor shape: {batch_tensor.shape}")
        
        # Run inference
        batch_size = batch_tensor.size(0)
        batch_max_length = self.opt.get('batch_max_length', 200)
        
        length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)
        
        with torch.no_grad():
            if self.opt.get('Prediction') == 'CTC':
                preds = self.model(batch_tensor, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, preds_size)
            else:  # Attention
                preds = self.model(batch_tensor, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)
            
            # Calculate confidence scores
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            # MATCH recognizer_surya_cy EXACTLY: list of dicts
            output = []
            for idx, (pred, pred_max_prob) in enumerate(zip(preds_str, preds_max_prob)):
                if 'Attn' in self.opt.get('Prediction', ''):
                    pred_EOS = pred.find('[s]')
                    if pred_EOS != -1:
                        pred_max_prob = pred_max_prob[:pred_EOS]
                    pred = pred.split('[s]')[0]
                
                # Apply Bangla normalization if enabled
                if self.bangla_normalize:
                    pred = self._normalize_bangla_text(pred)
                
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                except:
                    confidence_score = 0.0
                
                print(f"🔍 INFERENCE - Result {idx}: text='{pred}', conf={confidence_score:.4f}")
                
                # Simple dict - MATCHES recognizer_surya_cy line 119
                output.append({
                    "text": pred,
                    "confidence": confidence_score
                })
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"🔍 INFERENCE - Returning {len(output)} results")
        return output

    def postprocess(self, model_output):
        """Wrap in list - MATCHES recognizer_surya_cy line 142"""
        print(f"🔍 POSTPROCESS - Input: {len(model_output)} items")
        return [model_output]
    
    def _normalize_bangla_text(self, text):
        """Normalize Bangla text using the normalizer library"""
        if not BANGLA_NORMALIZER_AVAILABLE:
            return text
        
        try:
            return normalize(text, unicode_norm="NFKC", punct_replacement=None,
                           url_replacement=None, emoji_replacement=None,
                           apply_unicode_norm_last=True)
        except Exception as e:
            print(f"Warning: Bangla normalization failed: {e}")
            return text