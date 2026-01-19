# import easyocr

# reader = easyocr.Reader(['bn'], gpu=True, model_storage_directory='saved_models') 

# results = reader.readtext('test_data_english/334_385_693_428.png')

# for (bbox, text, prob) in results:
#     print(f"Text: {text:<20} | Confidence: {prob:.4f}")

import easyocr
import base64
import tempfile
import os
from PIL import Image
from io import BytesIO

# Initialize the reader
reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='saved_models') 

image_path = 'test_data_english/334_307_545_339.png'

# --- STEP 1: Create Base64 string (Simulating your input) ---
with open(image_path, "rb") as image_file:
    base64_encoded_data = base64.b64encode(image_file.read())
    base64_string = base64_encoded_data.decode('utf-8')

# --- STEP 2: Write Base64 to Temp File and Read ---

# Decode base64 string to bytes
image_bytes = base64.b64decode(base64_string)

try:
    # 1. Load with PIL and convert to RGB (Handler Preprocess Logic)
    img = Image.open(BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 2. Save to Temporary File (Handler Inference Logic)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img:
        img.save(temp_img.name)
        temp_img_path = temp_img.name
    
    # 3. Call recognize() with horizontal_list covering full image (Handler Inference Logic)
    width, height = img.size
    # horizontal_list format: [x_min, x_max, y_min, y_max]
    horizontal_list = [[0, width, 0, height]]

    # Pass the TEMPORARY FILE PATH to EasyOCR
    # Use recognize() instead of readtext() to match handler.py behavior
    results = reader.recognize(temp_img_path, horizontal_list=horizontal_list, free_list=[])

    for (_, text, prob) in results:
        print(f"Text: {text:<20} | Confidence: {prob:.4f}")

finally:
    # Clean up: Delete the temporary file
    if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
        os.remove(temp_img_path)
