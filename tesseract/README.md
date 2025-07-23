# Tesseract OCR TorchServe API Usage

This document describes how to make API calls to the Tesseract OCR model hosted on TorchServe.

---

## Request Body Parameters

- **images**: List of Base64-encoded images (e.g., `"<base_64_image>"`).
- **language**: (Optional) Tesseract language codes. Default: `ben+eng`.
- **output_type**: (Optional) Type of output to return:
  - `"txt"` – Plain text output.
  - `"box"` – Bounding boxes for each character.
  - `"data"` – Detailed OCR data (words, confidence, coordinates).
  - `"osd"` – Orientation and script detection.
  - `["txt", "box"]` – Multiple outputs in a single request.

---

## Example `curl` Calls

### 1. Plain Text Output
```bash
curl -X POST http://127.0.0.1:8080/predictions/tesseract-ocr   -H "Content-Type: application/json"   -d '{
        "images": ["<base_64_image>"],
        "language": "ben+eng",
        "output_type": "txt"
      }'
```

### 2. Bounding Box Output
```bash
curl -X POST http://127.0.0.1:8080/predictions/tesseract-ocr   -H "Content-Type: application/json"   -d '{
        "images": ["<base_64_image>"],
        "output_type": "box"
      }'
```

### 3. Multiple Outputs (Text + Box)
```bash
curl -X POST http://127.0.0.1:8080/predictions/tesseract-ocr   -H "Content-Type: application/json"   -d '{
        "images": ["<base_64_image>"],
        "output_type": ["txt", "box"]
      }'
```

---
