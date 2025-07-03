import os
import io
import json
from PIL import Image
import base64
import glob 

# Import your handler class from my_handler.py
# Make sure my_handler.py is in the same directory as test.py
from handler import QwenHandler

# --- Mock Objects for Handler Testing ---
class MockRequest:
    def __init__(self, body, headers=None):
        self.body = body
        self.headers = headers if headers is not None else {}

class MockContext:
    def __init__(self, model_dir="/models"):
        # TorchServe usually provides a model_dir where MAR contents are extracted
        self.system_properties = {"gpu_id": 0, "model_dir": model_dir}
        self.manifest = None
        os.makedirs(model_dir, exist_ok=True) # Ensure mock model_dir exists

# --- Helper to create a dummy image ---
def create_dummy_image_bytes(color='red', size=(60, 30), format="PNG"):
    dummy_image = Image.new('RGB', size, color=color)
    dummy_image_bytes = io.BytesIO()
    dummy_image.save(dummy_image_bytes, format=format)
    return dummy_image_bytes.getvalue()

if __name__ == "__main__":
    handler = QwenHandler()
    mock_context = MockContext()
    handler.initialize(mock_context)

    # dummy_image_bytes_1 = create_dummy_image_bytes(color='blue')
    image = Image.open("./data/2025-07-03_194328.png")
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_uri = f"data:image/jpeg;base64,{img_str}"

    test_body_dict = {
        "image": img_uri,
        "prompt": "OCR the image"
    }
    mock_request_dict = {"body": test_body_dict}
    results_dict = handler.handle([mock_request_dict], mock_context)
    print(f"\nTest Case 1 Results: {results_dict}")


# # --- Main Test Execution ---
# if __name__ == "__main__":
#     print("--- Starting Handler Debug Test Script ---")

#     # Instantiate your handler
#     handler = MyModelHandler()

#     # Create a mock context for initialization
#     mock_context = MockContext()
#     handler.initialize(mock_context)

#     # --- Test Case 1: Image bytes and prompt in a dictionary (simulating JSON body) ---
#     print("\n\n--- Running Test Case 1: Dictionary Body ---")
#     dummy_image_bytes_1 = create_dummy_image_bytes(color='blue')
#     test_body_dict = {
#         "image": dummy_image_bytes_1,
#         "prompt": "What is the primary color?",
#         "sys_prompt": "Answer precisely."
#     }
#     mock_request_dict = MockRequest(body=test_body_dict)

#     # Call the handle method (which internally calls preprocess and _run_inference)
#     results_dict = handler.handle([mock_request_dict], mock_context)
#     print(f"\nTest Case 1 Results: {results_dict}")

#     # Verify no temporary files remain in the default /tmp directory (or wherever they were created)
#     # This check is illustrative; adjust path if your temp_file creates elsewhere.
#     # In a real scenario, tempfile module's behavior is more robust.
#     temp_files_after_test1 = glob.glob("/tmp/input_image_*") # Your handler creates files like this
#     print(f"Temporary files after Test 1: {temp_files_after_test1}")
#     assert len(temp_files_after_test1) == 0, f"Temporary files were not cleaned up after Test 1: {temp_files_after_test1}"


#     # --- Test Case 2: Raw image bytes, prompt in header ---
#     print("\n\n--- Running Test Case 2: Raw Bytes with Headers ---")
#     dummy_image_bytes_2 = create_dummy_image_bytes(color='green')
#     mock_request_raw = MockRequest(
#         body=dummy_image_bytes_2,
#         headers={"X-Prompt": "Identify the object.", "X-Sys-Prompt": "Be brief."}
#     )
#     results_raw = handler.handle([mock_request_raw], mock_context)
#     print(f"\nTest Case 2 Results: {results_raw}")
#     temp_files_after_test2 = glob.glob("/tmp/input_image_*")
#     print(f"Temporary files after Test 2: {temp_files_after_test2}")
#     assert len(temp_files_after_test2) == 0, f"Temporary files were not cleaned up after Test 2: {temp_files_after_test2}"

#     # --- Test Case 3: JSON string with base64 encoded image ---
#     print("\n\n--- Running Test Case 3: JSON String with Base64 Image ---")
#     dummy_image_bytes_3 = create_dummy_image_bytes(color='yellow')
#     base64_image = base64.b64encode(dummy_image_bytes_3).decode('utf-8')
#     json_body_string = json.dumps({
#         "image_bytes": base64_image,
#         "prompt": "What shape is it?",
#         "sys_prompt": "Give a one-word answer."
#     }).encode('utf-8') # Encode the JSON string to bytes
#     mock_request_json_string = MockRequest(body=json_body_string)
#     results_json_string = handler.handle([mock_request_json_string], mock_context)
#     print(f"\nTest Case 3 Results: {results_json_string}")
#     temp_files_after_test3 = glob.glob("/tmp/input_image_*")
#     print(f"Temporary files after Test 3: {temp_files_after_test3}")
#     assert len(temp_files_after_test3) == 0, f"Temporary files were not cleaned up after Test 3: {temp_files_after_test3}"


#     # --- Test Case 4: Multiple Requests ---
#     print("\n\n--- Running Test Case 4: Multiple Requests ---")
#     req1_body = {
#         "image": create_dummy_image_bytes(color='orange'),
#         "prompt": "What fruit is this?",
#         "sys_prompt": "Provide a common fruit name."
#     }
#     req2_body = {
#         "image": create_dummy_image_bytes(color='purple'),
#         "prompt": "What vegetable is this?",
#         "sys_prompt": "Provide a common vegetable name."
#     }
#     mock_request_multi = [
#         MockRequest(body=req1_body),
#         MockRequest(body=req2_body)
#     ]
#     results_multi = handler.handle(mock_request_multi, mock_context)
#     print(f"\nTest Case 4 Results: {results_multi}")
#     temp_files_after_test4 = glob.glob("/tmp/input_image_*")
#     print(f"Temporary files after Test 4: {temp_files_after_test4}")
#     assert len(temp_files_after_test4) == 0, f"Temporary files were not cleaned up after Test 4: {temp_files_after_test4}"


#     print("\n--- All tests completed successfully (assuming mocks are fine)! ---")