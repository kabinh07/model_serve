import torch
import numpy as np
import cv2
import base64
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException
from scipy.spatial.distance import cosine
from retinaface import extract_faces
import os

class FaceNetTRTHandler(BaseHandler):
    def __init__(self):
        super(FaceNetTRTHandler, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def initialize(self, ctx):
        """Load the TensorRT model and initialize inference context."""
        properties = ctx.system_properties
        model_dir = properties.get('model_dir')
        model_path = os.path.join(model_dir, "facenet.pt")
        self.model = torch.jit.load(model_path, map_location=self.device)
        # self.model = torch.compile(torch_model, backend = "tensorrt")
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        """Preprocess images using OpenCV for face detection and TensorRT for face recognition."""
        data = data[0].get("body") or data[0].get("data")
        img_1 = data.get("img_1")
        img_2 = data.get("img_2")

        if not img_1 or not img_2:
            raise PredictionException("Invalid data provided for inference", 513)
        preprocessed_images = []
        image_data = [img_1, img_2]
        if not len(image_data) == 2:
            raise PredictionException("Enter two images for inference", 513)
        for image in image_data:
            split = image.strip().split(',')
            if len(split) < 2:
                raise PredictionException("Invalid image", 513)
            img_data = base64.b64decode(split[1])
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise PredictionException("Failed to decode image", 513)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            faces = extract_faces(image, align=True)
            
            if len(faces) == 0:
                raise PredictionException("No Face detected", 513)  # No faces detected
            if len(faces) > 1:
                raise PredictionException("Multiple faces detected, please provide a single face image", 513)
            
            for face in faces:
                preprocessed_images.append(cv2.resize(face, (160, 160)))

        if not preprocessed_images:
            return None
        
        return preprocessed_images

    def inference(self, data):
        """Run inference."""
        if data is None:
            return None
        output_tensor = []
        for image in data:
            with torch.no_grad():
                # if image.shape != (160, 160, 3):
                #     image = cv2.resize(image, (160, 160))
                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
                result = self.model(image)
                output_tensor.append(result.cpu().numpy())
        return output_tensor

    def calculate_similarity(self, emb1, emb2):
        """Calculate the cosine similarity between two embeddings."""
        return 1 - cosine(emb1[0], emb2[0])

    def postprocess(self, inference_output):
        """Postprocess results and compute similarity."""
        similarity = 0.0
        if inference_output is None or len(inference_output) < 2:
            return [{"similarity": similarity}]
        
        emb1, emb2 = inference_output[0], inference_output[1]
        similarity = self.calculate_similarity(emb1, emb2)

        # Ensure similarity is JSON serializable
        return [{"similarity": float(similarity)}]

    def handle(self, data, context):
        """Main handler function for TorchServe."""
        # Preprocess both images
        images = self.preprocess(data)
        
        # Get embeddings for both faces
        embeddings = self.inference(images)
        
        # Compute similarity between the two embeddings
        return self.postprocess(embeddings)