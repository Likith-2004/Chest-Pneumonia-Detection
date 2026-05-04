from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

MODEL_PATH = "pneumonia_unknown_model.pth"
CLASSES = ["Normal", "Pneumonia", "Unknown"]
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[:, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), target_class

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

    logger.info(f"Loading model from {MODEL_PATH}...")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    logger.info("✅ Model loaded successfully")
    return model

try:
    model = load_model()
    gradcam = GradCAM(model, model.layer4[1].conv2)
    MODEL_ERROR = None
except Exception as e:
    logger.error(str(e))
    model = None
    gradcam = None
    MODEL_ERROR = str(e)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def predict_and_visualize(image_path):
    if model is None:
        raise RuntimeError(MODEL_ERROR)

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    output = model(img_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

    predicted_class = predicted.item()
    confidence_score = confidence.item()

    cam, _ = gradcam.generate(img_tensor, predicted_class)
    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_array = np.array(img.resize((224, 224)))
    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

    buffered = BytesIO()
    Image.fromarray(overlay).save(buffered, format="PNG")

    return {
        "prediction": CLASSES[predicted_class],
        "confidence": confidence_score * 100,
        "gradcam": base64.b64encode(buffered.getvalue()).decode(),
        "original_image": image_to_base64(image_path)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    filepath = None

    try:
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_and_visualize(filepath)
        return jsonify(result)

    except Exception as e:
        logger.error(str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/health')
def health():
    return jsonify({
        "model_loaded": model is not None,
        "error": MODEL_ERROR
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
