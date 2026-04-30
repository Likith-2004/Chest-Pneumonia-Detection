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
import json
import urllib.request
import urllib.error
import sys
import logging

# Setup logging to stderr so it appears in Render logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "pneumonia_unknown_model.pth"
MODEL_URL = "https://github.com/Likith-2004/Chest-X-Ray-Pneumonia-Detection/releases/download/v1.0.0/pneumonia_unknown_model.pth"
CLASSES = ["Normal", "Pneumonia", "Unknown"]
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# ── lazy-load globals ──────────────────────────────────────────────────────────
_model = None
_gradcam = None
_model_error = None   # store the error message if loading failed


def download_model():
    """Download model from GitHub Releases if not present locally"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        logger.info(f"Model found at {MODEL_PATH} ({file_size:.2f} MB)")
        return True

    logger.warning(f"Model not found – downloading from GitHub…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        logger.info(f"Model downloaded. Size: {file_size:.2f} MB")
        return True
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        logger.error(f"URL Error: {e.reason}")
    except Exception as e:
        logger.error(f"Download error: {type(e).__name__}: {e}")
    return False


def _load_model_once():
    """Lazy-load the model on first prediction request."""
    global _model, _gradcam, _model_error

    if _model is not None:
        return _model, _gradcam

    if _model_error is not None:
        raise RuntimeError(_model_error)

    logger.info("Lazy-loading model…")
    try:
        if not download_model():
            raise RuntimeError("Could not download model from GitHub Releases.")

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")

        gradcam = GradCAM(model, model.layer4[1].conv2)
        logger.info("Grad-CAM initialised.")

        _model = model
        _gradcam = gradcam
        return _model, _gradcam

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        _model_error = str(e)
        raise


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_in, grad_out):
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


# ── transforms ────────────────────────────────────────────────────────────────
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
    model, gradcam = _load_model_once()   # raises if loading fails

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = predicted.item()
        confidence_score = confidence.item()

    logger.info(f"Prediction: {CLASSES[predicted_class]} ({confidence_score*100:.2f}%)")

    cam, _ = gradcam.generate(img_tensor, predicted_class)
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_array = np.array(img.resize((224, 224)))
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_rgb, 0.4, 0)
    overlay_img = Image.fromarray(overlay)

    buffered = BytesIO()
    overlay_img.save(buffered, format="PNG")
    overlay_base64 = base64.b64encode(buffered.getvalue()).decode()

    class_scores = {
        CLASSES[i]: float(probabilities[0][i].item()) * 100
        for i in range(len(CLASSES))
    }

    return {
        'prediction': CLASSES[predicted_class],
        'confidence': confidence_score * 100,
        'class_scores': class_scores,
        'gradcam': overlay_base64,
        'device': str(device)
    }


# ── routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    logger.info("POST /api/predict")
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = predict_and_visualize(filepath)
            result['original_image'] = image_to_base64(filepath)
        finally:
            # always clean up, even if prediction fails
            try:
                os.remove(filepath)
            except Exception:
                pass

        return jsonify(result), 200

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        # Always return valid JSON – this is what was missing before
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/info', methods=['GET'])
def info():
    return jsonify({
        'model': 'ResNet18',
        'classes': CLASSES,
        'device': str(device),
        'visualization': 'Grad-CAM',
        'model_loaded': _model is not None,
        'gradcam_ready': _gradcam is not None
    })


@app.route('/health', methods=['GET'])
def health():
    status = {
        'status': 'ok' if _model is not None else 'not_loaded_yet',
        'model_loaded': _model is not None,
        'gradcam_ready': _gradcam is not None,
        'device': str(device),
        'model_path_exists': os.path.exists(MODEL_PATH)
    }
    return jsonify(status), 200


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16 MB.'}), 413


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Error: {error}")
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
