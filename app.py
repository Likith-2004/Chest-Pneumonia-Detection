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
import sys
import logging

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
logger.info(f"Using device: {device}")

MODEL_PATH = "https://github.com/Likith-2004/Chest-X-Ray-Pneumonia-Detection/releases/download/v1.0.0/pneumonia_unknown_model.pth"
CLASSES = ["Normal", "Pneumonia", "Unknown"]
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


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
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Make sure build.sh ran successfully during the Render build phase."
        )
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded and ready.")
    return model


try:
    model = load_model()
    gradcam = GradCAM(model, model.layer4[1].conv2)
    logger.info("Grad-CAM initialised.")
    MODEL_ERROR = None
except Exception as e:
    import traceback
    logger.error(traceback.format_exc())
    model = None
    gradcam = None
    MODEL_ERROR = str(e)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def predict_and_visualize(image_path):
    if model is None:
        raise RuntimeError(f"Model not loaded: {MODEL_ERROR}")

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

    buffered = BytesIO()
    Image.fromarray(overlay).save(buffered, format="PNG")
    overlay_base64 = base64.b64encode(buffered.getvalue()).decode()

    return {
        'prediction': CLASSES[predicted_class],
        'confidence': confidence_score * 100,
        'class_scores': {CLASSES[i]: float(probabilities[0][i].item()) * 100 for i in range(len(CLASSES))},
        'gradcam': overlay_base64,
        'device': str(device)
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    logger.info("POST /api/predict")
    filepath = None
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

        result = predict_and_visualize(filepath)
        result['original_image'] = image_to_base64(filepath)
        return jsonify(result), 200

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'status': 'error'}), 500

    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass


@app.route('/api/info', methods=['GET'])
def info():
    return jsonify({
        'model': 'ResNet18',
        'classes': CLASSES,
        'device': str(device),
        'visualization': 'Grad-CAM',
        'model_loaded': model is not None,
        'gradcam_ready': gradcam is not None,
        'model_error': MODEL_ERROR
    })


@app.route('/health', methods=['GET'])
def health():
    if model is None:
        return jsonify({'status': 'error', 'model_loaded': False, 'error': MODEL_ERROR}), 503
    return jsonify({'status': 'ok', 'model_loaded': True, 'gradcam_ready': gradcam is not None, 'device': str(device)}), 200


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum 16 MB allowed.'}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
