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

def download_model():
    """Download model from GitHub Releases if not present locally"""
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model found at {MODEL_PATH}")
        file_size = os.path.getsize(MODEL_PATH) / (1024*1024)  # MB
        logger.info(f"Model size: {file_size:.2f} MB")
        return True
    
    logger.warning(f"Model not found at {MODEL_PATH}. Downloading from GitHub...")
    try:
        logger.info(f"Downloading from: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        file_size = os.path.getsize(MODEL_PATH) / (1024*1024)
        logger.info(f"Model downloaded successfully. Size: {file_size:.2f} MB")
        return True
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error during download: {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        logger.error(f"URL Error during download: {e.reason}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_model():
    logger.info("Starting model initialization...")
    try:
        logger.info("Creating ResNet18 architecture...")
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
        logger.info("ResNet18 architecture created successfully")
        
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}. Attempting download...")
            if not download_model():
                raise Exception("Could not download model from GitHub Releases after retry")
        
        logger.info(f"Loading model weights from {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        logger.info(f"State dict loaded. Keys: {list(state_dict.keys())[:3]}...")
        
        model.load_state_dict(state_dict)
        logger.info("State dict loaded into model")
        
        model = model.to(device)
        model.eval()
        logger.info("Model successfully loaded, moved to device, and set to eval mode")
        return model
    
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        raise Exception(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

try:
    logger.info("="*60)
    logger.info("INITIALIZING MODEL...")
    logger.info("="*60)
    model = load_model()
    logger.info("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    logger.error("="*60)
    logger.error(f"FATAL ERROR: Failed to load model on startup")
    logger.error(f"Error: {type(e).__name__}: {e}")
    logger.error("="*60)
    import traceback
    logger.error(traceback.format_exc())
    model = None

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

gradcam = None
if model is not None:
    try:
        logger.info("Initializing Grad-CAM...")
        gradcam = GradCAM(model, model.layer4[1].conv2)
        logger.info("Grad-CAM initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Grad-CAM: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        gradcam = None
else:
    logger.warning("Model is None, skipping Grad-CAM initialization")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Convert image to base64 string for display"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode()

def predict_and_visualize(image_path):
    """Make prediction and generate Grad-CAM visualization"""
    logger.info(f"Starting prediction for: {image_path}")
    
    if model is None:
        logger.error("Model is None - cannot make prediction")
        raise Exception("Model not loaded. Please check Render logs for model loading errors.")
    
    if gradcam is None:
        logger.error("Grad-CAM is None - cannot generate visualization")
        raise Exception("Grad-CAM not initialized")
    
    try:
        logger.info("Opening image...")
        img = Image.open(image_path).convert('RGB')
        logger.info(f"Image loaded. Size: {img.size}")
        
        logger.info("Applying transforms...")
        img_tensor = transform(img).unsqueeze(0).to(device)
        logger.info(f"Image tensor shape: {img_tensor.shape}")
        
        logger.info("Running inference...")
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidence_score = confidence.item()
        
        logger.info(f"Prediction: {CLASSES[predicted_class]} (confidence: {confidence_score*100:.2f}%)")
        
        logger.info("Generating Grad-CAM...")
        cam, _ = gradcam.generate(img_tensor, predicted_class)
        logger.info("Grad-CAM generated successfully")
        
        logger.info("Processing heatmap...")
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
        
        logger.info("Prediction completed successfully")
        return {
            'prediction': CLASSES[predicted_class],
            'confidence': confidence_score * 100,
            'class_scores': class_scores,
            'gradcam': overlay_base64,
            'device': str(device)
        }
    except Exception as e:
        logger.error(f"Error in predict_and_visualize: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    logger.info(f"POST /api/predict request received")
    try:
        if 'file' not in request.files:
            logger.warning("No file provided in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        logger.info(f"File received: {file.filename}")
        
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        logger.info("Calling predict_and_visualize...")
        result = predict_and_visualize(filepath)
        
        logger.info("Getting original image...")
        original_base64 = image_to_base64(filepath)
        result['original_image'] = original_base64
        
        logger.info("Cleaning up uploaded file...")
        try:
            os.remove(filepath)
        except:
            pass
        
        logger.info("Prediction successful, returning result")
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error in /api/predict: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/info', methods=['GET'])
def info():
    logger.info("GET /api/info request received")
    return jsonify({
        'model': 'ResNet18',
        'classes': CLASSES,
        'device': str(device),
        'visualization': 'Grad-CAM',
        'model_loaded': model is not None,
        'gradcam_ready': gradcam is not None
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    logger.info("GET /health request received")
    health_status = {
        'status': 'ok',
        'model_loaded': model is not None,
        'gradcam_ready': gradcam is not None,
        'device': str(device),
        'model_path_exists': os.path.exists(MODEL_PATH)
    }
    
    if model is None or gradcam is None:
        health_status['status'] = 'degraded'
        return jsonify(health_status), 503  # Service Unavailable
    
    return jsonify(health_status), 200

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Error: {error}")
    import traceback
    logger.error(traceback.format_exc())
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("FLASK APP STARTING UP")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Grad-CAM ready: {gradcam is not None}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
