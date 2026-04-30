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
        print(f"✅ Model found at {MODEL_PATH}")
        return True
    
    print(f"📥 Downloading model from GitHub Releases...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✅ Model downloaded successfully to {MODEL_PATH}")
        return True
    except urllib.error.URLError as e:
        print(f"❌ Failed to download model: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model not found locally. Attempting download...")
            if not download_model():
                raise Exception("Could not download model from GitHub Releases")
        
        print(f"Loading model from {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
        return model
    except FileNotFoundError:
        raise Exception(f"Model file not found at {MODEL_PATH} and download failed")
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading model: {e}")
        raise Exception(f"Error loading model: {e}")

try:
    model = load_model()
except Exception as e:
    print(f"❌ FATAL ERROR: Failed to load model on startup: {e}")
    print(f"❌ App is starting but predictions will fail")
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
        gradcam = GradCAM(model, model.layer4[1].conv2)
        print("✅ Grad-CAM initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Grad-CAM: {e}")
        gradcam = None

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
    if model is None:
        raise Exception("Model not loaded. Please check Render logs for model loading errors.")
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidence_score = confidence.item()
        
        if gradcam is None:
            raise Exception("Grad-CAM not initialized")
        
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
    except Exception as e:
        print(f"❌ Error in predict_and_visualize: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
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
        
        original_base64 = image_to_base64(filepath)
        result['original_image'] = original_base64
        
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"❌ Error in /api/predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/info', methods=['GET'])
def info():
    return jsonify({
        'model': 'ResNet18',
        'classes': CLASSES,
        'device': str(device),
        'visualization': 'Grad-CAM',
        'model_loaded': model is not None,
        'gradcam_ready': gradcam is not None
    })

@app.errorhandler(500)
def internal_error(error):
    print(f"❌ 500 Error: {error}")
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Flask app starting...")
    print(f"✅ Device: {device}")
    print(f"✅ Model loaded: {model is not None}")
    print(f"✅ Grad-CAM ready: {gradcam is not None}")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
