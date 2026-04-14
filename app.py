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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
MODEL_PATH = "pneumonia_unknown_model.pth"
CLASSES = ["Normal", "Pneumonia", "Unknown"]
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Load model
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        raise Exception(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

model = load_model()

# Grad-CAM implementation
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

gradcam = GradCAM(model, model.layer4[1].conv2)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Convert image to base64 string for display"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode()

def predict_and_visualize(image_path):
    """Make prediction and generate Grad-CAM visualization"""
    # Load and process image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = predicted.item()
        confidence_score = confidence.item()
    
    # Generate Grad-CAM
    cam, _ = gradcam.generate(img_tensor, predicted_class)
    
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (224, 224))
    
    # Create heatmap
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert to RGB
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    img_array = np.array(img.resize((224, 224)))
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_rgb, 0.4, 0)
    
    # Convert overlay to PIL Image
    overlay_img = Image.fromarray(overlay)
    
    # Convert to base64
    buffered = BytesIO()
    overlay_img.save(buffered, format="PNG")
    overlay_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Get confidence scores for all classes
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

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_and_visualize(filepath)
        
        # Get original image as base64
        original_base64 = image_to_base64(filepath)
        result['original_image'] = original_base64
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def info():
    return jsonify({
        'model': 'ResNet18',
        'classes': CLASSES,
        'device': str(device),
        'visualization': 'Grad-CAM'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
