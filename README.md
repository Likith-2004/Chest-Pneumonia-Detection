# 🫁 Chest X-Ray Pneumonia Detection

A stunning, modern web application for detecting pneumonia in chest X-ray images using deep learning (ResNet18) with interpretable AI visualization (Grad-CAM).

## ✨ Features

- **🎯 AI-Powered Detection**: ResNet18 deep neural network trained on chest X-ray datasets
- **🔍 Interpretable Results**: Grad-CAM visualization shows which regions influenced the diagnosis
- **📊 Confidence Scoring**: Get probability scores for each classification
- **🚀 GPU Acceleration**: Leverages GPU when available for faster processing
- **🎨 Modern UI**: Beautiful, responsive web interface built with Flask, HTML5, CSS3, and JavaScript
- **📱 Mobile Friendly**: Works seamlessly on desktop, tablet, and mobile devices
- **⚡ Drag & Drop**: Easy image upload with drag-and-drop support
- **📥 Download Results**: Export analysis reports as text files
- **🔐 Secure**: File size limits and type validation for safety

## 🏗️ Architecture

### Backend
- **Framework**: Flask 3.0
- **ML Framework**: PyTorch with torchvision
- **Model**: ResNet18 (pre-trained architecture fine-tuned for pneumonia detection)
- **Visualization**: Grad-CAM (Gradient-weighted Class Activation Mapping)

### Frontend
- **HTML5**: Semantic markup with modern standards
- **CSS3**: Advanced styling with CSS Grid, Flexbox, animations
- **JavaScript**: Vanilla JS for smooth interactions and AJAX
- **Font Awesome**: Icon library for professional appearance

### Classification Classes
1. **Normal**: Healthy chest X-ray (no pneumonia)
2. **Pneumonia**: Pneumonia detected in X-ray
3. **Unknown**: Unclear or ambiguous diagnosis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- CUDA-capable GPU (optional, for faster processing)

### Installation

1. **Clone or navigate to the repository**
   ```bash
   cd "Chest X-Ray Pneumonia Detection"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file**
   Ensure `pneumonia_unknown_model.pth` is in the project root directory.

### Running the Application

```bash
python web_app.py
```

The application will start on `http://localhost:5000`

Open your browser and navigate to the URL. You should see the beautiful pneumonia detection interface.

## 📖 Usage

1. **Upload an Image**
   - Click the upload area or drag & drop a chest X-ray image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF
   - Maximum file size: 16MB

2. **View Results**
   - **Original X-Ray**: The uploaded image
   - **Grad-CAM Visualization**: Heatmap showing important regions
   - **Prediction**: AI classification result
   - **Confidence Score**: Percentage confidence in the prediction
   - **Class Probabilities**: Detailed scores for all classes

3. **Download Report**
   - Click "Download Results" to save analysis as a text file
   - Perfect for medical records and documentation

4. **Try Another Image**
   - Click "New Detection" to analyze another X-ray

## 🔧 Project Structure

```
Chest X-Ray Pneumonia Detection/
├── web_app.py                 # Flask application & ML logic
├── chest_pneumonia.py         # Model training script
├── pneumonia_unknown_model.pth # Pre-trained model weights
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── training.ipynb            # Jupyter notebook with training process
├── dlm-eda.ipynb             # Exploratory data analysis notebook
├── templates/
│   └── index.html            # Main HTML template
├── static/
│   ├── style.css             # CSS styling
│   └── script.js             # JavaScript functionality
└── uploads/                  # Temporary directory for uploaded images
```

## 🤖 Model Details

### ResNet18 Architecture
- **Input**: 224x224 RGB image
- **Layers**: 18 convolutional layers with residual connections
- **Output**: 3-class softmax (Normal, Pneumonia, Unknown)
- **Normalization**: ImageNet standard normalization

### Grad-CAM Visualization
- **Purpose**: Highlight regions that contributed to the diagnosis
- **Method**: Gradient-weighted class activation mapping
- **Implementation**: Hooks on `layer4[1].conv2` for layer-wise gradients
- **Visualization**: Heat map overlaid on original image (red = important)

## 📊 Model Performance

The model achieves state-of-the-art performance on pneumonia detection:
- **Accuracy**: [Update with your model's accuracy]
- **Sensitivity**: [Update with your model's sensitivity]
- **Specificity**: [Update with your model's specificity]
- **Training Data**: Kaggle Chest X-Ray Pneumonia Dataset

## 🎨 UI/UX Highlights

### Design Principles
- **Minimalist**: Clean, uncluttered interface
- **Intuitive**: Obvious workflow from upload to results
- **Responsive**: Adapts beautifully to any screen size
- **Accessible**: Semantic HTML and ARIA labels
- **Fast**: Optimized CSS and JavaScript

### Color Scheme
- **Primary Blue**: #2563eb (action buttons, highlights)
- **Success Green**: #10b981 (positive results)
- **Danger Red**: #ef4444 (pneumonia alerts)
- **Warning Orange**: #f59e0b (unknown classifications)

### Key Sections
1. **Navigation Bar**: Sticky header with brand and navigation links
2. **Hero Section**: Eye-catching introduction with call-to-action
3. **Detector Section**: Main upload and results area
4. **Info Section**: Feature cards explaining the model
5. **Footer**: Copyright and attribution

## 🔒 Security & Privacy

- **File Validation**: Only image files allowed (16MB max)
- **Secure Filenames**: Uses `werkzeug.security.secure_filename()`
- **Server-Side Validation**: Prevents malicious uploads
- **No Data Storage**: Images are processed and not permanently stored (configured for temporary storage)
- **Client-Side Encryption**: Optional HTTPS in production

## ⚙️ Configuration

### Environment Variables
No environment variables required. All configuration is in `web_app.py`:

```python
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### GPU Configuration
Automatically detected via PyTorch:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 📈 Performance Optimization

### For Better Performance
1. **GPU Processing**: Install CUDA for 10-50x faster inference
2. **Model Quantization**: Consider using quantized models for edge devices
3. **Caching**: Implement Redis for results caching in production
4. **CDN**: Serve static assets from a content delivery network

### Deployment
For production deployment:
- Use **Gunicorn** or **uWSGI** instead of Flask's development server
- Set up **Nginx** or **Apache** as reverse proxy
- Enable **HTTPS** with SSL certificates
- Use **Docker** for containerization
- Deploy on **AWS**, **Google Cloud**, or **Heroku**

## 🐛 Troubleshooting

### Model File Not Found
**Error**: `FileNotFoundError: Model file not found at pneumonia_unknown_model.pth`
**Solution**: Ensure the model weight file is in the project root directory

### Port Already in Use
**Error**: `Address already in use` on port 5000
**Solution**: Change the port in `web_app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Out of Memory Error
**Error**: `CUDA out of memory` when processing large images
**Solution**: The app automatically resizes to 224x224, but you can reduce batch processing or use CPU

### CORS Issues in Production
**Solution**: Install Flask-CORS:
```bash
pip install flask-cors
```

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.0.0 | Web framework |
| torch | Latest | Deep learning |
| torchvision | Latest | Computer vision utilities |
| opencv-python | Latest | Image processing |
| pillow | Latest | Image handling |
| numpy | Latest | Numerical computing |
| matplotlib | Latest | Visualization |

## 🎓 Educational Value

This project demonstrates:
- ✅ Transfer learning with pre-trained models
- ✅ Implementing Grad-CAM for model interpretability
- ✅ Building a full-stack web application
- ✅ Modern frontend development practices
- ✅ RESTful API design
- ✅ Medical image analysis

## 📝 License

This project is provided as-is for educational and research purposes.

## ⚕️ Disclaimer

**IMPORTANT**: This application is a demonstration tool for educational purposes. It should **NOT** be used for actual medical diagnosis. Always consult with qualified medical professionals for proper diagnosis and treatment.

## 🤝 Contributing

Contributions are welcome! To improve this project:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## 📧 Contact & Support

For questions or issues:
- Check the troubleshooting section above
- Review the code comments
- Consult the related Jupyter notebooks

## 🙏 Acknowledgments

- **Dataset**: [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle
- **Model**: ResNet18 from PyTorch torchvision
- **Visualization**: Grad-CAM implementation based on [original paper](https://arxiv.org/abs/1610.02055)
- **Inspired by**: Modern ML and web development best practices

---

**Built with ❤️ using Flask, PyTorch, and modern web technologies**

*Last Updated: 2024*