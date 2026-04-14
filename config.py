# Configuration file for Flask application
import os

class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    
    # File upload
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    # Model
    MODEL_PATH = os.environ.get('MODEL_PATH', 'pneumonia_unknown_model.pth')
    CLASSES = ['Normal', 'Pneumonia', 'Unknown']
    
    # Device
    USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'
    
    # Flask
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ENV = 'development'
    FLASK_ENV = 'development'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    ENV = 'production'
    FLASK_ENV = 'production'
    
    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    UPLOAD_FOLDER = 'test_uploads'
    MODEL_PATH = 'pneumonia_unknown_model.pth'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """Get configuration object based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    return config.get(config_name, config['default'])
