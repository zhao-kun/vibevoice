"""
Flask application factory for VibeVoice backend
"""
import os
from flask import Flask, jsonify
from flask_cors import CORS
from pathlib import Path

from backend.config import get_config


def create_app(config_name=None):
    """
    Application factory pattern for creating Flask app

    Args:
        config_name: Configuration name ('development', 'production', 'testing')

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    config_class = get_config(config_name)
    app.config.from_object(config_class)

    # Initialize CORS
    CORS(app, origins=app.config['CORS_ORIGINS'], supports_credentials=True)

    # Ensure upload folder exists
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    upload_folder.mkdir(parents=True, exist_ok=True)

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    register_blueprints(app)

    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'version': '0.0.1',
            'service': 'vibevoice-backend'
        }), 200

    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint"""
        return jsonify({
            'service': 'VibeVoice API',
            'version': '0.0.1',
            'status': 'running'
        }), 200

    return app


def register_error_handlers(app):
    """Register error handlers for the application"""

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': str(error)
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource was not found'
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }), 500

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            'error': 'File Too Large',
            'message': f'Maximum file size is {app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)}MB'
        }), 413


def register_blueprints(app):
    """Register API blueprints"""
    from backend.api import api_bp

    app.register_blueprint(api_bp, url_prefix='/api')
