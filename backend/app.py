"""
Flask application factory for VibeVoice backend
"""
import os
from flask import Flask, jsonify, send_from_directory
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
    # Get the directory where dist folder is located (backend directory)
    backend_dir = Path(__file__).parent
    static_folder = backend_dir / 'dist'

    # Create Flask app WITHOUT automatic static serving
    # We'll handle static files manually in serve_spa()
    app = Flask(__name__,
                static_folder=str(static_folder),
                static_url_path=None)  # Disable Flask's built-in static serving

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

    # Serve frontend static files
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_spa(path):
        """
        Serve frontend SPA
        - API routes are handled by blueprints
        - Static files are served from dist folder
        - All other routes serve index.html for client-side routing
        """
        print(f"[DEBUG] serve_spa called with path: '{path}'")
        # If path exists as a static file, serve it
        if path and (static_folder / path).exists():
            print(f"[DEBUG] Found exact file: {path}")
            return send_from_directory(static_folder, path)
        # Check if .html version exists (for Next.js static export pages)
        html_path = f"{path}.html"
        if path and (static_folder / html_path).exists():
            print(f"[DEBUG] Found HTML file: {html_path}")
            return send_from_directory(static_folder, html_path)
        # Otherwise, serve index.html for SPA routing
        print(f"[DEBUG] Serving index.html for path: '{path}'")
        return send_from_directory(static_folder, 'index.html')

    return app


def register_error_handlers(app):
    """Register error handlers for the application"""
    from flask import request

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': str(error)
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        # If it's an API request, return JSON 404
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Not Found',
                'message': 'The requested resource was not found'
            }), 404
        # Otherwise, let the SPA handle it (this shouldn't happen with catch-all route)
        # But just in case, try to serve index.html
        try:
            backend_dir = Path(__file__).parent
            static_folder = backend_dir / 'dist'
            return send_from_directory(static_folder, 'index.html')
        except Exception:
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

    app.register_blueprint(api_bp, url_prefix='/api/v1')
