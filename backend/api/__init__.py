"""
API Blueprint for VibeVoice backend
"""
from flask import Blueprint, jsonify

# Create main API blueprint
api_bp = Blueprint('api', __name__)


@api_bp.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint for testing"""
    return jsonify({'message': 'pong', 'status': 'ok'}), 200


# Import route modules (will be created later)
# from backend.api import projects, speakers, generation
