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


# Import route modules
from backend.api import projects  # noqa: F401, E402
from backend.api import speakers  # noqa: F401, E402
from backend.api import dialog_sessions  # noqa: F401, E402
# from backend.api import generation  # To be implemented
