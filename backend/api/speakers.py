"""
Speakers API endpoints
"""
from flask import request, jsonify, current_app, send_file
from backend.api import api_bp
from backend.services.speaker_service import SpeakerService
from backend.services.project_service import ProjectService


def get_speaker_service(project_id: str) -> SpeakerService:
    """Get SpeakerService instance for a specific project"""
    # Get project service to find project directory
    project_service = ProjectService(
        workspace_dir=current_app.config['WORKSPACE_DIR'],
        meta_file_name=current_app.config['PROJECTS_META_FILE']
    )

    # Get project path
    project_path = project_service.get_project_path(project_id)
    if not project_path:
        return None

    # Return speaker service for project's voices directory
    return SpeakerService(project_path / 'voices')


@api_bp.route('/projects/<project_id>/speakers', methods=['GET'])
def list_speakers(project_id):
    """
    List all speaker roles for a project

    Args:
        project_id: Project identifier

    Returns:
        JSON response with list of speakers
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        speakers = service.list_speakers()

        return jsonify({
            'speakers': [s.to_dict() for s in speakers],
            'count': len(speakers)
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to list speakers',
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/speakers/<speaker_id>', methods=['GET'])
def get_speaker(project_id, speaker_id):
    """
    Get speaker role by ID

    Args:
        project_id: Project identifier
        speaker_id: Speaker identifier (e.g., "Speaker 1")

    Returns:
        JSON response with speaker data
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        speaker = service.get_speaker(speaker_id)
        if not speaker:
            return jsonify({
                'error': 'Speaker not found',
                'message': f'Speaker with ID "{speaker_id}" does not exist'
            }), 404

        return jsonify(speaker.to_dict()), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get speaker',
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/speakers', methods=['POST'])
def add_speaker(project_id):
    """
    Add a new speaker role with voice file

    Args:
        project_id: Project identifier

    Form data:
        name: Speaker name (required)
        description: Speaker description (optional)
        voice_file: Audio file (required)

    Returns:
        JSON response with created speaker data
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        # Get form data
        name = request.form.get('name')
        description = request.form.get('description', '')
        voice_file = request.files.get('voice_file')

        if not name:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Speaker name is required'
            }), 400

        if not voice_file:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Voice file is required'
            }), 400

        speaker = service.add_speaker(name, description, voice_file)

        return jsonify(speaker.to_dict()), 201

    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Failed to add speaker',
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/speakers/<speaker_id>', methods=['PUT'])
def update_speaker(project_id, speaker_id):
    """
    Update speaker role metadata

    Args:
        project_id: Project identifier
        speaker_id: Speaker identifier

    Request body:
        {
            "name": "Updated Name",
            "description": "Updated description"
        }

    Returns:
        JSON response with updated speaker data
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON'
            }), 400

        name = data.get('name')
        description = data.get('description')

        speaker = service.update_speaker(speaker_id, name, description)
        if not speaker:
            return jsonify({
                'error': 'Speaker not found',
                'message': f'Speaker with ID "{speaker_id}" does not exist'
            }), 404

        return jsonify(speaker.to_dict()), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to update speaker',
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/speakers/<speaker_id>', methods=['DELETE'])
def delete_speaker(project_id, speaker_id):
    """
    Delete speaker role and its voice file

    Args:
        project_id: Project identifier
        speaker_id: Speaker identifier

    Returns:
        JSON response confirming deletion
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        success = service.delete_speaker(speaker_id)
        if not success:
            return jsonify({
                'error': 'Speaker not found',
                'message': f'Speaker with ID "{speaker_id}" does not exist'
            }), 404

        return jsonify({
            'message': 'Speaker deleted successfully. Speaker IDs have been reindexed.',
            'speaker_id': speaker_id
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to delete speaker',
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/speakers/<speaker_id>/voice', methods=['GET'])
def download_voice_file(project_id, speaker_id):
    """
    Download speaker's voice file

    Args:
        project_id: Project identifier
        speaker_id: Speaker identifier

    Returns:
        Audio file or error
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        voice_file_path = service.get_voice_file_path(speaker_id)
        if not voice_file_path or not voice_file_path.exists():
            return jsonify({
                'error': 'Voice file not found',
                'message': f'Voice file for speaker "{speaker_id}" does not exist'
            }), 404

        return send_file(voice_file_path, as_attachment=True)

    except Exception as e:
        return jsonify({
            'error': 'Failed to download voice file',
            'message': str(e)
        }), 500
