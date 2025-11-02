"""
Speakers API endpoints
"""
from flask import request, jsonify, current_app, send_file
from backend.api import api_bp
from backend.services.speaker_service import SpeakerService
from backend.services.project_service import ProjectService
from backend.i18n import t


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
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        speakers = service.list_speakers()

        return jsonify({
            'speakers': [s.to_dict() for s in speakers],
            'count': len(speakers)
        }), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
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
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        speaker = service.get_speaker(speaker_id)
        if not speaker:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.speaker_not_found')
            }), 404

        return jsonify(speaker.to_dict()), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/speakers', methods=['POST'])
def add_speaker(project_id):
    """
    Add a new speaker role with voice file

    Args:
        project_id: Project identifier

    Form data:
        description: Speaker description (optional)
        voice_file: Audio file (required)

    Returns:
        JSON response with created speaker data
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        # Get form data
        description = request.form.get('description', '')
        voice_file = request.files.get('voice_file')

        if not voice_file:
            return jsonify({
                'error': t('errors.bad_request'),
                'message': t('errors.file_upload_error')
            }), 400

        speaker = service.add_speaker(description, voice_file)

        return jsonify(speaker.to_dict()), 201

    except ValueError as e:
        return jsonify({
            'error': t('errors.validation_error'),
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
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
            "description": "Updated description"
        }

    Returns:
        JSON response with updated speaker data
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        data = request.get_json()
        if not data:
            return jsonify({
                'error': t('errors.bad_request'),
                'message': 'Request body must be JSON'
            }), 400

        description = data.get('description')

        speaker = service.update_speaker(speaker_id, description)
        if not speaker:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.speaker_not_found')
            }), 404

        return jsonify(speaker.to_dict()), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
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
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        success = service.delete_speaker(speaker_id)
        if not success:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.speaker_not_found')
            }), 404

        return jsonify({
            'message': t('success.speaker_deleted'),
            'speaker_id': speaker_id
        }), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
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
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        voice_file_path = service.get_voice_file_path(speaker_id)
        if not voice_file_path or not voice_file_path.exists():
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.speaker_not_found')
            }), 404

        return send_file(voice_file_path, as_attachment=True)

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/speakers/<speaker_id>/voice', methods=['PUT'])
def update_voice_file(project_id, speaker_id):
    """
    Update speaker's voice file without changing speaker ID

    Args:
        project_id: Project identifier
        speaker_id: Speaker identifier

    Form data:
        voice_file: New audio file (required)

    Returns:
        JSON response with updated speaker data
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        voice_file = request.files.get('voice_file')
        if not voice_file:
            return jsonify({
                'error': t('errors.bad_request'),
                'message': t('errors.file_upload_error')
            }), 400

        speaker = service.update_voice_file(speaker_id, voice_file)
        if not speaker:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.speaker_not_found')
            }), 404

        return jsonify(speaker.to_dict()), 200

    except ValueError as e:
        return jsonify({
            'error': t('errors.validation_error'),
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/speakers/<speaker_id>/voice/trim', methods=['POST'])
def trim_voice_file(project_id, speaker_id):
    """
    Trim speaker's voice file to a specific time range

    Args:
        project_id: Project identifier
        speaker_id: Speaker identifier

    JSON body:
        start_time: Start time in seconds (required)
        end_time: End time in seconds (required)

    Returns:
        JSON response with updated speaker data
    """
    try:
        service = get_speaker_service(project_id)
        if not service:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        data = request.get_json()
        if not data:
            return jsonify({
                'error': t('errors.bad_request'),
                'message': 'JSON body is required'
            }), 400

        start_time = data.get('start_time')
        end_time = data.get('end_time')

        if start_time is None or end_time is None:
            return jsonify({
                'error': t('errors.bad_request'),
                'message': t('errors.validation_error')
            }), 400

        if start_time < 0 or end_time <= start_time:
            return jsonify({
                'error': t('errors.validation_error'),
                'message': 'Invalid time range: end_time must be greater than start_time and both must be non-negative'
            }), 400

        speaker = service.trim_voice_file(speaker_id, start_time, end_time)
        if not speaker:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.speaker_not_found')
            }), 404

        return jsonify(speaker.to_dict()), 200

    except ValueError as e:
        return jsonify({
            'error': t('errors.validation_error'),
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500
