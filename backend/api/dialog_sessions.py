"""
Dialog Sessions API endpoints
"""
from flask import request, jsonify, current_app, send_file
from backend.api import api_bp
from backend.services.dialog_session_service import DialogSessionService
from backend.services.speaker_service import SpeakerService
from backend.services.project_service import ProjectService
from backend.i18n import t


def get_dialog_session_service(project_id: str) -> DialogSessionService:
    """Get DialogSessionService instance for a specific project"""
    # Get project service to find project directory
    project_service = ProjectService(
        workspace_dir=current_app.config['WORKSPACE_DIR'],
        meta_file_name=current_app.config['PROJECTS_META_FILE']
    )

    # Get project path
    project_path = project_service.get_project_path(project_id)
    if not project_path:
        return None

    # Get speaker service for validation
    speaker_service = SpeakerService(project_path / 'voices')

    # Return dialog session service for project's scripts directory
    return DialogSessionService(project_path / 'scripts', speaker_service=speaker_service)


@api_bp.route('/projects/<project_id>/sessions', methods=['GET'])
def list_sessions(project_id):
    """
    List all dialog sessions for a project

    Args:
        project_id: Project identifier

    Returns:
        JSON response with list of sessions
    """
    try:
        service = get_dialog_session_service(project_id)
        if not service:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        sessions = service.list_sessions()

        return jsonify({
            'sessions': [s.to_dict() for s in sessions],
            'count': len(sessions)
        }), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/sessions/<session_id>', methods=['GET'])
def get_session(project_id, session_id):
    """
    Get dialog session by ID

    Args:
        project_id: Project identifier
        session_id: Session identifier

    Returns:
        JSON response with session data
    """
    try:
        service = get_dialog_session_service(project_id)
        if not service:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        session = service.get_session(session_id)
        if not session:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.session_not_found')
            }), 404

        return jsonify(session.to_dict()), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/sessions', methods=['POST'])
def create_session(project_id):
    """
    Create a new dialog session

    Args:
        project_id: Project identifier

    Request body:
        {
            "name": "Session name",
            "description": "Session description",
            "dialog_text": "Speaker 1: Hello\\n\\nSpeaker 2: Hi"
        }

    Returns:
        JSON response with created session data
    """
    try:
        service = get_dialog_session_service(project_id)
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

        name = data.get('name')
        description = data.get('description', '')
        dialog_text = data.get('dialog_text', '')  # Default to empty string if not provided

        if not name:
            return jsonify({
                'error': t('errors.bad_request'),
                'message': t('errors.validation_error')
            }), 400

        # Allow empty dialog_text - user can add dialogs later
        session = service.create_session(name, description, dialog_text)

        return jsonify(session.to_dict()), 201

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


@api_bp.route('/projects/<project_id>/sessions/<session_id>', methods=['PUT'])
def update_session(project_id, session_id):
    """
    Update dialog session metadata and/or text

    Args:
        project_id: Project identifier
        session_id: Session identifier

    Request body:
        {
            "name": "Updated name",
            "description": "Updated description",
            "dialog_text": "Speaker 1: Updated text"
        }

    Returns:
        JSON response with updated session data
    """
    try:
        service = get_dialog_session_service(project_id)
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

        name = data.get('name')
        description = data.get('description')
        dialog_text = data.get('dialog_text')

        session = service.update_session(session_id, name, description, dialog_text)
        if not session:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.session_not_found')
            }), 404

        return jsonify(session.to_dict()), 200

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


@api_bp.route('/projects/<project_id>/sessions/<session_id>', methods=['DELETE'])
def delete_session(project_id, session_id):
    """
    Delete dialog session and its text file

    Args:
        project_id: Project identifier
        session_id: Session identifier

    Returns:
        JSON response confirming deletion
    """
    try:
        service = get_dialog_session_service(project_id)
        if not service:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        success = service.delete_session(session_id)
        if not success:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.session_not_found')
            }), 404

        return jsonify({
            'message': t('success.session_deleted'),
            'session_id': session_id
        }), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/sessions/<session_id>/text', methods=['GET'])
def get_session_text(project_id, session_id):
    """
    Get dialog text content for a session

    Args:
        project_id: Project identifier
        session_id: Session identifier

    Returns:
        JSON response with dialog text content
    """
    try:
        service = get_dialog_session_service(project_id)
        if not service:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        text = service.get_session_text(session_id)
        if text is None:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.session_not_found')
            }), 404

        return jsonify({
            'session_id': session_id,
            'dialog_text': text
        }), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>/sessions/<session_id>/download', methods=['GET'])
def download_session_text(project_id, session_id):
    """
    Download dialog text file for a session

    Args:
        project_id: Project identifier
        session_id: Session identifier

    Returns:
        Text file or error
    """
    try:
        service = get_dialog_session_service(project_id)
        if not service:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        text_file_path = service.get_text_file_path(session_id)
        if not text_file_path or not text_file_path.exists():
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.session_not_found')
            }), 404

        return send_file(text_file_path, as_attachment=True, download_name=f'dialog_{session_id}.txt')

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500
