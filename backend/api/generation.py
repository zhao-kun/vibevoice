"""
Generation API endpoints
"""
from uuid import uuid4
from flask import request, jsonify, current_app, send_file
from backend.api import api_bp
from backend.models.generation import Generation
from backend.services.voice_gerneration_service import VoiceGenerationService
from backend.services.dialog_session_service import DialogSessionService
from backend.services.speaker_service import SpeakerService
from backend.services.project_service import ProjectService
from backend.gen_voice.task import gm
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

@api_bp.route('/projects/<project_id>/generations', methods=['POST'])
def get_voice_generation_service(project_id: str):
    """Get VoiceGenerationService instance for a specific project"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON'
            }), 400

        dialog_session_id = data.get('dialog_session_id')
        if dialog_session_id is None:
            return jsonify({
                'error': 'Bad Request',
                'message': 'dialog_session_id is required'
            }), 400

        request_id = uuid4().hex
        seeds = data.get('seeds', 42)
        cfg_scale = data.get('cfg_scale', 1.3)
        model_dtype = data.get('model_dtype', 'float8_e4m3fn')
        attn_implementation = data.get('attn_implementation', 'sdpa')

        # Get project service to find project directory
        project_service = ProjectService(workspace_dir=current_app.config['WORKSPACE_DIR'],
                                         meta_file_name=current_app.config['PROJECTS_META_FILE'])

        # Get project path
        project_path = project_service.get_project_path(project_id)
        if not project_path:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        # Get speaker service for validation
        speaker_service = SpeakerService(project_path / 'voices')
        dialog_service = DialogSessionService(project_path / 'scripts', speaker_service=speaker_service)
        # Return dialog session service for project's scripts directory
        service = VoiceGenerationService(project_path / 'output', speaker_service=speaker_service, dialog_service=dialog_service)

        generation: Generation = service.generation(dialog_session_id,
                                                    request_id,
                                                    seeds=seeds,
                                                    cfg_scale=cfg_scale,
                                                    model_dtype=model_dtype,
                                                    attn_implementation=attn_implementation)
        if not generation:
            return jsonify({
                'error': 'Generation Failed',
                'message': 'Failed to start voice generation due to existing active generation'
            }), 500
        return jsonify({
            'message': 'Voice generation started successfully',
            'request_id': generation.request_id,
            'generation': generation.to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Error occurred while starting voice generation: {e}")
        return jsonify({
            'error': 'Failed to start voice generation',
            'message': str(e)
        }), 500

@api_bp.route('/projects/generations/current', methods=['GET'])
def get_current_generation():
    """
    Get the current generation status for a project

    Args:
        project_id: Project identifier
    Returns:
        JSON response with current generation status (200 with null if none active)
    """
    generation: Generation = gm.get_current_generation()
    if not generation:
        return jsonify({
            'message': 'No active generation at the moment',
            'generation': None
        }), 200

    return jsonify({
        'message': 'Current generation status retrieved successfully',
        'generation': generation.to_dict()
    }), 200

@api_bp.route('/projects/<project_id>/generations', methods=['GET'])
def get_all_generations(project_id):
    """
    Get all generations for a project

    Args:
        project_id: Project identifier
    Returns:
        JSON response with list of generations for the project
    """
    try:
        # Get project service to find project directory
        project_service = ProjectService(workspace_dir=current_app.config['WORKSPACE_DIR'],
                                         meta_file_name=current_app.config['PROJECTS_META_FILE'])

        # Get project path
        project_path = project_service.get_project_path(project_id)
        if not project_path:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        # Get speaker service for validation
        speaker_service = SpeakerService(project_path / 'voices')
        dialog_service = DialogSessionService(project_path / 'scripts', speaker_service=speaker_service)
        # Return dialog session service for project's scripts directory
        service = VoiceGenerationService(project_path / 'output', speaker_service=speaker_service, dialog_service=dialog_service)

        generations = service.list_generations()

        return jsonify({
            'generations': [g.to_dict() for g in generations],
            'count': len(generations)
        }), 200

    except Exception as e:
        logger.error(f"Error occurred while retrieving generations: {e}")
        return jsonify({
            'error': 'Failed to retrieve generations',
            'message': str(e)
        }), 500

@api_bp.route('/projects/<project_id>/generations/<request_id>/download', methods=['GET'])
def download_generation_audio(project_id: str, request_id: str):
    """
    Download the generated audio file for a specific generation

    Args:
        project_id: Project identifier
        request_id: Generation request identifier
    Returns:
        Audio file as attachment or error message
    """
    try:
        # Get project service to find project directory
        project_service = ProjectService(workspace_dir=current_app.config['WORKSPACE_DIR'],
                                         meta_file_name=current_app.config['PROJECTS_META_FILE'])

        # Get project path
        project_path = project_service.get_project_path(project_id)
        if not project_path:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        # Get speaker service for validation
        speaker_service = SpeakerService(project_path / 'voices')
        dialog_service = DialogSessionService(project_path / 'scripts', speaker_service=speaker_service)
        # Return dialog session service for project's scripts directory
        service = VoiceGenerationService(project_path / 'output', speaker_service=speaker_service, dialog_service=dialog_service)

        generations = service.list_generations()
        generation = next((g for g in generations if g.request_id == request_id), None)
        if not generation:
            return jsonify({
                'error': 'Generation not found',
                'message': f'Generation with request ID "{request_id}" does not exist'
            }), 400

        audio_file_path = project_path / 'output' / generation.output_filename
        if not audio_file_path.exists():
            return jsonify({
                'error': 'Audio file not found',
                'message': f'Generated audio file for request ID "{request_id}" does not exist'
            }), 400

        return send_file(str(audio_file_path), as_attachment=True)

    except Exception as e:
        logger.error(f"Error occurred while downloading generated audio: {e}")
        return jsonify({
            'error': 'Failed to download generated audio',
            'message': str(e)
        }), 500
