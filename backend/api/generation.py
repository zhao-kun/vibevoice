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


def _validate_offloading_config(offloading: dict) -> dict:
    """
    Validate offloading configuration from request.

    Args:
        offloading: Offloading config dict from request (can be None)

    Returns:
        Validated offloading config dict or None if disabled/not provided

    Raises:
        ValueError: If config is invalid
    """
    # If not provided or disabled, return None (backward compatible)
    if not offloading or not offloading.get('enabled', False):
        return None

    mode = offloading.get('mode', 'preset')

    # Validate mode
    if mode not in ['preset', 'manual']:
        raise ValueError(f"Invalid offloading mode: '{mode}'. Must be 'preset' or 'manual'")

    if mode == 'preset':
        preset = offloading.get('preset', 'balanced')
        valid_presets = ['balanced', 'aggressive', 'extreme']
        if preset not in valid_presets:
            raise ValueError(f"Invalid preset: '{preset}'. Must be one of: {', '.join(valid_presets)}")

        return {
            'enabled': True,
            'mode': 'preset',
            'preset': preset
        }

    elif mode == 'manual':
        num_gpu_layers = offloading.get('num_gpu_layers')
        if num_gpu_layers is None:
            raise ValueError("num_gpu_layers is required for manual mode")

        if not isinstance(num_gpu_layers, int) or num_gpu_layers < 1 or num_gpu_layers > 28:
            raise ValueError(f"num_gpu_layers must be an integer between 1 and 28, got: {num_gpu_layers}")

        return {
            'enabled': True,
            'mode': 'manual',
            'num_gpu_layers': num_gpu_layers
        }


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

        # Parse and validate offloading configuration (NEW)
        offloading_config = data.get('offloading')
        try:
            validated_offloading = _validate_offloading_config(offloading_config)
        except ValueError as e:
            return jsonify({
                'error': 'Invalid offloading configuration',
                'message': str(e)
            }), 400

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
                                                    attn_implementation=attn_implementation,
                                                    project_id=project_id,
                                                    offloading_config=validated_offloading)
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
    Download or stream the generated audio file for a specific generation

    Query parameters:
        - download: If set to 'true', force download. Otherwise, serve inline for playback.

    Args:
        project_id: Project identifier
        request_id: Generation request identifier
    Returns:
        Audio file (inline or as attachment) or error message
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

        # Check if download parameter is set to true
        force_download = request.args.get('download', 'false').lower() == 'true'

        # Serve inline by default (for playback), or as attachment if requested
        return send_file(
            str(audio_file_path),
            mimetype='audio/wav',
            as_attachment=force_download,
            download_name=generation.output_filename if force_download else None
        )

    except Exception as e:
        logger.error(f"Error occurred while downloading generated audio: {e}")
        return jsonify({
            'error': 'Failed to download generated audio',
            'message': str(e)
        }), 500

@api_bp.route('/projects/<project_id>/generations/<request_id>', methods=['GET'])
def get_generation(project_id: str, request_id: str):
    """
    Get a specific generation by request ID

    Args:
        project_id: Project identifier
        request_id: Generation request identifier
    Returns:
        JSON response with generation data
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
        service = VoiceGenerationService(project_path / 'output', speaker_service=speaker_service, dialog_service=dialog_service)

        # Get all generations and find the one with matching request_id
        generations = service.list_generations()
        generation = next((g for g in generations if g.request_id == request_id), None)

        if not generation:
            return jsonify({
                'error': 'Generation not found',
                'message': f'Generation with request ID "{request_id}" does not exist'
            }), 404

        return jsonify({
            'generation': generation.to_dict()
        }), 200

    except Exception as e:
        logger.error(f"Error occurred while retrieving generation: {e}")
        return jsonify({
            'error': 'Failed to retrieve generation',
            'message': str(e)
        }), 500

@api_bp.route('/projects/<project_id>/generations/<request_id>', methods=['DELETE'])
def delete_generation(project_id: str, request_id: str):
    """
    Delete a specific generation and its audio file

    Args:
        project_id: Project identifier
        request_id: Generation request identifier
    Returns:
        JSON response with deletion status
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
        service = VoiceGenerationService(project_path / 'output', speaker_service=speaker_service, dialog_service=dialog_service)

        # Delete the generation
        success = service.delete_generation(request_id)
        if not success:
            return jsonify({
                'error': 'Generation not found',
                'message': f'Generation with request ID "{request_id}" does not exist'
            }), 404

        return jsonify({
            'message': 'Generation deleted successfully',
            'request_id': request_id
        }), 200

    except Exception as e:
        logger.error(f"Error occurred while deleting generation: {e}")
        return jsonify({
            'error': 'Failed to delete generation',
            'message': str(e)
        }), 500

@api_bp.route('/projects/<project_id>/generations/batch-delete', methods=['POST'])
def batch_delete_generations(project_id: str):
    """
    Delete multiple generations and their audio files

    Request body:
        {
            "request_ids": ["id1", "id2", ...]
        }

    Args:
        project_id: Project identifier
    Returns:
        JSON response with batch deletion results
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON'
            }), 400

        request_ids = data.get('request_ids', [])
        if not request_ids or not isinstance(request_ids, list):
            return jsonify({
                'error': 'Bad Request',
                'message': 'request_ids must be a non-empty array'
            }), 400

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
        service = VoiceGenerationService(project_path / 'output', speaker_service=speaker_service, dialog_service=dialog_service)

        # Delete generations in batch
        result = service.delete_generations_batch(request_ids)

        return jsonify({
            'message': f'Batch delete completed: {result["deleted_count"]} deleted, {result["failed_count"]} failed',
            'deleted_count': result['deleted_count'],
            'failed_count': result['failed_count'],
            'deleted_ids': result['deleted_ids'],
            'failed_ids': result['failed_ids']
        }), 200

    except Exception as e:
        logger.error(f"Error occurred while batch deleting generations: {e}")
        return jsonify({
            'error': 'Failed to batch delete generations',
            'message': str(e)
        }), 500
