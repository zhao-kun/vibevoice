"""
Generation API endpoints
"""
from flask import request, jsonify, current_app, send_file
from backend.api import api_bp
from backend.services.voice_gerneration_service import VoiceGenerationService
from backend.services.dialog_session_service import DialogSessionService
from backend.services.speaker_service import SpeakerService
from backend.services.project_service import ProjectService


def get_voice_generation_service(project_id: str) -> VoiceGenerationService:
    """Get VoiceGenerationService instance for a specific project"""
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

    dialog_service = DialogSessionService(project_path / 'scripts', speaker_service=speaker_service)

    # Return dialog session service for project's scripts directory
    return VoiceGenerationService(project_path / 'output', speaker_service=speaker_service, dialog_service=dialog_service)