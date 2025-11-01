"""
Projects API endpoints
"""
import re
from flask import request, jsonify, current_app
from backend.api import api_bp
from backend.services.project_service import ProjectService
from backend.i18n import t


def get_project_service() -> ProjectService:
    """Get ProjectService instance with current app config"""
    return ProjectService(
        workspace_dir=current_app.config['WORKSPACE_DIR'],
        meta_file_name=current_app.config['PROJECTS_META_FILE']
    )


def validate_project_name(name: str) -> tuple[bool, str]:
    """
    Validate project name according to rules:
    - Must start with an alphabet character (a-z, A-Z)
    - Can include: alphabet, numbers, underscore (_), hyphen (-), and space
    - Spaces can only appear in the middle (not at start or end)

    Args:
        name: Project name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, t('errors.project_name_required')

    # Check if name starts or ends with space
    if name.startswith(' ') or name.endswith(' '):
        return False, t('errors.project_name_invalid_chars')

    # Check if first character is an alphabet
    if not name[0].isalpha():
        return False, t('errors.invalid_project_name')

    # Check if all characters are valid (alphabet, number, _, -, or space)
    # Pattern: starts with letter, followed by any combination of letters, numbers, _, -, or spaces
    pattern = r'^[a-zA-Z][a-zA-Z0-9_\- ]*$'
    if not re.match(pattern, name):
        return False, t('errors.project_name_invalid_chars')

    return True, ""


@api_bp.route('/projects', methods=['GET'])
def list_projects():
    """
    List all projects

    Returns:
        JSON response with list of projects
    """
    try:
        service = get_project_service()
        projects = service.list_projects()

        return jsonify({
            'projects': [p.to_dict() for p in projects],
            'count': len(projects)
        }), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500


@api_bp.route('/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """
    Get project by ID

    Args:
        project_id: Project identifier

    Returns:
        JSON response with project data
    """
    try:
        service = get_project_service()
        project = service.get_project(project_id)

        if not project:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        return jsonify(project.to_dict()), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500


@api_bp.route('/projects', methods=['POST'])
def create_project():
    """
    Create a new project

    Request body:
        {
            "name": "Project Name",
            "description": "Optional description"
        }

    Returns:
        JSON response with created project data
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': t('errors.bad_request'),
                'message': 'Request body must be JSON'
            }), 400

        name = data.get('name')
        if not name:
            return jsonify({
                'error': t('errors.bad_request'),
                'message': t('errors.project_name_required')
            }), 400

        # Validate project name
        is_valid, error_message = validate_project_name(name)
        if not is_valid:
            return jsonify({
                'error': t('errors.validation_error'),
                'message': error_message
            }), 400

        description = data.get('description', '')

        service = get_project_service()
        project = service.create_project(name, description)

        return jsonify(project.to_dict()), 201

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


@api_bp.route('/projects/<project_id>', methods=['PUT'])
def update_project(project_id):
    """
    Update project metadata

    Args:
        project_id: Project identifier

    Request body:
        {
            "name": "Updated Name",
            "description": "Updated description"
        }

    Returns:
        JSON response with updated project data
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': t('errors.bad_request'),
                'message': 'Request body must be JSON'
            }), 400

        name = data.get('name')
        description = data.get('description')

        # Validate project name if provided
        if name is not None:
            is_valid, error_message = validate_project_name(name)
            if not is_valid:
                return jsonify({
                    'error': t('errors.validation_error'),
                    'message': error_message
                }), 400

        service = get_project_service()
        project = service.update_project(project_id, name, description)

        if not project:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        return jsonify(project.to_dict()), 200

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


@api_bp.route('/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """
    Delete project and its directory

    Args:
        project_id: Project identifier

    Returns:
        JSON response confirming deletion
    """
    try:
        service = get_project_service()
        success = service.delete_project(project_id)

        if not success:
            return jsonify({
                'error': t('errors.not_found'),
                'message': t('errors.project_not_found')
            }), 404

        return jsonify({
            'message': t('success.project_deleted'),
            'project_id': project_id
        }), 200

    except Exception as e:
        return jsonify({
            'error': t('errors.internal_error'),
            'message': str(e)
        }), 500
