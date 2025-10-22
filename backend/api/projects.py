"""
Projects API endpoints
"""
from flask import request, jsonify, current_app
from backend.api import api_bp
from backend.services.project_service import ProjectService


def get_project_service() -> ProjectService:
    """Get ProjectService instance with current app config"""
    return ProjectService(
        workspace_dir=current_app.config['WORKSPACE_DIR'],
        meta_file_name=current_app.config['PROJECTS_META_FILE']
    )


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
            'error': 'Failed to list projects',
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
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        return jsonify(project.to_dict()), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get project',
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
                'error': 'Bad Request',
                'message': 'Request body must be JSON'
            }), 400

        name = data.get('name')
        if not name:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Project name is required'
            }), 400

        description = data.get('description', '')

        service = get_project_service()
        project = service.create_project(name, description)

        return jsonify(project.to_dict()), 201

    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Failed to create project',
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
                'error': 'Bad Request',
                'message': 'Request body must be JSON'
            }), 400

        name = data.get('name')
        description = data.get('description')

        service = get_project_service()
        project = service.update_project(project_id, name, description)

        if not project:
            return jsonify({
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        return jsonify(project.to_dict()), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to update project',
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
                'error': 'Project not found',
                'message': f'Project with ID "{project_id}" does not exist'
            }), 404

        return jsonify({
            'message': 'Project deleted successfully',
            'project_id': project_id
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to delete project',
            'message': str(e)
        }), 500
