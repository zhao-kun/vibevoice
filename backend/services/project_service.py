"""
Project management service - handles business logic for projects
"""
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from backend.models.project import Project
from backend.utils.file_handler import FileHandler


class ProjectService:
    """Service for managing projects and their metadata"""

    def __init__(self, workspace_dir: Path, meta_file_name: str = 'projects.json'):
        """
        Initialize project service

        Args:
            workspace_dir: Root directory for all projects
            meta_file_name: Name of the metadata JSON file
        """
        self.workspace_dir = Path(workspace_dir)
        self.meta_file_path = self.workspace_dir / meta_file_name
        self.file_handler = FileHandler()

        # Ensure workspace directory exists
        self.file_handler.ensure_directory(self.workspace_dir)

        # Initialize metadata file if it doesn't exist
        if not self.meta_file_path.exists():
            self._save_metadata({})

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Load projects metadata from JSON file

        Returns:
            Dictionary mapping project_id to project data
        """
        try:
            return self.file_handler.read_json(self.meta_file_path)
        except FileNotFoundError:
            return {}
        except Exception as e:
            raise RuntimeError(f"Failed to load projects metadata: {str(e)}")

    def _save_metadata(self, metadata: Dict[str, Dict[str, Any]]) -> None:
        """
        Save projects metadata to JSON file

        Args:
            metadata: Dictionary mapping project_id to project data
        """
        try:
            self.file_handler.write_json(self.meta_file_path, metadata)
        except Exception as e:
            raise RuntimeError(f"Failed to save projects metadata: {str(e)}")

    def _generate_project_id(self, name: str) -> str:
        """
        Generate unique project ID from name

        Args:
            name: Project name

        Returns:
            Sanitized unique project ID
        """
        # Sanitize the name for use as directory name
        base_id = self.file_handler.sanitize_filename(name.lower().replace(' ', '-'))

        # Check if ID already exists
        metadata = self._load_metadata()
        if base_id not in metadata:
            return base_id

        # Append UUID suffix if name already exists
        return f"{base_id}-{str(uuid.uuid4())[:8]}"

    def list_projects(self) -> List[Project]:
        """
        List all projects

        Returns:
            List of Project objects
        """
        metadata = self._load_metadata()
        return [Project.from_dict(data) for data in metadata.values()]

    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get project by ID

        Args:
            project_id: Project identifier

        Returns:
            Project object or None if not found
        """
        metadata = self._load_metadata()
        project_data = metadata.get(project_id)

        if project_data:
            return Project.from_dict(project_data)
        return None

    def create_project(self, name: str, description: str = "") -> Project:
        """
        Create a new project

        Args:
            name: Project name
            description: Project description

        Returns:
            Created Project object

        Raises:
            ValueError: If project name is empty
            RuntimeError: If project creation fails
        """
        if not name or not name.strip():
            raise ValueError("Project name cannot be empty")

        # Generate unique project ID
        project_id = self._generate_project_id(name)

        # Create project directory
        project_dir = self.workspace_dir / project_id
        try:
            self.file_handler.ensure_directory(project_dir)

            # Create subdirectories for project resources
            (project_dir / 'voices').mkdir(exist_ok=True)
            (project_dir / 'scripts').mkdir(exist_ok=True)
            (project_dir / 'outputs').mkdir(exist_ok=True)

        except Exception as e:
            raise RuntimeError(f"Failed to create project directory: {str(e)}")

        # Create project metadata
        project = Project.create(project_id, name.strip(), description.strip())

        # Save to metadata file
        metadata = self._load_metadata()
        metadata[project_id] = project.to_dict()
        self._save_metadata(metadata)

        return project

    def update_project(self, project_id: str, name: Optional[str] = None,
                       description: Optional[str] = None) -> Optional[Project]:
        """
        Update project metadata

        Args:
            project_id: Project identifier
            name: New project name (optional)
            description: New project description (optional)

        Returns:
            Updated Project object or None if not found
        """
        metadata = self._load_metadata()
        project_data = metadata.get(project_id)

        if not project_data:
            return None

        # Load project and update
        project = Project.from_dict(project_data)
        project.update(name=name, description=description)

        # Save updated metadata
        metadata[project_id] = project.to_dict()
        self._save_metadata(metadata)

        return project

    def delete_project(self, project_id: str) -> bool:
        """
        Delete project and its directory

        Args:
            project_id: Project identifier

        Returns:
            True if deleted successfully, False if project not found
        """
        metadata = self._load_metadata()

        if project_id not in metadata:
            return False

        # Delete project directory
        project_dir = self.workspace_dir / project_id
        try:
            self.file_handler.delete_directory(project_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to delete project directory: {str(e)}")

        # Remove from metadata
        del metadata[project_id]
        self._save_metadata(metadata)

        return True

    def get_project_path(self, project_id: str) -> Optional[Path]:
        """
        Get absolute path to project directory

        Args:
            project_id: Project identifier

        Returns:
            Path to project directory or None if project doesn't exist
        """
        if self.get_project(project_id):
            return self.workspace_dir / project_id
        return None
