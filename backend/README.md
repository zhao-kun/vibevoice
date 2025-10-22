# VibeVoice Backend

Flask-based REST API backend for VibeVoice speech generation system.

## Architecture

```
backend/
 api/                    # API endpoints (blueprints)
    __init__.py        # Main API blueprint
    projects.py        # Project management endpoints
    speakers.py        # Speaker role endpoints
    generation.py      # Voice generation endpoints
 models/                 # Data models
 services/              # Business logic services
 utils/                 # Utility functions
 app.py                 # Flask application factory
 config.py              # Configuration management
 run.py                 # Development server
 .env.example          # Environment variables template
```

## Setup

### Install Dependencies

The backend shares dependencies with the main VibeVoice project. Install from the project root:

```bash
pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp backend/.env.example backend/.env
```

2. Edit `backend/.env` with your configuration

### Running the Development Server

From the project root:

```bash
python backend/run.py
```

Or using the module:

```bash
python -m backend.run
```

The server will start at `http://localhost:5000` by default.

## API Endpoints

### Health Check

```
GET /health
```

Returns server health status.

### API Base

```
GET /
```

Returns API information.

### Test Endpoint

```
GET /api/v1/ping
```

Simple ping endpoint for testing.

### Projects API

#### List Projects
```
GET /api/v1/projects
```

Returns all projects with metadata.

**Response:**
```json
{
  "projects": [
    {
      "id": "my-project",
      "name": "My Project",
      "description": "Project description",
      "created_at": "2025-10-22T03:18:58.969507",
      "updated_at": "2025-10-22T03:18:58.969507"
    }
  ],
  "count": 1
}
```

#### Get Project
```
GET /api/v1/projects/<project_id>
```

Get specific project by ID.

#### Create Project
```
POST /api/v1/projects
Content-Type: application/json

{
  "name": "Project Name",
  "description": "Optional description"
}
```

Creates a new project directory with subdirectories (`voices/`, `scripts/`, `outputs/`) and adds metadata entry.

**Response:** HTTP 201 with project data

#### Update Project
```
PUT /api/v1/projects/<project_id>
Content-Type: application/json

{
  "name": "Updated Name",
  "description": "Updated description"
}
```

Updates project metadata (name and/or description).

#### Delete Project
```
DELETE /api/v1/projects/<project_id>
```

Deletes project directory and removes from metadata.

**Response:**
```json
{
  "message": "Project deleted successfully",
  "project_id": "my-project"
}
```

## Configuration

Environment variables (see `.env.example`):

- `FLASK_ENV`: Environment (development/production/testing)
- `FLASK_HOST`: Server host (default: 0.0.0.0)
- `FLASK_PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: true)
- `SECRET_KEY`: Flask secret key
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)
- `WORKSPACE_DIR`: Root directory for all projects (default: ./workspace)
- `MODEL_PATH`: Path to VibeVoice model
- `MODEL_DEVICE`: Device for model inference (cuda/cpu)
- `UPLOAD_FOLDER`: Directory for uploaded files
- `MAX_CONTENT_LENGTH`: Maximum file upload size

## Development

### Project Structure

- **api/**: REST API endpoints organized by resource
- **models/**: Data models and schemas
- **services/**: Business logic and model integration
- **utils/**: Helper functions and utilities

### Adding New Endpoints

1. Create a new file in `api/` (e.g., `api/myresource.py`)
2. Define routes using Flask blueprints
3. Import and register the blueprint in `api/__init__.py`

## Project Directory Structure

Each project is stored as a directory under `WORKSPACE_DIR`:

```
workspace/
├── projects.json              # Metadata for all projects
└── my-project/               # Project directory (named by project ID)
    ├── voices/               # Speaker voice samples
    ├── scripts/              # Dialog scripts
    └── outputs/              # Generated audio files
```

## Future Implementation

The following endpoints are planned for implementation:

- **Speakers API**: Manage speaker roles and voice files
- **Generation API**: Speech generation from dialog sessions