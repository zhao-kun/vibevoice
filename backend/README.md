# VibeVoice Backend

Flask-based REST API backend for VibeVoice speech generation system.

## Architecture

```
backend/
├── api/                    # API endpoints (blueprints)
│   ├── __init__.py        # Main API blueprint
│   ├── projects.py        # Project management endpoints
│   ├── speakers.py        # Speaker role endpoints
│   ├── dialog_sessions.py # Dialog session endpoints
│   └── generation.py      # Voice generation endpoints (TBD)
├── models/                 # Data models
│   ├── project.py         # Project dataclass
│   ├── speaker.py         # SpeakerRole dataclass
│   └── dialog_session.py  # DialogSession dataclass
├── services/              # Business logic services
│   ├── project_service.py
│   ├── speaker_service.py
│   └── dialog_session_service.py
├── utils/                 # Utility functions
│   ├── file_handler.py
│   └── dialog_validator.py
├── app.py                 # Flask application factory
├── config.py              # Configuration management
├── run.py                 # Development server
└── .env.example          # Environment variables template
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

The server will start at `http://localhost:9527` by default.

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
- `FLASK_PORT`: Server port (default: 9527)
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
    │   ├── speakers.json    # Speaker metadata
    │   └── *.wav            # Voice sample files
    ├── scripts/              # Dialog scripts
    │   ├── sessions.json    # Dialog session metadata
    │   └── *.txt            # Dialog text files
    └── outputs/              # Generated audio files (TBD)
```

### Speakers API

#### List Speakers
```
GET /api/v1/projects/<project_id>/speakers
```

Returns all speaker roles for a project.

**Response:**
```json
{
  "speakers": [
    {
      "speaker_id": "Speaker 1",
      "name": "Alice",
      "description": "Main host",
      "voice_filename": "abc123.wav",
      "created_at": "2025-10-22T...",
      "updated_at": "2025-10-22T..."
    }
  ],
  "count": 1
}
```

#### Get Speaker
```
GET /api/v1/projects/<project_id>/speakers/<speaker_id>
```

Get specific speaker by ID (e.g., "Speaker 1").

#### Add Speaker
```
POST /api/v1/projects/<project_id>/speakers
Content-Type: multipart/form-data

name: Speaker Name
description: Speaker description (optional)
voice_file: <audio file> (.wav, .mp3, .m4a, .flac)
```

Creates a new speaker role with voice file upload. Speaker ID is automatically generated as "Speaker N" where N is sequential starting from 1.

**Response:** HTTP 201 with speaker data

#### Update Speaker
```
PUT /api/v1/projects/<project_id>/speakers/<speaker_id>
Content-Type: application/json

{
  "name": "Updated Name",
  "description": "Updated description"
}
```

Updates speaker metadata (name and/or description). Voice file cannot be changed after creation.

#### Delete Speaker
```
DELETE /api/v1/projects/<project_id>/speakers/<speaker_id>
```

Deletes speaker role and its voice file. Speaker IDs are automatically reindexed to maintain continuity (e.g., deleting "Speaker 1" renames "Speaker 2" to "Speaker 1").

**Response:**
```json
{
  "message": "Speaker deleted successfully. Speaker IDs have been reindexed.",
  "speaker_id": "Speaker 1"
}
```

#### Download Voice File
```
GET /api/v1/projects/<project_id>/speakers/<speaker_id>/voice
```

Download speaker's voice sample file.

### Dialog Sessions API

#### List Dialog Sessions
```
GET /api/v1/projects/<project_id>/sessions
```

Returns all dialog sessions for a project.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "uuid",
      "name": "Episode 1",
      "description": "First episode",
      "text_filename": "uuid.txt",
      "created_at": "2025-10-22T...",
      "updated_at": "2025-10-22T..."
    }
  ],
  "count": 1
}
```

#### Get Dialog Session
```
GET /api/v1/projects/<project_id>/sessions/<session_id>
```

Get specific dialog session by ID.

#### Create Dialog Session
```
POST /api/v1/projects/<project_id>/sessions
Content-Type: application/json

{
  "name": "Session Name",
  "description": "Session description",
  "dialog_text": "Speaker 1: Hello\n\nSpeaker 2: Hi there!"
}
```

Creates a new dialog session with text content. The dialog text must follow the format:
- `Speaker N: dialog text` (where N is a positive integer)
- Empty line between dialog entries
- Speaker IDs must exist in the speaker management system

The API validates:
1. Dialog text format (correct pattern)
2. Speaker IDs exist in the project

**Response:** HTTP 201 with session data

**Error Response (validation failed):**
```json
{
  "error": "Validation Error",
  "message": "Speaker ID validation failed: Invalid speaker IDs found: Speaker 99"
}
```

#### Update Dialog Session
```
PUT /api/v1/projects/<project_id>/sessions/<session_id>
Content-Type: application/json

{
  "name": "Updated Name",
  "description": "Updated description",
  "dialog_text": "Speaker 1: Updated content"
}
```

Updates session metadata and/or dialog text content. Same validation rules apply for dialog_text.

#### Delete Dialog Session
```
DELETE /api/v1/projects/<project_id>/sessions/<session_id>
```

Deletes dialog session and its text file.

**Response:**
```json
{
  "message": "Session deleted successfully",
  "session_id": "uuid"
}
```

#### Get Session Text
```
GET /api/v1/projects/<project_id>/sessions/<session_id>/text
```

Get dialog text content for a session.

**Response:**
```json
{
  "session_id": "uuid",
  "dialog_text": "Speaker 1: Hello\n\nSpeaker 2: Hi!"
}
```

#### Download Session Text File
```
GET /api/v1/projects/<project_id>/sessions/<session_id>/download
```

Download dialog text file.

## Dialog Text Format

Dialog sessions must follow this specific format (matching `demo/text_examples/*.txt`):

```
Speaker 1: First line of dialog from speaker 1

Speaker 2: Response from speaker 2

Speaker 1: Another line from speaker 1

Speaker 3: A third speaker joins
```

**Format rules:**
- Pattern: `Speaker N: dialog text` where N is a positive integer
- Empty line separates each dialog entry
- Speaker can appear multiple times in any order
- Speaker IDs must match existing speakers in the project
- Dialog text cannot be empty

**Example:**
```
Speaker 1: Hello and welcome to our show.

Speaker 2: Thanks for having me on today.

Speaker 1: Let's dive right in. What brings you here?

Speaker 2: I wanted to discuss our new project.
```

## Future Implementation

The following endpoints are planned for implementation:

- **Generation API**: Speech generation from dialog sessions