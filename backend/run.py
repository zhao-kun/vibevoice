#!/usr/bin/env python3
"""
Development server for VibeVoice backend

Usage:
    python backend/run.py
    or
    python -m backend.run
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app import create_app  # noqa: E402

def main():
    """Run the development server"""
    # Set environment for development
    os.environ.setdefault('FLASK_ENV', 'development')

    # Create application
    app = create_app('development')

    # Get configuration
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print(f"""
    TPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPW
    Q       VibeVoice Backend Server             Q
    `PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPc
    Q  Environment: Development                  Q
    Q  Server:      http://{host}:{port:<15}  Q
    Q  Debug mode:  {'Enabled' if debug else 'Disabled':<24}  Q
    ZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]
    """)

    # Run development server
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=True
    )


if __name__ == '__main__':
    main()
