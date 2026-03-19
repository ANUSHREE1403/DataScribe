#!/usr/bin/env python3
"""
DataScribe Launcher Script

This script launches the DataScribe web application.
Run it from the project root directory.
"""

import sys
import os
import traceback

# Force non-interactive matplotlib backend BEFORE anything imports matplotlib.
os.environ["MPLBACKEND"] = "Agg"

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from utils.config import settings

if __name__ == "__main__":
    try:
        # Use PORT env var on platforms like Render.
        port = int(os.environ.get("PORT", settings.port))
        host = "0.0.0.0" if os.environ.get("PORT") else settings.host

        # Never use autoreload in managed deployments.
        is_production_host = bool(os.environ.get("PORT")) or os.environ.get("RENDER") == "true"
        use_reload = False if is_production_host else settings.reload

        print(f"Starting {settings.app_name} v{settings.app_version}")
        print(f"{settings.app_subtitle}")
        print(f"Server: http://{host}:{port}")
        print(f"Debug mode: {settings.debug}")
        print(f"Auto-reload: {use_reload}")
        print(f"Upload directory: {settings.upload_dir}")
        print(f"Reports directory: {settings.reports_dir}")
        print("\n" + "=" * 60)

        uvicorn.run(
            "web.main:app",
            host=host,
            port=port,
            reload=use_reload,
            log_level="info",
        )
    except Exception as e:
        print(f"Startup failed: {e}")
        print(traceback.format_exc())
        raise
