#!/usr/bin/env python3
"""
DataScribe Launcher Script

This script launches the DataScribe web application.
Run it from the project root directory.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.main import app
import uvicorn
from utils.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"{settings.app_subtitle}")
    print(f"Server: http://{settings.host}:{settings.port}")
    print(f"Debug mode: {settings.debug}")
    print(f"Upload directory: {settings.upload_dir}")
    print(f"Reports directory: {settings.reports_dir}")
    print("\n" + "="*60)
    
    uvicorn.run(
        "web.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info" if settings.debug else "warning"
    )
