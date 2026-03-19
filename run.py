#!/usr/bin/env python3
"""DataScribe Launcher"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from utils.config import settings

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.port))
    host = "0.0.0.0" if os.environ.get("PORT") else settings.host
    use_reload = False if os.environ.get("PORT") else settings.reload

    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Server: http://{host}:{port}")
    print("=" * 50)

    uvicorn.run("web.main:app", host=host, port=port, reload=use_reload, log_level="info")
