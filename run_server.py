"""
Launcher script for the DSA Lecture Retrieval platform server.
Ensures the working directory and Python paths are properly configured regardless of launch location.
"""

import os
import sys
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    print(f"Starting DSA Lecture Retrieval Platform on {host}:{port} from {PROJECT_ROOT}")
    uvicorn.run("api.main:app", host=host, port=port, app_dir=str(PROJECT_ROOT))
