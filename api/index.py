import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app

# Vercel serverless function entrypoint
if __name__ == "__main__":
    app.run()
