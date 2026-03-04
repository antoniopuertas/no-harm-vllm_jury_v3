"""
Test configuration - set up Python path for imports.
"""

import sys
from pathlib import Path

# Add the project root to the path so src can be imported
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
