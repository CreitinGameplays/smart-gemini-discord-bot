import subprocess
import os

"""
git tag -a vx.x.x -m "Release version x.x.x"
git push origin x.x.x
"""

def get_version():
    """
    Gets the version from a .version file, then from git tags for local dev.
    Falls back to a default version if all methods fail.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Accommodate both local dev (.../bot/version.py) and Docker (/app/version.py) layouts
    if os.path.basename(script_dir) == 'bot':
        project_root = os.path.dirname(script_dir)
    else:
        project_root = script_dir

    version_file = os.path.join(project_root, '.version')

    # 1. Try to read from .version file (for deployments)
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            version = f.read().strip()
            if version:
                return version

    # 2. Try to get version from git (for local development)
    try:
        git_version = subprocess.check_output(
            ['git', 'describe', '--tags', '--always', '--dirty'],
            cwd=project_root,
            stderr=subprocess.DEVNULL
        ).strip().decode('utf-8')
        return git_version
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 3. Fallback if all else fails
        return "null"

__version__ = get_version()