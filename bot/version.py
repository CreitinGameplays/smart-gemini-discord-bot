import subprocess
import os

def get_version():
    """
    Gets the version from git tags.
    Falls back to a default version if git is not available or it fails.
    """
    try:
        # The root of the git repository is one level up from the 'bot' directory,
        # where the .git folder is located.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Run git describe to get a version string
        git_version = subprocess.check_output(
            ['git', 'describe', '--tags', '---always', '--dirty'],
            cwd=project_root,
            stderr=subprocess.DEVNULL
        ).strip().decode('utf-8')
        return git_version
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback if git is not installed or not a git repo
        return "v1.0.0-nogit"

__version__ = get_version()