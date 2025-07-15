import subprocess
import sys
import os
import atexit
import asyncio
from jupyter_client.manager import KernelManager
from queue import Empty

VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jupyter_env")
VENV_PYTHON = os.path.join(VENV_DIR, 'bin', 'python') if sys.platform != 'win32' else os.path.join(VENV_DIR, 'Scripts', 'python.exe')
VENV_PIP = os.path.join(VENV_DIR, 'bin', 'pip') if sys.platform != 'win32' else os.path.join(VENV_DIR, 'Scripts', 'pip.exe')

class JupyterManager:
    def __init__(self):
        self.kernel_manager = None
        self.client = None

    def _run_subprocess(self, command, **kwargs):
        """Runs a subprocess command."""
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True, **kwargs)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {' '.join(command)}")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
            raise

    def setup_environment(self):
        """Creates a virtual environment and installs necessary packages."""
        if not os.path.exists(VENV_DIR):
            print("Creating virtual environment...")
            self._run_subprocess([sys.executable, "-m", "venv", VENV_DIR])
            print("Virtual environment created.")
        
        print("Installing/upgrading jupyter_client and ipykernel...")
        self._run_subprocess([VENV_PIP, "install", "--upgrade", "pip"])
        self._run_subprocess([VENV_PIP, "install", "jupyter_client", "ipykernel", "nest_asyncio"])
        print("Jupyter packages installed.")
        print("Registering kernel spec...")
        self._run_subprocess([VENV_PYTHON, "-m", "ipykernel", "install", "--user", "--name", "python3", "--display-name", "Python 3 (Bot Env)"])
        print("Kernel spec registered.")

    async def install_library(self, library_name: str):
        """Installs a Python library into the virtual environment."""
        print(f"Installing library: {library_name}...")
        loop = asyncio.get_running_loop()
        try:
            # Running pip install can take time, so run in executor
            result = await loop.run_in_executor(None, self._run_subprocess, [VENV_PIP, "install", library_name])
            print(f"Successfully installed {library_name}.")
            return f"Successfully installed `{library_name}`.\n```\n{result.stdout}\n```"
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {library_name}.")
            return f"Failed to install `{library_name}`.\nError:\n```\n{e.stderr}\n```"

    def start_kernel(self):
        """Starts a Jupyter kernel in the virtual environment."""
        if self.kernel_manager and self.kernel_manager.is_alive():
            print("Kernel is already running.")
            return

        print("Starting Jupyter kernel...")
        self.kernel_manager = KernelManager(kernel_name='python3')
        self.kernel_manager.start_kernel()
        self.client = self.kernel_manager.client()
        self.client.start_channels()
        
        try:
            self.client.wait_for_ready(timeout=60)
            print("Kernel is ready.")
        except RuntimeError:
            print("Kernel did not start in time. Shutting down.")
            self.shutdown_kernel()
            raise

    def shutdown_kernel(self):
        """Shuts down the Jupyter kernel."""
        if self.client:
            self.client.stop_channels()
        if self.kernel_manager and self.kernel_manager.is_alive():
            print("Shutting down kernel...")
            self.kernel_manager.shutdown_kernel(now=True)
        print("Kernel shutdown complete.")

    def _execute_sync(self, code: str, timeout: int):
        """The synchronous part of code execution."""
        msg_id = self.client.execute(code)
        output = []

        while True:
            try:
                msg = self.client.get_iopub_msg(timeout=timeout)
            except Empty:
                self.kernel_manager.interrupt_kernel()
                output.append("\n[Execution timed out or no more output]")
                break
            
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            
            msg_type = msg['header']['msg_type']
            content = msg['content']

            if msg_type == 'status' and content['execution_state'] == 'idle':
                break
            elif msg_type == 'stream':
                output.append(content['text'])
            elif msg_type in ('display_data', 'execute_result'):
                if 'text/plain' in content['data']:
                    output.append(content['data']['text/plain'])
            elif msg_type == 'error':
                output.append('\n'.join(content['traceback']))
        
        return "".join(output)

    async def execute_code(self, code: str, timeout=60):
        """Executes code in the Jupyter kernel and returns the output."""
        if not self.client or not self.kernel_manager.is_alive():
            print("Kernel not running. Starting it now.")
            self.start_kernel()

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self._execute_sync, timeout)
            return result if result else "Code executed with no output."
        except Exception as e:
            return f"An error occurred during execution: {e}"

jupyter_manager = JupyterManager()

def setup_jupyter():
    """Initializes the Jupyter environment and kernel."""
    print("Setting up Jupyter environment...")
    jupyter_manager.setup_environment()
    jupyter_manager.start_kernel()
    atexit.register(jupyter_manager.shutdown_kernel)

def get_jupyter_manager():
    return jupyter_manager