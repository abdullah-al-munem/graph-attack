import os
import subprocess
import torch

# Set the environment variable for the PyTorch version
os.environ['TORCH'] = torch.__version__
print(f"PyTorch version: {torch.__version__}")

# Function to run shell commands
def run_shell_command(command):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {command}")
        print(e)

print(f"pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{os.environ['TORCH']}.html")
# Install torch-scatter
run_shell_command(f"pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{os.environ['TORCH']}.html")

# Install torch-sparse
run_shell_command(f"pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{os.environ['TORCH']}.html")

# Install pytorch-geometric from GitHub
run_shell_command("pip install -q git+https://github.com/pyg-team/pytorch_geometric.git")
