import os
import subprocess


def run_shell_command(command):
    """Run a shell command and handle errors."""
    try:
        print(f"\n>>> Running: {command}")
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while running: {command}")
        print(e)


# Step 1: Uninstall torch-related packages
run_shell_command("pip uninstall -y torch torchvision torchaudio")

# Step 2: Install DGL (adjust URL if CUDA version changes)
run_shell_command("pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html")

# # Step 3: Reinstall torch, torchvision, torchaudio (matching CUDA 12.1/12.4 builds)
# run_shell_command("pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 "
#                   "-f https://download.pytorch.org/whl/torch_stable.html")

# Step 4: Install PyTorch Geometric dependencies
import torch

torch_version = torch.__version__
print(f"\n✅ Using PyTorch {torch_version} (CUDA {torch.version.cuda})")

pyg_base_url = f"https://data.pyg.org/whl/torch-{torch_version}.html"

for pkg in ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"]:
    run_shell_command(f"pip install {pkg} -f {pyg_base_url}")

# Finally install torch-geometric
run_shell_command("pip install torch-geometric")

print("\n🎉 Setup complete! Test with: `python -c \"import torch_geometric as pyg; print(pyg.__version__)\"`")
