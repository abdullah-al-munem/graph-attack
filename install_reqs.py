import os
import subprocess
import sys

def run_shell_command(command):
    """Run a shell command and handle errors."""
    try:
        print(f"\n>>> Running: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while running: {command}")
        print(f"Exit code: {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False

def check_cuda_availability():
    """Check if CUDA is available on the system."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print(result.stdout.split('\n')[2])  # Show driver version line
            return True
        else:
            print("❌ NVIDIA GPU not detected or drivers not installed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found. CUDA may not be available.")
        return False

def uninstall_torch_ecosystem():
    """Uninstall all PyTorch related packages."""
    print("\n🗑️  Uninstalling existing PyTorch ecosystem...")
    
    # List of packages to uninstall (order matters - dependencies first)
    packages_to_uninstall = [
        "torch-geometric",
        "torch-scatter", 
        "torch-sparse", 
        "torch-cluster", 
        "torch-spline-conv",
        "dgl",
        "torchvision", 
        "torchaudio", 
        "torch"
    ]
    
    for package in packages_to_uninstall:
        print(f"\nUninstalling {package}...")
        run_shell_command(f"pip uninstall {package} -y")
    
    # Clean up any remaining torch files
    print("\nCleaning up remaining PyTorch files...")
    run_shell_command("pip cache purge")

def install_pytorch_24():
    """Install PyTorch 2.4 with CUDA 12.1+ support."""
    print("\n🔧 Installing PyTorch 2.4 with CUDA 12.1+...")
    
    # PyTorch 2.4 with CUDA 12.1 support
    pytorch_install_cmd = (
        "pip install torch==2.4.1 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    
    if not run_shell_command(pytorch_install_cmd):
        print("❌ Failed to install PyTorch. Trying alternative approach...")
        # Fallback to conda if available
        try:
            subprocess.run("conda --version", shell=True, check=True, capture_output=True)
            print("Conda detected, trying conda installation...")
            run_shell_command("conda install pytorch==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y")
        except:
            print("❌ Conda not available. Please install PyTorch manually.")
            return False
    
    return True

def verify_pytorch_installation():
    """Verify PyTorch installation and CUDA availability."""
    print("\n🔍 Verifying PyTorch installation...")
    
    verification_code = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available - CPU only installation")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", verification_code], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return "CUDA available: True" in result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch verification failed: {e.stderr}")
        return False

def install_dgl_compatible():
    """Install DGL compatible with PyTorch 2.4."""
    print("\n🔧 Installing DGL for PyTorch 2.4...")
    
    # Set environment variables for DGL
    os.environ["DGLBACKEND"] = "pytorch"
    os.environ["DGL_NO_GRAPHBOLT"] = "1"
    
    # DGL for PyTorch 2.4 with CUDA 12.1
    dgl_url = "https://data.dgl.ai/wheels/repo.html"
    
    if not run_shell_command(f"pip install dgl==1.1.2 -f {dgl_url}"):
        # Fallback to general DGL installation
        print("Trying general DGL installation...")
        run_shell_command("pip install dgl")

def install_pyg_compatible():
    """Install PyTorch Geometric compatible with PyTorch 2.4."""
    print("\n🔧 Installing PyTorch Geometric for PyTorch 2.4...")
    
    # Set torch version environment variable
    os.environ['TORCH'] = '2.4.1'
    
    # PyG wheel URL for PyTorch 2.4
    pyg_base_url = "https://data.pyg.org/whl/torch-2.4.1+cu121.html"
    
    # Install PyG dependencies
    pyg_packages = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"]
    
    for pkg in pyg_packages:
        print(f"\nInstalling {pkg}...")
        if not run_shell_command(f"pip install {pkg} -f {pyg_base_url}"):
            # Fallback installation
            print(f"Trying fallback installation for {pkg}...")
            run_shell_command(f"pip install {pkg}")
    
    # Install torch-geometric
    print("\nInstalling torch-geometric...")
    run_shell_command("pip install torch-geometric")

def verify_complete_installation():
    """Verify all packages are installed correctly."""
    print("\n🔍 Verifying complete installation...")
    
    verification_code = '''
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    import dgl
    print(f"✅ DGL: {dgl.__version__}")
    
    import torch_geometric
    print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
    
    # Test basic operations
    x = torch.randn(2, 3)
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        print("✅ CUDA tensor operations working")
    
    print("\\n🎉 All packages installed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
'''
    
    try:
        subprocess.run([sys.executable, "-c", verification_code], check=True)
    except subprocess.CalledProcessError:
        print("❌ Verification failed. Some packages may not be installed correctly.")

def main():
    """Main installation process."""
    print("🚀 Starting PyTorch 2.4 + CUDA 12.1+ Setup")
    print("=" * 50)
    
    # Step 0: Check CUDA availability
    cuda_available = check_cuda_availability()
    if not cuda_available:
        response = input("\n⚠️  CUDA not detected. Continue with CPU-only installation? (y/N): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return
    
    # Step 1: Uninstall existing PyTorch ecosystem
    uninstall_torch_ecosystem()
    
    # Step 2: Install PyTorch 2.4
    if not install_pytorch_24():
        print("❌ Failed to install PyTorch. Aborting.")
        return
    
    # Step 3: Verify PyTorch installation
    if not verify_pytorch_installation():
        print("❌ PyTorch verification failed. Aborting.")
        return
    
    # Step 4: Install DGL
    install_dgl_compatible()
    
    # Step 5: Install PyTorch Geometric
    install_pyg_compatible()
    
    # Step 6: Final verification
    verify_complete_installation()
    
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("\nYou can now test your installation with:")
    print('python -c "import torch; print(f\'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\')"')
    print('python -c "import torch_geometric as pyg; print(f\'PyG: {pyg.__version__}\')"')
    print('python -c "import dgl; print(f\'DGL: {dgl.__version__}\')"')

if __name__ == "__main__":
    main()