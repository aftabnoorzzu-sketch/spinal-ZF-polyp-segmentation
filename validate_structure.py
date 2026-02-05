"""
Structure Validation Script

Validates the project structure without running the model.
Useful for CI/CD and pre-deployment checks.
"""

import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False


def validate_structure():
    """Validate project structure"""
    print("=" * 60)
    print("Validating Project Structure")
    print("=" * 60)
    
    all_valid = True
    
    # Required files at root
    print("\n📁 Root Files:")
    all_valid &= check_file_exists("app.py", "Main application")
    all_valid &= check_file_exists("requirements.txt", "Requirements file")
    all_valid &= check_file_exists("README.md", "README")
    
    # Model files
    print("\n📁 Model Module:")
    all_valid &= check_file_exists("models/__init__.py", "Models init")
    all_valid &= check_file_exists("models/spinal_zf_model.py", "Model architecture")
    
    # Utils files
    print("\n📁 Utils Module:")
    all_valid &= check_file_exists("utils/__init__.py", "Utils init")
    all_valid &= check_file_exists("utils/preprocessing.py", "Preprocessing")
    all_valid &= check_file_exists("utils/visualization.py", "Visualization")
    all_valid &= check_file_exists("utils/explainability.py", "Explainability")
    
    # Data module
    print("\n📁 Data Module:")
    all_valid &= check_file_exists("data/__init__.py", "Data init")
    
    # Checkpoints directory
    print("\n📁 Checkpoints Directory:")
    all_valid &= check_file_exists("checkpoints/.gitkeep", "Checkpoints gitkeep")
    
    # Optional files
    print("\n📁 Optional Files:")
    check_file_exists("DEPLOYMENT.md", "Deployment guide")
    check_file_exists("packages.txt", "System packages")
    check_file_exists(".gitignore", "Git ignore")
    
    # Validate requirements.txt content
    print("\n📋 Validating requirements.txt:")
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
            
        required_packages = ["streamlit", "torch", "Pillow", "opencv-python-headless", "numpy", "matplotlib"]
        for pkg in required_packages:
            if pkg in content:
                print(f"✅ Found: {pkg}")
            else:
                print(f"⚠️ Missing recommended package: {pkg}")
        
        # Check for opencv-python (should NOT be there)
        if "opencv-python\n" in content or "opencv-python>" in content:
            print("⚠️ WARNING: Found 'opencv-python' - should use 'opencv-python-headless' for cloud deployment")
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        all_valid = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All required files present!")
        print("Project is ready for deployment.")
    else:
        print("❌ Some required files are missing!")
        print("Please fix the issues before deploying.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    validate_structure()
