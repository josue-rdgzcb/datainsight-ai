# Script to verify installation and versions of libraries listed in requirements-dev.txt

import importlib
import os

# Dictionary to map pip installation names to Python import names
LIB_MAPPING = {
    "scikit-learn": "sklearn",
    "fonttools": "fontTools",
    "pillow": "PIL",
    "python-dateutil": "dateutil",
    "python-dotenv": "dotenv"
}

requirements_file = "requirements.txt"
libraries = []

print("📂 Reading requirements.txt...\n")

# 1. Read and parse the requirements file dynamically
if os.path.exists(requirements_file):
    with open(requirements_file, "r") as file:
        for line in file:
            line = line.strip()
            # Ignore empty lines or comments
            if not line or line.startswith("#"):
                continue
            
            # Split the line at operators like ==, >=, <=, etc.
            # Example: "pandas==2.2.2" becomes "pandas"
            lib_name = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
            
            # Map to the correct import name if necessary
            import_name = LIB_MAPPING.get(lib_name.lower(), lib_name)
            
            if import_name not in libraries:
                libraries.append(import_name)
else:
    print(f"❌ Error: '{requirements_file}' not found.")
    exit(1)

print("🔍 Checking installed libraries...\n")

# 2. Verify each extracted library
for lib in libraries:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "Version keyword not found")
        print(f"✅ {lib}: {version}")
    except ImportError:
        print(f"❌ {lib}: Not installed")

print("\n🏁 Check complete.")