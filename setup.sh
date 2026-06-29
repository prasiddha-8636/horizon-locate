#!/usr/bin/env bash
# Exit immediately if any command exits with a non-zero status
set -e

echo "=========================================================="
echo " Starting Installation "
echo "=========================================================="

# 1. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in your PATH."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# 2. Create the master environment from the YAML file
echo -e "\n[1/3] Creating isolated Conda environment 'skyline_env'..."
conda env create -f environment.yml

# 3. Dynamically source Conda to allow environment activation within bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 4. Activate the newly created environment
echo -e "\n[2/3] Activating environment..."
conda activate skyline_env

# 5. Clone and compile HORAYZON C++ bindings locally
echo -e "\n[3/3] Fetching and compiling HORAYZON C++ bindings..."
if [ -d "HORAYZON" ]; then
    echo "HORAYZON folder already exists, updating..."
    cd HORAYZON
    git pull
    cd ..
else
    git clone https://github.com/ChristianSteger/HORAYZON.git
fi

cd HORAYZON
python -m pip install .
cd ..

echo -e "\n=========================================================="
echo " Setup complete!"
echo " "
echo " To activate your environment and begin working, run:"
echo "    conda activate skyline_env"
echo "=========================================================="