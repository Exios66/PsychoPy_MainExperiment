#!/bin/bash
# Installation script for PsychoPy WebGazer Integration

# Print colored messages
print_green() {
    echo -e "\033[0;32m$1\033[0m"
}

print_yellow() {
    echo -e "\033[0;33m$1\033[0m"
}

print_red() {
    echo -e "\033[0;31m$1\033[0m"
}

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_red "Python 3 is not installed. Please install Python 3.9 or later."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_red "Python version $python_version is not supported. Please install Python 3.9 or later."
    exit 1
fi

print_green "Python $python_version detected."

# Create virtual environment
print_yellow "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
print_yellow "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_yellow "Upgrading pip..."
pip install --upgrade pip

# Install setuptools and wheel
print_yellow "Installing setuptools and wheel..."
pip install setuptools wheel

# Install requirements
print_yellow "Installing requirements..."
pip install -r requirements.txt

# Install the package in development mode
print_yellow "Installing the package in development mode..."
pip install -e .

print_green "Installation complete!"
print_yellow "To activate the virtual environment, run:"
echo "source venv/bin/activate"
print_yellow "To run the demo experiment, run:"
echo "python -m PsychoPy.experiments.webgazer_demo"
print_yellow "To run the main application, run:"
echo "python PsychoPy/run_application.py" 