#!/bin/bash

# Detect OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    IS_WINDOWS=true
    PYTHON_CMD="python"
else
    IS_WINDOWS=false
    PYTHON_CMD="python3"
fi

# Function to check Python version
check_python_version() {
    echo "Checking Python version..."
    python_version=$($PYTHON_CMD --version)
    if [[ $python_version == *"Python 3.10.0"* ]]; then
        echo "✅ Python version is correct: $python_version"
        return 0
    else
        echo "❌ Python version mismatch. Expected 3.10.0, got: $python_version"
        return 1
    fi
}

# Function to check TensorFlow installation
check_tensorflow() {
    echo "Checking TensorFlow installation..."
    $PYTHON_CMD -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
    if [ $? -eq 0 ]; then
        echo "✅ TensorFlow is installed correctly"
        return 0
    else
        echo "❌ TensorFlow installation failed"
        return 1
    fi
}

# Function to check required packages
check_requirements() {
    echo "Checking required packages..."
    $PYTHON_CMD -c "
import sys
required_packages = {
    'torch': '2.0.0',
    'torchaudio': '2.0.0',
    'sounddevice': '0.4.6',
    'librosa': '0.10.0',
    'numpy': '1.24.0',
    'PyQt6': '6.4.0',
    'PyQt6-WebEngine': '6.4.0',
    'scipy': '1.10.0',
    'tensorflow': '2.12.0',
    'tensorflow-hub': '0.13.0',
    'tensorflow-io': '0.31.0',
    'tqdm': '4.65.0',
    'scikit-learn': '1.0.0',
    'geopy': '2.3.0',
    'pyproj': '3.6.0',
    'folium': '0.14.0',
    'branca': '0.6.0'
}

missing_packages = []
version_mismatches = []

for package, required_version in required_packages.items():
    try:
        module = __import__(package)
        installed_version = getattr(module, '__version__', 'unknown')
        if installed_version == 'unknown':
            print(f'⚠️  {package}: version unknown')
        elif installed_version < required_version:
            version_mismatches.append(f'{package} (required: {required_version}, installed: {installed_version})')
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print('❌ Missing packages:')
    for package in missing_packages:
        print(f'  - {package}')
    sys.exit(1)

if version_mismatches:
    print('❌ Version mismatches:')
    for mismatch in version_mismatches:
        print(f'  - {mismatch}')
    sys.exit(1)

print('✅ All required packages are installed with correct versions')
"
    return $?
}

# Function to check GPU availability
check_gpu() {
    echo "Checking GPU availability..."
    $PYTHON_CMD -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('✅ GPU is available:')
    for gpu in gpus:
        print(f'  - {gpu}')
else:
    print('⚠️  No GPU found, running on CPU')
"
    return 0
}

# Function to check environment variables
check_env_vars() {
    echo "Checking environment variables..."
    if [ "$IS_WINDOWS" = true ]; then
        # Windows environment variable check
        python_path=$(echo %PYTHONPATH%)
        tf_log_level=$(echo %TF_CPP_MIN_LOG_LEVEL%)
        tf_gpu_growth=$(echo %TF_FORCE_GPU_ALLOW_GROWTH%)
    else
        # Unix-like environment variable check
        python_path=$PYTHONPATH
        tf_log_level=$TF_CPP_MIN_LOG_LEVEL
        tf_gpu_growth=$TF_FORCE_GPU_ALLOW_GROWTH
    fi

    missing_vars=0
    if [ -z "$python_path" ]; then
        echo "❌ Missing environment variable: PYTHONPATH"
        missing_vars=1
    else
        echo "✅ PYTHONPATH is set"
    fi

    if [ -z "$tf_log_level" ]; then
        echo "❌ Missing environment variable: TF_CPP_MIN_LOG_LEVEL"
        missing_vars=1
    else
        echo "✅ TF_CPP_MIN_LOG_LEVEL is set"
    fi

    if [ -z "$tf_gpu_growth" ]; then
        echo "❌ Missing environment variable: TF_FORCE_GPU_ALLOW_GROWTH"
        missing_vars=1
    else
        echo "✅ TF_FORCE_GPU_ALLOW_GROWTH is set"
    fi

    return $missing_vars
}

# Main health check
echo "Starting health check..."
echo "======================="

# Run all checks
check_python_version
python_status=$?

check_tensorflow
tensorflow_status=$?

check_requirements
requirements_status=$?

check_gpu
gpu_status=$?

check_env_vars
env_status=$?

# Summary
echo "======================="
echo "Health Check Summary:"
echo "Python Version: $([ $python_status -eq 0 ] && echo "✅" || echo "❌")"
echo "TensorFlow: $([ $tensorflow_status -eq 0 ] && echo "✅" || echo "❌")"
echo "Requirements: $([ $requirements_status -eq 0 ] && echo "✅" || echo "❌")"
echo "GPU: $([ $gpu_status -eq 0 ] && echo "✅" || echo "❌")"
echo "Environment: $([ $env_status -eq 0 ] && echo "✅" || echo "❌")"

# Exit with error if any check failed
if [ $python_status -ne 0 ] || [ $tensorflow_status -ne 0 ] || [ $requirements_status -ne 0 ] || [ $gpu_status -ne 0 ] || [ $env_status -ne 0 ]; then
    exit 1
fi

exit 0 