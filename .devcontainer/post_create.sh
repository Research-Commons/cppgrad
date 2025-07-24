#!/bin/bash
set -e  # Exit immediately if any command fails

echo "[devcontainer] Starting post-create setup..."

# Initialize submodules if this is a Git repo
if [ -d .git ]; then
  echo "[devcontainer] Initializing git submodules..."
  git submodule update --init --recursive
else
  echo "[devcontainer] Warning: .git directory not found, skipping submodules."
fi

# Create and enter build directory
mkdir -p build
cd build

# Run CMake with environment overrides
cmake .. \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DUSE_PREBUILT_ARRAYFIRE=${USE_PREBUILT_ARRAYFIRE} \
  -DAF_BACKEND=${AF_BACKEND}

# Build the project
cmake --build . -j$(nproc)

echo "[devcontainer] Setup complete."
