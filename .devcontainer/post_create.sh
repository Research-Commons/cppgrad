#!/bin/bash
set -e  # Exit immediately if any command

BUILD_TYPE="${BUILD_TYPE:-Release}"
USE_PREBUILT_ARRAYFIRE="${USE_PREBUILT_ARRAYFIRE:-OFF}"
AF_BACKEND="${AF_BACKEND:-cpu}"
AF_BUILD_EXAMPLES="${AF_BUILD_EXAMPLES:-OFF}"
BUILD_TESTING="${BUILD_TESTING:-OFF}"

git config --global --add safe.directory /cppgrad
git config --global --add safe.directory /cppgrad/third_party/Catch2
git config --global --add safe.directory /cppgrad/third_party/arrayfire

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
  -DAF_BACKEND=${AF_BACKEND} \
  -DBUILD_TESTING=${BUILD_TESTING} \
  -DAF_BUILD_EXAMPLES=${AF_BUILD_EXAMPLES}

# Build the project
cmake --build . -j$(nproc)

echo "[devcontainer] Setup complete."
