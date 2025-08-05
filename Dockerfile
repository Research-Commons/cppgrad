# Use an official Ubuntu LTS base
FROM ubuntu:22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install core dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gdb \
        git \
        wget \
        curl \
        libfftw3-dev \
        libopenblas-dev \
        liblapack-dev \
        liblapacke-dev \
        libboost-all-dev \
        libunwind8 \
        libfreeimage-dev \
        graphviz \
        ca-certificates \
        libgl1 \
        libglfw3 \
        libglfw3-dev \
        libglew-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

        
# Set working directory
WORKDIR /cppgrad

# Copy everything
COPY . .


#-----------------------------------------
#Optional Commands (These get overriden if you are using a devcontainer, so just set them directly there)

# Build only if ure not running it from the devcontainer. Saves time as you would have to rebuild it anyways if using a devcontainer
ARG IS_DEVCONTAINER=OFF

# Set default build type
ARG BUILD_TYPE=Release

# ON or OFF
ARG USE_PREBUILT_ARRAYFIRE=OFF

# cpu, cuda, or metal
ARG AF_BACKEND=cpu

#ON or OFF
ARG AF_BUILD_EXAMPLES=OFF

#ON or OFF
ARG BUILD_TESTING=OFF

# Set env variables to be used later in post_create.sh for devcontainers
ENV BUILD_TYPE=${BUILD_TYPE}
ENV USE_PREBUILT_ARRAYFIRE=${USE_PREBUILT_ARRAYFIRE}
ENV AF_BACKEND=${AF_BACKEND}
ENV AF_BUILD_EXAMPLES=${AF_BUILD_EXAMPLES}
ENV BUILD_TESTING=${BUILD_TESTING}
#-----------------------------------------

# Create build directory and run cmake **only if not in a devcontainer**
RUN if [ "$IS_DEVCONTAINER" = "OFF" ]; then \
      echo "Building cppgrad (Docker build)"; \
      mkdir -p /cppgrad/build && \
      cd /cppgrad/build && \
      cmake .. \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DUSE_PREBUILT_ARRAYFIRE=${USE_PREBUILT_ARRAYFIRE} \
        -DAF_BACKEND=${AF_BACKEND} && \
        -DBUILD_TESTING=${BUILD_TESTING} \
        -DAF_BUILD_EXAMPLES=${AF_BUILD_EXAMPLES} && \
      cmake --build . --target all -j$(nproc); \
    else \
      echo "Skipping build (inside devcontainer)"; \
    fi

WORKDIR /cppgrad
# Default command if someone runs the container
ENTRYPOINT ["/bin/bash"]