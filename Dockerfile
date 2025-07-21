# Use an official Ubuntu LTS base
FROM ubuntu:22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install core dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        libfftw3-dev \
        libopenblas-dev \
        liblapack-dev \
        liblapacke-dev \
        libboost-all-dev \
        ca-certificates \
        libgl1 \
        libglfw3 \
        libglfw3-dev \
        libglew-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

        
# Set working directory
WORKDIR /cppgrad

# Copy everything (you can also selectively copy for caching benefits)
COPY . .

# Optional: Set default build type
ARG BUILD_TYPE=Debug

# Create build directory and build cppgrad
RUN mkdir build
WORKDIR /cppgrad/build
RUN cmake .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

RUN cmake --build . --target all -j$(nproc)

WORKDIR /cppgrad
# Default command if someone runs the container
CMD ["/bin/bash"]