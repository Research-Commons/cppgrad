# cppgrad
The official neural network library in C++
The header files and the source C++ files are stored separately to reduce build time and avoid redundancy.
The header files are the ones which act as your api layer. 
The source files are the ones having the core logic.
Because is C++ a compiled language, it needs to be built to run.


## Build Instructions

>git clone --recurse-submodules https://github.com/Research-Commons/cppgrad.git

### 1. ArrayFire Dependency

`cppgrad` depends on [ArrayFire](https://arrayfire.com/), a high-performance tensor and matrix library. You can use either:

- The **prebuilt ArrayFire binaries**, or
- The **bundled source code** in `third_party/arrayfire`.

> Make sure to change CMakeLists.txt based on your choice

---

### Option 1: Use Prebuilt ArrayFire Binaries 

Follow the official guide:  
➡️ https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire

---

### Option 2: Use Bundled Source (in `third_party/arrayfire`)

This option requires no external ArrayFire installation. Just build `cppgrad` normally, and the bundled `third_party/arrayfire` will be built automatically.


### Prerequisites (Fedora – CPU-only Build)

Install the required system packages using:

```bash
sudo dnf install \
    git cmake gcc-c++ \
    fftw-devel blas-devel lapack-devel \
    libpng-devel hdf5-devel \
    boost-devel glibc-devel glm-devel
```

For full instructions on building ArrayFire from source (including CUDA and Intel-mkl versions):
➡️ https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Linux

---

### Build cppgrad

```bash
cd /path/to/cppgrad
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 # or whatever you want
```

or just use your IDE to build it

---

## VSCode Support

You can use the following configuration files for a smooth development experience with Visual Studio Code. Tune it to your requirements

### `.vscode/launch.json (for debugging cppgrad)`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Tensor Example",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/build/examples/tensor_example",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "gdb",
      "miDebuggerPath": "/usr/bin/gdb",
      "setupCommands": [
        {
          "description": "Enable pretty-printing",
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        }
      ]
    }
  ]
}
```

### `.vscode/tasks.json (Cntrl + Shift + P -> Run tasks -> Select what you want)`

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Build cppgrad",
      "type": "shell",
      "command": "cmake --build build --target all -j",
      "group": "build",
      "problemMatcher": []
    },
    {
      "label": "Run Tensor Example",
      "type": "shell",
      "command": "./build/examples/tensor_example",
      "group": {
        "kind": "none",
        "isDefault": true
      }
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "./build/test/cppgrad_test",
      "group": {
        "kind": "test",
        "isDefault": false
      }
    }
  ]
}
```