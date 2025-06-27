# C++ Project with Makefile Build System

## Overview
This project provides a robust build system using Make for C++ projects. It features:
- Automated source file discovery
- Dependency tracking
- Cross-platform compatibility (Unix/Git Bash/MSYS2)
- Optimized build configuration

## Build Instructions

### Prerequisites
- GCC/G++ compiler with C++17 support
- Make utility
- Unix-like environment (Linux/macOS) or Windows with:
  - Git Bash, or
  - MSYS2

### Building the Library
```
make
```
- Compiles all source files in `src/` directory
- Generates object files in `build/` directory
- Preserves directory structure from source

### Cleaning Build Artifacts
```
make clean
```
- Removes the entire `build/` directory

### Running Examples
```
make run FILE=path/to/example.cpp
```
- Removes the entire `build/` directory

### Running Examples
```
make run FILE=path/to/example.cpp
```
- Compiles and executes a specific example file
- Example: `make run FILE=Examples/iris_classifier.cpp`

## Project Structure
```
├── include/ # Header files (.h, .hpp)
├── src/ # Source files (.cpp)
├── build/ # Build artifacts (auto-generated)
│ ├── *.o # Object files
│ ├── *.d # Dependency files
│ └── example.exe # Compiled examples
├── Makefile # Build system configuration
└── README.md # Project documentation
```


## Key Features
1. **Smart Dependency Tracking**  
   Automatically generates and includes dependency files (.d)

2. **Optimized Compilation**  
   Uses `-O2` optimization and strict warnings (`-Wall -Wextra`)

3. **Cross-Platform Support**  
   Compatible with:
   - Linux/macOS terminals
   - Git Bash on Windows
   - MSYS2 environments

4. **Structured Builds**  
   Preserves source directory structure in build output

5. **Temporary File Management**  
   Uses system temp directory (`~/AppData/Local/Temp` on Windows)

## Customization
Modify these variables in the Makefile as needed:

```
CXX := g++ # Compiler
CXXFLAGS := -std=c++17 -Wall # Compiler flags
SRC_DIR := src # Source directory
BUILD_DIR := build # Output directory
```


## Troubleshooting
**Common Issues:**
- **"FILE variable not set"**:  
  Specify example path: `make run FILE=path/to/example.cpp`
- **Missing dependencies**:  
  Install build-essential (Linux) or MinGW (Windows)
- **Permission errors**:  
  Run `chmod +x` on scripts if needed

## License
This build system is open-source. Modify and use as needed for your projects.

