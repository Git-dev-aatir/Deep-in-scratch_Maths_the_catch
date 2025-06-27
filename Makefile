# =======================
#        Makefile
#   Unix / Git Bash / MSYS2 Compatible
# =======================

# Set temporary directory
export TMP := $(HOME)/AppData/Local/Temp
export TEMP := $(TMP)

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude -O2 -MMD -MP

SRC_DIR := src
BUILD_DIR := build

# Find all .cpp files recursively
SRC_FILES := $(shell find $(SRC_DIR) -name '*.cpp' 2>/dev/null)

# Generate object files list
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))

# Default target
all: $(OBJ_FILES)
	@echo "✅ Library built successfully."

# Rule to build each object file
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Auto-include dependency files
-include $(OBJ_FILES:.o=.d)

# Run any example file
run: $(OBJ_FILES)
ifndef FILE
	$(error FILE variable is not set. Usage: make run-example FILE=path/to/example.cpp)
endif
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) "$(FILE)" $(OBJ_FILES) -o $(BUILD_DIR)/example.exe
	@./$(BUILD_DIR)/example.exe

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run-example clean
