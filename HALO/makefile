# ============================================================================
# Makefile for Halo CR Spectra C Library
# ============================================================================
#
# Builds spectra_halo.c into a shared library (spectra_halo.so) that can be
# loaded by Python via ctypes.
#
# Usage:
#   make           # Build the shared library
#   make clean     # Remove build artifacts
#   make test      # Run a simple test
#   make install   # Install to system (requires sudo)
#
# Requirements:
#   - GCC compiler
#   - OpenMP support
#   - GSL library (optional, for full version)
#
# ============================================================================

# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra -fopenmp -std=c11
LDFLAGS = -shared -fopenmp

# Optional: Add GSL if available (uncomment if needed)
# GSL_PREFIX = /usr/local
# CFLAGS += -I$(GSL_PREFIX)/include
# LDFLAGS += -L$(GSL_PREFIX)/lib -lgsl -lgslcblas -lm

# Standard math library
LDFLAGS += -lm

# Target library name
TARGET = spectra_halo.so

# Source files
SOURCES = spectra_halo.c
OBJECTS = $(SOURCES:.c=.o)

# Installation directory
INSTALL_DIR = /usr/local/lib

# ============================================================================
# Build rules
# ============================================================================

.PHONY: all clean test install help

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Linking shared library: $(TARGET)"
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "✓ Build complete: $(TARGET)"
	@echo ""
	@echo "To use in Python:"
	@echo "  from halo_init import HaloCalculator"
	@echo ""

%.o: %.c
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(TARGET)
	@echo "✓ Clean complete"

test: $(TARGET)
	@echo "Running basic library test..."
	@python3 -c "import ctypes; lib = ctypes.CDLL('./$(TARGET)'); lib.print_library_info(); print('✓ Library loads successfully')"

install: $(TARGET)
	@echo "Installing $(TARGET) to $(INSTALL_DIR)..."
	@if [ -w $(INSTALL_DIR) ]; then \
		cp $(TARGET) $(INSTALL_DIR)/; \
		echo "✓ Installed to $(INSTALL_DIR)/$(TARGET)"; \
	else \
		echo "Error: Cannot write to $(INSTALL_DIR). Try 'sudo make install'"; \
		exit 1; \
	fi

help:
	@echo "Halo CR Spectra Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make           - Build the shared library"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make test      - Test that library loads"
	@echo "  make install   - Install to system (may need sudo)"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Compiler: $(CC)"
	@echo "Flags: $(CFLAGS)"
	@echo ""
	@echo "Requirements:"
	@echo "  - GCC with OpenMP support"
	@echo "  - Python 3 with numpy and ctypes"
	@echo ""

# ============================================================================
# Advanced targets
# ============================================================================

# Build with debug symbols
debug: CFLAGS += -g -DDEBUG -O0
debug: clean $(TARGET)

# Build with profiling
profile: CFLAGS += -pg
profile: LDFLAGS += -pg
profile: clean $(TARGET)

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@command -v $(CC) >/dev/null 2>&1 || { echo "Error: GCC not found"; exit 1; }
	@echo "  ✓ GCC found: $$($(CC) --version | head -n1)"
	@$(CC) -fopenmp -xc -E - </dev/null >/dev/null 2>&1 || { echo "Error: OpenMP not supported"; exit 1; }
	@echo "  ✓ OpenMP supported"
	@command -v python3 >/dev/null 2>&1 || { echo "Error: Python3 not found"; exit 1; }
	@echo "  ✓ Python3 found: $$(python3 --version)"
	@python3 -c "import numpy" 2>/dev/null || { echo "Error: NumPy not found"; exit 1; }
	@echo "  ✓ NumPy found"
	@echo ""
	@echo "All dependencies satisfied"

# ============================================================================
# Platform-specific settings
# ============================================================================

# Detect platform
UNAME := $(shell uname -s)

ifeq ($(UNAME), Darwin)
    # macOS
    LDFLAGS += -undefined dynamic_lookup
    TARGET = spectra_halo.dylib
    INSTALL_DIR = /usr/local/lib
endif

ifeq ($(UNAME), Linux)
    # Linux
    LDFLAGS += -Wl,-soname,$(TARGET)
endif

# ============================================================================
# Information targets
# ============================================================================

info:
	@echo "Build Configuration:"
	@echo "  Platform: $(UNAME)"
	@echo "  Compiler: $(CC)"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  LDFLAGS: $(LDFLAGS)"
	@echo "  Target: $(TARGET)"
	@echo "  Install dir: $(INSTALL_DIR)"
	@echo ""
	@echo "Source files:"
	@for src in $(SOURCES); do echo "  - $$src"; done
	@echo ""

# Show estimated build time
benchmark: clean
	@echo "Building with timing..."
	@time $(MAKE) all --no-print-directory

# ============================================================================
# Documentation
# ============================================================================

# Generate function documentation
docs:
	@echo "Extracting function documentation from $(SOURCES)..."
	@grep -A 10 "^/\* ====" $(SOURCES) | sed 's/^/  /' || true
	@echo ""
	@echo "For full documentation, see comments in source files"

# ============================================================================
# Development targets
# ============================================================================

# Format source code (requires clang-format)
format:
	@command -v clang-format >/dev/null 2>&1 || { echo "clang-format not found"; exit 1; }
	clang-format -i $(SOURCES)
	@echo "✓ Code formatted"

# Static analysis (requires cppcheck)
lint:
	@command -v cppcheck >/dev/null 2>&1 || { echo "cppcheck not found"; exit 1; }
	cppcheck --enable=all --suppress=missingIncludeSystem $(SOURCES)

# ============================================================================
# End of Makefile
# ============================================================================