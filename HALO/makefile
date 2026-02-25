# ============================================================================
# Makefile for Halo CR Spectra - FULL PHYSICS VERSION
# ============================================================================
#
# This version includes complete GSL integration and links against your
# original physics modules for exact numerical accuracy.
#
# Prerequisites:
#   1. GSL library (libgsl-dev)
#   2. Your original code headers and compiled objects
#   3. OpenMP support
#
# Usage:
#   make                 # Build full physics version
#   make simplified      # Build lightweight version (no GSL)
#   make clean          # Remove build artifacts
#
# ============================================================================

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Location of your original code
ORIG_CODE_DIR = ../original_code

# Subdirectories in your original code
ORIG_INCLUDE = $(ORIG_CODE_DIR)/include
ORIG_SRC = $(ORIG_CODE_DIR)/c_core

# GSL installation (auto-detect or specify manually)
GSL_PREFIX = /usr/local
# GSL_PREFIX = /opt/homebrew  # For Apple Silicon Macs

# ============================================================================
# COMPILER AND FLAGS
# ============================================================================

CC = gcc
CXX = g++

# Base flags
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra -fopenmp -std=c11

# Include paths
CFLAGS += -I.
CFLAGS += -I$(ORIG_INCLUDE)
CFLAGS += $(shell gsl-config --cflags 2>/dev/null || echo "-I$(GSL_PREFIX)/include")

# Linker flags
LDFLAGS = -shared -fopenmp
LDFLAGS += $(shell gsl-config --libs 2>/dev/null || echo "-L$(GSL_PREFIX)/lib -lgsl -lgslcblas")
LDFLAGS += -lm

# ============================================================================
# SOURCE FILES
# ============================================================================

# Your halo-specific code
HALO_SOURCES = spectra_halo_full.c

# Original physics modules (compiled as needed)
ORIG_SOURCES = \
	$(ORIG_SRC)/CRe_steadystate.c \
	$(ORIG_SRC)/ionisation.c \
	$(ORIG_SRC)/synchrotron.c \
	$(ORIG_SRC)/bremsstrahlung.c \
	$(ORIG_SRC)/diffusion.c \
	$(ORIG_SRC)/inverse_Compton.c \
	$(ORIG_SRC)/spectra_funcs.c

# Check which source files actually exist
ORIG_SOURCES_EXIST = $(wildcard $(ORIG_SOURCES))

# Object files
HALO_OBJECTS = $(HALO_SOURCES:.c=.o)
ORIG_OBJECTS = $(ORIG_SOURCES_EXIST:.c=.o)

# Target library
TARGET = spectra_halo.so

# Simplified version (no GSL, no original code)
SIMPLE_TARGET = spectra_halo_simple.so
SIMPLE_SOURCE = spectra_halo.c

# ============================================================================
# BUILD RULES
# ============================================================================

.PHONY: all full simplified clean test install help check-deps info

# Default target: full physics version
all: full

# Full physics version with GSL and original code
full: check-deps $(TARGET)
	@echo ""
	@echo "✓ Full physics version built: $(TARGET)"
	@echo ""
	@echo "This version includes:"
	@echo "  - Complete GSL integration"
	@echo "  - Original CRe_steadystate_solve()"
	@echo "  - Exact loss timescales (plasma, sync, IC, BS)"
	@echo "  - Same physics as spectra.c"
	@echo ""

$(TARGET): $(HALO_OBJECTS) $(ORIG_OBJECTS)
	@echo "Linking full physics library..."
	$(CC) $(LDFLAGS) -o $@ $^

# Compile halo-specific code
%.o: %.c
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Compile original code modules
$(ORIG_SRC)/%.o: $(ORIG_SRC)/%.c
	@echo "Compiling original module: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Simplified version (lightweight, no dependencies)
simplified: $(SIMPLE_TARGET)
	@echo ""
	@echo "✓ Simplified version built: $(SIMPLE_TARGET)"
	@echo ""
	@echo "This version:"
	@echo "  - No GSL required"
	@echo "  - No original code dependencies"
	@echo "  - ~80-90% accuracy"
	@echo "  - Good for testing/prototyping"
	@echo ""

$(SIMPLE_TARGET): $(SIMPLE_SOURCE)
	@echo "Building simplified version..."
	$(CC) -O3 -march=native -fPIC -Wall -fopenmp -std=c11 \
		-shared -fopenmp -lm -o $@ $<

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

test: $(TARGET)
	@echo "Testing library..."
	@python3 -c "\
import ctypes; \
lib = ctypes.CDLL('./$(TARGET)'); \
lib.print_library_info(); \
print('✓ Library loads successfully')" || \
	(echo "Error: Library test failed"; exit 1)

# Validate that all required headers exist
check-deps:
	@echo "Checking dependencies..."
	@command -v $(CC) >/dev/null 2>&1 || \
		{ echo "Error: GCC not found"; exit 1; }
	@echo "  ✓ GCC found: $$($(CC) --version | head -n1)"
	@$(CC) -fopenmp -xc -E - </dev/null >/dev/null 2>&1 || \
		{ echo "Error: OpenMP not supported"; exit 1; }
	@echo "  ✓ OpenMP supported"
	@command -v gsl-config >/dev/null 2>&1 || \
		{ echo "Error: GSL not found. Install with:"; \
		  echo "  Ubuntu: sudo apt-get install libgsl-dev"; \
		  echo "  macOS: brew install gsl"; exit 1; }
	@echo "  ✓ GSL found: $$(gsl-config --version)"
	@test -f "$(ORIG_INCLUDE)/CRe_steadystate.h" || \
		{ echo "Error: Original headers not found at $(ORIG_INCLUDE)"; \
		  echo "Update ORIG_CODE_DIR in Makefile"; exit 1; }
	@echo "  ✓ Original headers found"
	@test -f "gsl_decs.h" || \
		{ echo "Error: gsl_decs.h not found in current directory"; exit 1; }
	@echo "  ✓ gsl_decs.h found"
	@command -v python3 >/dev/null 2>&1 || \
		{ echo "Warning: Python3 not found (needed for ctypes interface)"; }
	@echo ""

# ============================================================================
# INSTALLATION
# ============================================================================

INSTALL_DIR = /usr/local/lib

install: $(TARGET)
	@echo "Installing $(TARGET) to $(INSTALL_DIR)..."
	@if [ -w $(INSTALL_DIR) ]; then \
		cp $(TARGET) $(INSTALL_DIR)/; \
		echo "✓ Installed to $(INSTALL_DIR)/$(TARGET)"; \
	else \
		echo "Error: Cannot write to $(INSTALL_DIR)."; \
		echo "Try: sudo make install"; \
		exit 1; \
	fi

# ============================================================================
# CLEANUP
# ============================================================================

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(HALO_OBJECTS) $(ORIG_OBJECTS)
	rm -f $(TARGET) $(SIMPLE_TARGET)
	rm -f *.o
	@echo "✓ Clean complete"

# Deep clean (including backup files)
distclean: clean
	rm -f *~ *.bak
	rm -f .*.swp

# ============================================================================
# INFORMATION AND HELP
# ============================================================================

help:
	@echo "Halo CR Spectra Makefile - Full Physics Version"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build full physics version (default)"
	@echo "  make full         - Build full physics version"
	@echo "  make simplified   - Build lightweight version (no GSL)"
	@echo "  make test         - Test library loading"
	@echo "  make check-deps   - Verify all dependencies"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make install      - Install to system (may need sudo)"
	@echo "  make info         - Show build configuration"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Configuration:"
	@echo "  ORIG_CODE_DIR = $(ORIG_CODE_DIR)"
	@echo "  GSL_PREFIX = $(GSL_PREFIX)"
	@echo ""
	@echo "Requirements:"
	@echo "  - GCC with OpenMP"
	@echo "  - GSL library"
	@echo "  - Your original code headers"
	@echo "  - Python 3 with numpy and ctypes"
	@echo ""

info:
	@echo "Build Configuration:"
	@echo "  Platform: $$(uname -s)"
	@echo "  Compiler: $(CC)"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  LDFLAGS: $(LDFLAGS)"
	@echo "  Target: $(TARGET)"
	@echo ""
	@echo "Source files:"
	@echo "  Halo: $(HALO_SOURCES)"
	@echo "  Original (found): $(ORIG_SOURCES_EXIST)"
	@echo ""
	@echo "Paths:"
	@echo "  Original code: $(ORIG_CODE_DIR)"
	@echo "  Original headers: $(ORIG_INCLUDE)"
	@echo "  Original source: $(ORIG_SRC)"
	@echo "  GSL: $(GSL_PREFIX)"
	@echo ""

# Show which physics modules are available
show-modules:
	@echo "Available original physics modules:"
	@for src in $(ORIG_SOURCES); do \
		if [ -f "$$src" ]; then \
			echo "  ✓ $$src"; \
		else \
			echo "  ✗ $$src (missing)"; \
		fi; \
	done

# ============================================================================
# ADVANCED TARGETS
# ============================================================================

# Build with debug symbols
debug: CFLAGS += -g -DDEBUG -O0
debug: clean $(TARGET)

# Build with verbose output
verbose: CFLAGS += -v
verbose: clean $(TARGET)

# Build with maximum warnings
strict: CFLAGS += -Werror -pedantic
strict: clean $(TARGET)

# Profile-guided optimization (2-step process)
pgo-generate: CFLAGS += -fprofile-generate
pgo-generate: LDFLAGS += -fprofile-generate
pgo-generate: clean $(TARGET)

pgo-use: CFLAGS += -fprofile-use
pgo-use: LDFLAGS += -fprofile-use
pgo-use: $(TARGET)

# ============================================================================
# PLATFORM-SPECIFIC SETTINGS
# ============================================================================

UNAME := $(shell uname -s)

ifeq ($(UNAME), Darwin)
    # macOS
    LDFLAGS += -undefined dynamic_lookup
    TARGET_SUFFIX = .dylib
    INSTALL_DIR = /usr/local/lib
endif

ifeq ($(UNAME), Linux)
    # Linux
    LDFLAGS += -Wl,-soname,$(TARGET)
    TARGET_SUFFIX = .so
endif

# ============================================================================
# DOCUMENTATION
# ============================================================================

# Quick reference card
quickref:
	@echo "Quick Reference:"
	@echo ""
	@echo "1. First time setup:"
	@echo "   Edit Makefile: set ORIG_CODE_DIR to your original code location"
	@echo "   make check-deps"
	@echo ""
	@echo "2. Build:"
	@echo "   make              # Full physics"
	@echo "   make simplified   # Lightweight (no GSL)"
	@echo ""
	@echo "3. Use in Python:"
	@echo "   python3 halo_init.py galaxies.txt disc_dir/ output_dir/"
	@echo ""
	@echo "4. Troubleshooting:"
	@echo "   make check-deps   # Verify dependencies"
	@echo "   make test         # Test library loads"
	@echo "   make info         # Show configuration"
	@echo ""

# ============================================================================
# End of Makefile
# ============================================================================