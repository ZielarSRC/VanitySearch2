#---------------------------------------------------------------------
# Makefile for VanitySearch2 (Linux)
#
# This Makefile supports:
#  - CPU only build:        make all
#  - GPU build (CUDA):      make gpu=1 all
#  - Override CUDA path:    make gpu=1 CUDA=/usr/local/cuda-12.5 all
#  - Override CCAP (SM):    make gpu=1 CCAP=100 all   # Blackwell/B100 is compute capability 10.0 => 100
#  - Debug build:           make DEBUG=1 [gpu=1] all
#
# Notes:
#  - nvcc often requires an older host compiler than your system default. Use CXXCUDA to set it.

SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp Bech32.cpp Wildcard.cpp

OBJDIR = obj

# Toolchain
CXX      ?= g++
CUDA     ?= /usr/local/cuda
NVCC     ?= $(CUDA)/bin/nvcc
CXXCUDA  ?= $(CXX)   # host compiler used by nvcc (override if needed)

# Compute capability (SM). Examples: 75, 80, 86, 89, 90, 100 ...
# NVIDIA Blackwell (B100/B200) is compute capability 10.0 => use 100.
CCAP     ?= 100

# Common flags
INCLUDES  = -I. -I$(CUDA)/include
WARNINGS  = -Wno-write-strings
CPUFLAGS  = -m64 -mssse3

ifeq ($(DEBUG),1)
  OPTFLAGS = -O0 -g
else
  OPTFLAGS = -O2
endif

# Build with or without GPU
ifdef gpu
  DEFINES  = -DWITHGPU
  LFLAGS   = -lpthread -L$(CUDA)/lib64 -lcudart
else
  DEFINES  =
  LFLAGS   = -lpthread
endif

CXXFLAGS = $(DEFINES) $(CPUFLAGS) $(WARNINGS) $(OPTFLAGS) $(INCLUDES)

# CUDA flags
# Use both SASS (sm_*) and embedded PTX (compute_*) for forward-compatibility.
NVCCFLAGS = $(OPTFLAGS) --compiler-bindir=$(CXXCUDA) -lineinfo -maxrregcount=0 \
            --ptxas-options=-v --use_fast_math \
            -gencode arch=compute_$(CCAP),code=sm_$(CCAP) \
            -gencode arch=compute_$(CCAP),code=compute_$(CCAP)

# Object lists
ifdef gpu
  OBJET = $(addprefix $(OBJDIR)/, \
          Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
          IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
          GPU/GPUEngine.o hash/ripemd160.o hash/sha256.o hash/sha512.o \
          hash/ripemd160_sse.o hash/sha256_sse.o Bech32.o Wildcard.o)
else
  OBJET = $(addprefix $(OBJDIR)/, \
          Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
          IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
          hash/ripemd160.o hash/sha256.o hash/sha512.o \
          hash/ripemd160_sse.o hash/sha256_sse.o Bech32.o Wildcard.o)
endif

# Rules
all: VanitySearch

VanitySearch: $(OBJET)
	@echo Making VanitySearch...
	$(CXX) $(OBJET) $(LFLAGS) -o VanitySearch

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# CUDA compilation
$(OBJDIR)/GPU/GPUEngine.o : GPU/GPUEngine.cu | $(OBJDIR) $(OBJDIR)/GPU
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

# Ensure directories exist
$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p hash

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/hash/*.o
	@rm -f VanitySearch
