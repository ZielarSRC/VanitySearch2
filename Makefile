# Konfiguracja kompilatora
CXX := g++
NVCC := nvcc
CFLAGS := -O3 -march=native -std=c++17 -Wall -Wextra
LDFLAGS := -lm
CUDA_FLAGS := -arch=sm_86 -Xcompiler "-O3 -march=native"
CUDA_LIBS := -lcudart

# Pliki źródłowe
SRC_DIR := .
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(SRC:.cpp=.o)
DEP := $(OBJ:.o=.d)

# Cel domyślny
all: VanitySearch

# Kompilacja wersji CPU
VanitySearch: $(OBJ)
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Kompilacja zakończona powodzeniem! Uruchom: ./VanitySearch"

# Kompilacja wersji GPU
VanitySearchCUDA: CFLAGS += -DUSE_CUDA
VanitySearchCUDA: $(OBJ) GPUEngine.cu.o
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(CUDA_LIBS)
	@echo "Kompilacja CUDA zakończona! Uruchom: ./VanitySearchCUDA"

# Reguła dla plików CUDA
%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

# Generowanie zależności
%.o: %.cpp
	$(CXX) $(CFLAGS) -MMD -MP -c $< -o $@

# Czyszczenie
clean:
	rm -f VanitySearch VanitySearchCUDA $(OBJ) $(DEP) *.cu.o

# Instalacja
install:
	cp VanitySearch /usr/local/bin/vanitysearch

# Sprawdzenie CUDA
check-cuda:
	@which nvcc >/dev/null || (echo "Błąd: Zainstaluj CUDA Toolkit!"; exit 1)

# Dołącz zależności
-include $(DEP)

.PHONY: all clean install check-cuda