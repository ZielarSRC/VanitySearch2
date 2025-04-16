#!/bin/bash
set -euo pipefail

# Konfiguracja
REPO_URL="https://github.com/ZielarSRC/VanitySearch2.git"
INSTALL_DIR="/opt/vanitysearch"
CONFIG_DIR="$HOME/.vanitysearch"
MIN_DISK=20 # GB
MIN_RAM=8 # GB
CUDA_MIN=11.6

# Kolory
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Sprawdź uprawnienia
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}Uruchom skrypt z uprawnieniami sudo!${NC}"
        exit 1
    fi
}

# Nagłówek
header() {
    clear
    echo -e "${GREEN}"
    echo " ██▒   █▓ ▄▄▄       ██▓     ██░ ██  ▄▄▄       ██▓███  "
    echo "▓██░   █▒▒████▄    ▓██▒    ▓██░ ██▒▒████▄    ▓██░  ██▒"
    echo " ▓██  █▒░▒██  ▀█▄  ▒██░    ▒██▀▀██░▒██  ▀█▄  ▓██░ ██▓▒"
    echo "  ▒██ █░░░██▄▄▄▄██ ▒██░    ░▓█ ░██ ░██▄▄▄▄██ ▒██▄█▓▒ ▒"
    echo "   ▒▀█░   ▓█   ▓██▒░██████▒░▓█▒░██▓ ▓█   ▓██▒▒██▒ ░  ░"
    echo "   ░ ▐░   ▒▒   ▓▒█░░ ▒░▓  ░ ▒ ░░▒░▒ ▒▒   ▓▒█░▒▓▒░ ░  ░"
    echo "   ░ ░░    ▒   ▒▒ ░░ ░ ▒  ░ ▒ ░▒░ ░  ▒   ▒▒ ░░▒ ░     "
    echo "     ░░    ░   ▒     ░ ░    ░  ░░ ░  ░   ▒   ░░       "
    echo "      ░        ░  ░    ░  ░ ░  ░  ░      ░  ░         "
    echo -e "${NC}"
}

# Sprawdź wymagania systemowe
check_requirements() {
    echo -e "${YELLOW}[*] Sprawdzanie wymagań systemowych...${NC}"
    
    # Dysk
    local disk=$(df -BG --output=avail / | tail -1 | tr -d 'G ')
    if [ "$disk" -lt "$MIN_DISK" ]; then
        echo -e "${RED} Wymagane minimum ${MIN_DISK}GB wolnego miejsca!${NC}"
        exit 1
    fi

    # RAM
    local ram=$(free -g | awk '/Mem/{print $2}')
    if [ "$ram" -lt "$MIN_RAM" ]; then
        echo -e "${RED} Wymagane minimum ${MIN_RAM}GB RAM!${NC}"
        exit 1
    fi

    # CUDA
    if nvidia-smi &> /dev/null; then
        local cuda_ver=$(nvcc --version | awk '/release/{print $6}')
        if awk "BEGIN {exit !($cuda_ver >= $CUDA_MIN)}"; then
            echo -e "${GREEN} Wykryto CUDA $cuda_ver${NC}"
            CUDA_ENABLED=true
        else
            echo -e "${RED} Wymagane CUDA >= $CUDA_MIN!${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW} Brak wsparcia GPU - instalacja trybu CPU${NC}"
        CUDA_ENABLED=false
    fi
}

# Instalacja zależności
install_deps() {
    echo -e "${YELLOW}[*] Instalowanie zależności...${NC}"
    
    apt update
    apt install -y \
        build-essential \
        cmake \
        libboost-all-dev \
        libssl-dev \
        libncurses5-dev \
        ocl-icd-opencl-dev \
        nvidia-opencl-dev \
        nvidia-cuda-toolkit \
        git \
        python3-pip
        
    pip3 install psutil
}

# Kompilacja
compile_app() {
    echo -e "${YELLOW}[*] Kompilowanie aplikacji...${NC}"
    
    local build_flags=(-DUSE_AVX2=ON -DUSE_OPENMP=ON)
    
    if [ "$CUDA_ENABLED" = true ]; then
        build_flags+=(-DUSE_CUDA=ON)
    fi

    mkdir -p "$INSTALL_DIR/build"
    cd "$INSTALL_DIR/build"
    cmake "${build_flags[@]}" ..
    make -j$(nproc)
    
    ln -sf "$INSTALL_DIR/build/VanitySearch" /usr/local/bin/vanitysearch
}

# Konfiguracja
setup_config() {
    echo -e "${YELLOW}[*] Tworzenie konfiguracji...${NC}"
    
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/config.json" <<EOF
{
    "mode": "hybrid",
    "gpu_enabled": $CUDA_ENABLED,
    "cpu_threads": $(nproc),
    "gpu_devices": []
}
EOF
}

# Konfiguracja Docker (opcjonalnie)
setup_docker() {
    echo -e "${YELLOW}[*] Konfiguracja Docker...${NC}"
    
    if ! command -v docker &> /dev/null; then
        apt install -y docker.io
        systemctl enable --now docker
    fi

    docker build -t vanitysearch "$INSTALL_DIR"
}

# Główna funkcja
main() {
    check_root
    header
    check_requirements
    install_deps
    
    echo -e "${YELLOW}[*] Klonowanie repozytorium...${NC}"
    git clone "$REPO_URL" "$INSTALL_DIR"
    
    compile_app
    setup_config
    
    echo -e "${GREEN}[✔] Instalacja zakończona!${NC}"
    echo -e "Uruchom aplikację komendą: ${YELLOW}vanitysearch${NC}"
}

main
