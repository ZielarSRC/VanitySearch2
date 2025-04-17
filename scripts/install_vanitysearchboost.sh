#!/bin/bash
# install_vanitysearchboost.sh

# Sprawdzenie zależności
check_dependencies() {
    local missing=()
    for dep in cmake g++ ocl-icd-opencl-dev libssl-dev; do
        if ! dpkg -l | grep -q $dep; then
            missing+=($dep)
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo "Instalowanie wymaganych pakietów: ${missing[*]}"
        sudo apt-get install -y ${missing[@]}
    fi
}

# Kompilacja
build_project() {
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON
    make -j$(nproc)
}

# Instalacja systemowa
install_systemwide() {
    sudo cp VanitySearchBoost /usr/local/bin/
    sudo chmod +x /usr/local/bin/VanitySearchBoost
}

check_dependencies
build_project
install_systemwide

echo "Instalacja zakończona pomyślnie!"
VanitySearchBoost --version
