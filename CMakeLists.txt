cmake_minimum_required(VERSION 3.20)
project(VanitySearchBoost VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optymalizacje dla CPU
if(MSVC)
    add_compile_options(/O2 /Oi /GL /arch:AVX2 /fp:fast)
    add_link_options(/LTCG)
else()
    add_compile_options(-O3 -march=native -mtune=native -fopenmp -mavx2 -maes -mpclmul -flto)
    add_link_options(-flto -fuse-ld=gold)
endif()

# Opcja GPU
option(WITH_GPU "Build with GPU support" ON)

if(WITH_GPU)
    find_package(OpenCL REQUIRED)
    add_definitions(-DWITH_GPU)
endif()

# Pliki źródłowe
set(SOURCES
    src/main.cpp
    src/SECP256k1.cpp
    src/AVX256_SHA256.cpp
    src/WorkStealingScheduler.cpp
    src/PuzzleOptimizer.cpp
)

if(WITH_GPU)
    list(APPEND SOURCES
        src/GPUEngine.cpp
        src/GPUMemoryManager.cpp
    )
endif()

# Executable
add_executable(VanitySearchBoost ${SOURCES})

if(WITH_GPU)
    target_include_directories(VanitySearchBoost PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(VanitySearchBoost PRIVATE ${OpenCL_LIBRARIES})
endif()

# Installer (Linux)
if(UNIX AND NOT APPLE)
    include(GNUInstallDirs)
    install(TARGETS VanitySearchBoost
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    
    install(FILES scripts/vanitysearchboost.sh
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    )
    
    include(CPack)
    set(CPACK_GENERATOR "DEB;RPM")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Your Name")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libssl-dev, ocl-icd-opencl-dev")
    set(CPACK_RPM_PACKAGE_REQUIRES "openssl-devel, ocl-icd")
endif()
