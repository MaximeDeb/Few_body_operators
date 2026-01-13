cmake_minimum_required(VERSION 3.15)
project(MPO_Simulation_ITensor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# Optimisations
# ============================================================================

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /openmp")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# ============================================================================
# ITensor
# ============================================================================

# Chemin vers ITensor - MODIFIEZ selon votre installation
set(ITENSOR_DIR "C:/itensor" CACHE PATH "Path to ITensor installation")

if(EXISTS "${ITENSOR_DIR}/itensor/all.h")
    message(STATUS "ITensor found at: ${ITENSOR_DIR}")
    include_directories(${ITENSOR_DIR})
    
    # Chercher la bibliothèque ITensor
    find_library(ITENSOR_LIB 
        NAMES itensor
        PATHS ${ITENSOR_DIR}/lib
        NO_DEFAULT_PATH
    )
    
    if(ITENSOR_LIB)
        message(STATUS "ITensor library: ${ITENSOR_LIB}")
    else()
        message(WARNING "ITensor library not found in ${ITENSOR_DIR}/lib")
        message(STATUS "Will try to link anyway (might work if header-only)")
    endif()
    
else()
    message(FATAL_ERROR "
========================================================================
ITensor not found at ${ITENSOR_DIR}!

Please install ITensor:

1. Download:
   git clone https://github.com/ITensor/ITensor ~/itensor

2. Compile (in MSYS2 MinGW terminal):
   cd ~/itensor
   cp options.mk.sample options.mk
   make -j4

3. Then run cmake with:
   cmake .. -DITENSOR_DIR=/c/itensor

Or install via package manager:
   pacman -S mingw-w64-x86_64-itensor  (if available)
========================================================================
    ")
endif()

# ============================================================================
# LAPACK/BLAS (requis par ITensor)
# ============================================================================

if(MSVC)
    # Windows avec Visual Studio
    find_package(LAPACK)
else()
    # MSYS2/MinGW/Linux
    find_package(LAPACK REQUIRED)
    if(LAPACK_FOUND)
        message(STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
    endif()
endif()

# ============================================================================
# Exécutable
# ============================================================================

add_executable(mpo_sim_itensor src/simulation_itensor.cpp)

target_include_directories(mpo_sim_itensor PRIVATE 
    ${ITENSOR_DIR}
)

# Lier les bibliothèques
if(ITENSOR_LIB)
    target_link_libraries(mpo_sim_itensor PRIVATE ${ITENSOR_LIB})
endif()

if(LAPACK_FOUND)
    target_link_libraries(mpo_sim_itensor PRIVATE ${LAPACK_LIBRARIES})
endif()

# Pthread si nécessaire
if(NOT MSVC)
    target_link_libraries(mpo_sim_itensor PRIVATE pthread)
endif()

# ============================================================================
# Options de compilation spécifiques
# ============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(mpo_sim_itensor PRIVATE
        -Wall
        -Wno-unknown-pragmas
        -Wno-unused-variable
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(mpo_sim_itensor PRIVATE
        -Wall
        -Wno-unknown-pragmas
    )
elseif(MSVC)
    target_compile_options(mpo_sim_itensor PRIVATE
        /W3
        /wd4267  # Disable size_t conversion warnings
        /wd4244  # Disable conversion warnings
    )
endif()

# ============================================================================
# Installation (optionnel)
# ============================================================================

install(TARGETS mpo_sim_itensor DESTINATION bin)

# ============================================================================
# Informations de compilation
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "MPO Simulation with ITensor")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "ITensor directory: ${ITENSOR_DIR}")
if(ITENSOR_LIB)
    message(STATUS "ITensor library: ${ITENSOR_LIB}")
else()
    message(STATUS "ITensor library: Not found (header-only mode)")
endif()
if(LAPACK_FOUND)
    message(STATUS "LAPACK: ${LAPACK_LIBRARIES}")
else()
    message(STATUS "LAPACK: Not found")
endif()
message(STATUS "========================================")
message(STATUS "")