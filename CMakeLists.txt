cmake_minimum_required(VERSION 3.21.0)

project(BASIC_ML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library for ML components
add_library(ml
    src/matrix.cpp
    src/stats.cpp
    src/csv.cpp
    src/linreg.cpp
    src/poly.cpp
    src/plot.cpp
)

target_include_directories(ml
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

# ------------------------------------------------------------
#  Optional parallel-STL switch (default = ON)
# ------------------------------------------------------------
option(ENABLE_PAR "Build with C++17 parallel algorithms" ON)

if(ENABLE_PAR)
    message(STATUS "Parallel STL enabled (adds -DUSE_PAR)")
    add_compile_definitions(USE_PAR)
else()
    message(STATUS "Parallel STL disabled (serial code only)")
endif()

# Executable
add_executable(BASIC_ML src/main.cpp)
target_link_libraries(BASIC_ML PRIVATE ml)

# Copy dataset automatically to build folder
add_custom_command(TARGET BASIC_ML POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/datasets
            $<TARGET_FILE_DIR:BASIC_ML>/datasets
)
