set(TARGET den-cli)

# Find CUDA toolkit for include paths (den_cli.cpp includes <cuda_runtime.h>)
find_package(CUDAToolkit REQUIRED)

add_executable(${TARGET}
    ../../tools/den_cli.cpp
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${TARGET} PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT} CUDA::cudart)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

install(TARGETS ${TARGET} RUNTIME)
