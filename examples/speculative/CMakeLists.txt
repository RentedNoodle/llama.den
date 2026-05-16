set(TARGET speculative-bench)
add_executable(${TARGET} speculative_bench.cpp)
target_link_libraries(${TARGET} PRIVATE common llama)
target_include_directories(${TARGET} PRIVATE ../server)
