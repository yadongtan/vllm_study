cmake_minimum_required(VERSION 3.24)

# 只让 CMake 管理普通 C++。当前 PyTorch wheel 中的 CUDA toolkit 是
# nvcc/headers/runtime 的精简分发，不适合 project(... LANGUAGES CUDA)
# 的完整 CUDA 编译器探测；下面用 custom command 显式调用 nvcc。
project(cuda_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 当前 WSL 虚拟环境中 PyTorch 和 CUDA toolkit 的固定路径。
set(TORCH_ROOT
    "/opt/vllm_study/.venv/lib/python3.12/site-packages/torch")
set(CUDA_ROOT
    "/opt/vllm_study/.venv/lib/python3.12/site-packages/nvidia/cu13")
set(NVCC "${CUDA_ROOT}/bin/nvcc")
set(PYTHON_INCLUDE_DIR
    "/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12")
set(PYTHON_ROOT
    "/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu")

# nvcc 直接把 add_one.cu 编译成一个目标文件。
set(CUDA_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/add_one.cuda.o")
add_custom_command(
    OUTPUT "${CUDA_OBJECT}"
    COMMAND "${NVCC}"
            -std=c++17
            -O0
            -g
            -lineinfo
            -arch=sm_89
            -Xcompiler=-fPIC
            "-I${TORCH_ROOT}/include"
            "-I${TORCH_ROOT}/include/torch/csrc/api/include"
            "-I${CUDA_ROOT}/include"
            "-I${PYTHON_INCLUDE_DIR}"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/add_one.cu"
            -o "${CUDA_OBJECT}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/add_one.cu"
    COMMENT "Compiling add_one.cu with the PyTorch CUDA toolkit"
    VERBATIM
)

add_custom_target(cuda_demo_cuda_object DEPENDS "${CUDA_OBJECT}")

# C++ 主程序源文件与 nvcc 生成的 CUDA 目标文件一起链接。
add_executable(cuda_demo_cpp
    demo_cpp.cpp
    "${CUDA_OBJECT}"
)
add_dependencies(cuda_demo_cpp cuda_demo_cuda_object)

target_include_directories(cuda_demo_cpp PRIVATE
    "${TORCH_ROOT}/include"
    "${TORCH_ROOT}/include/torch/csrc/api/include"
    "${CUDA_ROOT}/include"
    "${PYTHON_INCLUDE_DIR}"
)

# 直接链接当前 PyTorch wheel 中的 LibTorch 和 CUDA Runtime 动态库。
target_link_libraries(cuda_demo_cpp PRIVATE
    "${TORCH_ROOT}/lib/libtorch.so"
    "${TORCH_ROOT}/lib/libtorch_cpu.so"
    "${TORCH_ROOT}/lib/libtorch_cuda.so"
    "${TORCH_ROOT}/lib/libc10.so"
    "${TORCH_ROOT}/lib/libc10_cuda.so"
    "${CUDA_ROOT}/lib/libcudart.so"
    "${PYTHON_ROOT}/lib/libpython3.12.so"
    pthread
    dl
)

# 必须与当前 PyTorch 的 C++ ABI 一致。
target_compile_definitions(cuda_demo_cpp PRIVATE
    _GLIBCXX_USE_CXX11_ABI=1
)

target_compile_options(cuda_demo_cpp PRIVATE
    -O0
    -g
)

# 让运行时能找到 PyTorch 和 CUDA 的动态库。
set_target_properties(cuda_demo_cpp PROPERTIES
    BUILD_RPATH
    "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib;${PYTHON_ROOT}/lib"
)
