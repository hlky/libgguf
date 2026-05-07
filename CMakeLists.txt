cmake_minimum_required(VERSION 3.18)

project(libgguf VERSION 0.1.0 LANGUAGES CXX)

option(LIBGGUF_BUILD_SHARED "Build the standalone libgguf shared library" ON)
option(LIBGGUF_BUILD_PYTHON "Build the libgguf Python extension" ON)
option(LIBGGUF_BUILD_BENCHMARKS "Build native libgguf benchmark binaries" OFF)
option(LIBGGUF_BUILD_TOOLS "Build native libgguf command line tools" ON)
set(LIBGGUF_BUILD_CUDA_KERNELS "AUTO" CACHE STRING "Build the optional libgguf_cuda Torch CUDA extension (AUTO, ON, or OFF)")
set_property(CACHE LIBGGUF_BUILD_CUDA_KERNELS PROPERTY STRINGS AUTO ON OFF)
set(LIBGGUF_CPU_BACKEND "REF" CACHE STRING "CPU row backend to compile (REF, SSE2, SSE4_1, or AVX2)")
set_property(CACHE LIBGGUF_CPU_BACKEND PROPERTY STRINGS REF SSE2 SSE4_1 AVX2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

string(TOUPPER "${LIBGGUF_CPU_BACKEND}" LIBGGUF_CPU_BACKEND_UPPER)
set(LIBGGUF_VALID_CPU_BACKENDS REF SSE2 SSE4_1 AVX2)
if(NOT LIBGGUF_CPU_BACKEND_UPPER IN_LIST LIBGGUF_VALID_CPU_BACKENDS)
  message(FATAL_ERROR "LIBGGUF_CPU_BACKEND must be one of: REF, SSE2, SSE4_1, AVX2")
endif()

set(LIBGGUF_X86_PROCESSOR OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86|x86|i[3-6]86)$")
  set(LIBGGUF_X86_PROCESSOR ON)
endif()

if(NOT LIBGGUF_CPU_BACKEND_UPPER STREQUAL "REF" AND NOT LIBGGUF_X86_PROCESSOR)
  message(FATAL_ERROR "LIBGGUF_CPU_BACKEND=${LIBGGUF_CPU_BACKEND_UPPER} requires an x86/x86_64 target")
endif()

string(TOLOWER "${LIBGGUF_CPU_BACKEND_UPPER}" LIBGGUF_CPU_BACKEND_LOWER)
message(STATUS "libgguf CPU backend: ${LIBGGUF_CPU_BACKEND_UPPER}")

file(
  GLOB_RECURSE LIBGGUF_NATIVE_SOURCES
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp"
)
list(FILTER LIBGGUF_NATIVE_SOURCES EXCLUDE REGEX "[/\\\\]_libgguf_module\\.cpp$")
list(FILTER LIBGGUF_NATIVE_SOURCES EXCLUDE REGEX "[/\\\\]quantize_gguf\\.cpp$")
foreach(backend_dir IN ITEMS sse2 sse4_1 avx2)
  if(NOT LIBGGUF_CPU_BACKEND_LOWER STREQUAL backend_dir)
    list(FILTER LIBGGUF_NATIVE_SOURCES EXCLUDE REGEX "[/\\\\](common|dequant|quant)[/\\\\]${backend_dir}[/\\\\].*\\.cpp$")
  endif()
endforeach()

add_library(libgguf_objects OBJECT ${LIBGGUF_NATIVE_SOURCES})
target_include_directories(
  libgguf_objects
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common"
)
target_compile_definitions(
  libgguf_objects
  PRIVATE
    NDEBUG=1
    LIBGGUF_CPU_BACKEND_${LIBGGUF_CPU_BACKEND_UPPER}=1
)

if(MSVC)
  target_compile_options(
    libgguf_objects
    PRIVATE
      /O2
      /EHsc
      /std:c++17
      /D_CRT_SECURE_NO_WARNINGS
  )
else()
  target_compile_options(
    libgguf_objects
    PRIVATE
      -O3
      -pthread
  )
  target_link_libraries(libgguf_objects PUBLIC m pthread)
endif()

if(LIBGGUF_X86_PROCESSOR)
  foreach(source IN LISTS LIBGGUF_NATIVE_SOURCES)
    get_filename_component(source_name "${source}" NAME)
    get_filename_component(source_dir "${source}" DIRECTORY)
    string(REPLACE "\\" "/" source_dir_normalized "${source_dir}")
    set(is_backend_source OFF)
    if(source_dir_normalized MATCHES "/(dequant|quant)/(sse2|sse4_1|avx2)$")
      set(is_backend_source ON)
    elseif(source_name MATCHES "^(libgguf_common_quant|libgguf_storage)\\.cpp$" AND source_dir_normalized MATCHES "/common/(sse2|sse4_1|avx2)$")
      set(is_backend_source ON)
    endif()

    if(is_backend_source AND source_dir_normalized MATCHES "/avx2$")
      if(MSVC)
        set_source_files_properties("${source}" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
      else()
        set_source_files_properties("${source}" PROPERTIES COMPILE_OPTIONS "-mavx2")
      endif()
    elseif(source_name STREQUAL "quantize_gguf_f16c.cpp")
      if(MSVC)
        set_source_files_properties("${source}" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
      else()
        set_source_files_properties("${source}" PROPERTIES COMPILE_OPTIONS "-mavx;-mf16c")
      endif()
    elseif(is_backend_source AND NOT MSVC AND source_dir_normalized MATCHES "/sse4_1$")
      set_source_files_properties("${source}" PROPERTIES COMPILE_OPTIONS "-msse4.1")
    elseif(is_backend_source AND NOT MSVC AND source_dir_normalized MATCHES "/sse2$")
      set_source_files_properties("${source}" PROPERTIES COMPILE_OPTIONS "-msse2")
    endif()
  endforeach()
endif()

if(LIBGGUF_BUILD_SHARED)
  add_library(libgguf SHARED $<TARGET_OBJECTS:libgguf_objects>)
  target_link_libraries(libgguf PRIVATE libgguf_objects)
  target_include_directories(
    libgguf
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common"
  )
  set_target_properties(libgguf PROPERTIES OUTPUT_NAME libgguf)
endif()

if(LIBGGUF_BUILD_PYTHON OR NOT LIBGGUF_BUILD_CUDA_KERNELS STREQUAL "OFF")
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
endif()

if(LIBGGUF_BUILD_PYTHON)
  Python_add_library(
    _libgguf
    MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/_libgguf_module.cpp"
    $<TARGET_OBJECTS:libgguf_objects>
    WITH_SOABI
  )
  target_include_directories(
    _libgguf
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common"
  )
  target_link_libraries(_libgguf PRIVATE libgguf_objects)
  if(MSVC)
    target_compile_options(
      _libgguf
      PRIVATE
        /O2
        /EHsc
        /std:c++17
        /D_CRT_SECURE_NO_WARNINGS
    )
  else()
    target_compile_options(_libgguf PRIVATE -O3 -pthread)
  endif()
  install(TARGETS _libgguf DESTINATION libgguf)
endif()

if(NOT LIBGGUF_BUILD_CUDA_KERNELS STREQUAL "OFF")
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE LIBGGUF_TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE LIBGGUF_TORCH_IMPORT_RESULT
  )

  set(LIBGGUF_CAN_BUILD_CUDA_KERNELS OFF)
  if(LIBGGUF_TORCH_IMPORT_RESULT EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${LIBGGUF_TORCH_CMAKE_PREFIX_PATH}")
    find_package(Torch QUIET)
    find_program(LIBGGUF_NVCC_EXECUTABLE nvcc)
    if(Torch_FOUND AND LIBGGUF_NVCC_EXECUTABLE)
      enable_language(CUDA)
      find_package(CUDAToolkit REQUIRED)
      set(LIBGGUF_CAN_BUILD_CUDA_KERNELS ON)
    endif()
  endif()

  if(LIBGGUF_CAN_BUILD_CUDA_KERNELS)
    set(
      LIBGGUF_CUDA_QUANT_KERNEL_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_kernels.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q8_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q1_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_mxfp4.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_nvfp4.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q4_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q4_1.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q5_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q5_1.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q2_k.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q3_k.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q4_k.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q5_k.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_iq4_nl.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_iq4_xs.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_iq2_xxs.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_iq2_xs.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_iq2_s.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_iq3_xxs.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_iq3_s.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_iq1_s.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_iq1_m.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_tq1_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_tq2_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize_q6_k.cu"
    )
    add_library(libgguf_cuda_quant_objects OBJECT ${LIBGGUF_CUDA_QUANT_KERNEL_SOURCES})
    set_target_properties(
      libgguf_cuda_quant_objects
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(
      libgguf_cuda_quant_objects
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common"
    )
    target_compile_definitions(libgguf_cuda_quant_objects PRIVATE LIBGGUF_CUDA_NO_C10=1)
    target_compile_options(libgguf_cuda_quant_objects PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

    add_library(
      libgguf_cuda_native
      STATIC
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/libgguf_cuda_native.cu"
      $<TARGET_OBJECTS:libgguf_cuda_quant_objects>
    )
    set_target_properties(
      libgguf_cuda_native
      PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        OUTPUT_NAME "gguf_cuda_native"
    )
    target_include_directories(
      libgguf_cuda_native
      PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc"
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common"
    )
    target_compile_options(libgguf_cuda_native PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(libgguf_cuda_native PUBLIC CUDA::cudart)

    Python_add_library(
      libgguf_cuda_C_gguf
      MODULE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/torch_bindings.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_kernels.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q4_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q4_1.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q5_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q5_1.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q8_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_bf16.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q1_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_tq1_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_tq2_0.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_mxfp4.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_nvfp4.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q2_k.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q3_k.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q4_k.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q5_k.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_q6_k.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_iq2_xxs.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_iq2_xs.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_iq2_s.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_iq3_xxs.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_iq3_s.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_iq1_s.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_iq1_m.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_iq4_nl.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_dequantize_iq4_xs.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc/cuda_quantize.cu"
      $<TARGET_OBJECTS:libgguf_cuda_quant_objects>
      WITH_SOABI
    )
    set_target_properties(
      libgguf_cuda_C_gguf
      PROPERTIES
        OUTPUT_NAME "_C_gguf"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(
      libgguf_cuda_C_gguf
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/libgguf/libgguf_cuda/csrc"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common"
        ${TORCH_INCLUDE_DIRS}
    )
    target_compile_definitions(libgguf_cuda_C_gguf PRIVATE TORCH_EXTENSION_NAME=_C_gguf)
    target_compile_options(libgguf_cuda_C_gguf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_compile_features(libgguf_cuda_C_gguf PRIVATE cxx_std_17)
    target_link_libraries(libgguf_cuda_C_gguf PRIVATE ${TORCH_LIBRARIES})
    install(TARGETS libgguf_cuda_C_gguf DESTINATION libgguf/libgguf_cuda)
  elseif(LIBGGUF_BUILD_CUDA_KERNELS STREQUAL "ON")
    message(FATAL_ERROR "LIBGGUF_BUILD_CUDA_KERNELS=ON requires importable torch, Torch CMake metadata, and nvcc")
  else()
    message(STATUS "Skipping libgguf_cuda Torch CUDA extension; torch or nvcc was not found")
  endif()
endif()

if(LIBGGUF_BUILD_TOOLS)
  set(LIBGGUF_TOOLS_RUNTIME_DESTINATION "bin")
  if(DEFINED SKBUILD_SCRIPTS_DIR)
    set(LIBGGUF_TOOLS_RUNTIME_DESTINATION "${SKBUILD_SCRIPTS_DIR}")
  endif()

  add_executable(
    libgguf_quantize_gguf
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantize_gguf.cpp"
    $<TARGET_OBJECTS:libgguf_objects>
  )
  target_include_directories(
    libgguf_quantize_gguf
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common"
  )
  target_link_libraries(libgguf_quantize_gguf PRIVATE libgguf_objects)
  if(MSVC)
    target_compile_options(
      libgguf_quantize_gguf
      PRIVATE
        /O2
        /EHsc
        /std:c++17
        /D_CRT_SECURE_NO_WARNINGS
    )
  else()
    target_compile_options(libgguf_quantize_gguf PRIVATE -O3 -pthread)
    target_link_libraries(libgguf_quantize_gguf PRIVATE m pthread)
  endif()
  install(TARGETS libgguf_quantize_gguf RUNTIME DESTINATION "${LIBGGUF_TOOLS_RUNTIME_DESTINATION}")
endif()

if(LIBGGUF_BUILD_BENCHMARKS)
  add_executable(
    libgguf_backend_bench
    "${CMAKE_CURRENT_SOURCE_DIR}/bench/backend_bench.cpp"
    $<TARGET_OBJECTS:libgguf_objects>
  )
  target_include_directories(
    libgguf_backend_bench
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common"
  )
  target_link_libraries(libgguf_backend_bench PRIVATE libgguf_objects)
  if(MSVC)
    target_compile_options(
      libgguf_backend_bench
      PRIVATE
        /O2
        /EHsc
        /std:c++17
        /D_CRT_SECURE_NO_WARNINGS
    )
  else()
    target_compile_options(libgguf_backend_bench PRIVATE -O3 -pthread)
    target_link_libraries(libgguf_backend_bench PRIVATE m pthread)
  endif()
endif()
