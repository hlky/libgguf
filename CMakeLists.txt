cmake_minimum_required(VERSION 3.21)

option(LIBGGUF_BUILD_GPU "Build libgguf GPU targets" OFF)
option(LIBGGUF_ENABLE_ROCM "Build GPU targets with HIP/ROCm" OFF)
option(LIBGGUF_ENABLE_CUDA "Build GPU targets with CUDA" OFF)

if(LIBGGUF_ENABLE_ROCM AND LIBGGUF_ENABLE_CUDA)
  message(FATAL_ERROR "Select either LIBGGUF_ENABLE_ROCM or LIBGGUF_ENABLE_CUDA, not both")
endif()

if(LIBGGUF_BUILD_GPU AND NOT LIBGGUF_ENABLE_ROCM AND NOT LIBGGUF_ENABLE_CUDA)
  message(FATAL_ERROR "LIBGGUF_BUILD_GPU requires LIBGGUF_ENABLE_ROCM=ON or LIBGGUF_ENABLE_CUDA=ON")
endif()

if((LIBGGUF_ENABLE_ROCM OR LIBGGUF_ENABLE_CUDA) AND NOT LIBGGUF_BUILD_GPU)
  message(FATAL_ERROR "LIBGGUF_ENABLE_ROCM and LIBGGUF_ENABLE_CUDA require LIBGGUF_BUILD_GPU=ON")
endif()

if(LIBGGUF_ENABLE_ROCM)
  include(cmake/LibGGUFROCmSdk.cmake)
  libgguf_configure_rocm_sdk()
endif()

project(libgguf VERSION 0.2.2 LANGUAGES CXX)

option(LIBGGUF_BUILD_SHARED "Build the standalone libgguf shared library" ON)
option(LIBGGUF_BUILD_PYTHON "Build the libgguf Python extension" ON)
option(LIBGGUF_BUILD_TORCH "Build the libgguf Torch GPU extension" OFF)
option(LIBGGUF_BUILD_BENCHMARKS "Build native libgguf benchmark binaries" OFF)
option(LIBGGUF_BUILD_TOOLS "Build native libgguf command line tools" ON)
option(LIBGGUF_BUILD_TESTS "Build native libgguf contract tests" OFF)
option(LIBGGUF_RUN_GPU_TESTS
       "Register libgguf tests that require an available GPU" OFF)
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

set(
  LIBGGUF_CORE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/formats.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cpu/threading.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantize/cpu.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantize/iq_tables.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/dequantize/cpu.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/cpu/common/tables.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/cpu/common/${LIBGGUF_CPU_BACKEND_LOWER}.cpp"
)

file(
  GLOB LIBGGUF_CPU_QUANTIZE_SOURCES
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/cpu/quantize/${LIBGGUF_CPU_BACKEND_LOWER}/*.cpp"
)
file(
  GLOB LIBGGUF_CPU_DEQUANTIZE_SOURCES
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/cpu/dequantize/${LIBGGUF_CPU_BACKEND_LOWER}/*.cpp"
)

set(
  LIBGGUF_CPU_BACKEND_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/cpu/common/${LIBGGUF_CPU_BACKEND_LOWER}.cpp"
  ${LIBGGUF_CPU_QUANTIZE_SOURCES}
  ${LIBGGUF_CPU_DEQUANTIZE_SOURCES}
)
list(APPEND LIBGGUF_CORE_SOURCES ${LIBGGUF_CPU_QUANTIZE_SOURCES} ${LIBGGUF_CPU_DEQUANTIZE_SOURCES})

add_library(libgguf_objects OBJECT ${LIBGGUF_CORE_SOURCES})
target_include_directories(
  libgguf_objects
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
)
target_compile_definitions(libgguf_objects PRIVATE NDEBUG=1)

if(MSVC)
  target_compile_options(libgguf_objects PRIVATE /O2 /EHsc /std:c++17 /D_CRT_SECURE_NO_WARNINGS)
elseif(WIN32)
  target_compile_options(libgguf_objects PRIVATE -O3)
  target_compile_definitions(libgguf_objects PRIVATE _CRT_SECURE_NO_WARNINGS=1)
else()
  target_compile_options(libgguf_objects PRIVATE -O3 -pthread)
  target_link_libraries(libgguf_objects PUBLIC m pthread)
endif()

if(LIBGGUF_CPU_BACKEND_UPPER STREQUAL "AVX2")
  if(MSVC)
    set_source_files_properties(${LIBGGUF_CPU_BACKEND_SOURCES} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(${LIBGGUF_CPU_BACKEND_SOURCES} PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c")
  endif()
elseif(LIBGGUF_CPU_BACKEND_UPPER STREQUAL "SSE4_1" AND NOT MSVC)
  set_source_files_properties(${LIBGGUF_CPU_BACKEND_SOURCES} PROPERTIES COMPILE_OPTIONS "-msse4.1")
elseif(LIBGGUF_CPU_BACKEND_UPPER STREQUAL "SSE2" AND NOT MSVC)
  set_source_files_properties(${LIBGGUF_CPU_BACKEND_SOURCES} PROPERTIES COMPILE_OPTIONS "-msse2")
endif()

if(LIBGGUF_BUILD_SHARED)
  add_library(libgguf SHARED)
  target_link_libraries(libgguf PRIVATE libgguf_objects)
  target_include_directories(libgguf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set_target_properties(libgguf PROPERTIES OUTPUT_NAME libgguf)
endif()

if(LIBGGUF_BUILD_PYTHON OR LIBGGUF_BUILD_TORCH)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
endif()

if(LIBGGUF_BUILD_TORCH AND NOT LIBGGUF_BUILD_GPU)
  message(FATAL_ERROR "LIBGGUF_BUILD_TORCH requires LIBGGUF_BUILD_GPU")
endif()

if(LIBGGUF_BUILD_PYTHON)
  Python_add_library(
    _libgguf
    MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings/python/module.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings/python/payload.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings/python/raw_ops.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/safetensors_payload.cpp"
    WITH_SOABI
  )
  target_include_directories(
    _libgguf
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  )
  target_link_libraries(_libgguf PRIVATE libgguf_objects)
  if(MSVC)
    target_compile_options(_libgguf PRIVATE /O2 /EHsc /std:c++17 /D_CRT_SECURE_NO_WARNINGS)
  elseif(WIN32)
    target_compile_options(_libgguf PRIVATE -O3)
    target_compile_definitions(_libgguf PRIVATE _CRT_SECURE_NO_WARNINGS=1)
  else()
    target_compile_options(_libgguf PRIVATE -O3 -pthread)
  endif()
  install(TARGETS _libgguf DESTINATION libgguf)
endif()

if(LIBGGUF_BUILD_GPU)
    if(LIBGGUF_ENABLE_ROCM)
      enable_language(HIP)
      find_package(hip REQUIRED)
      set(LIBGGUF_GPU_LANGUAGE HIP)
      set(LIBGGUF_GPU_DEVICE_RUNTIME_TARGET hip::device)
      set(LIBGGUF_GPU_HOST_RUNTIME_TARGET hip::host)
    else()
      enable_language(CUDA)
      find_package(CUDAToolkit REQUIRED)
      set(LIBGGUF_GPU_LANGUAGE CUDA)
      set(LIBGGUF_GPU_DEVICE_RUNTIME_TARGET CUDA::cudart)
      set(LIBGGUF_GPU_HOST_RUNTIME_TARGET CUDA::cudart)
    endif()

    file(
      GLOB LIBGGUF_GPU_QUANTIZE_KERNEL_SOURCES
      CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/gpu/quantize/*.cu"
    )
    file(
      GLOB LIBGGUF_GPU_DEQUANTIZE_KERNEL_SOURCES
      CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/gpu/dequantize/*.cu"
    )

    set(
      LIBGGUF_GPU_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantize/gpu.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/dequantize/gpu.cu"
      ${LIBGGUF_GPU_QUANTIZE_KERNEL_SOURCES}
      ${LIBGGUF_GPU_DEQUANTIZE_KERNEL_SOURCES}
    )
    set_source_files_properties(${LIBGGUF_GPU_SOURCES} PROPERTIES LANGUAGE ${LIBGGUF_GPU_LANGUAGE})

    add_library(
      libgguf_gpu_objects
      OBJECT
      ${LIBGGUF_GPU_SOURCES}
    )
    set_target_properties(libgguf_gpu_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(LIBGGUF_ENABLE_ROCM)
      set_target_properties(
        libgguf_gpu_objects
        PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
      )
    else()
      set_target_properties(
        libgguf_gpu_objects
        PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
      )
    endif()
    target_include_directories(
      libgguf_gpu_objects
      PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    )
    if(LIBGGUF_ENABLE_ROCM)
      target_compile_options(libgguf_gpu_objects PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
    else()
      target_compile_options(libgguf_gpu_objects PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    endif()
    target_link_libraries(libgguf_gpu_objects PRIVATE ${LIBGGUF_GPU_DEVICE_RUNTIME_TARGET})

    if(LIBGGUF_BUILD_TORCH)
      if(LIBGGUF_ENABLE_ROCM)
        set(ENV{PYTORCH_ROCM_ARCH} "${CMAKE_HIP_ARCHITECTURES}")
      endif()
      execute_process(
        COMMAND
          "${Python_EXECUTABLE}" -c
          "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE LIBGGUF_TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE LIBGGUF_TORCH_IMPORT_RESULT
      )
      if(NOT LIBGGUF_TORCH_IMPORT_RESULT EQUAL 0)
        message(FATAL_ERROR "LIBGGUF_BUILD_TORCH requires an importable Torch installation")
      endif()
      list(APPEND CMAKE_PREFIX_PATH "${LIBGGUF_TORCH_CMAKE_PREFIX_PATH}")
      find_package(Torch REQUIRED)

      if(WIN32 AND LIBGGUF_ENABLE_ROCM)
        foreach(LIBGGUF_TORCH_TARGET c10 c10_hip torch_cpu torch_hip)
          if(TARGET ${LIBGGUF_TORCH_TARGET})
            get_target_property(
              LIBGGUF_TORCH_COMPILE_OPTIONS
              ${LIBGGUF_TORCH_TARGET}
              INTERFACE_COMPILE_OPTIONS
            )
            if(LIBGGUF_TORCH_COMPILE_OPTIONS)
              list(
                FILTER LIBGGUF_TORCH_COMPILE_OPTIONS
                EXCLUDE REGEX
                  "^\\$<\\$<COMPILE_LANGUAGE:CXX>:$|/(permissive-|EHsc|bigobj|Z7)"
              )
              set_target_properties(
                ${LIBGGUF_TORCH_TARGET}
                PROPERTIES INTERFACE_COMPILE_OPTIONS "${LIBGGUF_TORCH_COMPILE_OPTIONS}"
              )
            endif()
          endif()
        endforeach()
      endif()

      set(
        LIBGGUF_GPU_TORCH_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings/torch/gpu/quantize.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings/torch/gpu/dequantize.cu"
      )
      set_source_files_properties(
        ${LIBGGUF_GPU_TORCH_SOURCES}
        PROPERTIES LANGUAGE ${LIBGGUF_GPU_LANGUAGE}
      )
      Python_add_library(
        libgguf_gpu_C_gguf
        MODULE
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings/torch/module.cpp"
        ${LIBGGUF_GPU_TORCH_SOURCES}
        WITH_SOABI
      )
      set_target_properties(libgguf_gpu_C_gguf PROPERTIES OUTPUT_NAME "_C_gguf")
      if(LIBGGUF_ENABLE_ROCM)
        set_target_properties(
          libgguf_gpu_C_gguf
          PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
        )
      else()
        set_target_properties(
          libgguf_gpu_C_gguf
          PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
        )
      endif()
      target_include_directories(
        libgguf_gpu_C_gguf
        PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/include"
          "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
          ${TORCH_INCLUDE_DIRS}
      )
      target_compile_definitions(libgguf_gpu_C_gguf PRIVATE TORCH_EXTENSION_NAME=_C_gguf)
      if(LIBGGUF_ENABLE_ROCM)
        target_compile_options(libgguf_gpu_C_gguf PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
      else()
        target_compile_options(libgguf_gpu_C_gguf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
      endif()
      target_compile_features(libgguf_gpu_C_gguf PRIVATE cxx_std_17)
      target_link_libraries(
        libgguf_gpu_C_gguf
        PRIVATE
          libgguf_objects
          libgguf_gpu_objects
          ${LIBGGUF_GPU_HOST_RUNTIME_TARGET}
          ${TORCH_LIBRARIES}
      )
      install(TARGETS libgguf_gpu_C_gguf DESTINATION libgguf/libgguf_gpu)
    endif()
endif()

if(LIBGGUF_BUILD_TOOLS)
  set(LIBGGUF_TOOLS_RUNTIME_DESTINATION "bin")
  if(DEFINED SKBUILD_SCRIPTS_DIR)
    set(LIBGGUF_TOOLS_RUNTIME_DESTINATION "${SKBUILD_SCRIPTS_DIR}")
  endif()

  add_executable(
    libgguf_quantize_gguf
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cli/quantize_gguf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cli/quantize/options.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cli/quantize/policy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/common.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/converter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/cpu_quantizer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/safetensors.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/writer.cpp"
  )
  target_include_directories(
    libgguf_quantize_gguf
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  )
  target_link_libraries(libgguf_quantize_gguf PRIVATE libgguf_objects)

  if(TARGET libgguf_gpu_objects)
    set_source_files_properties(
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/gpu_quantizer.cu"
      PROPERTIES LANGUAGE ${LIBGGUF_GPU_LANGUAGE}
    )
    target_sources(
      libgguf_quantize_gguf
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/gpu_quantizer.cu"
    )
    target_compile_definitions(libgguf_quantize_gguf PRIVATE LIBGGUF_HAS_GPU=1)
    target_link_libraries(
      libgguf_quantize_gguf
      PRIVATE libgguf_gpu_objects ${LIBGGUF_GPU_HOST_RUNTIME_TARGET}
    )
    if(LIBGGUF_ENABLE_ROCM)
      set_target_properties(
        libgguf_quantize_gguf
        PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
      )
    else()
      set_target_properties(
        libgguf_quantize_gguf
        PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
      )
    endif()
  endif()

  if(LIBGGUF_CPU_BACKEND_UPPER STREQUAL "AVX2")
    if(MSVC)
      set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/cpu_quantizer.cpp"
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
      )
      else()
        set_source_files_properties(
          "${CMAKE_CURRENT_SOURCE_DIR}/csrc/conversion/cpu_quantizer.cpp"
          PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c"
        )
      endif()
  endif()

  if(MSVC)
    target_compile_options(libgguf_quantize_gguf PRIVATE /O2 /EHsc /std:c++17 /D_CRT_SECURE_NO_WARNINGS)
  elseif(WIN32)
    target_compile_options(libgguf_quantize_gguf PRIVATE -O3)
    target_compile_definitions(libgguf_quantize_gguf PRIVATE _CRT_SECURE_NO_WARNINGS=1)
  else()
    target_compile_options(libgguf_quantize_gguf PRIVATE -O3 -pthread)
    target_link_libraries(libgguf_quantize_gguf PRIVATE m pthread)
  endif()
  install(TARGETS libgguf_quantize_gguf RUNTIME DESTINATION "${LIBGGUF_TOOLS_RUNTIME_DESTINATION}")
endif()

if(LIBGGUF_BUILD_BENCHMARKS)
  add_executable(
    libgguf_cpu_bench
    "${CMAKE_CURRENT_SOURCE_DIR}/bench/cpu_bench.cpp"
  )
  target_include_directories(
    libgguf_cpu_bench
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  )
  target_link_libraries(libgguf_cpu_bench PRIVATE libgguf_objects)
  if(MSVC)
    target_compile_options(libgguf_cpu_bench PRIVATE /O2 /EHsc /std:c++17 /D_CRT_SECURE_NO_WARNINGS)
  elseif(WIN32)
    target_compile_options(libgguf_cpu_bench PRIVATE -O3)
    target_compile_definitions(libgguf_cpu_bench PRIVATE _CRT_SECURE_NO_WARNINGS=1)
  else()
    target_compile_options(libgguf_cpu_bench PRIVATE -O3 -pthread)
    target_link_libraries(libgguf_cpu_bench PRIVATE m pthread)
  endif()
endif()

if(LIBGGUF_BUILD_TESTS)
  enable_testing()
  add_executable(libgguf_fragment_test tests/native/fragment_test.cpp)
  target_include_directories(libgguf_fragment_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(libgguf_fragment_test PRIVATE libgguf_objects)
  add_test(NAME libgguf_fragment_test COMMAND libgguf_fragment_test)
  if(LIBGGUF_BUILD_GPU)
    set_source_files_properties(tests/native/fragment_device.cu
      PROPERTIES LANGUAGE ${LIBGGUF_GPU_LANGUAGE})
    add_executable(libgguf_fragment_device_test
      tests/native/fragment_device.cu)
    target_include_directories(libgguf_fragment_device_test PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(libgguf_fragment_device_test PRIVATE
      ${LIBGGUF_GPU_HOST_RUNTIME_TARGET})
    if(LIBGGUF_ENABLE_ROCM)
      set_target_properties(libgguf_fragment_device_test PROPERTIES
        HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
    else()
      set_target_properties(libgguf_fragment_device_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    endif()
    if(LIBGGUF_RUN_GPU_TESTS)
      add_test(NAME libgguf_fragment_device_test
               COMMAND libgguf_fragment_device_test)
    endif()
    if(LIBGGUF_ENABLE_ROCM)
      list(GET CMAKE_HIP_ARCHITECTURES 0 LIBGGUF_FRAGMENT_TEST_ARCH)
      add_custom_target(libgguf_fragment_noruntime_test
        COMMAND "${CMAKE_HIP_COMPILER}"
          -std=c++17 -x hip --cuda-device-only -nogpuinc -nogpulib
          "--offload-arch=${LIBGGUF_FRAGMENT_TEST_ARCH}"
          "-I${CMAKE_CURRENT_SOURCE_DIR}/include" -fsyntax-only
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/fragment_noruntime.hip"
        VERBATIM)
      add_test(NAME libgguf_fragment_noruntime_test
        COMMAND "${CMAKE_HIP_COMPILER}"
          -std=c++17 -x hip --cuda-device-only -nogpuinc -nogpulib
          "--offload-arch=${LIBGGUF_FRAGMENT_TEST_ARCH}"
          "-I${CMAKE_CURRENT_SOURCE_DIR}/include" -fsyntax-only
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/fragment_noruntime.hip")
    endif()
  endif()
endif()
