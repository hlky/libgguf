include_guard(GLOBAL)

function(_libgguf_rocm_path in_path out_path)
  file(TO_CMAKE_PATH "${in_path}" _path)
  set(${out_path} "${_path}" PARENT_SCOPE)
endfunction()

function(_libgguf_rocm_command out_result out_output)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(${out_result} "${_result}" PARENT_SCOPE)
  set(${out_output} "${_output}" PARENT_SCOPE)
endfunction()

function(_libgguf_rocm_find_device_libs out_path root)
  set(_candidates
    "${root}/lib/llvm/amdgcn/bitcode"
    "${root}/llvm/amdgcn/bitcode"
    "${root}/amdgcn/bitcode"
  )
  file(GLOB _clang_candidates
    "${root}/lib/llvm/lib/clang/*/amdgcn/bitcode"
    "${root}/llvm/lib/clang/*/amdgcn/bitcode"
    "${root}/lib/clang/*/amdgcn/bitcode"
  )
  list(APPEND _candidates ${_clang_candidates})
  foreach(_candidate IN LISTS _candidates)
    if(IS_DIRECTORY "${_candidate}")
      set(${out_path} "${_candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_path} "" PARENT_SCOPE)
endfunction()

function(_libgguf_rocm_find_rc out_path)
  if(NOT WIN32)
    set(${out_path} "" PARENT_SCOPE)
    return()
  endif()

  find_program(_LIBGGUF_ROCM_RC NAMES rc.exe)
  if(NOT _LIBGGUF_ROCM_RC)
    file(GLOB_RECURSE _rc_candidates "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/rc.exe")
    if(_rc_candidates)
      list(SORT _rc_candidates COMPARE NATURAL ORDER DESCENDING)
      list(GET _rc_candidates 0 _LIBGGUF_ROCM_RC)
    endif()
  endif()
  set(${out_path} "${_LIBGGUF_ROCM_RC}" PARENT_SCOPE)
endfunction()

function(libgguf_configure_rocm_sdk)
  if(LIBGGUF_ROCM_ROOT AND IS_DIRECTORY "${LIBGGUF_ROCM_ROOT}")
    # A standalone TheRock SDK is sufficient; the Python development package
    # is needed only when discovering an SDK through rocm-sdk.
    set(_root "${LIBGGUF_ROCM_ROOT}")
    set(_cmake "${_root}/lib/cmake")
    set(_bin "${_root}/bin")
  else()
    find_program(LIBGGUF_ROCM_SDK_EXECUTABLE NAMES rocm-sdk rocm_sdk REQUIRED)

    execute_process(COMMAND "${LIBGGUF_ROCM_SDK_EXECUTABLE}" init ERROR_QUIET OUTPUT_QUIET)
    _libgguf_rocm_command(_root_result _root "${LIBGGUF_ROCM_SDK_EXECUTABLE}" path --root)
    _libgguf_rocm_command(_cmake_result _cmake "${LIBGGUF_ROCM_SDK_EXECUTABLE}" path --cmake)
    _libgguf_rocm_command(_bin_result _bin "${LIBGGUF_ROCM_SDK_EXECUTABLE}" path --bin)
    if(NOT _root_result EQUAL 0 OR NOT _cmake_result EQUAL 0 OR NOT _bin_result EQUAL 0)
      message(FATAL_ERROR "rocm-sdk did not provide the required root, CMake, and binary paths")
    endif()
  endif()

  _libgguf_rocm_path("${_root}" _root)
  _libgguf_rocm_path("${_cmake}" _cmake)
  _libgguf_rocm_path("${_bin}" _bin)

  set(_llvm_bin "${_root}/lib/llvm/bin")
  if(NOT WIN32 AND NOT EXISTS "${_llvm_bin}/clang++" AND EXISTS "${_root}/llvm/bin/clang++")
    set(_llvm_bin "${_root}/llvm/bin")
  endif()
  if(WIN32)
    set(_clangxx "${_llvm_bin}/clang++.exe")
  else()
    set(_clangxx "${_llvm_bin}/clang++")
  endif()
  if(NOT EXISTS "${_clangxx}")
    message(FATAL_ERROR "ROCm clang++ was not found at ${_clangxx}")
  endif()

  _libgguf_rocm_find_device_libs(_device_libs "${_root}")
  _libgguf_rocm_find_rc(_rc)
  set(LIBGGUF_ARCHITECTURE gfx1201 CACHE STRING "Explicit AMD GPU architecture pack")
  set_property(CACHE LIBGGUF_ARCHITECTURE PROPERTY STRINGS gfx1151 gfx1201 gfx942)
  if(NOT LIBGGUF_ARCHITECTURE MATCHES "^(gfx1151|gfx1201|gfx942)$")
    message(FATAL_ERROR "Select one supported LIBGGUF_ARCHITECTURE")
  endif()

  set(LIBGGUF_ROCM_ROOT "${_root}" CACHE PATH "ROCm SDK root" FORCE)
  set(LIBGGUF_ROCM_CMAKE_PREFIX "${_cmake}" CACHE PATH "ROCm CMake package directory" FORCE)
  set(LIBGGUF_ROCM_BIN "${_bin}" CACHE PATH "ROCm binary directory" FORCE)
  set(LIBGGUF_ROCM_LLVM_BIN "${_llvm_bin}" CACHE PATH "ROCm LLVM binary directory" FORCE)
  set(LIBGGUF_ROCM_DEVICE_LIB_DIR "${_device_libs}" CACHE PATH "ROCm device bitcode directory" FORCE)

  set(HIP_PLATFORM "amd" CACHE STRING "HIP platform" FORCE)
  set(ENV{HIP_PLATFORM} "amd")
  set(ENV{HIP_PATH} "${LIBGGUF_ROCM_ROOT}")
  set(ENV{ROCM_PATH} "${LIBGGUF_ROCM_ROOT}")

  set(_prefix_path "${LIBGGUF_ROCM_CMAKE_PREFIX};${LIBGGUF_ROCM_ROOT};${CMAKE_PREFIX_PATH}")
  set(CMAKE_PREFIX_PATH "${_prefix_path}" CACHE STRING "CMake package search path" FORCE)
  set(CMAKE_PREFIX_PATH "${_prefix_path}" PARENT_SCOPE)

  set(CMAKE_HIP_ARCHITECTURES "${LIBGGUF_ARCHITECTURE}" CACHE STRING "HIP architectures" FORCE)
  set(CMAKE_HIP_COMPILER_ROCM_ROOT "${LIBGGUF_ROCM_ROOT}" CACHE PATH "ROCm root for HIP compiler detection" FORCE)
  set(CMAKE_HIP_COMPILER "${_clangxx}" CACHE FILEPATH "HIP compiler" FORCE)
  if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${_clangxx}" CACHE FILEPATH "C++ compiler" FORCE)
  endif()
  if(WIN32 AND _rc AND NOT DEFINED CMAKE_RC_COMPILER)
    set(CMAKE_RC_COMPILER "${_rc}" CACHE FILEPATH "Windows resource compiler" FORCE)
  endif()

  set(_hip_flags "--rocm-path=${LIBGGUF_ROCM_ROOT}")
  if(LIBGGUF_ROCM_DEVICE_LIB_DIR)
    string(APPEND _hip_flags " --rocm-device-lib-path=${LIBGGUF_ROCM_DEVICE_LIB_DIR}")
  endif()
  set(CMAKE_HIP_FLAGS_INIT "${_hip_flags}" CACHE STRING "Initial HIP flags" FORCE)

  message(STATUS "libgguf ROCm root: ${LIBGGUF_ROCM_ROOT}")
  message(STATUS "libgguf ROCm CMake prefix: ${LIBGGUF_ROCM_CMAKE_PREFIX}")
  message(STATUS "libgguf ROCm LLVM bin: ${LIBGGUF_ROCM_LLVM_BIN}")
  message(STATUS "libgguf ROCm device libs: ${LIBGGUF_ROCM_DEVICE_LIB_DIR}")
  message(STATUS "libgguf ROCm arch: ${CMAKE_HIP_ARCHITECTURES}")
endfunction()
