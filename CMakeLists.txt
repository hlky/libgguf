cmake_minimum_required(VERSION 3.18)

project(libgguf VERSION 0.1.0 LANGUAGES CXX)

option(LIBGGUF_BUILD_SHARED "Build the standalone libgguf shared library" ON)
option(LIBGGUF_BUILD_PYTHON "Build the libgguf Python extension" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(
  GLOB_RECURSE LIBGGUF_NATIVE_SOURCES
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp"
)
list(FILTER LIBGGUF_NATIVE_SOURCES EXCLUDE REGEX "[/\\\\]_libgguf_module\\.cpp$")

add_library(libgguf_objects OBJECT ${LIBGGUF_NATIVE_SOURCES})
target_include_directories(
  libgguf_objects
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common"
)
target_compile_definitions(libgguf_objects PRIVATE NDEBUG=1)

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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86|x86|i[3-6]86)$")
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

if(LIBGGUF_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
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
