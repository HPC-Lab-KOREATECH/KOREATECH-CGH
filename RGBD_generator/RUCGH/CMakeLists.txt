cmake_minimum_required(VERSION 3.25.2)
project(RUCGH LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(OPTIX_INSTALL_DIR "" CACHE PATH "OptiX SDK root, or a parent directory containing the extracted OptiX SDK")

set(OPTIX_SEARCH_ROOTS)
if(OPTIX_INSTALL_DIR)
    list(APPEND OPTIX_SEARCH_ROOTS "${OPTIX_INSTALL_DIR}")
endif()
if(DEFINED ENV{OPTIX_INSTALL_DIR})
    list(APPEND OPTIX_SEARCH_ROOTS "$ENV{OPTIX_INSTALL_DIR}")
endif()

if(WIN32)
    list(APPEND OPTIX_SEARCH_ROOTS
        "$ENV{ProgramData}/NVIDIA Corporation/OptiX SDK 8.0.0"
        "$ENV{ProgramData}/NVIDIA Corporation/OptiX SDK 7.7.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
        "C:/NVIDIA-OptiX-SDK-8.0.0-win64"
    )
endif()

find_path(OPTIX_INCLUDE
    NAMES optix.h
    HINTS ${OPTIX_SEARCH_ROOTS}
    PATH_SUFFIXES
        include
        NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/include
        NVIDIA-OptiX-SDK-8.0.0-win64/include
        NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64/include
        NVIDIA-OptiX-SDK-7.7.0-win64/include
)

if(NOT OPTIX_INCLUDE)
    message(FATAL_ERROR
        "Could not find optix.h. Set OPTIX_INSTALL_DIR to the extracted OptiX SDK root "
        "or to its parent directory. Example on Windows: "
        "cmake -S . -B build -DOPTIX_INSTALL_DIR=\"C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0\"")
endif()

get_filename_component(OPTIX_DIR "${OPTIX_INCLUDE}/.." ABSOLUTE)
message(STATUS "OptiX include: ${OPTIX_INCLUDE}")

set(RUCGH_GDT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/common/gdt")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RUCGH_GDT_INCLUDE}
    ${OPTIX_INCLUDE}
)

add_library(RUCGH
    ParsingManager.cpp
    RTCGHParser.cpp
    RUCGH.cpp
    RayHandler.cu
)
set_property(TARGET RUCGH PROPERTY CXX_STANDARD 20)

if(MSVC)
    set(RUCGH_CUDA_OPENMP_FLAG "/openmp")
else()
    set(RUCGH_CUDA_OPENMP_FLAG "-fopenmp")
endif()

target_compile_definitions(RUCGH PUBLIC
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
)

if(MSVC)
    target_compile_options(RUCGH PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
    target_compile_options(RUCGH PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${RUCGH_CUDA_OPENMP_FLAG}>
    )
else()
    target_compile_options(RUCGH PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${RUCGH_CUDA_OPENMP_FLAG}>)
endif()

add_library(myptx OBJECT
    deviceProgramOrthogonal.cu
)
set_property(TARGET myptx PROPERTY CUDA_PTX_COMPILATION ON)
target_compile_options(myptx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
if(MSVC)
    target_compile_options(myptx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
target_include_directories(myptx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RUCGH_GDT_INCLUDE}
    ${OPTIX_INCLUDE}
)

target_link_libraries(RUCGH PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
)
target_include_directories(RUCGH PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RUCGH_GDT_INCLUDE}
    ${OPTIX_INCLUDE}
)

add_executable(RUCGH_exe main.cu)
set_property(TARGET RUCGH_exe PROPERTY CXX_STANDARD 20)
if(MSVC)
    target_compile_options(RUCGH_exe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${RUCGH_CUDA_OPENMP_FLAG}>
    )
else()
    target_compile_options(RUCGH_exe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${RUCGH_CUDA_OPENMP_FLAG}>)
endif()
target_link_libraries(RUCGH_exe PUBLIC
    RUCGH
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
)
target_include_directories(RUCGH_exe PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RUCGH_GDT_INCLUDE}
    ${OPTIX_INCLUDE}
)
