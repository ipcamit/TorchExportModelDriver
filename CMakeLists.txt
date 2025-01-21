cmake_minimum_required(VERSION 3.10..3.18)

# STD 17 required for torch export
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-std=c++17)

cmake_policy(SET CMP0074 NEW)
STRING(COMPARE EQUAL "${CMAKE_BUILD_TYPE}" "" flg)
if(flg)
    set(CMAKE_BUILD_TYPE Release)
endif()
message("Build Type: ${CMAKE_BUILD_TYPE}")

# Append to PREFIX PATH -----------------------------------------
find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

# KIM-API setup --------------------------------------------------
kim_api_items_setup_before_project(ITEM_TYPE "modelDriver")
project(TorchExport__MD_000000000000_000 LANGUAGES CXX)
kim_api_items_setup_after_project(ITEM_TYPE "modelDriver")

# Find Torch (env TORCH_ROOT) and Torch Geometric dependencies
find_package(Torch REQUIRED)

add_kim_api_model_driver_library(
  NAME                    ${PROJECT_NAME}
  CREATE_ROUTINE_NAME     "model_driver_create"
  CREATE_ROUTINE_LANGUAGE "cpp"
)

# Check if User needs MPI aware version of the model driver
if (DEFINED ENV{KIM_MODEL_MPI_AWARE})
    STRING(COMPARE EQUAL $ENV{KIM_MODEL_MPI_AWARE} "yes" flg)
    if(flg)
    # Find MPI
    find_package(MPI)
        if(MPI_FOUND)
            message("MPI found, using MPI compilers to compile")
            include_directories(SYSTEM ${MPI_INCLUDE_PATH})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI")
        else()
            message(FATAL_ERROR "MPI not found, you requested MPI aware build, either provide MPI LIB or set KIM_MODEL_MPI_AWARE to no")
        endif()
    endif()
endif()

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# Add ML model directory
add_subdirectory(MLModel)

target_sources(${PROJECT_NAME} PRIVATE
        TorchExportModelDriver.cpp
        TorchExportModelDriverImplementation.cpp
)

# Link against torch geometric dependencies
target_link_libraries("${PROJECT_NAME}" PRIVATE MLModel PRIVATE ${TORCH_LIBRARIES})
target_link_libraries("${PROJECT_NAME}" PUBLIC "/home/amit/Projects/COLABFIT/TorchExport/escn.so")

target_include_directories(
        "${PROJECT_NAME}" PRIVATE
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/MLModel")

# Suppress annoying Variable not used warnings, parameter still warns :(
add_compile_options(-Wno-unused-variable -Wno-unused-parameter)

# Link against MPI if needed
if (DEFINED ENV{KIM_MODEL_MPI_AWARE})
    STRING(COMPARE EQUAL $ENV{KIM_MODEL_MPI_AWARE} "yes" flg)
    if(flg)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_CXX_LIBRARIES})
    endif()
endif()
