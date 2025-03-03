set(ML_RUNTIME_TARGET_LIB MLModel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-std=c++17)

#list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_ROOT})
find_package(Torch REQUIRED)

add_library(${ML_RUNTIME_TARGET_LIB} STATIC MLModel.cpp
        ../yaml.hpp)

set_property(TARGET ${ML_RUNTIME_TARGET_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${ML_RUNTIME_TARGET_LIB} PROPERTY CXX_STANDARD 17)

target_link_libraries(${ML_RUNTIME_TARGET_LIB} PRIVATE ${TORCH_LIBRARIES})

target_include_directories(MLModel PRIVATE ${TORCH_INCLUDE_DIRS})
#include_directories(/opt/mambaforge/mambaforge/envs/torch2/lib/python3.10/site-packages/torch/include)

#target_link_libraries("${PROJECT_NAME}" PRIVATE MLModel PRIVATE ${TORCH_LIBRARIES})
#target_link_libraries("${PROJECT_NAME}" PUBLIC "/home/amit/Projects/COLABFIT/TorchExport/escn.so")
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
