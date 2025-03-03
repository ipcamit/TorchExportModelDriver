cmake_minimum_required(VERSION 3.10..3.18)

# Append to PREFIX PATH -----------------------------------------
find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

# KIM-API setup --------------------------------------------------
kim_api_items_setup_before_project(ITEM_TYPE "modelDriver")
project(TorchExport__MD_000000000000_000 LANGUAGES CXX)
kim_api_items_setup_after_project(ITEM_TYPE "modelDriver")

add_kim_api_model_driver_library(
  NAME                    ${PROJECT_NAME}
  CREATE_ROUTINE_NAME     "model_driver_create"
  CREATE_ROUTINE_LANGUAGE "cpp"
)

target_sources(${PROJECT_NAME} PRIVATE
        TorchExportModelDriver.cpp
        TorchExportModelDriverImplementation.cpp
)

add_subdirectory(MLModel)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
#link_directories(/home/amit/Projects/COLABFIT/TorchExport)
target_link_libraries(${PROJECT_NAME} PRIVATE MLModel)




set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#set(CMAKE_INSTALL_RPATH "/home/amit/Projects/COLABFIT/TorchExport")

