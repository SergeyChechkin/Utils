cmake_minimum_required(VERSION 3.8)
project(utils_lib)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")

  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})

set(SOURCES
    src/CeresUtils.cpp
    src/PoseUtils.cpp
    
    src/image/ColorConvertion.cpp

    src/geometry/Triangulation.cpp

    src/features/KLT.cpp
    src/features/InverseLK_Pose.cpp
    src/features/HomographySolver.cpp
    src/features/MinFeatureExtractor.cpp
    
    src/solver/PnPSolver.cpp

    src/dataset_reader/EuRoC_DatasetReader.cpp
    src/dataset_reader/TUM_DatasetReader.cpp
)

set(HEADERS
)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_LIBS} Ceres::ceres)

install(
  DIRECTORY include/utils
  DESTINATION include
)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

add_subdirectory(test)