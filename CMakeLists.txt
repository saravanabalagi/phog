cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cv2_example)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0074 NEW)
endif(COMMAND cmake_policy)

find_package(OpenCV REQUIRED)
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# set(ZLIB_ROOT "../../libs/zlib/dist")
# find_package(ZLIB REQUIRED)

# set(CNPY_ROOT "../../dist")
# find_package(CNPY REQUIRED)

include_directories(
    # ${CNPY_INCLUDE_DIRS}
    # ${ZLIB_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(phog compute_phog.cpp)
target_link_libraries(phog 
    # CNPY 
    # ${ZLIB_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
set_property(TARGET phog PROPERTY CXX_STANDARD 17)
