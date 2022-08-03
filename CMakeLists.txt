cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(phog)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0074 NEW)
endif(COMMAND cmake_policy)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(ZLIB REQUIRED)                                     # apt install zlib1g-dev
find_package(Boost REQUIRED COMPONENTS system filesystem)       # apt install libboost-all-dev

# Follow instructions given at
# https://github.com/saravanabalagi/cnpy
set(CNPY_ROOT "/libs/cnpy/dist")
find_package(CNPY REQUIRED)

include_directories(
    include
    ${CNPY_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

add_executable(phog compute_phog.cpp)
target_link_libraries(phog 
    CNPY 
    ${ZLIB_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
)
set_property(TARGET phog PROPERTY CXX_STANDARD 17)
