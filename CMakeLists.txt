cmake_minimum_required(VERSION 3.20)

project(thread LANGUAGES CXX CUDA)
set(TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)


set(SOURCES
main.cpp
build/stb_image.h
)

add_executable(thread  ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${CUDA_LIBS}
)
