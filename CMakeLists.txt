cmake_minimum_required(VERSION 3.20)

project(thread LANGUAGES CXX CUDA)
set(TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
set(IMGUI_SOURCES
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.h"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.h"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/imgui.cpp"
)

find_package(OpenGL REQUIRED)
set( imgui_impl_files
    ${IMGUI_DIR}/examples/imgui_impl_glfw.h
    ${IMGUI_DIR}/examples/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/examples/imgui_impl_opengl3.h
    ${IMGUI_DIR}/examples/imgui_impl_opengl3.cpp
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glfw/")
set(GLFW_LIBRARY_TYPE STATIC)

add_subdirectory(${GLFW_DIR})


set(SOURCES
main.cpp
external/stb_image.h
external/stbi_write.h
external/threadPool.h
external/Game.h
external/GameCuda.h
cudaCode/kernel.cu
${IMGUI_SOURCES}
)

message(STATUS "====== IMGUI Located: ${IMGUI_DIR}")
message(STATUS "====== OpenGL Located: ${OPENGL_LIBRARIES}")


add_executable(thread  ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_INCLUDE_DIRS}
    "${GLFW_DIR}/include"
    "${IMGUI_DIR}"
    "${IMGUI_DIR}/backends" 
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    glfw
    ${OPENGL_LIBRARIES}
    ${CUDA_LIBS}
)
