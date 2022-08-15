# Set cmake version
cmake_minimum_required(VERSION "3.24")

# Name Project
project("MJPEG_Motion_Detector")

# Set output directory to bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Set C/C++ Version
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Local Includes
set(LOCAL_INCLUDES "include")
include_directories(${LOCAL_INCLUDES})

# Grab all .cc Files In "src"
AUX_SOURCE_DIRECTORY("src" SRC_CC_FILES) 


# Set Executable Name
set(EXE_NAME "exec")
set(TEST_EXE_NAME "test")

# Create Executables
add_executable(${EXE_NAME} "src/main.cc" ${SRC_CC_FILES})
# Remove main.cc and open_cl_interface.cc from SRC_CC_Files for use in tests
list(REMOVE_ITEM SRC_CC_FILES "src/main.cc")
add_executable(${TEST_EXE_NAME} "tests/tests.cc" ${SRC_CC_FILES})

# Find OpenCL
find_package(OpenCL REQUIRED)
# Link library
target_link_libraries(${EXE_NAME} PRIVATE OpenCL::OpenCL)
target_link_libraries(${TEST_EXE_NAME} PRIVATE OpenCL::OpenCL)

# Find libjpeg-turbo
find_package(libjpeg-turbo REQUIRED)
# Link library
target_link_libraries(${EXE_NAME} PRIVATE libjpeg-turbo::turbojpeg)
target_include_directories(${EXE_NAME} PRIVATE ${JPEG_INCLUDE_DIR})
target_link_libraries(${TEST_EXE_NAME} PRIVATE libjpeg-turbo::turbojpeg)
target_include_directories(${TEST_EXE_NAME} PRIVATE ${JPEG_INCLUDE_DIR})

# Find Catch2
find_package(Catch2 CONFIG REQUIRED)
# Link library
target_link_libraries(${TEST_EXE_NAME} PRIVATE Catch2::Catch2 Catch2::Catch2WithMain)