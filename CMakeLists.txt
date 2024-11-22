make_minimum_required(VERSION 3.15.4)
project(
    pc 
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(PC_SOURCES
        src/transpose.cpp
        src/check_symm.cpp
)
set(PC_HEADERS include)
set(PC_COMPILE_OPTIONS -Wall -Wextra -Wpedantic
		       -Werror -Wconversion -Wshadow
		       -ftemplate-depth=4096
		       -fconstexpr-depth=4096
               -fexceptions
)
set(PC_TEST_SOURCES
        tests/transpose_test.cpp
        tests/check_symm_test.cpp
        fuzz/transpose_fuzz.cpp
        benchmarks/benchmarks.cpp
)
set(PC_TEST_HEADERS)
set(PC_LINK_LIBRARIES pthread)

option(PC_USE_CLANG "Use clang" OFF)
option(PC_BUILD_TESTS "Build tests" ON)
option(PC_BUILD_SHARED "Build shared library" OFF)
option(PC_BUILD_STATIC "Build static library" OFF)
option(PC_BUILD_OPTIMIZED_O1 "Build with optimizations -O1" OFF)
option(PC_BUILD_OPTIMIZED_O2 "Build with optimizations -O2" OFF)
option(PC_BUILD_OPTIMIZED_O3 "Build with optimizations -O3" OFF)

set(ENV{CPM_SOURCE_CACHE} ${CMAKE_CURRENT_SOURCE_DIR}/deps)
include(cmake/CPM.cmake) # <3
CPMUsePackageLock(package-lock.cmake)
CPMAddPackage(
  NAME Ccache.cmake
  GITHUB_REPOSITORY TheLartians/Ccache.cmake
  VERSION 1.2
)
CPMAddPackage(
    NAME valfuzz
    GITHUB_REPOSITORY San7o/valFuzz
    GIT_TAG v1.1.1
    DOWNLOAD_ONLY True
)
CPMAddPackage(
    NAME tenno
    GITHUB_REPOSITORY San7o/tenno-tl
    GIT_TAG v0.1.1
    DOWNLOAD_ONLY True
)
if (valfuzz_ADDED)
   message("Using valfuzz")
    list(APPEND PC_TEST_HEADERS ${valfuzz_SOURCE_DIR}/include)
    add_subdirectory(${valfuzz_SOURCE_DIR})
    list(APPEND PC_LINK_LIBRARIES valfuzz_shared)
endif()
if (tenno_ADDED)
   message("Using tenno")
    list(APPEND PC_HEADERS ${tenno_SOURCE_DIR}/include)
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("Compiling with OpenMP")
    list(APPEND PC_LINK_LIBRARIES OpenMP::OpenMP_CXX)
    list(APPEND PC_COMPILE_OPTIONS -D openMP
                                      -fopenmp -fopenmp-simd)
else()
    message(ERROR "OpenMP not found in your system")
endif()

if(PC_USE_CLANG)
    message("Building with clang")
    set(CMAKE_CXX_COMPILER clang++)
    list(APPEND PC_COMPILE_OPTIONS -std=c++23 -fexperimental-library)
    list(APPEND PC_LINK_LIBRARIES -fexperimental-library)
endif()
if(PC_BUILD_SHARED)
    message("Building shared library")
    add_library(pc_shared SHARED ${PC_SOURCES})
    target_include_directories(pc_shared PRIVATE ${PC_HEADERS})
    target_compile_options(pc_shared PRIVATE ${PC_COMPILE_OPTIONS})
    target_link_libraries(pc_shared PRIVATE ${PC_LINK_LIBRARIES})
endif()

if(PC_BUILD_STATIC)
    message("Building static library")
    add_library(pc_static STATIC ${PC_SOURCES})
    target_include_directories(pc_static PRIVATE ${PC_HEADERS})
    target_compile_options(pc_static PRIVATE ${PC_COMPILE_OPTIONS})
    target_link_libraries(pc_static PRIVATE ${PC_LINK_LIBRARIES})
endif()

if(PC_BUILD_TESTS AND valfuzz_ADDED)
    message("Building tests")
    add_executable(tests ${PC_SOURCES} ${PC_TEST_SOURCES})
    target_include_directories(tests PRIVATE ${PC_HEADERS} ${PC_TEST_HEADERS})
    target_compile_options(tests PRIVATE ${PC_COMPILE_OPTIONS})
    target_link_libraries(tests PRIVATE ${PC_LINK_LIBRARIES})

    if (PC_BUILD_OPTIMIZED_O1)
        message("Building optimized target -O1")
        add_executable(tests_opt_o1 ${PC_SOURCES} ${PC_TEST_SOURCES})
        target_include_directories(tests_opt_o1 PRIVATE ${PC_HEADERS} ${PC_TEST_HEADERS})
        target_compile_options(tests_opt_o1 PRIVATE ${PC_COMPILE_OPTIONS} -O1)
        target_link_libraries(tests_opt_o1 PRIVATE ${PC_LINK_LIBRARIES})
    endif()
    if (PC_BUILD_OPTIMIZED_O2)
        message("Building optimized target -O2")
        add_executable(tests_opt_o2 ${PC_SOURCES} ${PC_TEST_SOURCES})
        target_include_directories(tests_opt_o2 PRIVATE ${PC_HEADERS} ${PC_TEST_HEADERS})
        target_compile_options(tests_opt_o2 PRIVATE ${PC_COMPILE_OPTIONS} -O2)
        target_link_libraries(tests_opt_o2 PRIVATE ${PC_LINK_LIBRARIES})
    endif()
    if (PC_BUILD_OPTIMIZED_O3)
        message("Building optimized target -O3 -march=native -Ofast")
        add_executable(tests_opt_o3 ${PC_SOURCES} ${PC_TEST_SOURCES})
        target_include_directories(tests_opt_o3 PRIVATE ${PC_HEADERS} ${PC_TEST_HEADERS})
        target_compile_options(tests_opt_o3 PRIVATE ${PC_COMPILE_OPTIONS} -O3 -march=native -Ofast)
        target_link_libraries(tests_opt_o3 PRIVATE ${PC_LINK_LIBRARIES})
    endif()
endif()
