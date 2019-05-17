cmake_minimum_required(VERSION 2.8)

#As per http://permalink.gmane.org/gmane.comp.programming.tools.cmake.user/15952
IF(DEFINED CMAKE_BUILD_TYPE)
   IF((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "Release") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo") OR (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel"))
      MESSAGE("-- Build type is defined as '${CMAKE_BUILD_TYPE}'")
   ELSE()
      SET(CMAKE_BUILD_TYPE "Release")   
      MESSAGE("-- Invalid option for BUILD TYPE, defaulted to '${CMAKE_BUILD_TYPE}' build")
   ENDIF()
ELSE()
   SET(CMAKE_BUILD_TYPE "Release")
   MESSAGE("-- Build type wasn't defined, defaulted to '${CMAKE_BUILD_TYPE}' build")
ENDIF()
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
  MESSAGE(WARNING "You are compiling the backend in debugging mode... things may be slower than usual. Overriding all optimizations to 'OFF'")
ENDIF()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake")
#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set(BUILD_SHARED_LIBS ON) #we always want shared libs out of this for python extensions
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_NATIVE_TUNING "Enables support for optimized code generation (-march=native)" ON)
option(ENABLE_FAST_MATH "Enables -fast-math option (breaks IEEE754 callbacks)" ON)

if (${ENABLE_NATIVE_TUNING})
  set(TUNING_FLAGS "-march=native -mtune=native")
  message(WARNING "Instructing gcc to use your native supported instruction set. This will break .so portability and may raise the ILLEGAL_INSTRUCTION signal on incompatible chipsets")
else (${ENABLE_NATIVE_TUNING})

  set(TUNING_FLAGS "")
endif (${ENABLE_NATIVE_TUNING})
if (${ENABLE_FAST_MATH})
  message("-ffast-math is enabled")
  set(FAST_MATH_FLAGS "-ffast-math")
else (${ENABLE_FAST_MATH})
  set(FAST_MATH_FLAGS "")
endif (${ENABLE_FAST_MATH})

#Find necessary modules
find_package(PythonLibs 2.7 REQUIRED)
find_package(NumPy REQUIRED)
find_package(OpenMP REQUIRED)
find_package(RT REQUIRED)
find_package(pybind11 REQUIRED)

if (NOT ${PYTHON_FOUND})
  message(FATAL_ERROR "Failed to find Python, ensure python-dev is installed")
endif (NOT ${PYTHON_FOUND})
if (NOT ${NUMPY_FOUND})
  message(FATAL_ERROR "Failed to find Numpy, ensure python-numpy is installed")
endif (NOT ${NUMPY_FOUND})
if (NOT ${OPENMP_FOUND})
  message(FATAL_ERROR "Failed to find OpenMP.")
endif (NOT ${OPENMP_FOUND})
if (NOT ${HAVE_RT})
  message(FATAL_ERROR "Failed to find librt and header.")
endif (NOT ${HAVE_RT})
if (NOT ${PYTHON_PYBIND11_FOUND})
  message(FATAL_ERROR "Failed to find Pybind11, ensure pybind11 is installed from pip. If it still can't find it downgrade to pip 9.0.3 and redo the installation.")
endif (NOT ${PYTHON_PYBIND11_FOUND})

#Update include paths with found includes above
include_directories(${PYTHON_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIR}/numpy ${RT_INCLUDES} ${CASACORE_INCLUDE_DIR} ${PYTHON_PYBIND11_INCLUDE_DIR})

set(PYBIND11_CPP_STANDARD -std=c++14)

#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

#Add more options for debug and release builds
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -ggdb3 -std=c99 -pedantic -W -Wall -Wconversion ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -ggdb3 -std=c++14 -pedantic -W -Wall -Wconversion -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ggdb3 -fmax-errors=1 -Wconversion -W -Wall -std=c99 ${OpenMP_C_FLAGS} ${VECTORIZATION_FLAGS} ${TUNING_FLAGS} ${FAST_MATH_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ggdb3 -fmax-errors=1 -std=c++14 -pedantic -W -Wall -Wconversion -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ${OpenMP_CXX_FLAGS} ${VECTORIZATION_FLAGS} ${TUNING_FLAGS} ${FAST_MATH_FLAGS}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_RELEASE}")
message("Debug CFLAGS are ${CMAKE_C_FLAGS_DEBUG}")
message("Release CFLAGS are ${CMAKE_C_FLAGS_RELEASE}")

add_subdirectory(Array)
add_subdirectory(Gridder)
add_subdirectory(Predict)