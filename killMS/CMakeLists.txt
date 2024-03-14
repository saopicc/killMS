cmake_minimum_required(VERSION 2.8)
project(kmsbackend)

set(BUILD_SHARED_LIBS ON) #we always want shared libs out of this for python extensions
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_NATIVE_TUNING "Enables support for optimized code generation (-march=native)" ON)
option(ENABLE_FAST_MATH "Enables -fast-math option (breaks IEEE754 callbacks)" ON)
option(ENABLE_PYTHON_2 "Enables support for Python 2.7" ON)
option(ENABLE_PYTHON_3 "Enables support for Python 3.x" ON)

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
  set(ENABLE_FAST_MATH "OFF")
  set(ENABLE_NATIVE_TUNING "OFF")
ENDIF()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake")
#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

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

#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
install(FILES __init__.py DESTINATION killMS/cbuild)

#Add more options for debug and release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp -std=c++14 -Wall -fmax-errors=1 -ggdb3 -pedantic -W -Wall -Wconversion -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -std=c++14 -ggdb3 -fmax-errors=1 -pedantic -W -Wall -Wconversion -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ${OpenMP_CXX_FLAGS} ${VECTORIZATION_FLAGS} ${TUNING_FLAGS} ${FAST_MATH_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -fopenmp -ggdb3 -std=c99 -pedantic -W -Wall -fmax-errors=1 -Wconversion ${OpenMP_C_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ggdb3 -fopenmp -fmax-errors=1 -Wconversion -W -Wall -std=c99 ${OpenMP_C_FLAGS} ${VECTORIZATION_FLAGS} ${TUNING_FLAGS} ${FAST_MATH_FLAGS}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_RELEASE}")
message("-- Debug CXXFLAGS are ${CMAKE_CXX_FLAGS_DEBUG}")
message("-- Release CXXFLAGS are ${CMAKE_CXX_FLAGS_RELEASE}")


add_subdirectory(Predict)
add_subdirectory(Gridder)
add_subdirectory(Array)