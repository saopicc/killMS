cmake_minimum_required(VERSION 2.8)

function(build_be PYVER)
  #Find necessary modules
  set(PYREQ_VER 3.0)
  set(PYABI .cpython-30)
  find_package(PythonInterp ${PYREQ_VER} REQUIRED)    
  find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} EXACT REQUIRED) #must match
  
  find_package(NumPy REQUIRED)
  find_package(OpenMP REQUIRED)
  find_package(RT REQUIRED)

  if (NOT ${PYTHON_FOUND})
    message(FATAL_ERROR "Failed to find Python, ensure python-dev and/or python3-dev is installed")
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

  #Update include paths with found includes above
  include_directories(${PYTHON_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIR}/numpy ${RT_INCLUDES} ${CASACORE_INCLUDE_DIR} ${PYTHON_PYBIND11_INCLUDE_DIR})

  set(PYBIND11_CPP_STANDARD -std=c++14)

  project(arraydot)
  #Ensure __init__.py is added to out-of-source build directory
  execute_process(COMMAND touch __init__.py
		  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  
  #compile and link predict.so
  add_library(predict${PYVER} predict.c predict.h)
  set_target_properties(predict${PYVER} PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
  target_link_libraries(predict${PYVER} ${RT_LIBRARIES} ${PYTHON_LIBRARY})
  install(TARGETS predict${PYVER} LIBRARY DESTINATION killMS/cbuild/Predict)
endfunction(build_be)

if (${ENABLE_PYTHON_3})
  build_be("3x")
endif (${ENABLE_PYTHON_3})

# ensure __init__ gets propagated to non-editable install
install(FILES __init__.py DESTINATION killMS/cbuild/Predict)