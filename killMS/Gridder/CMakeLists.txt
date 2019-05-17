cmake_minimum_required(VERSION 2.8)

project(arraydot)
#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

#compile and link _pyGridder.so
add_library(_pyGridder Gridder.c Gridder.h)
set_target_properties(_pyGridder PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
target_link_libraries(_pyGridder ${RT_LIBRARIES} ${PYTHON_LIBRARY})