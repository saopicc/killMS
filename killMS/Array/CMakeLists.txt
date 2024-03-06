cmake_minimum_required(VERSION 2.8)

#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(Dot)
install(FILES __init__.py DESTINATION killMS/cbuild/Array)