cmake_minimum_required(VERSION 2.8)

project(arraydot)
#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

#compile and link predict.so
add_library(predict predict.c predict.h)
set_target_properties(predict PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
target_link_libraries(predict ${RT_LIBRARIES} ${PYTHON_LIBRARY})

#compile and link predict.so
add_library(predict_np19 predict_np19.c predict_np19.h)
set_target_properties(predict_np19 PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
target_link_libraries(predict_np19 ${RT_LIBRARIES} ${PYTHON_LIBRARY})