cmake_minimum_required(VERSION 2.8)

project( upsampling )

find_package ( OpenCV REQUIRED )

add_executable ( upsampling src/main.cpp )
target_link_libraries ( upsampling ${OpenCV_LIBS} )