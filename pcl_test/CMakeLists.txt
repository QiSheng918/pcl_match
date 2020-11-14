cmake_minimum_required(VERSION 2.8.3)
project(pcl_test)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
   roscpp
   rospy
   std_msgs
   sensor_msgs
   cv_bridge
   image_transport
   pcl_conversions
   pcl_ros
   geometry_msgs
   visualization_msgs
   tf
)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
# MESSAGE(STATUS "This is SOURCE dir" ${PROJECT_SOURCE_DIR})









################################################
## Declare ROS dynamic reconfigure parameters ##

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES agv_description
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    # include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    
)
# MESSAGE(FATAL_ERROR 
# ${PCL_INCLUDE_DIRS})
## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/agv_description.cpp
# )


# add_executable(icp_test_node ${PROJECT_SOURCE_DIR}/src/icp_test.cpp)
# target_link_libraries(icp_test_node
#    ${PCL_LIBRARIES} 
#    ${catkin_LIBRARIES}
#    ${Eigen3_INCLUDE_DIRS}

# )

# add_executable(pcl_save_node ${PROJECT_SOURCE_DIR}/src/pcl_save.cpp)
# target_link_libraries(pcl_save_node
#    ${PCL_LIBRARIES} 
#    ${catkin_LIBRARIES}
#    ${Eigen3_INCLUDE_DIRS}

# )


# add_executable(fpfh_test_node ${PROJECT_SOURCE_DIR}/src/fpfh_test.cpp)
# target_link_libraries(fpfh_test_node
#    ${PCL_LIBRARIES} 
#    ${catkin_LIBRARIES}
#    ${Eigen3_INCLUDE_DIRS}

# )


add_executable(pcl_display_node src/pcl_display.cpp)
target_link_libraries(pcl_display_node
   ${PCL_LIBRARIES} 
   ${catkin_LIBRARIES}
   ${Eigen3_INCLUDE_DIRS}
)


# add_executable(pcl_filter_node src/pcl_filter.cpp)
# target_link_libraries(pcl_filter_node
#    ${PCL_LIBRARIES} 
#    ${catkin_LIBRARIES}
#    ${Eigen3_INCLUDE_DIRS}
# )






