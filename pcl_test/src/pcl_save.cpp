
#include <iostream> 

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>  
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <string>



ros::Subscriber sub;
int i=0;

std::string pcl_topic_name;

void pclCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromROSMsg (*input, *scene);

	// 保存为ply格式文件
	std::string file_name="wangxu"+std::to_string(i);
	pcl::io::savePLYFile("/home/seven/"+file_name+".ply", *scene);
	i++;
	if(i==1){
		sub.shutdown();
		ros::shutdown();
	}


	// 保存为pcd格式文件
	// std::string file_name="wangxu"+std::to_string(i);
	// pcl::io::savePCDFile("/home/seven/"+file_name+".pcd", *scene);
	// i++;
	// if(i==1){
	// 	sub.shutdown();
	// 	ros::shutdown();
	// }

    
}
	
int main(int argc, char **argv)
{
	ros::init(argc, argv, "pcl_save_node");
	ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe (pcl_topic_name, 1, pclCallback);
	ros::spin();
	return (0);
}
