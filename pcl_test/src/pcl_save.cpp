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
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <iostream> 
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>  
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <string>



ros::Subscriber sub;
ros::Publisher pub;
int i=0;

void pclCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
	

	
	// if (pcl::io::loadPLYFile<PointT>("/home/seven/pointcloud_temp.ply", *object) == -1) //* load the file 
    // {
    //     PCL_ERROR("Couldn't read file test_pcd.pcd \n");
    //     return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromROSMsg (*input, *scene);

	// pcl::PassThrough<pcl::PointXYZ> pass;

    // pass.setInputCloud (scene);            //设置输入点云
    // pass.setFilterFieldName ("z");         //设置过滤时所需要点云类型的Z字段
    // pass.setFilterLimits (0.5, 1.8);        //设置在过滤字段的范围
    // pass.filter (*scene);  
	std::string file_name="wangxu"+std::to_string(i);
	pcl::io::savePLYFile("/home/seven/"+file_name+".ply", *scene);
	i++;
	if(i==1){
		sub.shutdown();
		ros::shutdown();
	}

    
}
	
															 // Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv)
{
	ros::init(argc, argv, "filter");
	ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe ("/cloud_fpfh", 1, pclCallback);
	// Point clouds
	ros::spin();
	return (0);
}
