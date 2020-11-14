#include <iostream> 
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>  
#include <pcl/filters/passthrough.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/ia_ransac.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/PointIndices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>





typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::FPFHSignature33 FeatureT;                                     //FPFH特征描述子
typedef pcl::FPFHEstimation<PointNT, PointNT, FeatureT> FeatureEstimationT;  //FPFH特征估计类
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT; //自定义颜色句柄

ros::Subscriber sub;
ros::Publisher pub;




void pclCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
	
	PointCloudT::Ptr object(new PointCloudT);
	PointCloudNT::Ptr object_normal(new PointCloudNT);
	PointCloudNT::Ptr object_aligned(new PointCloudNT);

	PointCloudT::Ptr scene(new PointCloudT);
	PointCloudNT::Ptr scene_normal(new PointCloudNT);

	FeatureCloudT::Ptr object_features(new FeatureCloudT);
	FeatureCloudT::Ptr scene_features(new FeatureCloudT);

	
// if (pcl::io::loadPLYFile<PointT>("/home/seven/pointcloud_temp.ply", *object) == -1) //* load the file 
    // {
    //     PCL_ERROR("Couldn't read file test_pcd.pcd \n");
    //     return;
    // }
	// PointCloudT::Ptr object(new PointCloudT);
	if (pcl::io::loadPCDFile<PointT>("/home/seven/template.pcd", *object) == -1) //* load the file 
    {
        PCL_ERROR("Couldn't read file test_pcd.pcd \n");
        return;
    }

	pcl::fromROSMsg (*input, *scene);



	pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (scene);            //设置输入点云
    pass.setFilterFieldName ("z");         //设置过滤时所需要点云类型的Z字段
    pass.setFilterLimits (0.5, 1.8);        //设置在过滤字段的范围
    pass.filter (*scene);  



	// 下采样：使用0.005提速分辨率对目标物体和场景点云进行空间下采样
	pcl::console::print_highlight("Downsampling...\n");
	pcl::VoxelGrid<PointT> grid;
	float leaf = 0.005f;
	grid.setLeafSize(leaf, leaf, leaf);
	grid.setInputCloud(scene);
	grid.filter(*scene);
	grid.setInputCloud(object);
	grid.filter(*object);


	std::cout<<"降采样后的点云数量"<<object->points.size()<<",\t"<<scene->points.size()<<std::endl;

	pcl::PointCloud<pcl::Normal>::Ptr object_n(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr scene_n(new pcl::PointCloud<pcl::Normal>);

	// 估计场景法线
	pcl::console::print_highlight("Estimating scene normals...\n");
	pcl::NormalEstimation<PointT, pcl::Normal> nest;
	//pcl::NormalEstimationOMP<PointNT, PointNT> nest;
	nest.setRadiusSearch(0.025f);
	nest.setInputCloud(scene);
	nest.compute(*scene_n);

	nest.setRadiusSearch(0.025f);
	nest.setInputCloud(object);
	nest.compute(*object_n);

	pcl::concatenateFields(*scene, *scene_n, *scene_normal);
	pcl::concatenateFields(*object, *object_n, *object_normal);

	


	// 特征估计
	pcl::console::print_highlight("Estimating features...\n");
	FeatureEstimationT fest;
	fest.setRadiusSearch(0.025f);//该搜索半径决定FPFH特征描述的范围，一般设置为分辨率10倍以上
	fest.setInputCloud(object_normal);
	fest.setInputNormals(object_normal);
	fest.compute(*object_features);

	fest.setInputCloud(scene_normal);
	fest.setInputNormals(scene_normal);
	fest.compute(*scene_features);

	// 实施配准
	pcl::console::print_highlight("Starting alignment...\n");

		
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, FeatureT> sac_ia;
	pcl::PointCloud<pcl::PointXYZ>::Ptr align(new pcl::PointCloud<pcl::PointXYZ>);
	sac_ia.setInputSource(scene);			//设置源点云
	sac_ia.setSourceFeatures(scene_features);	//设置源点云特征
	sac_ia.setInputTarget(object);			//设置目标点云
	sac_ia.setTargetFeatures(object_features);	//设置目标点云特征

	// sac_ia.setInputSource(object);			//设置源点云
	// sac_ia.setSourceFeatures(object_features);	//设置源点云特征
	// sac_ia.setInputTarget(scene);			//设置目标点云
	// sac_ia.setTargetFeatures(scene_features);	//设置目标点云特征

	sac_ia.setNumberOfSamples(10);			//设置每次迭代计算中使用样本数量
	sac_ia.setCorrespondenceRandomness(6);	//设置计算协方差时选择多少临近点
	sac_ia.setMaximumIterations(1000);		//设置最大迭代次数
	sac_ia.setTransformationEpsilon(0.01);	//设置最大误差
	sac_ia.align(*align);					//配准

	std::cout<<sac_ia.getFinalTransformation().inverse()<<std::endl;
	Eigen::Matrix4f transformation = sac_ia.getFinalTransformation().inverse();
	Eigen::Matrix3f rotation_matrix=transformation.block(0,0,3,3);
	Eigen::Quaternionf q(rotation_matrix);
	Eigen::Vector3f eulerAngle1 = rotation_matrix.eulerAngles(2,1,0);
	std::cout<<eulerAngle1<<std::endl;
	std::cout<<q.x()<<","<<q.y()<<","<<q.z()<<","<<q.w()<<std::endl;


	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud (*object, *transformed_cloud, transformation);


	sensor_msgs::PointCloud2 output;	



	pcl::toROSMsg(*transformed_cloud, output);
	output.header.stamp=ros::Time::now();
    output.header.frame_id = input->header.frame_id;
    pub.publish(output);

  
	

	// pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  	// icp.setInputCloud(transformed_cloud);
  	// icp.setInputTarget(scene);
  	// pcl::PointCloud<pcl::PointXYZ> Final;
  	// icp.align(Final);
  	// std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  	// icp.getFitnessScore() << std::endl;
  	// std::cout << icp.getFinalTransformation() << std::endl;

    // pcl::transformPointCloud (*transformed_cloud, *transformed_cloud, icp.getFinalTransformation());
	// sensor_msgs::PointCloud2 output;	

	// pcl::toROSMsg(*transformed_cloud, output);
	// output.header.stamp=ros::Time::now();
    // output.header.frame_id = input->header.frame_id;
    // pub.publish(output);
	// pcl::SampleConsensusPrerejective<PointNT, PointNT, FeatureT> align;//基于采样一致性的位姿估计
	// align.setInputSource(object_normal);
	// align.setSourceFeatures(object_features);
	// align.setInputTarget(scene_normal);
	// align.setTargetFeatures(scene_features);
	// align.setMaximumIterations(200000);  //  采样一致性迭代次数
	// align.setNumberOfSamples(3);          //  创建假设所需的样本数，为了正常估计至少需要3点
	// align.setCorrespondenceRandomness(5); //  使用的临近特征点的数目
	// align.setSimilarityThreshold(0.9f);   //  多边形边长度相似度阈值
	// align.setMaxCorrespondenceDistance(2.5f * 0.005); //  判断是否为内点的距离阈值
	// align.setInlierFraction(0.1f);       //接受位姿假设所需的内点比例
	// {
	// 	pcl::ScopeTime t("Alignment");
	// 	align.align(*object_aligned);
	// }

	// if (align.hasConverged())
	// {
	// 	// pcl::IndicesConstPtr 
	// 	// align.
	// 	pcl::IndicesPtr index_ptr=align.getIndices();
	// 	pcl::ExtractIndices<pcl::PointXYZ> extract;
	//      // Extract the inliers
    //  	extract.setInputCloud(scene);
    //  	extract.setIndices(index_ptr);
    //  	extract.setNegative(false);//如果设为true,可以提取指定index之外的点云
    //  	extract.filter(*scene);
	// 	// Print results
	// 	printf("\n");
	// 	Eigen::Matrix4f transformation = align.getFinalTransformation();
	// 	Eigen::Matrix3f rotation_matrix=transformation.block(0,0,3,3);
	// 	Eigen::Quaternionf q(rotation_matrix);
	// 	// Eigen::Vector3f eulerAngle1 = rotation_matrix.eulerAngles(2,1,0);
	// 	// std::cout<<eulerAngle1<<std::endl;
	// 	std::cout<<q.x()<<","<<q.y()<<","<<q.z()<<","<<q.w()<<std::endl;
	// 	pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(0, 0), transformation(0, 1), transformation(0, 2));
	// 	pcl::console::print_info("R = | %6.3f %6.3f %6.3f | \n", transformation(1, 0), transformation(1, 1), transformation(1, 2));
	// 	pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(2, 0), transformation(2, 1), transformation(2, 2));
	// 	pcl::console::print_info("\n");
	// 	pcl::console::print_info("t = < %0.3f, %0.3f, %0.3f >\n", transformation(0, 3), transformation(1, 3), transformation(2, 3));
	// 	pcl::console::print_info("\n");
	// 	pcl::console::print_info("Inliers: %i/%i\n", align.getInliers().size(), object->size());
	// 	pcl::console::print_info("FitnessScore: %6.6f\n",align.getFitnessScore());	
	// 	// // Show alignment
	// 	// pcl::visualization::PCLVisualizer visu("点云库PCL学习教程第二版-鲁棒位姿估计");
	// 	// int v1(0), v2(0);
	// 	// visu.createViewPort(0, 0, 0.5, 1, v1);
	// 	// visu.createViewPort(0.5, 0, 1, 1, v2);
	// 	// visu.setBackgroundColor(255, 255, 255, v1);
	// 	// visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 255.0, 0.0), "scene", v1);
	// 	// visu.addPointCloud(object_aligned, ColorHandlerT(object_aligned, 0.0, 0.0, 255.0), "object_aligned", v1);

	// 	// visu.addPointCloud(object, ColorHandlerT(object, 0.0, 255.0, 0.0), "object_before_aligned", v2);
	// 	// visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 0.0, 255.0), "scene_v2", v2);
	// 	// visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene");
	// 	// visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "object_aligned");
	// 	// visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "object_before_aligned");
	// 	// visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene_v2");
	// 	// visu.spin();
	// 	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  	// 	icp.setInputCloud(object);
  	// 	icp.setInputTarget(scene);
  	// 	pcl::PointCloud<pcl::PointXYZ> Final;
  	// 	icp.align(Final);
  	// 	std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  	// 	icp.getFitnessScore() << std::endl;
  	// 	std::cout << icp.getFinalTransformation() << std::endl;
	// }
	// else
	// {
	// 	pcl::console::print_error("Alignment failed!\n");
	// 	return;
	// }
	// sub.shutdown();
    
}
	
															 // Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv)
{
	ros::init(argc, argv, "filter");
	ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe ("/camera/depth/points", 1, pclCallback);
	pub= nh.advertise<sensor_msgs::PointCloud2> ("cloud_preprocessed", 1);
	ros::spin();
	return (0);
}
