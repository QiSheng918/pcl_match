#include <iostream> 
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
// #include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>  
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/ia_ransac.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/PointIndices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>





typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::FPFHSignature33 FeatureT;                                     //FPFH特征描述子
typedef pcl::FPFHEstimation<PointNT, PointNT, FeatureT> FeatureEstimationT;  //FPFH特征估计类
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT; //自定义颜色句柄


ros::Subscriber sub;
ros::Publisher filter_pub;
ros::Publisher fpfh_pub;
ros::Publisher icp_pub;

PointCloudT::Ptr object(new PointCloudT);
PointCloudNT::Ptr object_normal(new PointCloudNT);
FeatureCloudT::Ptr object_features(new FeatureCloudT);


void loadPclFiles(PointCloudT::Ptr &cloud_ptr){
	// if (pcl::io::loadPLYFile<PointT>("/home/seven/pointcloud_temp.ply", *object) == -1) //* load the file 
    // {
    //     PCL_ERROR("Couldn't read file test_pcd.pcd \n");
    //     return;
    // }
	// PointCloudT::Ptr object(new PointCloudT);
	if (pcl::io::loadPCDFile<PointT>("/home/seven/template_part.pcd", *cloud_ptr) == -1) //* load the file 
    {
        PCL_ERROR("Couldn't read file test_pcd.pcd \n");
        return;
    }
}

void downSizeFilter(PointCloudT::Ptr &cloud_ptr){
	pcl::VoxelGrid<PointT> grid;
	float leaf = 0.01f;
	grid.setLeafSize(leaf, leaf, leaf);
	grid.setInputCloud(cloud_ptr);
	grid.filter(*cloud_ptr);
}

void outerPointFilter(PointCloudT::Ptr &cloud_ptr,int K_means=30){
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud_ptr);
	sor.setMeanK(K_means); //K近邻搜索点个数
	sor.setStddevMulThresh(1.0); //标准差倍数
	sor.setNegative(false); //保留未滤波点（内点）
	sor.filter(*cloud_ptr);  //保存滤波结果到cloud_filter
}

void normalCalculate(PointCloudT::Ptr &cloud_ptr,PointCloudNT::Ptr &cloud_normal_ptr){
	pcl::PointCloud<pcl::Normal>::Ptr cloud_n(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<PointT, pcl::Normal> nest;
	
	//pcl::NormalEstimationOMP<PointNT, PointNT> nest;
	nest.setRadiusSearch(0.02f);
	nest.setInputCloud(cloud_ptr);
	nest.compute(*cloud_n);
	pcl::concatenateFields(*cloud_ptr, *cloud_n, *cloud_normal_ptr);
}

void featuresEstimate(PointCloudNT::Ptr &cloud_normal_ptr,FeatureCloudT::Ptr &cloud_normal_features_ptr){
	FeatureEstimationT fest;
	fest.setRadiusSearch(0.05f);//该搜索半径决定FPFH特征描述的范围，一般设置为分辨率10倍以上
	fest.setInputCloud(cloud_normal_ptr);
	fest.setInputNormals(cloud_normal_ptr);
	fest.compute(*cloud_normal_features_ptr);
}

Eigen::Matrix4f sacIaMatch(PointCloudT::Ptr &source,PointCloudT::Ptr &target,FeatureCloudT::Ptr &source_feature,FeatureCloudT::Ptr &target_feature){
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, FeatureT> sac_ia;
	pcl::PointCloud<pcl::PointXYZ>::Ptr align(new pcl::PointCloud<pcl::PointXYZ>);
	sac_ia.setInputSource(source);			//设置源点云
	sac_ia.setSourceFeatures(source_feature);	//设置源点云特征
	sac_ia.setInputTarget(target);			//设置目标点云
	sac_ia.setTargetFeatures(target_feature);	//设置目标点云特征
	sac_ia.setNumberOfSamples(10);			//设置每次迭代计算中使用样本数量
	sac_ia.setCorrespondenceRandomness(5);	//设置计算协方差时选择多少临近点
	sac_ia.setMaximumIterations(10000);		//设置最大迭代次数
	sac_ia.setTransformationEpsilon(0.01);	//设置最大误差
	sac_ia.align(*align);					//配准
	if(sac_ia.hasConverged()){
		return sac_ia.getFinalTransformation();
	}
	return Eigen::Matrix4f::Identity();
}

Eigen::Matrix4f ranscMatch(PointCloudT::Ptr &source,PointCloudT::Ptr &target,FeatureCloudT::Ptr &source_feature,FeatureCloudT::Ptr &target_feature){
	pcl::SampleConsensusPrerejective<PointT, PointT, FeatureT> ransc;//基于采样一致性的位姿估计
	ransc.setInputSource(source);
	ransc.setSourceFeatures(source_feature);
	ransc.setInputTarget(target);
	ransc.setTargetFeatures(target_feature);
	// ransc.setMaximumIterations(4000000);  //  采样一致性迭代次数
	// ransc.setNumberOfSamples(3);          //  创建假设所需的样本数，为了正常估计至少需要3点
	// ransc.setCorrespondenceRandomness(6); //  使用的临近特征点的数目
	// ransc.setSimilarityThreshold(0.9f);   //  多边形边长度相似度阈值
	// ransc.setMaxCorrespondenceDistance(0.004f ); //  判断是否为内点的距离阈值
	ransc.setInlierFraction(0.2f);       //接受位姿假设所需的内点比例
	// ransc.setTransformationEstimation
	pcl::PointCloud<pcl::PointXYZ>::Ptr align(new pcl::PointCloud<pcl::PointXYZ>);
	ransc.align(*align);
	if(ransc.hasConverged()){
		return ransc.getFinalTransformation();
	}
	return Eigen::Matrix4f::Identity();
}


void planeWithNormalSeg(PointCloudT::Ptr &cloud_ptr,PointCloudNT::Ptr &cloud_normal_ptr){
	pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::PointNormal> seg;
	pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);

	pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

	pcl::ExtractIndices<pcl::PointXYZ> extract;


	seg.setOptimizeCoefficients(true);//设置对估计的模型系数需要进行优化
	seg.setModelType(pcl::SACMODEL_NORMAL_PLANE); //设置分割模型
	seg.setNormalDistanceWeight(0.1);//设置表面法线权重系数
	seg.setMethodType(pcl::SAC_RANSAC);//设置采用RANSAC作为算法的参数估计方法
	seg.setMaxIterations(20000); //设置迭代的最大次数
	seg.setDistanceThreshold(0.07f); //设置内点到模型的距离允许最大值
	seg.setInputCloud(cloud_ptr);
	seg.setInputNormals(cloud_normal_ptr);

	seg.segment(*inliers_plane, *coefficients_plane);

	// std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;
	extract.setInputCloud(cloud_ptr);
	extract.setIndices(inliers_plane);
	extract.setNegative(true);
	extract.filter(*cloud_ptr);

	pcl::ExtractIndices<pcl::PointNormal> extract_normal;
	extract_normal.setInputCloud(cloud_normal_ptr);
	extract_normal.setIndices(inliers_plane);
	extract_normal.setNegative(true);
	extract_normal.filter(*cloud_normal_ptr);
}

void planeWithoutNormalSeg(PointCloudT::Ptr &cloud_ptr,PointCloudNT::Ptr &cloud_normal_ptr){

 	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);

	pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
	seg.setOptimizeCoefficients (true);
	//必须设置
	seg.setModelType(pcl::SACMODEL_PLANE); //设置模型类型，检测平面
  	seg.setMethodType(pcl::SAC_RANSAC);		//设置方法【聚类或随机样本一致性】
  	seg.setDistanceThreshold (0.03);
  	seg.setInputCloud(cloud_ptr);
  	seg.segment (*inliers_plane, *coefficients_plane);	//分割操作

	pcl::ExtractIndices<pcl::PointXYZ> extract;
	// std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;
	extract.setInputCloud(cloud_ptr);
	extract.setIndices(inliers_plane);
	extract.setNegative(true);
	extract.filter(*cloud_ptr);

	pcl::ExtractIndices<pcl::PointNormal> extract_normal;

	extract_normal.setInputCloud(cloud_normal_ptr);
	extract_normal.setIndices(inliers_plane);
	extract_normal.setNegative(true);
	extract_normal.filter(*cloud_normal_ptr);
}

void pclCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
	PointCloudT::Ptr scene(new PointCloudT);
	PointCloudNT::Ptr scene_normal(new PointCloudNT);
	FeatureCloudT::Ptr scene_features(new FeatureCloudT);

	pcl::fromROSMsg (*input, *scene);
	
	
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud (scene);            //设置输入点云
	pass.setFilterFieldName ("z");         //设置过滤时所需要点云类型的Z字段
	pass.setFilterLimits (0.2, 1);        //设置在过滤字段的范围
	pass.filter (*scene);  

	pass.setInputCloud (scene);            //设置输入点云
	pass.setFilterFieldName ("x");         //设置过滤时所需要点云类型的Z字段
	pass.setFilterLimits (-0.4, 0.4);        //设置在过滤字段的范围
	pass.filter (*scene);  

	pass.setInputCloud (scene);            //设置输入点云
	pass.setFilterFieldName ("y");         //设置过滤时所需要点云类型的Z字段
	pass.setFilterLimits (-0.4, 0.4);        //设置在过滤字段的范围
	pass.filter (*scene);  


	// sensor_msgs::PointCloud2 filter_output;	
	// pcl::toROSMsg(*scene, filter_output);
	// filter_output.header.stamp=ros::Time::now();
    // filter_output.header.frame_id = input->header.frame_id;
    // filter_pub.publish(filter_output);
	
	outerPointFilter(scene,50);
	std::cout<<"preProcessing"<<std::endl;
	downSizeFilter(scene);
	// outerPointFilter(scene);

	std::cout<<"normal vector calcalating"<<std::endl;
	normalCalculate(scene,scene_normal);

	planeWithNormalSeg(scene,scene_normal);

	// planeWithNormalSeg(scene,scene_normal);


 	Eigen::Vector4f centroid;
 



	std::cout<<scene->points.size()<<","<<scene_normal->points.size()<<std::endl;
	outerPointFilter(scene,30);

	scene_normal->clear();

	std::cout<<scene->points.size()<<","<<scene_normal->points.size()<<std::endl;

	normalCalculate(scene,scene_normal);
	
	pcl::compute3DCentroid(*scene, centroid);
	std::cout << "点云质心是（"<< centroid[0] << ","<< centroid[1] << ","<< centroid[2] << ")." << std::endl;

	std::cout<<scene->points.size()<<","<<scene_normal->points.size()<<std::endl;

	sensor_msgs::PointCloud2 filter_output;	
	pcl::toROSMsg(*scene, filter_output);
	filter_output.header.stamp=ros::Time::now();
    filter_output.header.frame_id = input->header.frame_id;
    filter_pub.publish(filter_output);
	// sensor_msgs::PointCloud2 fpfh_output;
	// pcl::toROSMsg(*scene, fpfh_output);
	// fpfh_output.header.stamp=ros::Time::now();
    // fpfh_output.header.frame_id = input->header.frame_id;
    // fpfh_pub.publish(fpfh_output);

	std::cout<<"fpfh features estimating"<<std::endl;
	featuresEstimate(scene_normal,scene_features);
	
	std::cout<<"matching"<<std::endl;


	// Eigen::Matrix4f transform_matrix=ranscMatch(object,scene,object_features,scene_features);

	// std::cout<<transform_matrix<<std::endl;
	
	// PointCloudT::Ptr transformed_cloud(new PointCloudT);

    // pcl::transformPointCloud(*object, *transformed_cloud,transform_matrix);

	// sensor_msgs::PointCloud2 fpfh_output;
	// pcl::toROSMsg(*transformed_cloud, fpfh_output);
	// fpfh_output.header.stamp=ros::Time::now();
    // fpfh_output.header.frame_id = input->header.frame_id;
    // fpfh_pub.publish(fpfh_output);



	

	Eigen::Matrix4f transform_matrix=ranscMatch(scene,object,scene_features,object_features).inverse();
	std::cout<<transform_matrix<<std::endl;
	
	PointCloudT::Ptr transformed_cloud(new PointCloudT);

    pcl::transformPointCloud (*object, *transformed_cloud,transform_matrix);

	sensor_msgs::PointCloud2 fpfh_output;
	pcl::toROSMsg(*transformed_cloud, fpfh_output);
	fpfh_output.header.stamp=ros::Time::now();
    fpfh_output.header.frame_id = input->header.frame_id;
    fpfh_pub.publish(fpfh_output);


	pcl::visualization::PCLVisualizer visu("点云库PCL学习教程第二版-鲁棒位姿估计");
	int v1(0), v2(0);
	visu.createViewPort(0, 0, 0.5, 1, v1);
	visu.createViewPort(0.5, 0, 1, 1, v2);
	visu.setBackgroundColor(255, 255, 255, v1);
	visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 255.0, 0.0), "scene", v1);

	visu.addPointCloud(transformed_cloud, ColorHandlerT(transformed_cloud, 255.0, 0.0, 0.0), "transformed_cloud", v1);

	visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene");
	visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "transformed_cloud");

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(scene);
	icp.setInputTarget(transformed_cloud);
	// icp.setMaximumIterations(1000);
	pcl::PointCloud<pcl::PointXYZ> Final;
	icp.align(Final);
	// std::cout << "has converged:" << icp.hasConverged() << " score: " <<
	// icp.getFitnessScore() << std::endl;
	Eigen::Matrix4f transform_matrix2=icp.getFinalTransformation().inverse();
	Eigen::Matrix3f R=transform_matrix2.block(0,0,3,3);
	std::cout<<R.eulerAngles(2,1,0)<<std::endl;
	// Eigen::AngleAxisd V2;
    // V2.fromRotationMatrix(R);
	// std::cout<<V2<<std::endl;
	sensor_msgs::PointCloud2 icp_output;
	pcl::toROSMsg(*transformed_cloud, icp_output);
	icp_output.header.stamp=ros::Time::now();
    icp_output.header.frame_id = input->header.frame_id;
    icp_pub.publish(icp_output);
	

	pcl::transformPointCloud (*transformed_cloud, *transformed_cloud,transform_matrix2);

	visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 255.0, 0.0), "scene_icp", v2);

	visu.addPointCloud(transformed_cloud, ColorHandlerT(transformed_cloud, 255.0, 0.0, 0.0), "transformed_cloud_icp", v2);

	visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene_icp");
	visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "transformed_cloud_icp");

	visu.spin();
	std::cout<<transform_matrix*transform_matrix2<<std::endl;
}
	
															 // Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv)
{
	ros::init(argc, argv, "filter");
	ros::NodeHandle nh;
	// object=new PointCloudT;
	// object_normal=new PointCloudNT;
	// object_features=new FeatureCloudT;
	// std::cout<<object->points.size()<<std::endl;
	loadPclFiles(object);
	std::cout<<object->points.size()<<std::endl;
	downSizeFilter(object);
	std::cout<<object->points.size()<<std::endl;

	normalCalculate(object,object_normal);
	std::cout<<object_normal->points.size()<<std::endl;
	featuresEstimate(object_normal,object_features);

    ros::Subscriber sub = nh.subscribe ("/camera/depth/color/points", 1, pclCallback);
	filter_pub= nh.advertise<sensor_msgs::PointCloud2> ("cloud_filter", 1);
	fpfh_pub= nh.advertise<sensor_msgs::PointCloud2> ("cloud_fpfh", 1);
	icp_pub= nh.advertise<sensor_msgs::PointCloud2> ("cloud_icp", 1);
	ros::spin();
	return (0);
}
