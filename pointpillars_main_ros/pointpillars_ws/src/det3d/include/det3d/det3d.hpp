
#pragma once
#include <math.h>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

// cuda
#include "cuda_runtime.h"
// ros
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <pcl/filters/passthrough.h>
//#include "autoware_msgs/DetectedObjectArray.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf/transform_listener.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
// pcl
#include "pcl/point_cloud.h"
#include "pcl_conversions/pcl_conversions.h"
// pointpillars headers
#include "params.h"
#include "pointpillar.h"

#define __APP_NAME__ "Det3D"

class Det3D {
 public:
  Det3D(ros::NodeHandle nh, ros::NodeHandle pnh) : nh_(nh), pnh_(pnh) {
    pnh_.param("topic", topic_, std::string("/camera/depth/color/points"));
    pnh_.param("cloud_topic", cloud_topic, std::string("/cloud_points"));
    pnh_.param("result_topic", result_topic, std::string("/rslidar_points/result"));
    pnh_.param("model_path", model_path_, std::string("/home/lenovo/Downloads/3Ddetection/pointpillars-main/pointpillars_ws/src/det3d/model/pointpillar.onnx"));
    pnh_.param("score_threshold", score_threshold_, static_cast<float>(0.8));
  }
  void start();

 private:
  // callback
  void get_info(void);
  void pub_box_pred(std::vector<Bndbox> boxes);
  void callback(const sensor_msgs::PointCloud2ConstPtr &msg);
  int load_data(const sensor_msgs::PointCloud2ConstPtr &msg, void **data, unsigned int *length);

 private:
  // ros
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // roslaunch params
  std::string topic_;
  float score_threshold_;
  std::string model_path_;
  std::string cloud_topic;
  std::string result_topic;

  // cuda
  cudaEvent_t start_, stop_;
  cudaStream_t stream_ = NULL;

  // pointpillars
  Params params_;
  float elapsed_time_ = 0.0f;
  std::vector<Bndbox> nms_pred_;
  std::unique_ptr<PointPillar> pointpillar_ptr_;

  // msgs subscriber
  ros::Subscriber sub_;
  ros::Publisher pub_;
  ros::Publisher pub_cloud_;

  // temp
  std::string frame_id_ = "map";
};
