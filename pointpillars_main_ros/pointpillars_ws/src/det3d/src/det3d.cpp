

#include <fstream>
#include <iostream>
#include <sstream>

#include "cuda_runtime.h"
// pointpillars headers
#include "params.h"
#include "pointpillar.h"
// local
#include "det3d/det3d.hpp"
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#define checkCudaErrors(status)                                       \
  {                                                                   \
    if (status != 0) {                                                \
      std::cout << "Cuda failure: " << cudaGetErrorString(status)     \
                << " at line " << __LINE__ << " in file " << __FILE__ \
                << " error status: " << status << std::endl;          \
      abort();                                                        \
    }                                                                 \
  }

void Det3D::start() {
  // pointpillars init
  get_info();
  checkCudaErrors(cudaEventCreate(&stop_));
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaStreamCreate(&stream_));

  nms_pred_.reserve(100);
  pointpillar_ptr_.reset(new PointPillar(model_path_, stream_));
  // PointPillar pointpillar(model_path_, stream_);
  // ros init
  sub_ = nh_.subscribe(topic_, 1, &Det3D::callback, this);
  pub_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>(result_topic, 1);
  pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(cloud_topic, 1, true);

  ros::spin();
}

void Det3D::callback(const sensor_msgs::PointCloud2ConstPtr &msg) {
  // load points cloud from ros msg
  frame_id_ = msg->header.frame_id;
  unsigned int length = 0;
  void *data = NULL;
  std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
  load_data(msg, &data, &length);
  buffer.reset((char *)data);

  // format data to 4 channels
  float *points = (float *)buffer.get();
  size_t points_size = length / sizeof(float) / 4;

  ROS_DEBUG_STREAM("find points num: " << points_size);

  float *points_data = nullptr;
  unsigned int points_data_size = points_size * 4 * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
  checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
  checkCudaErrors(cudaDeviceSynchronize());
  cudaEventRecord(start_, stream_);

  pointpillar_ptr_->doinfer(points_data, points_size, nms_pred_);
  cudaEventRecord(stop_, stream_);
  cudaEventSynchronize(stop_);
  cudaEventElapsedTime(&elapsed_time_, start_, stop_);
  ROS_DEBUG_STREAM("TIME: pointpillar: " << elapsed_time_ << " ms.");

  checkCudaErrors(cudaFree(points_data));
  // std::string save_file_name = Save_Dir + index_str + ".txt";
  pub_box_pred(nms_pred_);
  nms_pred_.clear();

  ROS_DEBUG_STREAM(">>>>>>>>>>>");
}

void Det3D::get_info(void) {

  cudaDeviceProp prop;
  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

int Det3D::load_data(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, void **data, unsigned int *length) {

  int pointBytes = cloud_msg->point_step;
  int offset_x, offset_y, offset_z, offset_int;

  for (int f = 0; f < cloud_msg->fields.size(); f++) {
    if (cloud_msg->fields[f].name == "x") {
      offset_x = cloud_msg->fields[f].offset;
    }
    if (cloud_msg->fields[f].name == "y") {
      offset_y = cloud_msg->fields[f].offset;
    }
    if (cloud_msg->fields[f].name == "z") {
      offset_z = cloud_msg->fields[f].offset;
    }
    if (cloud_msg->fields[f].name == "intensity") {
      offset_int = cloud_msg->fields[f].offset;
    }
  }

  std::cout << "x: " << offset_x << ", " << "y: " << offset_y << ", " << "z: " << offset_z << ", " << "intensity: " << offset_int << std::endl;
  int num_channels = 4;
  int num_points = cloud_msg->height * cloud_msg->width;
  *length = num_points * num_channels * sizeof(float);
  *data = malloc(*length);
  float *data_ptr = (float *)(*data);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_pcl(new pcl::PointCloud<pcl::PointXYZI>);
  
  for (int p = 0; p < num_points; ++p) {

    pcl::PointXYZI newPoints;
    newPoints.x = *(float*)(&cloud_msg->data[0] + (pointBytes * p) + offset_x);
    newPoints.y = *(float*)(&cloud_msg->data[0] + (pointBytes * p) + offset_y);
    newPoints.z = *(float*)(&cloud_msg->data[0] + (pointBytes * p) + offset_z);
    newPoints.intensity = *(unsigned char*)(&cloud_msg->data[0] + (pointBytes * p) + offset_int);

    if (newPoints.x > 50 || newPoints.x < 0 || newPoints.y > 10 || newPoints.y < -10 || newPoints.z > 30 || newPoints.z < -10) {
      continue;
    }

    cloud_pcl->push_back(newPoints);
    data_ptr[p * num_channels + 0] = newPoints.x;
    data_ptr[p * num_channels + 1] = newPoints.y;
    data_ptr[p * num_channels + 2] = newPoints.z;
    data_ptr[p * num_channels + 3] = newPoints.intensity;
  }

  sensor_msgs::PointCloud2 cloud_msg_;
  pcl::toROSMsg(*cloud_pcl, cloud_msg_);
  cloud_msg_.header.stamp = ros::Time::now();
  cloud_msg_.header.frame_id = frame_id_;
  pub_cloud_.publish(cloud_msg_);

  return 0;
}

// int Det3D::load_data(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, void **data, unsigned int *length) {
//   // convert sensor_msgs::PointCloud to array
//   pcl::PCLPointCloud2 pcl_pc2;
//   pcl_conversions::toPCL(*msg, pcl_pc2);
  
//   // use the first 4 channels
//   int num_points = cloud_pcl_z->points.size();
//   int num_channels = 4;
//   *length = num_points * num_channels * sizeof(float);
//   *data = malloc(*length);
//   float *data_ptr = (float *)(*data);
//   for (int i = 0; i < num_points; i++) {
//     data_ptr[i * num_channels + 0] = pcl_pc->points[i].x;
//     data_ptr[i * num_channels + 1] = pcl_pc->points[i].y;
//     data_ptr[i * num_channels + 2] = pcl_pc->points[i].z;
//     data_ptr[i * num_channels + 3] = pcl_pc->points[i].intensity / 255.0;
//   }
//   return 0;
// }

// int Det3D::load_data(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, void **data, unsigned int *length) {
//   // convert sensor_msgs::PointCloud to array
//   // pcl::PCLPointCloud2 cloud_msgs;
//   // pcl_conversions::toPCL(*cloud_msg, cloud_msgs);

//   // pcl::PassThrough<pcl::PointXYZI> pass;
//   // pass.setInputCloud (cloud_msg); 
//   // pass.setFilterFieldName ("z");
//   // pass.setFilterLimits (0.0, 1.0);
//   // pass.filter (*cloud_filtered);

//   int pointBytes = cloud_msg->point_step;
//   int offset_x, offset_y, offset_z, offset_int;

//   for (int f = 0; f < cloud_msg->fields.size(); f++) {
//     if (cloud_msg->fields[f].name == "x") {
//       offset_x = cloud_msg->fields[f].offset;
//     }
//     if (cloud_msg->fields[f].name == "y") {
//       offset_y = cloud_msg->fields[f].offset;
//     }
//     if (cloud_msg->fields[f].name == "z") {
//       offset_z = cloud_msg->fields[f].offset;
//     }
//     if (cloud_msg->fields[f].name == "intensity") {
//       offset_int = cloud_msg->fields[f].offset;
//     }
//   }

//   std::cout << "x: " << offset_x << ", " << "y: " << offset_y << ", " << "z: " << offset_z << ", " << "intensity: " << offset_int << std::endl;
//   int num_points = cloud_msg->height * cloud_msg->width;
//   int num_channels = 4;
//   *length = num_points * num_channels * sizeof(float);
//   *data = malloc(*length);
//   float *data_ptr = (float *)(*data);

//   // for (int p=0; p<cloud_msgs.width*cloud_msgs.height; ++p) {
//   //   data_ptr[p * num_channels + 0] = *(float*)(&cloud_msgs.data[0] + (pointBytes*p) + offset_x);
//   //   data_ptr[p * num_channels + 1] = *(float*)(&cloud_msgs.data[0] + (pointBytes*p) + offset_y);
//   //   data_ptr[p * num_channels + 2] = *(float*)(&cloud_msgs.data[0] + (pointBytes*p) + offset_z);
//   //   data_ptr[p * num_channels + 3] = *(unsigned char*)(&cloud_msgs.data[0] + (pointBytes*p) + offset_int);
//   // }
  
//   for (int p = 0; p < num_points; ++p) {
    
//     if ((p / cloud_msg->width % 2) != 0) {
//       continue;
//     }

//     pcl::PointXYZI newPoints;
//     newPoints.x = *(float*)(&cloud_msg->data[0] + (pointBytes * p) + offset_x);
//     newPoints.y = *(float*)(&cloud_msg->data[0] + (pointBytes * p) + offset_y);
//     newPoints.z = *(float*)(&cloud_msg->data[0] + (pointBytes * p) + offset_z);
//     newPoints.intensity = *(unsigned char*)(&cloud_msg->data[0] + (pointBytes*p) + offset_int);

//     // std::cout<< "newPoints.intensity: " << newPoints.intensity << std::endl;
//     // std::cout<< "newPoints:" << newPoints << std::endl;

//     // pcl_pc->points[p].x = newPoints.x;
//     // pcl_pc->points[p].y = newPoints.y;
//     // pcl_pc->points[p].z = newPoints.z;
//     // pcl_pc->points[p].intensity = newPoints.intensity;

//     // data_ptr[p * num_channels + 0] = pcl_pc->points[p].x;
//     // data_ptr[p * num_channels + 1] = pcl_pc->points[p].y;
//     // data_ptr[p * num_channels + 2] = pcl_pc->points[p].z;
//     // data_ptr[p * num_channels + 3] = pcl_pc->points[p].intensity;

//     // pcl_pc->points[p].x = newPoints.x;
//     // pcl_pc->points[p].y = newPoints.y;
//     // pcl_pc->points[p].z = newPoints.z;
//     // pcl_pc->points[p].intensity = newPoints.intensity;
//     // data_ptr[p] = &newPoints;

//     data_ptr[p * num_channels + 0] = newPoints.x;
//     data_ptr[p * num_channels + 1] = newPoints.y;
//     data_ptr[p * num_channels + 2] = newPoints.z;
//     data_ptr[p * num_channels + 3] = newPoints.intensity;
//   }

//   return 0;
// }

void Det3D::pub_box_pred(std::vector<Bndbox> boxes) {
  jsk_recognition_msgs::BoundingBoxArray objects;
  objects.header.stamp = ros::Time::now();
  objects.header.frame_id = frame_id_;
  objects.header.seq = 0;
  for (const auto box : boxes) {

    jsk_recognition_msgs::BoundingBox obj;
    if (box.score <= score_threshold_) {
      continue;
    }

    std::cout << "score:" << box.score << std::endl;
    obj.header = objects.header;
    // obj.valid = true;
    obj.value = box.score;
    obj.label = box.id;  // class id
    obj.pose.position.x = box.x;
    obj.pose.position.y = box.y;
    obj.pose.position.z = box.z;
    std::cout << "obj.pose.position.x:" << obj.pose.position.x << std::endl;
    std::cout << "obj.pose.position.y:" << obj.pose.position.y << std::endl;
    std::cout << "obj.pose.position.z:" << obj.pose.position.z << std::endl;
    // NOTE: box 的 l w h 对应的是车的 宽 长 高
    // 我们 dimensions 的 x y z 对应的是车的 长 宽 高
    // 所以这里要交换一下
    obj.dimensions.x = box.w;
    obj.dimensions.y = box.l;
    obj.dimensions.z = box.h;
    std::cout << "obj.dimensions.l:" << obj.dimensions.x << std::endl;
    std::cout << "obj.dimensions.w:" << obj.dimensions.y << std::endl;
    std::cout << "obj.dimensions.h:" << obj.dimensions.z << std::endl;

    // 将yaw角转换为四元数
    float yaw = box.rt;
    tf2::Quaternion quat_tf;
    quat_tf.setRPY(0, 0, yaw);
    obj.pose.orientation = tf2::toMsg(quat_tf);
    // std::cout << "obj.pose.orientation: " << obj.pose.orientation << std::endl;
    objects.boxes.push_back(obj);
  }

  ROS_DEBUG_STREAM("Bndbox objs: " << objects.boxes.size());
  pub_.publish(objects);
  return;
}
