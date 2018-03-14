/*
 * KinfuServer.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: mklingen
 */

#include <yak/ros/kinfu_server.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

namespace kfusion
{
    
    KinFuServer::KinFuServer(RosRGBDCamera* camera, const std::string& fixedFrame, const std::string& camFrame) :
            should_exit_(false), camera_(camera), baseFrame_(fixedFrame), cameraFrame_(camFrame)
    {
        cloud_pub_ = camera->nodeHandle.advertise<sensor_msgs::PointCloud2>("out_cloud", 1, true);
        raycastImgPublisher_ = camera->nodeHandle.advertise<sensor_msgs::Image>("raycast_image", 10);
        get_tsdf_server_ = camera->nodeHandle.advertiseService("get_tsdf", &KinFuServer::GetTSDF,  this);
        get_sparse_tsdf_server_ = camera->nodeHandle.advertiseService("get_sparse_tsdf", &KinFuServer::GetSparseTSDF,  this);
        publish_cloud_server_ = camera->nodeHandle.advertiseService("publish_cloud", &KinFuServer::PublishCloud,  this);

        if (!camera->nodeHandle.getParam("use_pose_hints", use_pose_hints_)) {
          ROS_INFO("Failed to get use_pose_hints flag!");
        }
        ROS_INFO_STREAM("Use pose hints set to " << use_pose_hints_);
        if (use_pose_hints_) {
          tfListener_.waitForTransform(baseFrame_, cameraFrame_, ros::Time::now(), ros::Duration(0.5));
          tfListener_.lookupTransform(baseFrame_, cameraFrame_, ros::Time(0), previous_volume_to_sensor_transform_);
        }

        cloud_pub_ = camera->nodeHandle.advertise<sensor_msgs::PointCloud2>("out_cloud", 1, true);
    }

    void KinFuServer::PublishRaycastImage()
    {
        const int mode = 3;
        kinfu_->renderImage(viewDevice_, mode);
        
        viewHost_.create(viewDevice_.rows(), viewDevice_.cols(), CV_8UC4);
        viewDevice_.download(viewHost_.ptr<void>(), viewHost_.step);
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        header.frame_id = cameraFrame_;
        cv_bridge::CvImage image(header, std::string("rgba8"), viewHost_);
        raycastImgPublisher_.publish(image.toImageMsg());
    }

    Affine3f KinFuServer::TransformToAffine(tf::Transform input) {
      Eigen::Affine3d inputAsEigen;
      tf::transformTFToEigen(input, inputAsEigen);
      cv::Mat inputAsCV(4,4, CV_32F);
      cv::eigen2cv(inputAsEigen.cast<float>().matrix(), inputAsCV);
      return Affine3f(inputAsCV);
    }
    
    void KinFuServer::Update()
    {
        // Try to grab an image from the camera. If it doesn't work, spin once and try again.
        bool has_frame = camera_->Grab(lastDepth_, lastColor_);

        if (!has_frame)
        {
            return;
        }
        // TODO:  when looking up the TF frame, we probably need to use the time at which the depth image was acquired,
        //        rather than 'ros::Time::now()'.  This works okay for now, especially if ICP is turned on.
        ros::Time t = ros::Time::now();

        // Once we have a new image, find the transform between the poses where the current image and the previous image were captured.
        if (use_pose_hints_)
        {
          tfListener_.waitForTransform(baseFrame_, cameraFrame_, ros::Time::now(), ros::Duration(0.5));
          tfListener_.lookupTransform(baseFrame_, cameraFrame_, t, current_volume_to_sensor_transform_);

          // calculate camera motion
          tf::Transform past_to_current_sensor = previous_volume_to_sensor_transform_.inverse() * current_volume_to_sensor_transform_;

          // update transforms (This seems redundant
          currentCameraMotionHint_ = KinFuServer::TransformToAffine(past_to_current_sensor);
          currentCameraPoseHint_ = KinFuServer::TransformToAffine(current_volume_to_sensor_transform_);

          previous_volume_to_sensor_transform_ = current_volume_to_sensor_transform_;
        }
        else
        {
          currentCameraMotionHint_ = Affine3f::Identity();
        }

        // currentCameraPoseHint_ is ONLY used if ICP is OFF
        bool has_image = KinFu(currentCameraMotionHint_, currentCameraPoseHint_, lastDepth_, lastColor_);

        if (has_image)
        {
            PublishRaycastImage();
        }

        // If we are using TF, then the frames are already being published.
        // If not, we need to publish them (for visualization only)
        if(!use_pose_hints_)
        {
          PublishTransform();
        }
    }

    bool KinFuServer::ExecuteBlocking()
    {
        double time_ms = 0;
        bool has_image = false;

        Initialize();

        kfusion::KinFu& kinfu = *kinfu_;


        // TODO: Need to change this to reflect actual sensor refresh rate? (e.g. Ensenso at ~4Hz)
        ROS_INFO("Starting tracking...\n");
        ros::Rate trackHz(30.0f);
        for (int i = 0; !should_exit_ && ros::ok(); ++i)
        {
            Update();
            ros::spinOnce();
            trackHz.sleep();
        }
        return true;
    }

    bool KinFuServer::KinFu(const Affine3f& cameraMotionHint, const Affine3f& currentCameraPoseHint, const cv::Mat& depth, const cv::Mat& color)
    {
        depthDevice_.upload(depth.data, depth.step, depth.rows, depth.cols);
        return(* kinfu_)(cameraMotionHint, currentCameraPoseHint, depthDevice_);
    }

    bool KinFuServer::ConnectCamera()
    {
        bool cameraConnected = false;

        ROS_INFO("Connecting to camera...\n");
        ros::Rate connectionHz(100.0f);
        ros::Timer printTimer;
        printTimer.setPeriod(ros::Duration(1.0));
        printTimer.start();
        while (!cameraConnected)
        {
            cameraConnected = camera_->hasNewDepth;
            ros::spinOnce();
            connectionHz.sleep();
            if (printTimer.hasPending())
            {
                ROS_INFO("No devices connected yet...\n");
            }
        }
        ROS_INFO("Connected.\n");
        return cameraConnected;
    }

    bool KinFuServer::LoadParams()
    {
        KinFuParams params = KinFuParams::default_params();
        params.cols = camera_->GetDepthWidth();
        params.rows = camera_->GetDepthHeight();
        params.intr = camera_->GetDepthIntrinsics();

        LoadParam(params.bilateral_kernel_size, "bilateral_kernel_size");
        LoadParam(params.bilateral_sigma_depth, "bilateral_sigma_depth");
        LoadParam(params.bilateral_sigma_spatial, "bilateral_sigma_spatial");
        LoadParam(params.gradient_delta_factor, "gradient_delta_factor");
        LoadParam(params.icp_angle_thres, "icp_angle_thresh");
        LoadParam(params.icp_dist_thres, "icp_dist_thresh");
        LoadParam(params.icp_iter_num, "icp_iter_num");
        LoadParam(params.icp_truncate_depth_dist, "icp_truncate_depth_dist");
        LoadParam(params.raycast_step_factor, "raycast_step_factor");
        LoadParam(params.tsdf_max_weight, "tsdf_max_weight");
        LoadParam(params.tsdf_min_camera_movement, "tsdf_min_camera_movement");
        LoadParam(params.tsdf_trunc_dist, "tsdf_trunc_dist");

        // TSDF volume parameters
        LoadParam(params.volume_dims.val[0], "volume_dims_x");
        LoadParam(params.volume_dims.val[1], "volume_dims_y");
        LoadParam(params.volume_dims.val[2], "volume_dims_z");
        LoadParam(params.volume_resolution, "volume_resolution");

        float volPosX, volPosY, volPosZ;
        LoadParam(volPosX, "volume_pos_x");
        LoadParam(volPosY, "volume_pos_y");
        LoadParam(volPosZ, "volume_pos_z");
        ROS_INFO_STREAM("volPos (params): " << volPosX << ", " << volPosY << ", " << volPosZ);
        params.volume_pose.translation(Vec3f(volPosX, volPosY, volPosZ));
        params.camera_pose = Affine3f::Identity();


        LoadParam(params.use_icp, "use_icp");
        LoadParam(params.use_pose_hints, "use_pose_hints");
        LoadParam(params.update_via_sensor_motion, "update_via_sensor_motion");

        if (params.use_pose_hints)
        {
          tf::Transform initialPose = previous_volume_to_sensor_transform_;
          params.camera_pose.translation(Vec3f(initialPose.getOrigin().x(), initialPose.getOrigin().y(), initialPose.getOrigin().z()));

          float temp_array[9];
          tf::Matrix3x3 temp_matrix = initialPose.getBasis();
          float* ptr = &temp_array[0];
          for(int i = 0; i < 3; ++i)
          {
            for(int j = 0; j < 3; ++j)
            {
              *ptr = temp_matrix.getColumn(i)[j];
              ++ptr;
            }
          }
          int t = 3;
          const cv::Mat matrix = cv::Mat(t,t, CV_32FC1, &temp_array[0]);

          ROS_INFO_STREAM( matrix.at<float>(0) << " " << matrix.at<float>(1) << " " << matrix.at<float>(2));
          ROS_INFO_STREAM( matrix.at<float>(3) << " " << matrix.at<float>(4) << " " << matrix.at<float>(5) );
          ROS_INFO_STREAM( matrix.at<float>(6) << " " << matrix.at<float>(7) << " " << matrix.at<float>(8) << "\n");
          params.camera_pose.rotation(matrix);
        }

        kinfu_ = KinFu::Ptr(new kfusion::KinFu(params));
        return true;
    }

    bool KinFuServer::PublishTransform()
    {
      // DONE-ISH: Relate baseFrame to actual frame of volume. It's not currently related to global-relative coordinates.
//        tf::StampedTransform currTf;

//        currTf.child_frame_id_ = cameraFrame_;
//        currTf.frame_id_ = baseFrame_;
//        currTf.stamp_ = ros::Time::now();
//        cv::Affine3f currPose = kinfu_->getCameraPose();
//        currTf.setOrigin(tf::Vector3(currPose.translation().val[0], currPose.translation().val[1], currPose.translation().val[2]));
//        cv::Affine3f::Mat3 rot = currPose.rotation();
//        tf::Matrix3x3 tfRot(rot.val[0], rot.val[1], rot.val[2], rot.val[3], rot.val[4], rot.val[5], rot.val[6], rot.val[7], rot.val[8]);
//        tf::Quaternion tfQuat;

//        tfRot.getRotation(tfQuat);
//        currTf.setRotation(tfQuat);
//        tfBroadcaster_.sendTransform(currTf);
//        return true;

      tf::Transform currTf;

      cv::Affine3f currPose = kinfu_->getCameraPose();
      currTf.setOrigin(tf::Vector3(currPose.translation().val[0], currPose.translation().val[1], currPose.translation().val[2]));
      cv::Affine3f::Mat3 rot = currPose.rotation();
      tf::Matrix3x3 tfRot(rot.val[0], rot.val[1], rot.val[2], rot.val[3], rot.val[4], rot.val[5], rot.val[6], rot.val[7], rot.val[8]);
      tf::Quaternion tfQuat;

      tfRot.getRotation(tfQuat);
      currTf.setRotation(tfQuat);

//      tf::Transform temp(KinFuServer::SwitchToVolumeFrame(currTf));
//      tf::Transform temp(currTf);

      tf::Transform temp = currTf;

//      tf::Transform temp = tf::Transform(tf::Quaternion(tf::Vector3(0,0,1), tfScalar(3.14159)), tf::Vector3(0,0,0)) * currTf;
      tf::StampedTransform output;
      output.setRotation(temp.getRotation());
      output.setOrigin(temp.getOrigin());
      output.child_frame_id_ = cameraFrame_;
      output.frame_id_ = baseFrame_;
      output.stamp_ = ros::Time::now();

      tfBroadcaster_.sendTransform(output);
      return true;
    }

    bool KinFuServer::PublishCloud(std_srvs::Empty::Request &req, std_srvs::Empty::Response &resp)
    {

      pcl::PointCloud<pcl::PointXYZ> cloud;
      cloud = kinfu_->getCloud();
      sensor_msgs::PointCloud2 msg;
      pcl::toROSMsg(cloud, msg);
      msg.header.stamp = ros::Time::now();
      msg.header.frame_id = baseFrame_;

      ROS_INFO_STREAM("publishing cloud w/ " << cloud.size() << " points.");
      cloud_pub_.publish(msg);
      return true;
    }

    bool KinFuServer::GetTSDF(yak::GetTSDFRequest& req, yak::GetTSDFResponse& res)
    {
        res.tsdf.header.stamp = ros::Time::now();
        res.tsdf.header.frame_id = baseFrame_;
        const kfusion::cuda::TsdfVolume& volume = kinfu_->tsdf();
        res.tsdf.max_weight = volume.getMaxWeight();
        res.tsdf. num_voxels_x = volume.getDims().val[0];
        res.tsdf. num_voxels_y = volume.getDims().val[1];
        res.tsdf. num_voxels_z = volume.getDims().val[2];
        res.tsdf.size_x = volume. getSize().val[0];
        res.tsdf.size_y = volume. getSize().val[1];
        res.tsdf.size_z = volume. getSize().val[2];
        res.tsdf.truncation_dist = volume.getTruncDist();
        res.tsdf.pose.position.x = volume.getPose().translation().val[0];
        res.tsdf.pose.position.y = volume.getPose().translation().val[1];
        res.tsdf.pose.position.z = volume.getPose().translation().val[2];
        cv::Affine3f::Mat3 rot = volume.getPose().rotation();
        tf::Matrix3x3 tfRot(rot.val[0], rot.val[1], rot.val[2], rot.val[3], rot.val[4], rot.val[5], rot.val[6], rot.val[7], rot.val[8]);
        tf::Quaternion tfQuat;
        tfRot.getRotation(tfQuat);
        res.tsdf.pose.orientation.x = tfQuat.x();
        res.tsdf.pose.orientation.y = tfQuat.y();
        res.tsdf.pose.orientation.z = tfQuat.z();
        res.tsdf.pose.orientation.w = tfQuat.w();
        res.tsdf.data.resize(res.tsdf.num_voxels_x * res.tsdf.num_voxels_y * res.tsdf.num_voxels_z, 0);
        ROS_INFO("About to download TSDF data...");
        volume.data().download(res.tsdf.data.data());
        ROS_INFO("Just downloaded TSDF data");
        return true;
    }

    bool KinFuServer::GetSparseTSDF(yak::GetSparseTSDFRequest& req, yak::GetSparseTSDFResponse& res)
    {
        res.tsdf.header.stamp = ros::Time::now();
        res.tsdf.header.frame_id = baseFrame_;
        const kfusion::cuda::TsdfVolume& volume = kinfu_->tsdf();
        res.tsdf.max_weight = volume.getMaxWeight();
        res.tsdf. num_voxels_x = volume.getDims().val[0];
        res.tsdf. num_voxels_y = volume.getDims().val[1];
        res.tsdf. num_voxels_z = volume.getDims().val[2];
        res.tsdf.size_x = volume. getSize().val[0];
        res.tsdf.size_y = volume. getSize().val[1];
        res.tsdf.size_z = volume. getSize().val[2];
        res.tsdf.truncation_dist = volume.getTruncDist();
        res.tsdf.pose.position.x = volume.getPose().translation().val[0];
        res.tsdf.pose.position.y = volume.getPose().translation().val[1];
        res.tsdf.pose.position.z = volume.getPose().translation().val[2];
        cv::Affine3f::Mat3 rot = volume.getPose().rotation();
        tf::Matrix3x3 tfRot(rot.val[0], rot.val[1], rot.val[2], rot.val[3], rot.val[4], rot.val[5], rot.val[6], rot.val[7], rot.val[8]);
        tf::Quaternion tfQuat;
        tfRot.getRotation(tfQuat);
        res.tsdf.pose.orientation.x = tfQuat.x();
        res.tsdf.pose.orientation.y = tfQuat.y();
        res.tsdf.pose.orientation.z = tfQuat.z();
        res.tsdf.pose.orientation.w = tfQuat.w();

        ROS_INFO("Making array to hold data...");
        std::vector<uint32_t> dataTemp;
        dataTemp.resize(res.tsdf.num_voxels_x * res.tsdf.num_voxels_y * res.tsdf.num_voxels_z);
        ROS_INFO("About to download sparse TSDF data...");
        volume.data().download(dataTemp.data());

        ROS_INFO("Just downloaded TSDF data");
        ROS_INFO_STREAM("Volume data is of length " << dataTemp.size());
        ROS_INFO_STREAM("Volume claims to have dimensions X=" << res.tsdf.num_voxels_x <<" ,Y=" << res.tsdf.num_voxels_y << " ,Z=" << res.tsdf.num_voxels_z);

        std::vector<uint32_t> dataOut;
        std::vector<uint16_t> rows;
        std::vector<uint16_t> cols;
        std::vector<uint16_t> sheets;

        ROS_INFO("Iterating through volume to find nonzero voxels...");
        for (uint16_t i = 0; i < res.tsdf.num_voxels_x; i++) {
          for (uint16_t j = 0; j < res.tsdf.num_voxels_y; j++) {
            for (uint16_t k = 0; k < res.tsdf.num_voxels_z; k++) {
              // Get the contents of every element of the serialized TSDF volume.
              int dataIndex = /*sheet index =*/res.tsdf.num_voxels_x*res.tsdf.num_voxels_y*k + /*column index within sheet =*/res.tsdf.num_voxels_x*j + /*row index within column =*/i;
              uint32_t currentData = dataTemp[dataIndex];
              half_float::half currentValue;
              uint16_t currentWeight;
              KinFuServer::GetTSDFData(currentData, currentValue, currentWeight);

              // If the weight is nonzero and voxel is within the TSDF distance, save its data and its coordinates.
              if (currentWeight > 0 && currentValue < 1) {
                dataOut.push_back(currentData);
                rows.push_back(i);
                cols.push_back(j);
                sheets.push_back(k);
              }
            }
          }
        }

        // Resize the arrays in the message to match the actual lengths of the vectors.
        // Assign data to the SparseTSDF message.
        ROS_INFO("Assigning data to result...");
        res.tsdf.data.resize(dataOut.size());
        std::vector<uint32_t>::iterator itA;
        itA = dataOut.begin();
        res.tsdf.data.assign(itA, dataOut.end());

        res.tsdf.rows.resize(rows.size());
        std::vector<uint16_t>::iterator itB;
        itB = rows.begin();
        res.tsdf.rows.assign(itB, rows.end());

        res.tsdf.cols.resize(cols.size());
        itB = cols.begin();
        res.tsdf.cols.assign(itB, cols.end());

        res.tsdf.sheets.resize(sheets.size());
        itB = sheets.begin();
        res.tsdf.sheets.assign(itB, sheets.end());
        ROS_INFO("Created sparse TSDF structure");
        ROS_INFO_STREAM("# Voxels: " << dataOut.size() << "  # Indices: " << rows.size() << " Message size: " << (4*dataOut.size() + 2 * 3 * rows.size())/1000 << "ish kB");

        return true;
    }

    bool KinFuServer::GetTSDFData(uint32_t input,  half_float::half& voxelValue, uint16_t& voxelWeight) {
      std::memcpy(&voxelValue, &input, 2);
      std::memcpy(&voxelWeight, ((char*)(&input)) + 2, 2);
      return true;
    }
} /* namespace kfusion */


