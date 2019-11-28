This is the implementation of MCC-EKF which is evaluated on the popular KITTI dataset.
Clone the repository to your ROS workspace.

git clone https://github.com/aralab-unr/MCC-EKF-SLAM

Collect the Lidar odometry using any one of the Lidar slam algorithms(HDL graph slam,rtabmap,icp etc..) by reording a rosbag.
play the rosbag in one terminal.

Remember to publish the topic of Lidar odometry in /rtabmap_lidar/odom topic.

Run the mcc-rkf slam node in another terminal

rosrun mcc_ekf_slam kitti_main_node 

Open one more terminal to open the rviz.

rosrun rviz rviz -d ~ROS_WORKSPACE/mcc_ekf_slam/rviz/mcc_rviz.rviz

attacks are already implemented in the node which introduces attacks to the Lidar odometry.

