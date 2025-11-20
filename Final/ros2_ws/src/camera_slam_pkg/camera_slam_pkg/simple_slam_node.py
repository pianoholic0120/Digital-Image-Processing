#!/usr/bin/env python3
"""
简单的视觉里程计节点 - 使用特征点匹配进行基本的 SLAM
这是一个简化版本，用于演示。生产环境建议使用 RTAB-Map 或 ORB-SLAM3
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# tf2_ros 是可选的，如果不可用则跳过 TF 发布
try:
    from tf2_ros import TransformBroadcaster
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False
    print("警告: tf2_ros 不可用，将跳过 TF 发布功能")


class SimpleSLAMNode(Node):
    """简单的视觉里程计节点"""
    
    def __init__(self):
        super().__init__('simple_slam_node')
        
        # 订阅摄像头图像
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # 发布位姿
        self.pose_pub = self.create_publisher(PoseStamped, '/slam/pose', 10)
        
        # TF 广播器（如果可用）
        if TF2_AVAILABLE:
            self.tf_broadcaster = TransformBroadcaster(self)
        else:
            self.tf_broadcaster = None
            self.get_logger().warn('tf2_ros 不可用，TF 发布功能已禁用')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # SLAM 状态
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.scale = 1.0
        
        # 特征检测器（ORB）
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # 特征匹配器
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.get_logger().info('简单 SLAM 节点已启动')
    
    def image_callback(self, msg):
        """处理接收到的图像"""
        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
            
            # 检测特征点
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if self.prev_frame is not None and descriptors is not None and self.prev_descriptors is not None:
                # 匹配特征点
                matches = self.matcher.knnMatch(descriptors, self.prev_descriptors, k=2)
                
                # 应用 Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) > 10:
                    # 提取匹配点
                    src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([self.prev_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # 计算单应性矩阵（用于平面运动估计）
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        # 从单应性矩阵提取旋转和平移
                        # 注意：这是简化版本，实际 SLAM 需要更复杂的处理
                        h00, h01, h02 = H[0]
                        h10, h11, h12 = H[1]
                        h20, h21, h22 = H[2]
                        
                        # 计算平移（简化）
                        dx = h02 / h22
                        dy = h12 / h22
                        
                        # 计算旋转角度
                        angle = math.atan2(h01, h00)
                        
                        # 更新位姿（简化，没有尺度恢复）
                        scale_factor = 0.1  # 需要根据实际场景调整
                        self.current_pose[0] += dx * scale_factor
                        self.current_pose[1] += dy * scale_factor
                        self.current_pose[2] += angle
                        
                        # 发布位姿
                        self.publish_pose(msg.header.stamp)
                        
                        # 发布 TF
                        self.publish_tf(msg.header.stamp)
            
            # 更新前一帧
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
        except Exception as e:
            self.get_logger().error(f'处理图像时出错: {e}')
    
    def publish_pose(self, stamp):
        """发布位姿"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(self.current_pose[0])
        pose_msg.pose.position.y = float(self.current_pose[1])
        pose_msg.pose.position.z = 0.0
        
        # 将角度转换为四元数
        qx = 0.0
        qy = 0.0
        qz = math.sin(self.current_pose[2] / 2.0)
        qw = math.cos(self.current_pose[2] / 2.0)
        
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        
        self.pose_pub.publish(pose_msg)
    
    def publish_tf(self, stamp):
        """发布 TF 变换"""
        if not TF2_AVAILABLE or self.tf_broadcaster is None:
            return
        
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = float(self.current_pose[0])
        t.transform.translation.y = float(self.current_pose[1])
        t.transform.translation.z = 0.0
        
        qz = math.sin(self.current_pose[2] / 2.0)
        qw = math.cos(self.current_pose[2] / 2.0)
        
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleSLAMNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

