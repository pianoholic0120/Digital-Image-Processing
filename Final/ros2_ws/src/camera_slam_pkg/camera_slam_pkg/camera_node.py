#!/usr/bin/env python3
"""
摄像头节点 - 从 MacBook 内置摄像头或外接摄像头获取图像并发布到 ROS2 话题
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os
import signal
import sys


class CameraNode(Node):
    """摄像头发布节点"""
    
    def __init__(self):
        super().__init__('camera_node')
        
        # 创建发布者
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        # 创建定时器（30 FPS）
        timer_period = 1.0 / 30.0  # 30 FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 摄像头参数
        self.declare_parameter('camera_index', 0)  # 0 为内置摄像头，1+ 为外接摄像头
        self.declare_parameter('camera_config_path', '')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        
        camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        config_path = self.get_parameter('camera_config_path').get_parameter_value().string_value
        width = self.get_parameter('width').get_parameter_value().integer_value
        height = self.get_parameter('height').get_parameter_value().integer_value
        
        # 加载摄像头标定参数
        self.camera_info = self.load_camera_info(config_path, width, height)
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 {camera_index}')
            raise RuntimeError(f'无法打开摄像头 {camera_index}')
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 获取实际分辨率
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'摄像头已打开: {camera_index}, 分辨率: {actual_width}x{actual_height}')
        
        # 更新 camera_info 以匹配实际分辨率
        if actual_width != width or actual_height != height:
            self.camera_info = self.load_camera_info(config_path, actual_width, actual_height)
    
    def load_camera_info(self, config_path, width, height):
        """加载摄像头标定参数"""
        camera_info = CameraInfo()
        camera_info.width = width
        camera_info.height = height
        
        # 如果提供了配置文件路径，尝试加载
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    cam_params = config.get('Camera', {})
                    
                    # 设置内参矩阵 K
                    fx = cam_params.get('fx', width * 0.8)
                    fy = cam_params.get('fy', height * 0.8)
                    cx = cam_params.get('cx', width / 2.0)
                    cy = cam_params.get('cy', height / 2.0)
                    
                    camera_info.k = [
                        fx, 0.0, cx,
                        0.0, fy, cy,
                        0.0, 0.0, 1.0
                    ]
                    
                    # 设置畸变参数 D
                    params = cam_params.get('params', {})
                    k1 = params.get('k1', 0.0)
                    k2 = params.get('k2', 0.0)
                    p1 = params.get('p1', 0.0)
                    p2 = params.get('p2', 0.0)
                    k3 = params.get('k3', 0.0)
                    
                    camera_info.d = [k1, k2, p1, p2, k3]
                    
                    # 设置投影矩阵 P (与 K 相同，因为没有外参)
                    camera_info.p = [
                        fx, 0.0, cx, 0.0,
                        0.0, fy, cy, 0.0,
                        0.0, 0.0, 1.0, 0.0
                    ]
                    
                    # R 矩阵（单位矩阵）
                    camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                    
                    self.get_logger().info(f'已加载摄像头标定参数: {config_path}')
            except Exception as e:
                self.get_logger().warn(f'无法加载摄像头配置: {e}，使用默认参数')
        else:
            # 使用默认参数（假设针孔相机模型）
            fx = fy = width * 0.8  # 粗略估计
            cx = width / 2.0
            cy = height / 2.0
            
            camera_info.k = [
                fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0
            ]
            camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            camera_info.p = [
                fx, 0.0, cx, 0.0,
                0.0, fy, cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            self.get_logger().info('使用默认摄像头参数')
        
        camera_info.distortion_model = 'plumb_bob'
        return camera_info
    
    def timer_callback(self):
        """定时器回调函数 - 捕获并发布图像"""
        ret, frame = self.cap.read()
        if ret:
            # 更新时间戳
            stamp = self.get_clock().now().to_msg()
            self.camera_info.header.stamp = stamp
            self.camera_info.header.frame_id = 'camera_frame'
            
            # 发布 CameraInfo
            self.camera_info_pub.publish(self.camera_info)
            
            # 转换并发布图像
            try:
                # 转换为 RGB（OpenCV 默认是 BGR）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_msg = self.bridge.cv2_to_imgmsg(frame_rgb, 'rgb8')
                img_msg.header.stamp = stamp
                img_msg.header.frame_id = 'camera_frame'
                self.image_pub.publish(img_msg)
                
                # 定期打印状态（每5秒一次）
                current_time = self.get_clock().now().nanoseconds
                if not hasattr(self, 'last_log_time') or (current_time - self.last_log_time) > 5e9:
                    self.get_logger().info(f'发布图像: {frame_rgb.shape[1]}x{frame_rgb.shape[0]}, encoding: rgb8')
                    self.last_log_time = current_time
            except Exception as e:
                self.get_logger().error(f'发布图像时出错: {e}', exc_info=True)
        else:
            self.get_logger().warn('无法读取摄像头帧，检查摄像头连接和权限')
    
    def destroy_node(self):
        """清理资源 - 确保相机被正确释放"""
        self.get_logger().info('正在关闭摄像头节点...')
        
        # 停止定时器
        if hasattr(self, 'timer'):
            self.timer.cancel()
        
        # 释放摄像头资源
        if hasattr(self, 'cap') and self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
                self.get_logger().info('摄像头已释放')
            self.cap = None
        
        # 清理 CV Bridge
        if hasattr(self, 'bridge'):
            self.bridge = None
        
        self.get_logger().info('摄像头节点已关闭')
        super().destroy_node()


def signal_handler(sig, frame):
    """信号处理函数 - 确保 Ctrl+C 时正确清理"""
    print('\n收到中断信号，正在关闭摄像头...')
    sys.exit(0)


def main(args=None):
    # 注册信号处理器，确保 Ctrl+C 时正确清理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    rclpy.init(args=args)
    camera_node = None
    
    try:
        camera_node = CameraNode()
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        print('\n收到键盘中断 (Ctrl+C)')
    except Exception as e:
        print(f'错误: {e}')
    finally:
        # 确保资源被释放
        if camera_node is not None:
            try:
                camera_node.destroy_node()
            except Exception as e:
                print(f'清理节点时出错: {e}')
        
        # 确保 ROS2 正确关闭
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f'关闭 ROS2 时出错: {e}')
        
        # 额外检查：确保相机被释放
        import gc
        gc.collect()
        print('摄像头资源已清理完成')


if __name__ == '__main__':
    main()

