#!/usr/bin/env python3
"""
Launch file - 使用 ORB-SLAM3 进行高精度单目SLAM（包含 RViz2）
ORB-SLAM3 是业界最精准的视觉SLAM系统之一
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    camera_config_path = os.path.join(project_root, 'camera.yaml')
    orb_slam3_config_path = os.path.join(project_root, 'orb_slam3_camera.yaml')
    
    # 获取包目录
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    rviz_config_path = os.path.join(package_dir, 'config', 'orb_slam3.rviz')
    
    # ORB-SLAM3 默认路径
    orb_slam3_dir = os.path.expanduser('~/ORB_SLAM3')
    vocab_path_default = os.path.join(orb_slam3_dir, 'Vocabulary', 'ORBvoc.txt')
    
    # Launch arguments
    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='Camera index (0=built-in, 1+=external)'
    )
    
    camera_config_arg = DeclareLaunchArgument(
        'camera_config',
        default_value=camera_config_path,
        description='Camera calibration config file path'
    )
    
    vocab_path_arg = DeclareLaunchArgument(
        'vocab_path',
        default_value=vocab_path_default,
        description='ORB vocabulary file path'
    )
    
    settings_path_arg = DeclareLaunchArgument(
        'settings_path',
        default_value=orb_slam3_config_path,
        description='ORB-SLAM3 settings file path'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 for visualization'
    )
    
    # Camera node
    camera_node = Node(
        package='camera_slam_pkg',
        executable='camera_node',
        name='camera_node',
        parameters=[{
            'camera_index': LaunchConfiguration('camera_index'),
            'camera_config_path': LaunchConfiguration('camera_config'),
            'width': 640,
            'height': 480,
            'fps': 30.0,
        }],
        output='screen'
    )
    
    # Static TF transform
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'camera_frame'],
        output='screen'
    )
    
    # ORB-SLAM3 Mono node
    orb_slam3_node = Node(
        package='orb_slam3_ros',
        executable='mono',
        name='orb_slam3_mono',
        parameters=[{
            'vocab_path': LaunchConfiguration('vocab_path'),
            'settings_path': LaunchConfiguration('settings_path'),
            'frame_id': 'camera_frame',
            'map_frame_id': 'map',
            'odom_frame_id': 'odom',
        }],
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
        ],
        output='screen'
    )
    
    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )
    
    return LaunchDescription([
        camera_index_arg,
        camera_config_arg,
        vocab_path_arg,
        settings_path_arg,
        use_rviz_arg,
        camera_node,
        static_tf,
        orb_slam3_node,
        rviz_node,
    ])

