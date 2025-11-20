#!/usr/bin/env python3
"""
Launch file - Start camera node and RTAB-Map for high-quality 3D reconstruction
RTAB-Map is the most accurate real-time 3D reconstruction package for ROS2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    # Get project root directory (where camera.yaml is located)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    camera_config_path = os.path.join(project_root, 'camera.yaml')
    
    # Get RTAB-Map config file path
    rtabmap_config_file = os.path.join(
        os.path.dirname(__file__),
        '..', 'config', 'rtabmap.yaml'
    )
    
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
    
    # Static TF transform - from base_link to camera_frame
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'camera_frame'],
        output='screen'
    )
    
    # RTAB-Map node (for high-quality 3D reconstruction with DBoW2 and g2o)
    # Reference: https://wiki.ros.org/rtabmap_ros
    # Note: RTAB-Map works best with RGB-D cameras, but can work with monocular
    rtabmap_node = Node(
        package='rtabmap_ros',
        executable='rtabmap',
        name='rtabmap',
        parameters=[{
            # Frame IDs
            'frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'map_frame_id': 'map',
            
            # Subscriptions
            'subscribe_depth': False,  # Monocular camera (no depth)
            'subscribe_rgb': True,
            'subscribe_rgbd': False,
            'subscribe_scan': False,
            'subscribe_odom_info': False,
            'subscribe_user_data': False,
            
            # QoS
            'qos_image': 1,
            'qos_camera_info': 1,
            'qos_imu': 1,
            
            # Load RTAB-Map config file (contains DBoW2 and g2o settings)
            'config_path': rtabmap_config_file,
        }],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
        ],
        output='screen'
    )
    
    # RTAB-Map visualization node (provides GUI)
    rtabmap_viz_node = Node(
        package='rtabmap_ros',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        parameters=[{
            'frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'subscribe_depth': False,
            'subscribe_rgb': True,
            'subscribe_rgbd': False,
            'subscribe_scan': False,
            'subscribe_odom_info': False,
            'qos_image': 1,
            'qos_camera_info': 1,
        }],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
        ],
        output='screen'
    )
    
    return LaunchDescription([
        camera_index_arg,
        camera_config_arg,
        camera_node,
        static_tf,
        rtabmap_node,
        rtabmap_viz_node,
    ])

