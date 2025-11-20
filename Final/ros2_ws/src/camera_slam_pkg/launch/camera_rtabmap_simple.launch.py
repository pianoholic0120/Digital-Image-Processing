#!/usr/bin/env python3
"""
Simple RTAB-Map Launch File - Direct integration with RTAB-Map core
Uses RTAB-Map's ROS2 nodes directly for highest quality SLAM
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    camera_config_path = os.path.join(project_root, 'camera.yaml')
    
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
    
    # Camera node - publishes to standard RTAB-Map topics
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
        remappings=[
            ('image_raw', '/camera/rgb/image_rect_color'),
            ('camera_info', '/camera/rgb/camera_info'),
        ],
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
    
    # RTAB-Map Visual Odometry (if available)
    # Note: This node may not be available if rtabmap_odom package didn't compile
    # In that case, RTAB-Map SLAM will use its internal odometry
    rtabmap_odom_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rtabmap_odom',
        parameters=[{
            'frame_id': 'camera_frame',
            'odom_frame_id': 'odom',
            'publish_tf': True,
            'wait_for_transform': 0.2,
            'approx_sync': True,
            'approx_sync_max_interval': 0.01,
            'qos_image': 1,
            'qos_camera_info': 1,
            'subscribe_rgbd': False,
        }],
        remappings=[
            ('rgb/image', '/camera/rgb/image_rect_color'),
            ('rgb/camera_info', '/camera/rgb/camera_info'),
            ('odom', '/rtabmap/odom'),
        ],
        output='screen'
    )
    
    # RTAB-Map SLAM node (main SLAM processing)
    rtabmap_slam_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        parameters=[{
            'frame_id': 'camera_frame',
            'odom_frame_id': 'odom',
            'map_frame_id': 'map',
            'publish_tf': True,
            'subscribe_depth': False,
            'subscribe_rgb': True,
            'subscribe_rgbd': False,
            'subscribe_scan': False,
            'subscribe_odom_info': True,
            'wait_for_transform': 0.2,
            'approx_sync': True,
            'qos_image': 1,
            'qos_camera_info': 1,
            'qos_odom': 1,
            # High-quality monocular SLAM parameters
            'Mem/IncrementalMemory': 'true',
            'Mem/InitWMWithAllNodes': 'true',
            'RGBD/LoopClosureReextractFeatures': 'True',
            'Reg/Strategy': '0',  # Visual registration
            'Grid/3D': 'true',
            'Grid/RangeMax': '20.0',
            'Grid/RangeMin': '0.1',
            'Grid/CellSize': '0.05',
        }],
        remappings=[
            ('rgb/image', '/camera/rgb/image_rect_color'),
            ('rgb/camera_info', '/camera/rgb/camera_info'),
            ('odom', '/rtabmap/odom'),
            ('map', '/rtabmap/map'),
        ],
        output='screen'
    )
    
    return LaunchDescription([
        camera_index_arg,
        camera_config_arg,
        LogInfo(msg='=' * 60),
        LogInfo(msg='Starting RTAB-Map SLAM System'),
        LogInfo(msg='RTAB-Map is the industry-standard real-time 3D SLAM solution'),
        LogInfo(msg='=' * 60),
        camera_node,
        static_tf,
        rtabmap_odom_node,
        rtabmap_slam_node,
    ])
