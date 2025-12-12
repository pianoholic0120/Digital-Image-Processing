#!/usr/bin/env python3
"""
Use RTAB-Map Official Launch File - The highest quality SLAM package for ROS2
This launch file uses RTAB-Map's official launch system for monocular camera SLAM
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
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
        remappings=[
            ('image_raw', '/camera/rgb/image_rect_color'),
            ('camera_info', '/camera/rgb/camera_info'),
        ],
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
    
    # Include RTAB-Map official launch file
    # This uses the industry-standard RTAB-Map configuration
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('rtabmap_launch'),
                'launch',
                'rtabmap.launch.py'
            ])
        ]),
        launch_arguments={
            # Monocular camera configuration
            'stereo': 'false',
            'depth': 'false',  # No depth camera
            'subscribe_rgb': 'true',
            'subscribe_depth': 'false',
            'subscribe_rgbd': 'false',
            'subscribe_scan': 'false',
            
            # Topic remappings for our camera
            'rgb_topic': '/camera/rgb/image_rect_color',
            'camera_info_topic': '/camera/rgb/camera_info',
            
            # Frame IDs
            'frame_id': 'camera_frame',
            'odom_frame_id': 'odom',
            'map_frame_id': 'map',
            
            # Enable visualization
            'rtabmap_viz': 'false',  # Set to 'true' to enable RTAB-Map GUI
            'rviz': 'false',  # Set to 'true' to enable RViz2
            
            # Quality settings for monocular SLAM
            'visual_odometry': 'true',
            'icp_odometry': 'false',
            
            # RTAB-Map parameters (passed via args)
            'args': '--delete_db_on_start',  # Start fresh each time
        }.items()
    )
    
    return LaunchDescription([
        camera_index_arg,
        camera_config_arg,
        LogInfo(msg='Starting RTAB-Map SLAM system using official launch file'),
        LogInfo(msg='RTAB-Map is the industry-standard real-time 3D SLAM solution'),
        camera_node,
        static_tf,
        rtabmap_launch,
    ])





