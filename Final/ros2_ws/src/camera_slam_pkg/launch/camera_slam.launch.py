#!/usr/bin/env python3
"""
Launch file - Start camera node and SLAM nodes
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Get package path
    pkg_share = FindPackageShare('camera_slam_pkg')
    
    # Get project root directory (where camera.yaml is located)
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
    
    # SLAM type argument
    slam_type_arg = DeclareLaunchArgument(
        'slam_type',
        default_value='visual',
        description='SLAM type: simple, visual, or none'
    )
    
    # RViz2 is not launched automatically - user should launch manually
    
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
    
    # Simple SLAM node (demo)
    simple_slam_node = Node(
        package='camera_slam_pkg',
        executable='simple_slam_node',
        name='simple_slam_node',
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('slam_type'), "' == 'simple'"
        ])),
        output='screen'
    )
    
    # Visual SLAM node (complete with map building)
    visual_slam_node = Node(
        package='camera_slam_pkg',
        executable='visual_slam_node',
        name='visual_slam_node',
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('slam_type'), "' == 'visual'"
        ])),
        parameters=[{
            'n_features': 2000,
            'scale_factor': 1.2,
            'n_levels': 8,
        }],
        output='screen'
    )
    
    return LaunchDescription([
        camera_index_arg,
        camera_config_arg,
        slam_type_arg,
        camera_node,
        static_tf,
        simple_slam_node,
        visual_slam_node,
    ])
