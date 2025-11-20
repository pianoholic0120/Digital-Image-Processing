#!/usr/bin/env python3
"""
Complete RTAB-Map Launch File - Using the highest quality SLAM package for ROS2
RTAB-Map is the industry-standard real-time 3D SLAM solution with loop closure detection
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
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
    
    use_rtabmap_viz_arg = DeclareLaunchArgument(
        'use_rtabmap_viz',
        default_value='false',
        description='Launch RTAB-Map visualization GUI'
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
            ('image_raw', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
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
    
    # Get RTAB-Map config file path
    config_file = os.path.join(
        os.path.dirname(__file__),
        '..', 'config', 'rtabmap.yaml'
    )
    
    # RTAB-Map Visual Odometry (for monocular camera)
    # This provides high-quality visual odometry estimation
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
            'config_path': config_file,  # Use RTAB-Map config file
            'qos_image': 1,  # Reliable
            'qos_camera_info': 1,  # Reliable
            'subscribe_rgbd': False,  # We have separate RGB and depth topics
        }],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
            ('odom', '/rtabmap/odom'),
        ],
        output='screen',
        condition=IfCondition('true')  # Always launch if package exists
    )
    
    # RTAB-Map SLAM node (main SLAM processing)
    # This provides loop closure detection and 3D map building with DBoW2 and g2o
    rtabmap_slam_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        parameters=[{
            'frame_id': 'camera_frame',
            'odom_frame_id': 'odom',
            'map_frame_id': 'map',
            'publish_tf': True,
            'subscribe_depth': False,  # Monocular (no depth)
            'subscribe_rgb': True,
            'subscribe_rgbd': False,
            'subscribe_scan': False,
            'subscribe_odom_info': True,  # Use odometry info from rtabmap_odom
            'wait_for_transform': 0.2,
            'approx_sync': True,
            'qos_image': 1,
            'qos_camera_info': 1,
            'qos_odom': 1,
            # Load RTAB-Map config file (contains DBoW2 and g2o settings)
            'config_path': config_file,
        }],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
            ('odom', '/rtabmap/odom'),
            ('map', '/rtabmap/map'),
        ],
        output='screen',
        condition=IfCondition('true')
    )
    
    # RTAB-Map Visualization (optional GUI)
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        parameters=[{
            'frame_id': 'camera_frame',
            'odom_frame_id': 'odom',
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
        }],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
            ('odom', '/rtabmap/odom'),
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rtabmap_viz'))
    )
    
    # Point cloud publisher (converts RTAB-Map output to PointCloud2)
    point_cloud_node = Node(
        package='rtabmap_util',
        executable='point_cloud_xyzrgb',
        name='point_cloud_xyzrgb',
        parameters=[{
            'decimation': 4,  # Reduce point cloud density for performance
            'voxel_size': 0.01,  # 1cm voxel size
            'approx_sync': True,
            'approx_sync_max_interval': 0.01,
            'qos': 1,
            'qos_camera_info': 1,
        }],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
            ('cloud', '/rtabmap/cloud'),
        ],
        output='screen',
        condition=IfCondition('true')
    )
    
    return LaunchDescription([
        camera_index_arg,
        camera_config_arg,
        use_rtabmap_viz_arg,
        LogInfo(msg='Starting RTAB-Map SLAM system (highest quality SLAM package for ROS2)'),
        camera_node,
        static_tf,
        rtabmap_odom_node,
        rtabmap_slam_node,
        rtabmap_viz_node,
        point_cloud_node,
    ])

