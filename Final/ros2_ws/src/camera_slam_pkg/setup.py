from setuptools import setup
import os
from glob import glob

package_name = 'camera_slam_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Camera and SLAM package for ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = camera_slam_pkg.camera_node:main',
            'simple_slam_node = camera_slam_pkg.simple_slam_node:main',
            'visual_slam_node = camera_slam_pkg.visual_slam_node:main',
        ],
    },
)
