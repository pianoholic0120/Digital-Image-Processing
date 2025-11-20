#!/bin/bash
# 清理 ROS2 workspace 的编译产物（保留源码）

cd "$(dirname "$0")"

echo "清理 build, install, log 目录..."
rm -rf build/* install/* log/*

echo "✓ 清理完成！现在可以重新编译："
echo "  colcon build --packages-select camera_slam_pkg"
