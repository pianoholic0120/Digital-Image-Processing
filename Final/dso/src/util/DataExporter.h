/**
* Data Exporter for DSO
* Exports point clouds, camera poses, and video
*/

#pragma once

#include "FullSystem/FullSystem.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "util/FrameShell.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace dso
{
namespace IOWrap
{

class DataExporter
{
public:
    DataExporter() {}
    ~DataExporter() {}
    
    // Export camera poses to file (TUM format: timestamp tx ty tz qx qy qz qw)
    static void exportCameraPoses(FullSystem* fullSystem, const std::string& filename);
    
    // Export point cloud to PLY format
    static void exportPointCloud(FullSystem* fullSystem, const std::string& filename);
    
    // Export point cloud from Pangolin viewer
    static void exportPointCloudFromViewer(IOWrap::PangolinDSOViewer* viewer, const std::string& filename);
    
    // Export video from captured frames
    static void exportVideo(const std::vector<cv::Mat>& frames, const std::string& filename, double fps = 30.0);
    
    // Export all data (poses, point cloud, video)
    static void exportAll(FullSystem* fullSystem, 
                         IOWrap::PangolinDSOViewer* viewer,
                         const std::vector<cv::Mat>& frames,
                         const std::string& outputDir,
                         double fps = 30.0);
};

} // namespace IOWrap
} // namespace dso

