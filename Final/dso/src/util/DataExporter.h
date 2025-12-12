/**
* Data Exporter for DSO
* Exports point clouds, camera poses, and video
*/

#pragma once

#include "FullSystem/FullSystem.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/Pangolin/DualPangolinDSOViewer.h"
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
    
    // Export point cloud from DualPangolin viewer (for raw or pipeline path)
    static void exportPointCloudFromDualViewer(IOWrap::DualPangolinDSOViewer* viewer, const std::string& filename, bool isRawPath);
    
    // Export video from captured frames
    static void exportVideo(const std::vector<cv::Mat>& frames, const std::string& filename, double fps = 30.0);
    
    // Export all data (poses, point cloud, video)
    static void exportAll(FullSystem* fullSystem, 
                         IOWrap::PangolinDSOViewer* viewer,
                         const std::vector<cv::Mat>& frames,
                         const std::string& outputDir,
                         double fps = 30.0);
    
    // Export all data for dual mode (with DualPangolin viewer)
    static void exportAllDual(FullSystem* fullSystem, 
                              IOWrap::DualPangolinDSOViewer* viewer,
                              const std::vector<cv::Mat>& frames,
                              const std::string& outputDir,
                              double fps,
                              bool isRawPath);
    
    // Calculate and export quantitative metrics
    static void exportQuantitativeMetrics(FullSystem* fullSystem,
                                         IOWrap::DualPangolinDSOViewer* viewer,
                                         const std::string& outputFile,
                                         bool isRawPath,
                                         int totalFrames,
                                         double totalTime);
    
    // Calculate trajectory consistency and reprojection error from exported files
    // This should be called after Pangolin is closed
    static void calculateAndUpdateMetricsFromFiles(const std::string& outputDir,
                                                   const std::string& metricsFile,
                                                   bool isRawPath);
};

} // namespace IOWrap
} // namespace dso

