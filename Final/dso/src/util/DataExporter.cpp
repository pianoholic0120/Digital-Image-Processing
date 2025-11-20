/**
* Data Exporter for DSO
* Implementation
*/

#include "util/DataExporter.h"
#include "FullSystem/FullSystem.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/Pangolin/KeyFrameDisplay.h"
#include "util/FrameShell.h"
#include "util/settings.h"
#include "util/globalCalib.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_types.hpp>

namespace dso
{
namespace IOWrap
{

void DataExporter::exportCameraPoses(FullSystem* fullSystem, const std::string& filename)
{
    if(fullSystem == nullptr)
    {
        printf("ERROR: fullSystem is null, cannot export camera poses!\n");
        return;
    }
    
    std::ofstream file(filename);
    if(!file.is_open())
    {
        printf("ERROR: Cannot open file %s for writing camera poses!\n", filename.c_str());
        return;
    }
    
    file << std::fixed << std::setprecision(6);
    
    // Temporarily disable onlyLogKFPoses to export all poses
    bool originalOnlyLogKFPoses = setting_onlyLogKFPoses;
    setting_onlyLogKFPoses = false;
    
    // Use printResult to generate the pose file, then read it
    // This ensures we get all poses, not just keyframes
    std::string tempFile = filename + ".tmp";
    fullSystem->printResult(tempFile);
    
    // Restore original setting
    setting_onlyLogKFPoses = originalOnlyLogKFPoses;
    
    // Read the temp file and copy to output file
    std::ifstream inFile(tempFile);
    
    if(!inFile.is_open())
    {
        printf("ERROR: Cannot open temp file for reading camera poses!\n");
        file.close();
        boost::filesystem::remove(tempFile);
        return;
    }
    
    std::string line;
    int poseCount = 0;
    while(std::getline(inFile, line))
    {
        if(line.empty()) continue;
        
        // printResult format: timestamp tx ty tz qx qy qz qw
        // This is already in TUM format, so just copy it
        file << line << "\n";
        poseCount++;
    }
    
    inFile.close();
    file.close();
    
    // Remove temp file
    boost::filesystem::remove(tempFile);
    
    printf("Exported %d camera poses to %s\n", poseCount, filename.c_str());
}

void DataExporter::exportPointCloud(FullSystem* fullSystem, const std::string& filename)
{
    if(fullSystem == nullptr)
    {
        printf("ERROR: fullSystem is null, cannot export point cloud!\n");
        return;
    }
    
    std::ofstream file(filename);
    if(!file.is_open())
    {
        printf("ERROR: Cannot open file %s for writing point cloud!\n", filename.c_str());
        return;
    }
    
    // Note: This function requires a viewer to access point cloud data
    // For now, create empty point cloud
    // Use exportPointCloudFromViewer instead if you have a viewer
    int numPoints = 0;
    
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << numPoints << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    file.close();
    printf("Exported %d points to %s (Note: Use exportPointCloudFromViewer for full point cloud)\n", numPoints, filename.c_str());
}

void DataExporter::exportPointCloudFromViewer(IOWrap::PangolinDSOViewer* viewer, const std::string& filename)
{
    if(viewer == nullptr)
    {
        printf("ERROR: viewer is null, cannot export point cloud!\n");
        return;
    }
    
    std::ofstream file(filename);
    if(!file.is_open())
    {
        printf("ERROR: Cannot open file %s for writing point cloud!\n", filename.c_str());
        return;
    }
    
    // Export point cloud from viewer
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> colors;
    viewer->exportPointCloud(points, colors);
    
    int numPoints = points.size();
    
    // Write PLY file
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << numPoints << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    for(size_t i = 0; i < points.size(); i++)
    {
        file << points[i].x() << " " << points[i].y() << " " << points[i].z() << " "
             << (int)colors[i].x() << " " << (int)colors[i].y() << " " << (int)colors[i].z() << "\n";
    }
    
    file.close();
    printf("Exported %d points to %s\n", numPoints, filename.c_str());
}

void DataExporter::exportVideo(const std::vector<cv::Mat>& frames, const std::string& filename, double fps)
{
    if(frames.empty())
    {
        printf("WARNING: No frames to export video!\n");
        return;
    }
    
    if(frames[0].empty())
    {
        printf("ERROR: First frame is empty!\n");
        return;
    }
    
    cv::Size frameSize = frames[0].size();
    cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frameSize);
    
    if(!writer.isOpened())
    {
        printf("ERROR: Cannot open video file %s for writing!\n", filename.c_str());
        return;
    }
    
    for(const cv::Mat& frame : frames)
    {
        if(frame.empty()) continue;
        
        cv::Mat frameToWrite = frame;
        if(frame.channels() == 1)
        {
            cv::cvtColor(frame, frameToWrite, cv::COLOR_GRAY2BGR);
        }
        
        if(frameToWrite.size() != frameSize)
        {
            cv::resize(frameToWrite, frameToWrite, frameSize);
        }
        
        writer.write(frameToWrite);
    }
    
    writer.release();
    printf("Exported %zu frames to video %s (fps=%.2f)\n", frames.size(), filename.c_str(), fps);
}

void DataExporter::exportAll(FullSystem* fullSystem, 
                             IOWrap::PangolinDSOViewer* viewer,
                             const std::vector<cv::Mat>& frames,
                             const std::string& outputDir,
                             double fps)
{
    printf("Starting data export...\n");
    
    // Create output directory
    boost::filesystem::path dir(outputDir);
    if(!boost::filesystem::exists(dir))
    {
        boost::filesystem::create_directories(dir);
        printf("Created output directory: %s\n", outputDir.c_str());
    }
    
    // Export camera poses (fast)
    printf("Exporting camera poses...\n");
    std::string posesFile = outputDir + "/camera_poses.txt";
    exportCameraPoses(fullSystem, posesFile);
    
    // Export point cloud (use viewer if available, may take time)
    printf("Exporting point cloud (this may take a moment)...\n");
    std::string pointCloudFile = outputDir + "/point_cloud.ply";
    if(viewer != nullptr)
    {
        exportPointCloudFromViewer(viewer, pointCloudFile);
    }
    else
    {
        exportPointCloud(fullSystem, pointCloudFile);
    }
    
    // Export video (may take time if many frames)
    if(!frames.empty())
    {
        printf("Exporting video (%zu frames)...\n", frames.size());
        std::string videoFile = outputDir + "/output_video.mp4";
        exportVideo(frames, videoFile, fps);
    }
    
    printf("All data exported to %s\n", outputDir.c_str());
}

} // namespace IOWrap
} // namespace dso

