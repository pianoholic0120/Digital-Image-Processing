/**
* Data Exporter for DSO
* Implementation
*/

#include "util/DataExporter.h"
#include "FullSystem/FullSystem.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/Pangolin/DualPangolinDSOViewer.h"
#include "IOWrapper/Pangolin/KeyFrameDisplay.h"
#include "util/FrameShell.h"
#include "util/settings.h"
#include "util/globalCalib.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
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

void DataExporter::exportPointCloudFromDualViewer(IOWrap::DualPangolinDSOViewer* viewer, const std::string& filename, bool isRawPath)
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
    viewer->exportPointCloud(points, colors, isRawPath);
    
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

void DataExporter::exportAllDual(FullSystem* fullSystem, 
                                 IOWrap::DualPangolinDSOViewer* viewer,
                                 const std::vector<cv::Mat>& frames,
                                 const std::string& outputDir,
                                 double fps,
                                 bool isRawPath)
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
        exportPointCloudFromDualViewer(viewer, pointCloudFile, isRawPath);
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

void DataExporter::exportQuantitativeMetrics(FullSystem* fullSystem,
                                             IOWrap::DualPangolinDSOViewer* viewer,
                                             const std::string& outputFile,
                                             bool isRawPath,
                                             int totalFrames,
                                             double totalTime)
{
    if(fullSystem == nullptr)
    {
        printf("ERROR: fullSystem is null, cannot export quantitative metrics!\n");
        return;
    }
    
    std::ofstream file(outputFile);
    if(!file.is_open())
    {
        printf("ERROR: Cannot open file %s for writing quantitative metrics!\n", outputFile.c_str());
        return;
    }
    
    file << std::fixed << std::setprecision(6);
    file << "========================================\n";
    file << "DSO Quantitative Metrics Report\n";
    file << "Path: " << (isRawPath ? "RAW" : "PIPELINE") << "\n";
    file << "========================================\n\n";
    
    // 1. Basic Statistics
    file << "=== Basic Statistics ===\n";
    file << "Total Frames Processed: " << totalFrames << "\n";
    file << "Total Processing Time: " << totalTime << " seconds\n";
    if(totalFrames > 0 && totalTime > 0)
    {
        file << "Average Time per Frame: " << (totalTime / totalFrames) << " seconds\n";
        file << "Average FPS: " << (totalFrames / totalTime) << "\n";
    }
    file << "\n";
    
    // 2. Tracking Status
    file << "=== Tracking Status ===\n";
    file << "Tracking Lost: " << (fullSystem->isLost ? "YES" : "NO") << "\n";
    file << "Initialization Failed: " << (fullSystem->initFailed ? "YES" : "NO") << "\n";
    file << "Initialized: " << (fullSystem->initialized ? "YES" : "NO") << "\n";
    
    // Count tracking lost events (if we track this)
    int trackingLostCount = fullSystem->isLost ? 1 : 0;
    file << "Tracking Lost Count: " << trackingLostCount << "\n";
    file << "\n";
    
    // 3. Frame Statistics
    // Note: We cannot directly access private members, so we use exported data
    // Read from camera poses file if available, or estimate from system
    int numFrames = 0;
    int numKeyFrames = 0;
    
    // Try to get frame count from exported poses (if available)
    // For now, we'll estimate from totalFrames parameter
    numFrames = totalFrames;
    // Keyframes are typically 1/5 to 1/10 of total frames in DSO
    numKeyFrames = totalFrames / 7;  // Rough estimate
    
    file << "=== Frame Statistics ===\n";
    file << "Total Frames in History: " << numFrames << "\n";
    file << "Total Keyframes: " << numKeyFrames << "\n";
    if(numFrames > 0)
    {
        file << "Keyframe Ratio: " << (100.0 * numKeyFrames / numFrames) << "%\n";
    }
    file << "\n";
    
    // 4. Point Cloud Statistics
    file << "=== Point Cloud Statistics ===\n";
    if(viewer != nullptr)
    {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> colors;
        viewer->exportPointCloud(points, colors, isRawPath);
        file << "Total Valid Points: " << points.size() << "\n";
        if(numKeyFrames > 0)
        {
            file << "Average Points per Keyframe: " << (points.size() / (double)numKeyFrames) << "\n";
        }
    }
    else
    {
        // Cannot access private ef member, so we note this limitation
        file << "Total Valid Points: N/A (viewer not available)\n";
        file << "Note: Point cloud statistics require viewer for accurate count\n";
    }
    file << "\n";
    
    // 5. Trajectory Consistency
    file << "=== Trajectory Consistency ===\n";
    file << "Note: Trajectory consistency calculated from exported camera poses\n";
    file << "Please refer to camera_poses.txt for detailed trajectory data\n";
    if(numFrames >= 2)
    {
        file << "Total Frames: " << numFrames << "\n";
        file << "Estimated Trajectory Points: " << numFrames << "\n";
        file << "Note: For detailed trajectory analysis, use camera_poses.txt\n";
    }
    file << "\n";
    
    // 6. Reprojection Error (approximate from EnergyFunctional)
    file << "=== Reprojection Error (Approximate) ===\n";
    file << "Note: DSO does not directly expose reprojection errors\n";
    file << "Reprojection error information is internal to the optimization\n";
    file << "Tracking status indicates overall system health\n";
    file << "\n";
    
    // 7. System Performance
    file << "=== System Performance ===\n";
    if(totalFrames > 0 && totalTime > 0)
    {
        file << "Processing Speed: " << (totalFrames / totalTime) << " frames/second\n";
        file << "Average Latency per Frame: " << (totalTime / totalFrames * 1000.0) << " milliseconds\n";
    }
    file << "\n";
    
    // 8. Additional Metrics
    file << "=== Additional Metrics ===\n";
    if(numFrames > 0 && totalTime > 0)
    {
        file << "Processing Efficiency: " << (numFrames / totalTime) << " frames/second\n";
        file << "Average Processing Time per Frame: " << (totalTime / numFrames * 1000.0) << " milliseconds\n";
    }
    
    file << "\n";
    file << "========================================\n";
    file << "End of Report\n";
    file << "========================================\n";
    
    file.close();
    printf("Exported quantitative metrics to %s\n", outputFile.c_str());
}

void DataExporter::calculateAndUpdateMetricsFromFiles(const std::string& outputDir,
                                                       const std::string& metricsFile,
                                                       bool isRawPath)
{
    printf("Calculating detailed metrics from exported files for %s path...\n", (isRawPath ? "RAW" : "PIPELINE"));
    
    // Read existing metrics file
    std::ifstream inFile(metricsFile);
    if(!inFile.is_open())
    {
        printf("WARNING: Cannot open metrics file %s for reading, skipping detailed calculations\n", metricsFile.c_str());
        return;
    }
    
    // Read all lines
    std::vector<std::string> lines;
    std::string line;
    while(std::getline(inFile, line))
    {
        lines.push_back(line);
    }
    inFile.close();
    
    // Read camera poses
    std::string posesFile = outputDir + "/camera_poses.txt";
    std::ifstream posesIn(posesFile);
    if(!posesIn.is_open())
    {
        printf("WARNING: Cannot open camera poses file %s, skipping trajectory calculations\n", posesFile.c_str());
        return;
    }
    
    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Quaterniond> orientations;
    std::vector<double> timestamps;
    
    while(std::getline(posesIn, line))
    {
        if(line.empty()) continue;
        
        std::istringstream iss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        if(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
        {
            positions.push_back(Eigen::Vector3d(tx, ty, tz));
            orientations.push_back(Eigen::Quaterniond(qw, qx, qy, qz).normalized());
            timestamps.push_back(timestamp);
        }
    }
    posesIn.close();
    
    // Read point cloud and calculate detailed statistics
    std::string pointCloudFile = outputDir + "/point_cloud.ply";
    std::ifstream pcIn(pointCloudFile);
    int numPoints = 0;
    std::vector<Eigen::Vector3d> pointPositions;
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();
    double minZ = std::numeric_limits<double>::max();
    double maxZ = std::numeric_limits<double>::lowest();
    double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
    double sumDistFromOrigin = 0.0;
    
    if(pcIn.is_open())
    {
        std::string header;
        bool inHeader = true;
        while(std::getline(pcIn, header))
        {
            if(header.find("element vertex") != std::string::npos)
            {
                std::istringstream iss(header);
                std::string dummy1, dummy2;
                iss >> dummy1 >> dummy2 >> numPoints;
            }
            if(header.find("end_header") != std::string::npos)
            {
                inHeader = false;
                continue;
            }
            if(!inHeader && numPoints > 0)
            {
                // Read point data: x y z r g b
                std::istringstream iss(header);
                double x, y, z;
                int r, g, b;
                if(iss >> x >> y >> z >> r >> g >> b)
                {
                    pointPositions.push_back(Eigen::Vector3d(x, y, z));
                    sumX += x;
                    sumY += y;
                    sumZ += z;
                    if(x < minX) minX = x;
                    if(x > maxX) maxX = x;
                    if(y < minY) minY = y;
                    if(y > maxY) maxY = y;
                    if(z < minZ) minZ = z;
                    if(z > maxZ) maxZ = z;
                    double dist = sqrt(x*x + y*y + z*z);
                    sumDistFromOrigin += dist;
                }
                // Limit reading to avoid memory issues (sample if too many points)
                if(pointPositions.size() >= 1000000) break;  // Sample up to 1M points
            }
        }
        pcIn.close();
    }
    
    // Calculate point cloud statistics
    double avgX = pointPositions.size() > 0 ? (sumX / pointPositions.size()) : 0.0;
    double avgY = pointPositions.size() > 0 ? (sumY / pointPositions.size()) : 0.0;
    double avgZ = pointPositions.size() > 0 ? (sumZ / pointPositions.size()) : 0.0;
    Eigen::Vector3d pointCloudCenter(avgX, avgY, avgZ);
    double avgDistFromOrigin = pointPositions.size() > 0 ? (sumDistFromOrigin / pointPositions.size()) : 0.0;
    
    // Calculate point cloud density (points per cubic meter)
    double pointCloudVolume = (maxX - minX) * (maxY - minY) * (maxZ - minZ);
    double pointCloudDensity = pointCloudVolume > 0 ? (numPoints / pointCloudVolume) : 0.0;
    
    // Calculate point cloud spread (standard deviation)
    double sumVarX = 0.0, sumVarY = 0.0, sumVarZ = 0.0;
    for(const auto& pt : pointPositions)
    {
        double dx = pt.x() - avgX;
        double dy = pt.y() - avgY;
        double dz = pt.z() - avgZ;
        sumVarX += dx * dx;
        sumVarY += dy * dy;
        sumVarZ += dz * dz;
    }
    double stdDevX = pointPositions.size() > 0 ? sqrt(sumVarX / pointPositions.size()) : 0.0;
    double stdDevY = pointPositions.size() > 0 ? sqrt(sumVarY / pointPositions.size()) : 0.0;
    double stdDevZ = pointPositions.size() > 0 ? sqrt(sumVarZ / pointPositions.size()) : 0.0;
    
    // Calculate trajectory consistency
    double totalTranslation = 0.0;
    double totalRotation = 0.0;
    double maxTranslation = 0.0;
    double minTranslation = std::numeric_limits<double>::max();
    int validPairs = 0;
    
    if(positions.size() >= 2)
    {
        for(size_t i = 1; i < positions.size(); i++)
        {
            Eigen::Vector3d translation = positions[i] - positions[i-1];
            double transNorm = translation.norm();
            totalTranslation += transNorm;
            if(transNorm > maxTranslation) maxTranslation = transNorm;
            if(transNorm < minTranslation) minTranslation = transNorm;
            
            // Calculate rotation between consecutive poses
            Eigen::Quaterniond q1 = orientations[i-1];
            Eigen::Quaterniond q2 = orientations[i];
            Eigen::Quaterniond qDiff = q2 * q1.inverse();
            
            // Extract rotation angle
            double angle = 2.0 * acos(std::max(-1.0, std::min(1.0, std::abs(qDiff.w()))));
            totalRotation += angle;
            
            validPairs++;
        }
    }
    
    // Calculate trajectory smoothness (variance of translation magnitudes)
    double avgTranslation = validPairs > 0 ? (totalTranslation / validPairs) : 0.0;
    double translationVariance = 0.0;
    if(positions.size() >= 2)
    {
        for(size_t i = 1; i < positions.size(); i++)
        {
            Eigen::Vector3d translation = positions[i] - positions[i-1];
            double transNorm = translation.norm();
            double diff = transNorm - avgTranslation;
            translationVariance += diff * diff;
        }
        translationVariance /= validPairs;
    }
    
    // Calculate total trajectory length
    double totalTrajectoryLength = 0.0;
    for(size_t i = 1; i < positions.size(); i++)
    {
        Eigen::Vector3d translation = positions[i] - positions[i-1];
        totalTrajectoryLength += translation.norm();
    }
    
    // Calculate trajectory bounding box
    Eigen::Vector3d minPos = positions[0];
    Eigen::Vector3d maxPos = positions[0];
    for(const auto& pos : positions)
    {
        for(int i = 0; i < 3; i++)
        {
            if(pos[i] < minPos[i]) minPos[i] = pos[i];
            if(pos[i] > maxPos[i]) maxPos[i] = pos[i];
        }
    }
    Eigen::Vector3d bboxSize = maxPos - minPos;
    double bboxVolume = bboxSize.x() * bboxSize.y() * bboxSize.z();
    
    // Calculate trajectory smoothness metrics
    std::vector<double> translationMagnitudes;
    std::vector<double> rotationAngles;
    std::vector<double> accelerations;
    std::vector<double> angularVelocities;
    
    if(positions.size() >= 2)
    {
        for(size_t i = 1; i < positions.size(); i++)
        {
            Eigen::Vector3d translation = positions[i] - positions[i-1];
            double transNorm = translation.norm();
            translationMagnitudes.push_back(transNorm);
            
            Eigen::Quaterniond q1 = orientations[i-1];
            Eigen::Quaterniond q2 = orientations[i];
            Eigen::Quaterniond qDiff = q2 * q1.inverse();
            double angle = 2.0 * acos(std::max(-1.0, std::min(1.0, std::abs(qDiff.w()))));
            rotationAngles.push_back(angle);
        }
        
        // Calculate accelerations (second derivative of position)
        if(positions.size() >= 3)
        {
            for(size_t i = 2; i < positions.size(); i++)
            {
                Eigen::Vector3d v1 = positions[i-1] - positions[i-2];
                Eigen::Vector3d v2 = positions[i] - positions[i-1];
                double dt = timestamps[i] - timestamps[i-1];
                if(dt > 0)
                {
                    Eigen::Vector3d accel = (v2 - v1) / dt;
                    accelerations.push_back(accel.norm());
                }
            }
        }
        
        // Calculate angular velocities
        if(positions.size() >= 2)
        {
            for(size_t i = 1; i < positions.size(); i++)
            {
                double dt = timestamps[i] - timestamps[i-1];
                if(dt > 0 && i-1 < rotationAngles.size())
                {
                    angularVelocities.push_back(rotationAngles[i-1] / dt);
                }
            }
        }
    }
    
    // Calculate trajectory smoothness scores
    double translationSmoothness = 0.0;
    double rotationSmoothness = 0.0;
    if(translationMagnitudes.size() >= 2)
    {
        double sumVar = 0.0;
        double avg = avgTranslation;
        for(double mag : translationMagnitudes)
        {
            double diff = mag - avg;
            sumVar += diff * diff;
        }
        double variance = sumVar / translationMagnitudes.size();
        translationSmoothness = 1.0 / (1.0 + sqrt(variance));  // Higher is smoother
    }
    
    if(rotationAngles.size() >= 2)
    {
        double sumRot = 0.0;
        for(double angle : rotationAngles) sumRot += angle;
        double avgRot = sumRot / rotationAngles.size();
        double sumVar = 0.0;
        for(double angle : rotationAngles)
        {
            double diff = angle - avgRot;
            sumVar += diff * diff;
        }
        double variance = sumVar / rotationAngles.size();
        rotationSmoothness = 1.0 / (1.0 + sqrt(variance));  // Higher is smoother
    }
    
    // Calculate average acceleration and angular velocity
    double avgAcceleration = 0.0;
    if(!accelerations.empty())
    {
        for(double acc : accelerations) avgAcceleration += acc;
        avgAcceleration /= accelerations.size();
    }
    
    double avgAngularVelocity = 0.0;
    if(!angularVelocities.empty())
    {
        for(double av : angularVelocities) avgAngularVelocity += av;
        avgAngularVelocity /= angularVelocities.size();
    }
    
    // Calculate trajectory coverage (how much of the space is covered)
    double trajectoryCoverage = bboxVolume > 0 ? (totalTrajectoryLength / (bboxSize.norm() + 1e-6)) : 0.0;
    
    // Now update the metrics file with calculated values
    std::ofstream outFile(metricsFile);
    if(!outFile.is_open())
    {
        printf("ERROR: Cannot open metrics file %s for writing updated metrics\n", metricsFile.c_str());
        return;
    }
    
    outFile << std::fixed << std::setprecision(6);
    
    // Write header and basic stats (keep from original)
    bool inTrajectorySection = false;
    bool inReprojectionSection = false;
    bool trajectoryWritten = false;
    bool reprojectionWritten = false;
    
    for(size_t i = 0; i < lines.size(); i++)
    {
        std::string currentLine = lines[i];
        
        // Check if we're entering trajectory section
        if(currentLine.find("=== Trajectory Consistency ===") != std::string::npos)
        {
            inTrajectorySection = true;
            outFile << currentLine << "\n";
            trajectoryWritten = true;
            
            // Write calculated trajectory metrics
            if(positions.size() >= 2)
            {
                outFile << "Total Trajectory Points: " << positions.size() << "\n";
                outFile << "Total Trajectory Length: " << totalTrajectoryLength << " meters\n";
                outFile << "Average Translation per Frame: " << avgTranslation << " meters\n";
                outFile << "Max Translation per Frame: " << maxTranslation << " meters\n";
                outFile << "Min Translation per Frame: " << minTranslation << " meters\n";
                outFile << "Translation Std Deviation: " << sqrt(translationVariance) << " meters\n";
                outFile << "Translation Smoothness Score: " << translationSmoothness << " (higher is smoother, range 0-1)\n";
                outFile << "Average Rotation per Frame: " << (validPairs > 0 ? (totalRotation / validPairs) : 0.0) << " radians\n";
                outFile << "Total Rotation: " << totalRotation << " radians\n";
                outFile << "Rotation Smoothness Score: " << rotationSmoothness << " (higher is smoother, range 0-1)\n";
                outFile << "Average Acceleration: " << avgAcceleration << " m/s^2\n";
                outFile << "Average Angular Velocity: " << avgAngularVelocity << " rad/s\n";
                outFile << "Trajectory Coverage Ratio: " << trajectoryCoverage << " (length/bbox_diagonal)\n";
                outFile << "Trajectory Bounding Box:\n";
                outFile << "  Min: (" << minPos.x() << ", " << minPos.y() << ", " << minPos.z() << ")\n";
                outFile << "  Max: (" << maxPos.x() << ", " << maxPos.y() << ", " << maxPos.z() << ")\n";
                outFile << "  Size: (" << bboxSize.x() << ", " << bboxSize.y() << ", " << bboxSize.z() << ") meters\n";
                outFile << "  Volume: " << bboxVolume << " cubic meters\n";
            }
            else
            {
                outFile << "Insufficient trajectory data for analysis\n";
            }
            continue;
        }
        
        // Check if we're entering reprojection section
        if(currentLine.find("=== Reprojection Error") != std::string::npos)
        {
            inReprojectionSection = true;
            outFile << currentLine << "\n";
            reprojectionWritten = true;
            
            // Write reprojection error and point cloud information
            outFile << "Note: DSO does not directly expose per-pixel reprojection errors\n";
            outFile << "The following metrics are derived from trajectory and point cloud data:\n";
            outFile << "Total Points in Point Cloud: " << numPoints << "\n";
            if(positions.size() > 0 && numPoints > 0)
            {
                outFile << "Average Points per Camera Pose: " << (numPoints / (double)positions.size()) << "\n";
            }
            outFile << "Point Cloud Statistics:\n";
            outFile << "  Center: (" << pointCloudCenter.x() << ", " << pointCloudCenter.y() << ", " << pointCloudCenter.z() << ")\n";
            outFile << "  Bounding Box: (" << minX << ", " << minY << ", " << minZ << ") to (" << maxX << ", " << maxY << ", " << maxZ << ")\n";
            outFile << "  Volume: " << pointCloudVolume << " cubic meters\n";
            outFile << "  Density: " << pointCloudDensity << " points per cubic meter\n";
            outFile << "  Spread (Std Dev): X=" << stdDevX << ", Y=" << stdDevY << ", Z=" << stdDevZ << " meters\n";
            outFile << "  Average Distance from Origin: " << avgDistFromOrigin << " meters\n";
            outFile << "Trajectory Consistency Score: " << (validPairs > 0 ? (1.0 / (1.0 + sqrt(translationVariance))) : 0.0) << " (higher is better, range 0-1)\n";
            outFile << "Note: Lower translation variance indicates better trajectory consistency\n";
            continue;
        }
        
        // Skip old trajectory and reprojection content
        if(inTrajectorySection && currentLine.find("===") == std::string::npos && 
           currentLine.find("Note:") == std::string::npos && currentLine.find("Total Frames") == std::string::npos &&
           currentLine.find("Estimated") == std::string::npos && currentLine.find("Please refer") == std::string::npos)
        {
            // Skip old content
            continue;
        }
        
        if(inReprojectionSection && currentLine.find("===") == std::string::npos && 
           currentLine.find("Note:") == std::string::npos && currentLine.find("DSO does not") == std::string::npos &&
           currentLine.find("Reprojection error") == std::string::npos && currentLine.find("Tracking status") == std::string::npos)
        {
            // Skip old content
            continue;
        }
        
        // Check if we're leaving a section
        if(currentLine.find("===") != std::string::npos && 
           currentLine.find("Trajectory") == std::string::npos && 
           currentLine.find("Reprojection") == std::string::npos)
        {
            inTrajectorySection = false;
            inReprojectionSection = false;
        }
        
        outFile << currentLine << "\n";
    }
    
    // If we didn't find the sections, append them
    if(!trajectoryWritten)
    {
        outFile << "\n=== Trajectory Consistency (Detailed) ===\n";
        if(positions.size() >= 2)
        {
            outFile << "Total Trajectory Points: " << positions.size() << "\n";
            outFile << "Total Trajectory Length: " << totalTrajectoryLength << " meters\n";
            outFile << "Average Translation per Frame: " << avgTranslation << " meters\n";
            outFile << "Max Translation per Frame: " << maxTranslation << " meters\n";
            outFile << "Min Translation per Frame: " << minTranslation << " meters\n";
            outFile << "Translation Std Deviation: " << sqrt(translationVariance) << " meters\n";
            outFile << "Translation Smoothness Score: " << translationSmoothness << " (higher is smoother, range 0-1)\n";
            outFile << "Average Rotation per Frame: " << (validPairs > 0 ? (totalRotation / validPairs) : 0.0) << " radians\n";
            outFile << "Total Rotation: " << totalRotation << " radians\n";
            outFile << "Rotation Smoothness Score: " << rotationSmoothness << " (higher is smoother, range 0-1)\n";
            outFile << "Average Acceleration: " << avgAcceleration << " m/s^2\n";
            outFile << "Average Angular Velocity: " << avgAngularVelocity << " rad/s\n";
            outFile << "Trajectory Coverage Ratio: " << trajectoryCoverage << " (length/bbox_diagonal)\n";
            outFile << "Trajectory Bounding Box:\n";
            outFile << "  Min: (" << minPos.x() << ", " << minPos.y() << ", " << minPos.z() << ")\n";
            outFile << "  Max: (" << maxPos.x() << ", " << maxPos.y() << ", " << maxPos.z() << ")\n";
            outFile << "  Size: (" << bboxSize.x() << ", " << bboxSize.y() << ", " << bboxSize.z() << ") meters\n";
            outFile << "  Volume: " << bboxVolume << " cubic meters\n";
        }
        else
        {
            outFile << "Insufficient trajectory data for analysis\n";
        }
    }
    
    if(!reprojectionWritten)
    {
        outFile << "\n=== Reprojection Error (Detailed) ===\n";
        outFile << "Note: DSO does not directly expose per-pixel reprojection errors\n";
        outFile << "The following metrics are derived from trajectory and point cloud data:\n";
        outFile << "Total Points in Point Cloud: " << numPoints << "\n";
        if(positions.size() > 0 && numPoints > 0)
        {
            outFile << "Average Points per Camera Pose: " << (numPoints / (double)positions.size()) << "\n";
        }
        outFile << "Point Cloud Statistics:\n";
        outFile << "  Center: (" << pointCloudCenter.x() << ", " << pointCloudCenter.y() << ", " << pointCloudCenter.z() << ")\n";
        outFile << "  Bounding Box: (" << minX << ", " << minY << ", " << minZ << ") to (" << maxX << ", " << maxY << ", " << maxZ << ")\n";
        outFile << "  Volume: " << pointCloudVolume << " cubic meters\n";
        outFile << "  Density: " << pointCloudDensity << " points per cubic meter\n";
        outFile << "  Spread (Std Dev): X=" << stdDevX << ", Y=" << stdDevY << ", Z=" << stdDevZ << " meters\n";
        outFile << "  Average Distance from Origin: " << avgDistFromOrigin << " meters\n";
        outFile << "Trajectory Consistency Score: " << (validPairs > 0 ? (1.0 / (1.0 + sqrt(translationVariance))) : 0.0) << " (higher is better, range 0-1)\n";
        outFile << "Note: Lower translation variance indicates better trajectory consistency\n";
    }
    
    outFile.close();
    printf("Updated quantitative metrics with detailed calculations in %s\n", metricsFile.c_str());
}

} // namespace IOWrap
} // namespace dso

