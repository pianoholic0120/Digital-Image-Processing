/**
* Dual Pangolin DSO Viewer
* Displays two DSO reconstructions side by side (left: raw, right: pipeline)
*/

#pragma once
#include <pangolin/pangolin.h>
#include <boost/thread/thread.hpp>
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include <map>
#include <deque>

namespace dso
{
class FrameHessian;
class CalibHessian;
class FrameShell;

namespace IOWrap
{
class KeyFrameDisplay;

class DualPangolinDSOViewer : public Output3DWrapper
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    DualPangolinDSOViewer(int w, int h, bool startRunThread=true);
    virtual ~DualPangolinDSOViewer();

    void run();
    void close();

    // ==================== Output3DWrapper Functionality ======================
    virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override;
    virtual void publishKeyframes(std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override;
    virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override;
    virtual void pushLiveFrame(FrameHessian* image) override;
    virtual void pushDepthImage(MinimalImageB3* image) override;
    virtual bool needPushDepthImage() override;
    virtual void join() override;
    virtual void reset() override;

    // Set which system this viewer is for (raw or pipeline)
    void setSystemType(bool isRaw);  // true for raw, false for pipeline
    
    // Set main viewer reference (for wrapper viewers to forward data)
    void setMainViewer(DualPangolinDSOViewer* mainViewer);
    
    // Export point cloud (for raw or pipeline path)
    void exportPointCloud(std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector3f>& colors, bool isRawPath);

private:
    bool needReset;
    void reset_internal();
    void drawConstraints();

    boost::thread runThread;
    bool running;
    int w, h;
    bool isRawSystem;  // true for raw, false for pipeline (mutable for wrapper forwarding)
    
    // Reference to main viewer (for wrapper viewers to forward data)
    DualPangolinDSOViewer* mainViewer;

    // 3D model rendering (separate for each system)
    boost::mutex model3DMutex;
    KeyFrameDisplay* currentCamRaw;
    KeyFrameDisplay* currentCamPipeline;
    std::vector<KeyFrameDisplay*> keyframesRaw;
    std::vector<KeyFrameDisplay*> keyframesPipeline;
    std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>> allFramePosesRaw;
    std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>> allFramePosesPipeline;
    std::map<int, KeyFrameDisplay*> keyframesByKFIDRaw;
    std::map<int, KeyFrameDisplay*> keyframesByKFIDPipeline;

    // Render settings
    bool settings_showKFCameras;
    bool settings_showCurrentCamera;
    bool settings_showTrajectory;
    bool settings_showFullTrajectory;
    bool settings_showActiveConstraints;
    bool settings_showAllConstraints;

    float settings_scaledVarTH;
    float settings_absVarTH;
    int settings_pointCloudMode;
    float settings_minRelBS;
    int settings_sparsity;
    
    // Image display for raw and pipeline
    boost::mutex openImagesMutex;
    MinimalImageB3* internalVideoImgRaw;
    MinimalImageB3* internalVideoImgPipeline;
    bool videoImgChangedRaw;
    bool videoImgChangedPipeline;
};

}  // namespace IOWrap
}  // namespace dso


