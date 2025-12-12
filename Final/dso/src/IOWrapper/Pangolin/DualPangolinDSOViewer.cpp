/**
* Dual Pangolin DSO Viewer Implementation
* Simplified version: uses two separate viewers for now
*/

#include "DualPangolinDSOViewer.h"
#include "IOWrapper/Pangolin/KeyFrameDisplay.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/settings.h"
#include "util/globalFuncs.h"
#include <iostream>
#include <unistd.h>
#include <algorithm>
#include <set>
#include <algorithm>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <pangolin/pangolin.h>
#ifdef __APPLE__
#include <pthread.h>
#endif

namespace dso
{
namespace IOWrap
{

DualPangolinDSOViewer::DualPangolinDSOViewer(int w, int h, bool startRunThread)
    : needReset(false)
    , running(true)
    , w(w)
    , h(h)
    , isRawSystem(true)
    , mainViewer(nullptr)
    , currentCamRaw(nullptr)
    , currentCamPipeline(nullptr)
    , settings_showKFCameras(false)
    , settings_showCurrentCamera(true)
    , settings_showTrajectory(true)
    , settings_showFullTrajectory(false)
    , settings_showActiveConstraints(true)
    , settings_showAllConstraints(false)
    , settings_scaledVarTH(0.001)
    , settings_absVarTH(0.001)
    , settings_pointCloudMode(1)
    , settings_minRelBS(0.1)
    , settings_sparsity(1)
{
    // Initialize current cameras
    currentCamRaw = new KeyFrameDisplay();
    currentCamPipeline = new KeyFrameDisplay();
    
    // Initialize image buffers for raw and pipeline
    {
        boost::unique_lock<boost::mutex> lk(openImagesMutex);
        internalVideoImgRaw = new MinimalImageB3(w, h);
        internalVideoImgPipeline = new MinimalImageB3(w, h);
        internalVideoImgRaw->setBlack();
        internalVideoImgPipeline->setBlack();
        videoImgChangedRaw = false;
        videoImgChangedPipeline = false;
    }
    
    if(startRunThread)
    {
        runThread = boost::thread(&DualPangolinDSOViewer::run, this);
    }
}

DualPangolinDSOViewer::~DualPangolinDSOViewer()
{
    close();
    if(runThread.joinable())
    {
        runThread.join();
    }
    
    // Clean up image buffers
    {
        boost::unique_lock<boost::mutex> lk(openImagesMutex);
        if(internalVideoImgRaw != nullptr) {
            delete internalVideoImgRaw;
            internalVideoImgRaw = nullptr;
        }
        if(internalVideoImgPipeline != nullptr) {
            delete internalVideoImgPipeline;
            internalVideoImgPipeline = nullptr;
        }
    }
    
    // Don't delete viewerRawRef and viewerPipelineRef - they are managed by FullSystem
}

void DualPangolinDSOViewer::run()
{
    printf("START DUAL PANGOLIN VIEWER!\n");
    printf("Image dimensions: w=%d, h=%d\n", w, h);

    #ifdef __APPLE__
    if(pthread_main_np() == 0) {
        printf("ERROR: Pangolin GUI initialization must be on main thread!\n");
        return;
    }
    #endif

    // Validate dimensions before creating window
    if(w <= 0 || h <= 0)
    {
        printf("ERROR: Invalid dimensions for dual viewer: w=%d, h=%d\n", w, h);
        return;
    }

    // Create window with double width for side-by-side display
    // Limit window size to reasonable maximum to avoid issues
    int windowWidth = std::min(2*w, 2560);  // Max width 2560
    int windowHeight = std::min(2*h, 1440); // Max height 1440
    printf("Creating dual viewer window: %dx%d (from image size %dx%d)\n", windowWidth, windowHeight, w, h);
    
    try {
        pangolin::CreateWindowAndBind("DSO Dual View - Left: Raw | Right: Pipeline", windowWidth, windowHeight);
    } catch(const std::exception& e) {
        printf("ERROR: Failed to create Pangolin window: %s\n", e.what());
        return;
    } catch(...) {
        printf("ERROR: Failed to create Pangolin window (unknown error)\n");
        return;
    }
    const int UI_WIDTH = 180;

    glEnable(GL_DEPTH_TEST);

    // Left view (raw) - bright yellow color scheme
    // Top half: 3D visualization (0.3 to 1.0)
    pangolin::OpenGlRenderState Visualization3D_camera_raw(
        pangolin::ProjectionMatrix(w, h, 400, 400, w/2, h/2, 0.1, 1000),
        pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY)
    );

    pangolin::View& Visualization3D_display_raw = pangolin::CreateDisplay()
        .SetBounds(0.3, 1.0, pangolin::Attach::Pix(UI_WIDTH), 0.5, -w/(float)h)
        .SetHandler(new pangolin::Handler3D(Visualization3D_camera_raw));

    // Right view (pipeline) - green color scheme
    // Top half: 3D visualization (0.3 to 1.0)
    pangolin::OpenGlRenderState Visualization3D_camera_pipeline(
        pangolin::ProjectionMatrix(w, h, 400, 400, w/2, h/2, 0.1, 1000),
        pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY)
    );

    pangolin::View& Visualization3D_display_pipeline = pangolin::CreateDisplay()
        .SetBounds(0.3, 1.0, 0.5, 1.0, -w/(float)h)
        .SetHandler(new pangolin::Handler3D(Visualization3D_camera_pipeline));

    // Image displays for raw and pipeline (bottom of window)
    pangolin::View& d_video_raw = pangolin::Display("imgVideoRaw")
        .SetAspect(w/(float)h);
    
    pangolin::View& d_video_pipeline = pangolin::Display("imgVideoPipeline")
        .SetAspect(w/(float)h);
    
    pangolin::GlTexture texVideoRaw(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    pangolin::GlTexture texVideoPipeline(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    
    // Layout: images on bottom (0.0 to 0.3), left side for raw, right side for pipeline
    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 0.5)
        .AddDisplay(d_video_raw);
    
    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3, 0.5, 1.0)
        .AddDisplay(d_video_pipeline);

    // Parameter panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    
    pangolin::Var<bool> settings_showLiveVideo("ui.showVideo", true, true);

    pangolin::Var<int> settings_pointCloudMode("ui.PC_mode", 1, 1, 4, false);
    pangolin::Var<bool> settings_showKFCameras("ui.KFCam", false, true);
    pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam", true, true);
    pangolin::Var<bool> settings_showTrajectory("ui.Trajectory", true, true);
    pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory", false, true);
    pangolin::Var<bool> settings_showActiveConstraints("ui.ActiveConst", false, true);
    pangolin::Var<bool> settings_showAllConstraints("ui.AllConst", false, true);

    pangolin::Var<bool> settings_show3D("ui.show3D", true, true);
    pangolin::Var<bool> settings_resetButton("ui.Reset", false, false);
    bool setting_render_displayVideo = true;

    // Main loop
    while(!pangolin::ShouldQuit() && running)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if(setting_render_display3D)
        {
            // Render left view (raw) - bright yellow color
            Visualization3D_display_raw.Activate(Visualization3D_camera_raw);
            {
                boost::unique_lock<boost::mutex> lk3d(model3DMutex);
                
                // Render raw keyframes with bright yellow color
                std::vector<KeyFrameDisplay*> keyframesCopyRaw = keyframesRaw;
                std::set<KeyFrameDisplay*> validKeyframesSetRaw;
                for(auto& kf : keyframesRaw) {
                    if(kf != nullptr) {
                        validKeyframesSetRaw.insert(kf);
                    }
                }
                lk3d.unlock();
                
                int refreshed = 0;
                for(KeyFrameDisplay* fh : keyframesCopyRaw)
                {
                    if(fh == nullptr) continue;
                    if(validKeyframesSetRaw.find(fh) == validKeyframesSetRaw.end()) continue;
                    
                    try {
                        float yellow[3] = {1.0f, 1.0f, 0.0f};  // Bright yellow for raw
                        if(settings_showKFCameras) fh->drawCam(1, yellow, 0.1);
                        
                        refreshed += (int)(fh->refreshPC(refreshed < 10, settings_scaledVarTH, settings_absVarTH,
                                settings_pointCloudMode, settings_minRelBS, settings_sparsity));
                        fh->drawPC(1);
                    } catch (...) {
                        continue;
                    }
                }
                
                lk3d.lock();
                // Draw trajectory for raw path
                if(settings_showTrajectory && !allFramePosesRaw.empty())
                {
                    glColor3f(1.0f, 1.0f, 0.0f);  // Bright yellow for raw trajectory
                    glLineWidth(2.0f);
                    glBegin(GL_LINE_STRIP);
                    for(unsigned int i = 0; i < allFramePosesRaw.size(); i++)
                    {
                        if(i >= allFramePosesRaw.size()) break; // Safety check
                        glVertex3f((float)allFramePosesRaw[i][0],
                                (float)allFramePosesRaw[i][1],
                                (float)allFramePosesRaw[i][2]);
                    }
                    glEnd();
                    glLineWidth(1.0f);
                }
                
                if(settings_showCurrentCamera && currentCamRaw != nullptr) {
                    try {
                        currentCamRaw->drawCam(2, 0, 0.2);
                    } catch (...) {}
                }
                lk3d.unlock();
            }

            // Render right view (pipeline) - green color
            Visualization3D_display_pipeline.Activate(Visualization3D_camera_pipeline);
            {
                boost::unique_lock<boost::mutex> lk3d(model3DMutex);
                
                // Render pipeline keyframes with green color
                std::vector<KeyFrameDisplay*> keyframesCopyPipeline = keyframesPipeline;
                std::set<KeyFrameDisplay*> validKeyframesSetPipeline;
                for(auto& kf : keyframesPipeline) {
                    if(kf != nullptr) {
                        validKeyframesSetPipeline.insert(kf);
                    }
                }
                lk3d.unlock();
                
                int refreshed = 0;
                for(KeyFrameDisplay* fh : keyframesCopyPipeline)
                {
                    if(fh == nullptr) continue;
                    if(validKeyframesSetPipeline.find(fh) == validKeyframesSetPipeline.end()) continue;
                    
                    try {
                        float green[3] = {0, 1, 0};  // Green for pipeline
                        if(settings_showKFCameras) fh->drawCam(1, green, 0.1);
                        
                        refreshed += (int)(fh->refreshPC(refreshed < 10, settings_scaledVarTH, settings_absVarTH,
                                settings_pointCloudMode, settings_minRelBS, settings_sparsity));
                        fh->drawPC(1);
                    } catch (...) {
                        continue;
                    }
                }
                
                lk3d.lock();
                // Draw trajectory for pipeline path
                if(settings_showTrajectory && !allFramePosesPipeline.empty())
                {
                    glColor3f(0, 1, 0);  // Green for pipeline trajectory
                    glLineWidth(2.0f);
                    glBegin(GL_LINE_STRIP);
                    for(unsigned int i = 0; i < allFramePosesPipeline.size(); i++)
                    {
                        if(i >= allFramePosesPipeline.size()) break; // Safety check
                        glVertex3f((float)allFramePosesPipeline[i][0],
                                (float)allFramePosesPipeline[i][1],
                                (float)allFramePosesPipeline[i][2]);
                    }
                    glEnd();
                    glLineWidth(1.0f);
                }
                
                if(settings_showCurrentCamera && currentCamPipeline != nullptr) {
                    try {
                        currentCamPipeline->drawCam(2, 0, 0.2);
                    } catch (...) {}
                }
                lk3d.unlock();
            }
        }

        // Update and render video images
        {
            boost::unique_lock<boost::mutex> lk(openImagesMutex);
            
            if(videoImgChangedRaw && internalVideoImgRaw != nullptr) {
                try {
                    texVideoRaw.Upload(internalVideoImgRaw->data, GL_BGR, GL_UNSIGNED_BYTE);
                } catch (...) {
                    // Skip if upload fails
                }
            }
            if(videoImgChangedPipeline && internalVideoImgPipeline != nullptr) {
                try {
                    texVideoPipeline.Upload(internalVideoImgPipeline->data, GL_BGR, GL_UNSIGNED_BYTE);
                } catch (...) {
                    // Skip if upload fails
                }
            }
            videoImgChangedRaw = false;
            videoImgChangedPipeline = false;
        }
        
        // Update settings
        setting_render_displayVideo = settings_showLiveVideo.Get();
        setting_render_display3D = settings_show3D.Get();
        this->settings_showTrajectory = settings_showTrajectory.Get();
        
        // Render video images
        if(setting_render_displayVideo)
        {
            try {
                // Left: Raw image
                d_video_raw.Activate();
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                texVideoRaw.RenderToViewportFlipY();
                
                // Right: Pipeline image
                d_video_pipeline.Activate();
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                texVideoPipeline.RenderToViewportFlipY();
            } catch (...) {
                // Skip if OpenGL context error
            }
        }

        if(settings_resetButton)
        {
            settings_resetButton = false;
            reset();
        }

        pangolin::FinishFrame();

        if(needReset)
        {
            reset_internal();
        }

        usleep(30000);  // ~30 FPS
    }

    running = false;
}

void DualPangolinDSOViewer::close()
{
    running = false;
}

void DualPangolinDSOViewer::publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity)
{
    // Graph connections are handled per system in publishKeyframes
    // This can be extended if needed
}

void DualPangolinDSOViewer::publishKeyframes(std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib)
{
    if(!setting_render_display3D) return;
    if(frames.empty() || HCalib == nullptr) return;
    
    // If this is a wrapper viewer, forward to main viewer with system type info
    if(mainViewer != nullptr)
    {
        // Temporarily set main viewer's system type to match this wrapper
        bool oldType = mainViewer->isRawSystem;
        mainViewer->isRawSystem = isRawSystem;
        mainViewer->publishKeyframes(frames, final, HCalib);
        mainViewer->isRawSystem = oldType;  // Restore
        return;
    }
    
    // Otherwise, this is the main viewer - store keyframes based on system type
    boost::unique_lock<boost::mutex> lk(model3DMutex);
    
    // Store keyframes based on system type
    std::vector<KeyFrameDisplay*>* targetKeyframes = isRawSystem ? &keyframesRaw : &keyframesPipeline;
    std::map<int, KeyFrameDisplay*>* targetKeyframesByKFID = isRawSystem ? &keyframesByKFIDRaw : &keyframesByKFIDPipeline;
    
    for(FrameHessian* fh : frames)
    {
        if(fh == nullptr) continue;
        
        if(targetKeyframesByKFID->find(fh->frameID) == targetKeyframesByKFID->end())
        {
            KeyFrameDisplay* kfd = new KeyFrameDisplay();
            if(kfd == nullptr) continue;
            (*targetKeyframesByKFID)[fh->frameID] = kfd;
            targetKeyframes->push_back(kfd);
        }
        
        if(targetKeyframesByKFID->find(fh->frameID) != targetKeyframesByKFID->end() && 
           (*targetKeyframesByKFID)[fh->frameID] != nullptr)
        {
            try {
                (*targetKeyframesByKFID)[fh->frameID]->setFromKF(fh, HCalib);
            } catch (...) {
                continue;
            }
        }
    }
}

void DualPangolinDSOViewer::publishCamPose(FrameShell* frame, CalibHessian* HCalib)
{
    if(!setting_render_display3D) return;
    if(frame == nullptr || HCalib == nullptr) return;
    
    // If this is a wrapper viewer, forward to main viewer with system type info
    if(mainViewer != nullptr)
    {
        // Temporarily set main viewer's system type to match this wrapper
        bool oldType = mainViewer->isRawSystem;
        mainViewer->isRawSystem = isRawSystem;
        mainViewer->publishCamPose(frame, HCalib);
        mainViewer->isRawSystem = oldType;  // Restore
        return;
    }
    
    // Otherwise, this is the main viewer - store camera pose based on system type
    boost::unique_lock<boost::mutex> lk(model3DMutex);
    
    // Store camera pose based on system type
    KeyFrameDisplay** targetCurrentCam = isRawSystem ? &currentCamRaw : &currentCamPipeline;
    std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>>* targetAllFramePoses = isRawSystem ? &allFramePosesRaw : &allFramePosesPipeline;
    
    if(*targetCurrentCam == nullptr)
    {
        *targetCurrentCam = new KeyFrameDisplay();
    }
    
    if(*targetCurrentCam != nullptr)
    {
        try {
            (*targetCurrentCam)->setFromF(frame, HCalib);
            targetAllFramePoses->push_back(frame->camToWorld.translation().cast<float>());
        } catch (...) {
            // Skip if setFromF fails
        }
    }
}

void DualPangolinDSOViewer::pushLiveFrame(FrameHessian* image)
{
    if(image == nullptr) return;
    
    // If this is a wrapper viewer, forward to main viewer with correct system type
    if(mainViewer != nullptr)
    {
        bool oldType = mainViewer->isRawSystem;
        mainViewer->isRawSystem = isRawSystem;  // Set correct system type
        mainViewer->pushLiveFrame(image);
        mainViewer->isRawSystem = oldType;  // Restore
        return;
    }
    
    // This is the main viewer - update the appropriate image buffer based on isRawSystem
    boost::unique_lock<boost::mutex> lk(openImagesMutex);
    
    try {
        if(image->dI != nullptr) {
            MinimalImageB3* targetImg = isRawSystem ? internalVideoImgRaw : internalVideoImgPipeline;
            bool* changedFlag = isRawSystem ? &videoImgChangedRaw : &videoImgChangedPipeline;
            
            if(targetImg != nullptr) {
                for(int i = 0; i < w * h; i++) {
                    float val = image->dI[i][0] * 0.8f;
                    val = val > 255.0f ? 255.0f : val;
                    targetImg->data[i][0] = targetImg->data[i][1] = targetImg->data[i][2] = (unsigned char)val;
                }
                *changedFlag = true;
            }
        }
    } catch (...) {
        // Skip if copy fails
    }
}

void DualPangolinDSOViewer::pushDepthImage(MinimalImageB3* image)
{
    // Not implemented for dual viewer (can be added if needed)
}

bool DualPangolinDSOViewer::needPushDepthImage()
{
    return false;
}

void DualPangolinDSOViewer::join()
{
    // Nothing to join for dual viewer
}

void DualPangolinDSOViewer::exportPointCloud(std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector3f>& colors, bool isRawPath)
{
    points.clear();
    colors.clear();
    
    boost::unique_lock<boost::mutex> lock(model3DMutex);
    
    // Select which keyframes to export based on path
    std::vector<KeyFrameDisplay*>& keyframesToExport = isRawPath ? keyframesRaw : keyframesPipeline;
    
    for(KeyFrameDisplay* fh : keyframesToExport)
    {
        if(fh == nullptr || !fh->active) continue;
        
        std::vector<Eigen::Vector3f> kfPoints;
        std::vector<Eigen::Vector3f> kfColors;
        
        // Export points from this keyframe
        fh->exportPointCloud(kfPoints, kfColors, 
                             settings_scaledVarTH, settings_absVarTH, 
                             settings_pointCloudMode, settings_minRelBS, settings_sparsity);
        
        // Append to global point cloud
        points.insert(points.end(), kfPoints.begin(), kfPoints.end());
        colors.insert(colors.end(), kfColors.begin(), kfColors.end());
    }
}

void DualPangolinDSOViewer::reset()
{
    needReset = true;
}

void DualPangolinDSOViewer::reset_internal()
{
    needReset = false;
    
    boost::unique_lock<boost::mutex> lk(model3DMutex);
    
    // Reset raw system
    for(size_t i = 0; i < keyframesRaw.size(); i++) {
        if(keyframesRaw[i] != nullptr) {
            delete keyframesRaw[i];
        }
    }
    keyframesRaw.clear();
    allFramePosesRaw.clear();
    keyframesByKFIDRaw.clear();
    if(currentCamRaw != nullptr) {
        delete currentCamRaw;
        currentCamRaw = nullptr;
    }
    
    // Reset pipeline system
    for(size_t i = 0; i < keyframesPipeline.size(); i++) {
        if(keyframesPipeline[i] != nullptr) {
            delete keyframesPipeline[i];
        }
    }
    keyframesPipeline.clear();
    allFramePosesPipeline.clear();
    keyframesByKFIDPipeline.clear();
    if(currentCamPipeline != nullptr) {
        delete currentCamPipeline;
        currentCamPipeline = nullptr;
    }
}

void DualPangolinDSOViewer::setSystemType(bool isRaw)
{
    isRawSystem = isRaw;
}

void DualPangolinDSOViewer::setMainViewer(DualPangolinDSOViewer* viewer)
{
    mainViewer = viewer;
}

void DualPangolinDSOViewer::drawConstraints()
{
    // TODO: Implement constraint drawing for both views
}

}  // namespace IOWrap
}  // namespace dso


