/**
* Pipeline Processor for DSO-SLAM
* Implements exposure compensation, gradient enhancement, optional CLAHE, and denoising
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <deque>
#include <mutex>
#include <vector>
#include <cmath>

class PipelineProcessor
{
public:
    PipelineProcessor(bool enableCLAHE = false, float gradientStrength = 0.15f);
    ~PipelineProcessor();

    // Process BGR frame, returns BGR frame
    // Processing order: Exposure Compensation → Gradient Enhancement → (optional)CLAHE → Denoise
    cv::Mat processFrame(const cv::Mat& frame_bgr);

    // Runtime control for CLAHE
    void setCLAHEEnabled(bool enabled);
    bool isCLAHEEnabled() const;

    // Runtime adjustment of gradient enhancement strength
    void setGradientStrength(float strength);  // Range: 0.0-0.3

    // Reset state (for scene changes)
    void reset();

private:
    // Exposure compensation state
    float refMeanLuma;
    float currentGain;
    float targetGain;
    float exposureSmoothAlpha;
    float minGain;
    float maxGain;
    std::deque<float> intensityHistory;
    std::deque<float> gainHistory;
    int historyLength;
    int frameCount;
    int sceneStableFrames;
    
    // Motion detection
    std::deque<cv::Mat> histHistory;
    int motionFreezeThreshold;
    
    // Gradient enhancement
    float gradientStrength;
    
    // CLAHE
    bool claheEnabled;
    cv::Ptr<cv::CLAHE> clahe;
    
    // Thread safety
    std::mutex processMutex;

    // Internal methods
    cv::Mat applyExposureCompensation(const cv::Mat& frame_bgr);
    cv::Mat enhanceGradients(const cv::Mat& gray, float strength);
    cv::Mat toGrayscaleBT709(const cv::Mat& frame_bgr);
    float computeMeanLuma(const cv::Mat& gray);
    float detectMotion(const cv::Mat& gray);
    bool detectSceneChange(float currentIntensity, float motionScore);
    float computeTargetGain(float currentIntensity, float saturationRatio, float informationContent);
    float smoothGainTransition(float targetGain, float motionScore, bool sceneChanged);
};

