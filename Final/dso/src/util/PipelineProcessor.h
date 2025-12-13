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
    // Constructor with DSO-optimized defaults
    // enableCLAHE: false by default (destroys grayscale consistency)
    // gradientStrength: 0.0 by default (DSO doesn't need us to "make gradients", just "don't destroy them")
    // gamma: 2.2 by default (standard gamma for linearization)
    // useFixedGain: true by default (maintains photometric consistency)
    // enableConservativeAE: false by default (very conservative auto exposure if enabled)
    // enableMildLogIntensity: false by default (experimental, disabled by default)
    PipelineProcessor(bool enableCLAHE = false, 
                     float gradientStrength = 0.0f,
                     float gamma = 2.2f,
                     bool useFixedGain = true,
                     bool enableConservativeAE = false,
                     bool enableMildLogIntensity = false);
    ~PipelineProcessor();

    // Process BGR frame, returns BGR frame
    // Processing order (Mode B - Photometric-stable):
    //   Gamma linearization → Fixed gain exposure → Grayscale → Bilateral filter
    // Processing order (Mode C - Aggressive, for ablation):
    //   Gamma linearization → Fixed gain exposure → Grayscale → (optional)Mild log → (optional)Gradient → (optional)CLAHE → Bilateral filter
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
    
    // Gamma correction
    float gamma;
    
    // Exposure compensation mode
    bool useFixedGain;
    bool enableConservativeAE;
    float fixedGainValue;
    
    // Gradient enhancement (disabled by default)
    float gradientStrength;
    
    // Mild log intensity (experimental, disabled by default)
    bool enableMildLogIntensity;
    
    // CLAHE (disabled by default)
    bool claheEnabled;
    cv::Ptr<cv::CLAHE> clahe;
    
    // Thread safety
    std::mutex processMutex;

    // Internal methods
    cv::Mat linearizeImage(const cv::Mat& frame_bgr);  // Gamma correction/linearization
    cv::Mat applyExposureCompensation(const cv::Mat& frame_bgr);
    cv::Mat applyBilateralFilter(const cv::Mat& gray);  // Edge-preserving denoising
    cv::Mat applyMildLogIntensity(const cv::Mat& gray);  // Experimental illumination normalization
    cv::Mat enhanceGradients(const cv::Mat& gray, float strength);
    cv::Mat toGrayscaleBT709(const cv::Mat& frame_bgr);
    float computeMeanLuma(const cv::Mat& gray);
    float computeFixedGain(const cv::Mat& gray);  // Compute fixed gain from initial frames
    float detectMotion(const cv::Mat& gray);
    bool detectSceneChange(float currentIntensity, float motionScore);
    float computeTargetGain(float currentIntensity, float saturationRatio, float informationContent);
    float smoothGainTransition(float targetGain, float motionScore, bool sceneChanged);
};


