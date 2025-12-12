/**
* Pipeline Processor Implementation
*/

#include "PipelineProcessor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>
#include <vector>
#include <deque>

PipelineProcessor::PipelineProcessor(bool enableCLAHE, float gradientStrength)
    : claheEnabled(enableCLAHE)
    , gradientStrength(std::max(0.0f, std::min(0.3f, gradientStrength)))
    , refMeanLuma(-1.0f)
    , currentGain(1.0f)
    , targetGain(1.0f)
    , exposureSmoothAlpha(0.05f)  // Conservative adaptation rate
    , minGain(0.8f)
    , maxGain(1.2f)
    , historyLength(15)
    , frameCount(0)
    , sceneStableFrames(0)
    , motionFreezeThreshold(0.3f)
{
    // Initialize CLAHE if enabled
    if(claheEnabled)
    {
        clahe = cv::createCLAHE(2.0, cv::Size(8, 8));  // Clip limit 2.0, tile size 8x8
    }
}

PipelineProcessor::~PipelineProcessor()
{
    // OpenCV smart pointers handle cleanup automatically
}

cv::Mat PipelineProcessor::processFrame(const cv::Mat& frame_bgr)
{
    std::lock_guard<std::mutex> lock(processMutex);
    
    if(frame_bgr.empty())
    {
        return frame_bgr;
    }

    // 1. Exposure Compensation
    cv::Mat exposed = applyExposureCompensation(frame_bgr);

    // 2. Convert to grayscale (for subsequent processing)
    cv::Mat gray = toGrayscaleBT709(exposed);

    // 3. Gradient Enhancement (fixed step)
    cv::Mat enhanced = enhanceGradients(gray, gradientStrength);

    // 4. CLAHE (optional)
    if(claheEnabled && clahe)
    {
        clahe->apply(enhanced, enhanced);
    }

    // 5. Light denoising
    cv::Mat denoised;
    cv::GaussianBlur(enhanced, denoised, cv::Size(3, 3), 0);

    // 6. Convert back to BGR (for consistency, though we could return grayscale)
    cv::Mat result;
    cv::cvtColor(denoised, result, cv::COLOR_GRAY2BGR);

    frameCount++;
    return result;
}

cv::Mat PipelineProcessor::applyExposureCompensation(const cv::Mat& frame_bgr)
{
    cv::Mat gray = toGrayscaleBT709(frame_bgr);
    float meanLuma = computeMeanLuma(gray);

    // Initialize reference on first frame
    if(refMeanLuma < 0.0f)
    {
        refMeanLuma = meanLuma;
        intensityHistory.push_back(meanLuma);
        return frame_bgr;
    }

    // Update history
    intensityHistory.push_back(meanLuma);
    if(intensityHistory.size() > historyLength)
    {
        intensityHistory.pop_front();
    }

    // Compute saturation ratio
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    float saturationRatio = 0.0f;
    int saturatedPixels = cv::countNonZero((gray <= 5) | (gray >= 250));
    saturationRatio = (float)saturatedPixels / (gray.rows * gray.cols);

    // Compute information content (entropy)
    cv::Mat hist;
    int histSize = 64;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    hist /= (gray.rows * gray.cols);  // Normalize
    float entropy = 0.0f;
    for(int i = 0; i < histSize; i++)
    {
        float val = hist.at<float>(i);
        if(val > 0.0f)
        {
            entropy -= val * log2f(val + 1e-10f);
        }
    }
    float informationContent = std::min(1.0f, entropy / 6.0f);  // Normalize to [0,1]

    // Detect motion
    float motionScore = detectMotion(gray);

    // Detect scene change
    bool sceneChanged = detectSceneChange(meanLuma, motionScore);

    if(sceneChanged)
    {
        // Reset for new scene
        intensityHistory.clear();
        intensityHistory.push_back(meanLuma);
        gainHistory.clear();
        sceneStableFrames = 0;
        histHistory.clear();
    }
    else
    {
        sceneStableFrames++;
    }

    // Compute target gain
    targetGain = computeTargetGain(meanLuma, saturationRatio, informationContent);

    // Smooth gain transition
    float gain = smoothGainTransition(targetGain, motionScore, sceneChanged);

    // Update state
    gainHistory.push_back(gain);
    if(gainHistory.size() > 3)
    {
        gainHistory.pop_front();
    }
    currentGain = gain;

    // Early exit if no correction needed
    if(std::abs(gain - 1.0f) < 0.02f)
    {
        return frame_bgr;
    }

    // Apply global gain
    cv::Mat result;
    frame_bgr.convertTo(result, CV_32F, gain);
    result = cv::max(0.0, cv::min(255.0, result));
    result.convertTo(result, CV_8U);

    return result;
}

cv::Mat PipelineProcessor::enhanceGradients(const cv::Mat& gray, float strength)
{
    cv::Mat blurred, sharpened;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    cv::addWeighted(gray, 1.0f + strength, blurred, -strength, 0, sharpened);
    return sharpened;
}

cv::Mat PipelineProcessor::toGrayscaleBT709(const cv::Mat& frame_bgr)
{
    cv::Mat gray;
    if(frame_bgr.channels() == 3)
    {
        std::vector<cv::Mat> channels;
        cv::split(frame_bgr, channels);
        // BT.709: Y = 0.2126*R + 0.7152*G + 0.0722*B
        gray = 0.2126f * channels[2] + 0.7152f * channels[1] + 0.0722f * channels[0];
        gray.convertTo(gray, CV_8U);
    }
    else
    {
        gray = frame_bgr.clone();
    }
    return gray;
}

float PipelineProcessor::computeMeanLuma(const cv::Mat& gray)
{
    // Use robust statistics: exclude extreme values
    cv::Mat mask = (gray > 10) & (gray < 245);
    cv::Scalar mean = cv::mean(gray, mask);
    return (float)mean[0];
}

float PipelineProcessor::detectMotion(const cv::Mat& gray)
{
    // Compute histogram
    cv::Mat hist;
    int histSize = 32;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // Normalize
    hist /= (gray.rows * gray.cols);

    if(histHistory.empty())
    {
        histHistory.push_back(hist.clone());
        return 0.0f;
    }

    // Compare with previous histogram (Chi-squared distance)
    cv::Mat prevHist = histHistory.back();
    float chi2 = 0.0f;
    for(int i = 0; i < histSize; i++)
    {
        float h1 = hist.at<float>(i);
        float h2 = prevHist.at<float>(i);
        float num = (h1 - h2) * (h1 - h2);
        float den = h1 + h2 + 1e-10f;
        chi2 += num / den;
    }

    histHistory.push_back(hist.clone());
    if(histHistory.size() > 3)
    {
        histHistory.pop_front();
    }

    float motionScore = std::min(1.0f, chi2 * 10.0f);
    return motionScore;
}

bool PipelineProcessor::detectSceneChange(float currentIntensity, float motionScore)
{
    if(intensityHistory.size() < 5)
    {
        return false;
    }

    // Don't detect scene changes during motion
    if(motionScore > 0.3f)
    {
        return false;
    }

    // Compute median and MAD
    std::vector<float> hist(intensityHistory.begin(), intensityHistory.end());
    std::sort(hist.begin(), hist.end());
    float median = hist[hist.size() / 2];
    
    std::vector<float> deviations;
    for(float val : hist)
    {
        deviations.push_back(std::abs(val - median));
    }
    std::sort(deviations.begin(), deviations.end());
    float mad = deviations[deviations.size() / 2];

    float threshold = 5.0f * (mad + 5.0f);
    float deviation = std::abs(currentIntensity - median);

    return deviation > threshold;
}

float PipelineProcessor::computeTargetGain(float currentIntensity, float saturationRatio, float informationContent)
{
    if(currentIntensity < 1e-3f)
    {
        return maxGain;
    }

    float target = 128.0f / currentIntensity;  // Target intensity 128

    // Penalize if too much saturation
    if(saturationRatio > 0.05f)
    {
        float penalty = std::pow(0.7f, (saturationRatio - 0.05f) * 10.0f);
        target = std::min(target, currentGain * penalty);
    }

    // Encourage information content (gently)
    if(informationContent < 0.70f)
    {
        float entropyBoost = 1.0f + 0.1f * (0.70f - informationContent);
        target *= entropyBoost;
    }

    // Clamp to safe range
    return std::max(minGain, std::min(maxGain, target));
}

float PipelineProcessor::smoothGainTransition(float targetGain, float motionScore, bool sceneChanged)
{
    // Freeze during very fast motion
    if(motionScore > motionFreezeThreshold)
    {
        return currentGain;
    }

    // Choose adaptation rate
    float alpha;
    if(sceneChanged)
    {
        alpha = 0.25f;  // Fast adaptation for scene changes
    }
    else
    {
        // Slower during motion
        float motionPenalty = 0.5f * std::min(1.0f, motionScore);
        alpha = exposureSmoothAlpha * (1.0f - motionPenalty);
        alpha = std::max(0.03f, std::min(exposureSmoothAlpha, alpha));
    }

    // Exponential moving average
    float smoothGain = alpha * targetGain + (1.0f - alpha) * currentGain;

    // Limit acceleration (jerk control)
    if(gainHistory.size() >= 2)
    {
        float prevVelocity = gainHistory.back() - gainHistory[gainHistory.size() - 2];
        float newVelocity = smoothGain - currentGain;
        float acceleration = newVelocity - prevVelocity;

        if(std::abs(acceleration) > 0.05f)
        {
            // Limit acceleration
            float limitedVelocity = prevVelocity + std::copysign(0.05f, acceleration);
            smoothGain = currentGain + limitedVelocity;
        }
    }

    // Additional temporal smoothing
    float finalGain = 0.15f * smoothGain + 0.85f * currentGain;

    return std::max(minGain, std::min(maxGain, finalGain));
}

void PipelineProcessor::setCLAHEEnabled(bool enabled)
{
    std::lock_guard<std::mutex> lock(processMutex);
    claheEnabled = enabled;
    if(enabled && !clahe)
    {
        clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    }
}

bool PipelineProcessor::isCLAHEEnabled() const
{
    return claheEnabled;
}

void PipelineProcessor::setGradientStrength(float strength)
{
    std::lock_guard<std::mutex> lock(processMutex);
    gradientStrength = std::max(0.0f, std::min(0.3f, strength));
}

void PipelineProcessor::reset()
{
    std::lock_guard<std::mutex> lock(processMutex);
    refMeanLuma = -1.0f;
    currentGain = 1.0f;
    targetGain = 1.0f;
    intensityHistory.clear();
    gainHistory.clear();
    histHistory.clear();
    frameCount = 0;
    sceneStableFrames = 0;
}

