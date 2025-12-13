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

PipelineProcessor::PipelineProcessor(bool enableCLAHE, 
                                     float gradientStrength,
                                     float gamma,
                                     bool useFixedGain,
                                     bool enableConservativeAE,
                                     bool enableMildLogIntensity)
    : claheEnabled(enableCLAHE)
    , gradientStrength(std::max(0.0f, std::min(0.3f, gradientStrength)))
    , gamma(gamma)
    , useFixedGain(useFixedGain)
    , enableConservativeAE(enableConservativeAE)
    , fixedGainValue(-1.0f)  // Will be computed from first frame(s)
    , enableMildLogIntensity(enableMildLogIntensity)
    , refMeanLuma(-1.0f)
    , currentGain(1.0f)
    , targetGain(1.0f)
    , exposureSmoothAlpha(enableConservativeAE ? 0.005f : 0.05f)  // Very conservative if enabled
    , minGain(enableConservativeAE ? 0.95f : 0.8f)  // Narrower range for conservative AE
    , maxGain(enableConservativeAE ? 1.05f : 1.2f)
    , historyLength(15)
    , frameCount(0)
    , sceneStableFrames(0)
    , motionFreezeThreshold(0.3f)  // float, not int
{
    // Initialize CLAHE if enabled
    if(claheEnabled)
    {
        clahe = cv::createCLAHE(2.0, cv::Size(8, 8));  // Clip limit 2.0, tile size 8x8
    }
    
    printf("PipelineProcessor initialized:\n");
    printf("  CLAHE: %s\n", claheEnabled ? "ENABLED" : "DISABLED");
    printf("  Gradient Strength: %.2f\n", gradientStrength);
    printf("  Gamma: %.2f\n", gamma);
    printf("  Fixed Gain Mode: %s\n", useFixedGain ? "ENABLED" : "DISABLED");
    printf("  Conservative AE: %s\n", enableConservativeAE ? "ENABLED" : "DISABLED");
    printf("  Mild Log Intensity: %s\n", enableMildLogIntensity ? "ENABLED (experimental)" : "DISABLED");
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

    // Mode B (Photometric-stable) processing order:
    // 1. Gamma correction/linearization
    cv::Mat linearized = linearizeImage(frame_bgr);
    
    // 2. Fixed gain exposure compensation
    cv::Mat exposed = applyExposureCompensation(linearized);
    
    // 3. Convert to grayscale (BT.709)
    cv::Mat gray = toGrayscaleBT709(exposed);
    
    // 4. Optional: Mild log intensity (experimental, Mode C only)
    cv::Mat processed = gray;
    if(enableMildLogIntensity)
    {
        processed = applyMildLogIntensity(gray);
    }
    else
    {
        processed = gray;
    }
    
    // 5. Optional: Gradient enhancement (Mode C only, disabled by default)
    cv::Mat enhanced = processed;
    if(gradientStrength > 0.0f)
    {
        enhanced = enhanceGradients(processed, gradientStrength);
    }
    
    // 6. Optional: CLAHE (Mode C only, disabled by default)
    if(claheEnabled && clahe)
    {
        clahe->apply(enhanced, enhanced);
    }
    
    // 7. Bilateral filter (edge-preserving denoising, replaces Gaussian blur)
    cv::Mat denoised = applyBilateralFilter(enhanced);
    
    // Return grayscale directly (no need to convert back to BGR since CameraReader will use grayscale anyway)
    frameCount++;
    return denoised;
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

    // Fixed gain mode: compute gain from first frame(s) and use it for all frames
    if(useFixedGain)
    {
        if(fixedGainValue < 0.0f)
        {
            // Compute fixed gain from first frame or first N frames
            fixedGainValue = computeFixedGain(gray);
            printf("Fixed gain computed: %.4f (will be used for all frames)\n", fixedGainValue);
        }
        
        // Use fixed gain for all frames
        gain = fixedGainValue;
        
        // Early exit if gain is close to 1.0
        if(std::abs(gain - 1.0f) < 0.02f)
        {
            return frame_bgr;
        }
    }
    else if(enableConservativeAE)
    {
        // Very conservative AE mode: gain range 0.95-1.05, very slow adaptation
        // Gain is already computed above with conservative limits
        // Early exit if no correction needed
        if(std::abs(gain - 1.0f) < 0.02f)
        {
            return frame_bgr;
        }
    }
    else
    {
        // Dynamic AE mode (original behavior, not recommended for DSO)
        // Early exit if no correction needed
        if(std::abs(gain - 1.0f) < 0.02f)
        {
            return frame_bgr;
        }
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
    // Only enhance if strength > 0 (disabled by default for DSO)
    if(strength <= 0.0f)
    {
        return gray.clone();
    }
    
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
    fixedGainValue = -1.0f;  // Reset fixed gain to recompute
    intensityHistory.clear();
    gainHistory.clear();
    histHistory.clear();
    frameCount = 0;
    sceneStableFrames = 0;
}

// Gamma correction/linearization: convert gamma-encoded image to linear space
// If camera response calibration (pcalib) is available, it replaces the fixed gamma model.
cv::Mat PipelineProcessor::linearizeImage(const cv::Mat& frame_bgr)
{
    if(frame_bgr.empty())
    {
        return frame_bgr;
    }
    
    // Note: If pcalib is available, this step should be skipped or the pcalib
    // response function should be used instead. The pcalib is typically applied
    // in the photometric undistorter (Undistort class), so we use fixed gamma here
    // as a fallback when pcalib is not available.
    
    cv::Mat result;
    frame_bgr.convertTo(result, CV_32F, 1.0 / 255.0);  // Normalize to [0, 1]
    
    // Apply gamma correction: I_linear = I_raw^gamma
    cv::pow(result, gamma, result);
    
    // Convert back to [0, 255]
    result.convertTo(result, CV_8U, 255.0);
    
    return result;
}

// Bilateral filter: edge-preserving denoising (replaces Gaussian blur)
cv::Mat PipelineProcessor::applyBilateralFilter(const cv::Mat& gray)
{
    if(gray.empty())
    {
        return gray;
    }
    
    cv::Mat result;
    // Parameters: d=5 (diameter), sigmaColor=20, sigmaSpace=20
    // This preserves edges while reducing noise
    cv::bilateralFilter(gray, result, 5, 20, 20);
    
    return result;
}

// Mild log intensity: experimental illumination normalization
// WARNING: This is experimental and disabled by default. It may break brightness constancy.
cv::Mat PipelineProcessor::applyMildLogIntensity(const cv::Mat& gray)
{
    if(gray.empty())
    {
        return gray;
    }
    
    // Very mild log transformation to reduce extreme illumination variations
    // I_log = log(1 + I) / log(256) * 255
    cv::Mat result;
    gray.convertTo(result, CV_32F);
    cv::log(result + 1.0, result);
    result = result / log(256.0) * 255.0;
    result.convertTo(result, CV_8U);
    
    return result;
}

// Compute fixed gain from initial frame(s)
float PipelineProcessor::computeFixedGain(const cv::Mat& gray)
{
    if(gray.empty())
    {
        return 1.0f;
    }
    
    // Compute mean luma of the frame
    float meanLuma = computeMeanLuma(gray);
    
    // Target intensity: 128 (middle of [0, 255])
    float targetIntensity = 128.0f;
    
    // Compute gain to reach target intensity
    if(meanLuma < 1e-3f)
    {
        return maxGain;  // Avoid division by zero
    }
    
    float gain = targetIntensity / meanLuma;
    
    // Clamp to safe range
    gain = std::max(minGain, std::min(maxGain, gain));
    
    return gain;
}


