/**
* Camera Reader for DSO
* Reads images from a connected camera device
*/

#pragma once
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "util/Undistort.h"
#include "util/PipelineProcessor.h"
#include "IOWrapper/ImageRW.h"

#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>
#include <mutex>
#include <unistd.h>

using namespace dso;

class CameraReader
{
public:
	// Constructor for camera input
	CameraReader(int cameraIndex, std::string calibFile, std::string gammaFile, std::string vignetteFile, bool enableDualMode = false, bool enableCLAHE = false, bool configureCamera = true, double fixedExposureUs = -1);
	
	// Constructor for video file input
	CameraReader(std::string videoFile, std::string calibFile, std::string gammaFile, std::string vignetteFile, bool enableDualMode = false, bool enableCLAHE = false, bool configureCamera = true, double fixedExposureUs = -1);
	
	~CameraReader();

	Eigen::VectorXf getOriginalCalib()
	{
		if (undistort == nullptr) return Eigen::VectorXf(); // Return empty or default if null
		return undistort->getOriginalParameter().cast<float>();
	}

	Eigen::Vector2i getOriginalDimensions()
	{
		if (undistort == nullptr) return Eigen::Vector2i(0,0); // Return default if null
		return undistort->getOriginalSize();
	}

	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		if (undistort == nullptr) {
			K.setZero(); w = 0; h = 0; return;
		}
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0];
		h = undistort->getSize()[1];
	}

	void setGlobalCalibration()
	{
		if (undistort == nullptr) return;
		int w_out, h_out;
		Eigen::Matrix3f K;
		getCalibMono(K, w_out, h_out);
		setGlobalCalib(w_out, h_out, K);
	}

	int getNumImages()
	{
		// For camera/video, return frame count or large number for continuous stream
		if(isVideoFile)
		{
			std::lock_guard<std::mutex> lock(captureMutex);
			return (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
		}
		return 1000000;  // Continuous stream for camera
	}

	double getTimestamp(int id)
	{
		// Return timestamp based on frame count
		return id * 0.033; // Assuming ~30 FPS
	}

	ImageAndExposure* getImage(int id, bool forceLoadDirectly=false)
	{
		return getImage_internal(0);
	}

	// Get raw image (no undistortion, no vignette, only photometric calibration)
	ImageAndExposure* getImageRaw(int id);

	// Get pipeline processed image (full processing: pipeline + undistortion + photometric calibration)
	ImageAndExposure* getImagePipeline(int id);
	
	// Get original BGR frame from camera (for video recording)
	// Returns empty Mat if not available
	cv::Mat getOriginalBGRFrame();
	
	// Get camera FPS (for video recording)
	double getFPS() const { return fps; }
	
	// Get camera resolution (for video recording)
	cv::Size getResolution() const { return cv::Size(widthOrg, heightOrg); }
	
	// Check if reading from video file (public access for main)
	bool isVideoFile;

	inline float* getPhotometricGamma()
	{
		if(undistort==0 || undistort->photometricUndist==0) return 0;
		return undistort->photometricUndist->getG();
	}

	// Camera hardware control methods
	bool configureCameraForDSO();  // Configure camera for DSO (returns success status)
	void setExposureTime(double exposureUs);  // Set fixed exposure time (microseconds)
	void setGain(double gain);  // Set fixed gain/ISO
	bool verifyCameraSettings();  // Verify hardware control settings

	Undistort* undistort;
	Undistort* undistort_pipeline;  // For pipeline path (with undistortion)
	PipelineProcessor* pipelineProcessor;
	bool enableDualMode;
	int width, height;
	int widthOrg, heightOrg;
	int frameCount;
	bool running;
	bool hardwareControlEnabled;  // Whether hardware control succeeded
	double fps;  // FPS for video recording

private:
	int cameraIndex;
	std::string videoFile;  // Video file path (if using video input)
	std::string calibfile;
	std::string gammaFile;
	std::string vignetteFile;
	cv::VideoCapture capture;
	std::mutex captureMutex;  // Mutex for thread-safe camera/video access
	cv::Mat lastCapturedFrame;  // Store last captured BGR frame for video recording
	std::mutex lastFrameMutex;  // Mutex for lastCapturedFrame access

	// Helper to capture frame from camera/video
	cv::Mat captureFrame();

	ImageAndExposure* getImage_internal(int unused);
	
	// Common initialization code
	void initializeCalibration(int camWidth, int camHeight, bool enableDualMode, bool enableCLAHE);
};

