/**
* Camera Reader Implementation
*/

#include "CameraReader.h"
#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"
#include <iostream>
#include <fstream>
#include <algorithm>

// Constructor implementation for camera input
CameraReader::CameraReader(int cameraIndex, std::string calibFile, std::string gammaFile, std::string vignetteFile, bool enableDualMode, bool enableCLAHE, bool configureCamera, double fixedExposureUs)
{
	this->cameraIndex = cameraIndex;
	this->videoFile = "";
	this->isVideoFile = false;
	this->calibfile = calibFile;
	this->gammaFile = gammaFile;
	this->vignetteFile = vignetteFile;
	this->enableDualMode = enableDualMode;
	this->hardwareControlEnabled = false;
	
	// Initialize camera
	capture.open(cameraIndex);
	if(!capture.isOpened())
	{
		printf("ERROR: Cannot open camera with index %d!\n", cameraIndex);
		exit(1);
	}

	capture.set(cv::CAP_PROP_FPS, 30);
	
	// Get camera FPS
	fps = capture.get(cv::CAP_PROP_FPS);
	if(fps <= 0 || fps > 120) {
		fps = 30.0;  // Default to 30 FPS if invalid
	}
	printf("Camera FPS: %.2f\n", fps);

	// Get actual camera resolution
	int camWidth = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
	int camHeight = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	printf("Camera opened: resolution %d x %d\n", camWidth, camHeight);
	
	// Configure camera for DSO if requested
	if(configureCamera)
	{
		hardwareControlEnabled = configureCameraForDSO();
		if(fixedExposureUs > 0)
		{
			setExposureTime(fixedExposureUs);
		}
	}
	
	// Initialize calibration
	initializeCalibration(camWidth, camHeight, enableDualMode, enableCLAHE);
	
	frameCount = 0;
	running = true;
}

// Constructor for video file input
CameraReader::CameraReader(std::string videoFile, std::string calibFile, std::string gammaFile, std::string vignetteFile, bool enableDualMode, bool enableCLAHE, bool configureCamera, double fixedExposureUs)
{
	this->cameraIndex = -1;
	this->videoFile = videoFile;
	this->isVideoFile = true;
	this->calibfile = calibFile;
	this->gammaFile = gammaFile;
	this->vignetteFile = vignetteFile;
	this->enableDualMode = enableDualMode;
	this->hardwareControlEnabled = false;  // Video files cannot control hardware
	
	// Initialize video file
	capture.open(videoFile);
	if(!capture.isOpened())
	{
		printf("ERROR: Cannot open video file: %s!\n", videoFile.c_str());
		exit(1);
	}

	// Get video properties
	int camWidth = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
	int camHeight = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	this->fps = capture.get(cv::CAP_PROP_FPS);
	if(this->fps <= 0 || this->fps > 120) {
		this->fps = 30.0;  // Default to 30 FPS if invalid
	}
	int totalFrames = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
	printf("Video file opened: %s\n", videoFile.c_str());
	printf("Resolution: %d x %d, FPS: %.2f, Total frames: %d\n", camWidth, camHeight, this->fps, totalFrames);
	
	// Initialize calibration
	initializeCalibration(camWidth, camHeight, enableDualMode, enableCLAHE);
	
	frameCount = 0;
	running = true;
}

// Common initialization code for calibration
void CameraReader::initializeCalibration(int camWidth, int camHeight, bool enableDualMode, bool enableCLAHE)
{
	// Calibration parameters in relative format (as per DSO documentation)
	// Original calibration values (absolute pixels for 640x480):
	// fx=657.7274, fy=658.6482, cx=339.5179, cy=239.1852
	// Convert to relative format (divide by image dimensions):
	float fx_rel = 657.7274f / 640.0f;  // 1.0277
	float fy_rel = 658.6482f / 480.0f;  // 1.3722
	float cx_rel = 339.5179f / 640.0f;  // 0.5305
	float cy_rel = 239.1852f / 480.0f;  // 0.4983
	
	printf("Using relative calibration format: fx=%.6f, fy=%.6f, cx=%.6f, cy=%.6f (for %dx%d)\n", 
	       fx_rel, fy_rel, cx_rel, cy_rel, camWidth, camHeight);

	// For raw path: create Undistort that only does photometric calibration (no geometric undistortion)
	// We'll create a minimal calibration file or use passthrough mode
	if(enableDualMode)
	{
		// Create pipeline processor
		// Create pipeline processor with optimized defaults for DSO
		// CLAHE disabled by default, gradient strength = 0.0 (disabled)
		pipelineProcessor = new PipelineProcessor(enableCLAHE, 0.0f);
		
		// For raw path: create Undistort that only does photometric calibration (no geometric correction)
		// We need to create a passthrough calibration file
		std::string rawCalibFile = calibfile + "_raw_passthrough";
		std::ofstream rawCalibOut(rawCalibFile);
		if(rawCalibOut.is_open())
		{
		// Create passthrough calibration (no distortion)
		// Format: RadTan fx fy cx cy k1 k2 r1 r2
		//         input_width input_height
		//         rectification_mode ("none" = no rectification)
		//         output_width output_height
		// For raw path, use absolute pixel values directly (no rectification)
		// Convert relative values back to absolute pixels for the actual camera resolution
		rawCalibOut << "RadTan " << (fx_rel * camWidth) << " " << (fy_rel * camHeight) << " "
		            << (cx_rel * camWidth) << " " << (cy_rel * camHeight) << " 0 0 0 0\n";
		rawCalibOut << camWidth << " " << camHeight << "\n";
		rawCalibOut << "none\n";  // No rectification for raw path
		rawCalibOut << camWidth << " " << camHeight << "\n";
			rawCalibOut.close();
		}
		
		undistort = Undistort::getUndistorterForFile(rawCalibFile, gammaFile, "");  // No vignette for raw
		if(undistort == nullptr)
		{
			printf("ERROR: Failed to create raw undistorter!\n");
			printf("Check calibration file: %s\n", rawCalibFile.c_str());
			exit(1);
		}
		printf("Raw undistorter created successfully: %dx%d -> %dx%d\n",
		       undistort->getOriginalSize()[0], undistort->getOriginalSize()[1],
		       undistort->getSize()[0], undistort->getSize()[1]);
		// Load photometric calibration but skip vignette
		undistort->loadPhotometricCalibration(gammaFile, "", "");
		
		// For pipeline path: create full Undistort with geometric correction and photometric calibration
		// Create a temporary camera.txt file with specified parameters
		std::string pipelineCalibFile = calibfile + "_pipeline";
		std::ofstream calibOut(pipelineCalibFile);
		if(calibOut.is_open())
		{
		// Write RadTan model with specified parameters
		// Format: RadTan fx fy cx cy k1 k2 r1 r2
		//         input_width input_height
		//         rectification_mode ("none" = no rectification, or "crop"/"full" for auto rectification)
		//         output_width output_height
		// For pipeline path, use absolute pixel values directly (no rectification)
		// Convert relative values back to absolute pixels for the actual camera resolution
		calibOut << "RadTan " << (fx_rel * camWidth) << " " << (fy_rel * camHeight) << " "
		         << (cx_rel * camWidth) << " " << (cy_rel * camHeight)
		         << " -0.19778828 -0.12460651 -0.00059336 0.00270068\n";
		calibOut << camWidth << " " << camHeight << "\n";
		calibOut << "none\n";  // No rectification - use original distortion model
		calibOut << camWidth << " " << camHeight << "\n";
			calibOut.close();
		}
		
		undistort_pipeline = Undistort::getUndistorterForFile(pipelineCalibFile, gammaFile, vignetteFile);
		if(undistort_pipeline == nullptr)
		{
			printf("ERROR: Failed to create pipeline undistorter!\n");
			printf("Check calibration file: %s\n", pipelineCalibFile.c_str());
			exit(1);
		}
		printf("Pipeline undistorter created successfully: %dx%d -> %dx%d\n",
		       undistort_pipeline->getOriginalSize()[0], undistort_pipeline->getOriginalSize()[1],
		       undistort_pipeline->getSize()[0], undistort_pipeline->getSize()[1]);
		
		widthOrg = undistort_pipeline->getOriginalSize()[0];
		heightOrg = undistort_pipeline->getOriginalSize()[1];
		width = undistort_pipeline->getSize()[0];
		height = undistort_pipeline->getSize()[1];
		
		// Also set dimensions for raw path (use same dimensions)
		if(undistort != nullptr)
		{
			// Raw path uses same dimensions as pipeline for consistency
		}
	}
	else
	{
		// Single mode: use original behavior
		pipelineProcessor = nullptr;
		undistort_pipeline = nullptr;
		undistort = Undistort::getUndistorterForFile(calibfile, gammaFile, vignetteFile);
		
		if(undistort == nullptr)
		{
			printf("ERROR: Failed to create undistorter!\n");
			exit(1);
		}
		
		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		width = undistort->getSize()[0];
		height = undistort->getSize()[1];
	}

	printf("CameraReader initialized with calibration from %s\n", calibfile.c_str());
	if(enableDualMode)
	{
		printf("Dual mode enabled: Raw path (photometric only), Pipeline path (full processing)\n");
	}
	printf("Expected resolution: %d x %d\n", widthOrg, heightOrg);

	// Check if camera/video resolution matches calibration
	if(camWidth != widthOrg || camHeight != heightOrg)
	{
		printf("WARNING: Video/Camera resolution (%d x %d) doesn't match calibration (%d x %d)\n", 
			camWidth, camHeight, widthOrg, heightOrg);
		printf("Images will be resized to match calibration.\n");
	}
}

// Original getImage_internal implementation (for backward compatibility)
ImageAndExposure* CameraReader::getImage_internal(int unused)
{
	if(!running || !capture.isOpened())
	{
		return nullptr;
	}

	cv::Mat frame = captureFrame();
	if(frame.empty())
	{
		printf("WARNING: Failed to read frame from camera after retries!\n");
		return nullptr;
	}

	// Convert to grayscale if needed
	cv::Mat gray;
	try {
		if(frame.channels() == 3)
		{
			cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		}
		else
		{
			gray = frame;
		}

		// Create MinimalImageB from OpenCV Mat
		MinimalImageB* minimg = new MinimalImageB(gray.cols, gray.rows);
		if(minimg == nullptr)
		{
			printf("ERROR: Failed to allocate MinimalImageB!\n");
			return nullptr;
		}
		memcpy(minimg->data, gray.data, gray.rows * gray.cols);

		// Undistort the image
		if(undistort == nullptr)
		{
			printf("ERROR: Undistort is null!\n");
			delete minimg;
			return nullptr;
		}

		ImageAndExposure* ret = undistort->undistort<unsigned char>(
			minimg,
			1.0f,  // exposure
			frameCount * 0.033  // timestamp (assuming ~30 FPS)
		);

		delete minimg;
		frameCount++;

		return ret;
	}
	catch(const std::exception& e)
	{
		printf("ERROR: Exception in getImage_internal: %s\n", e.what());
		return nullptr;
	}
	catch(...)
	{
		printf("ERROR: Unknown exception in getImage_internal!\n");
		return nullptr;
	}
}

// Destructor
cv::Mat CameraReader::getOriginalBGRFrame()
{
	std::lock_guard<std::mutex> lock(lastFrameMutex);
	if(lastCapturedFrame.empty())
	{
		return cv::Mat();
	}
	return lastCapturedFrame.clone();  // Return a copy
}

CameraReader::~CameraReader()
{
	running = false;
	{
		std::lock_guard<std::mutex> lock(captureMutex);
		if(capture.isOpened())
		{
			capture.release();
		}
	}
	if(undistort != nullptr)
	{
		delete undistort;
		undistort = nullptr;
	}
	if(undistort_pipeline != nullptr)
	{
		delete undistort_pipeline;
		undistort_pipeline = nullptr;
	}
	if(pipelineProcessor != nullptr)
	{
		delete pipelineProcessor;
		pipelineProcessor = nullptr;
	}
}

// Helper to capture frame
cv::Mat CameraReader::captureFrame()
{
	cv::Mat frame;
	bool success = false;
	
	{
		std::lock_guard<std::mutex> lock(captureMutex);
		
		if(!capture.isOpened())
		{
			return cv::Mat();
		}
		
		int retries = 3;
		while(retries > 0 && !success)
		{
			if(capture.read(frame))
			{
				success = true;
			}
			else
			{
				// For video files, if we can't read, we've reached the end
				if(isVideoFile)
				{
					printf("Reached end of video file.\n");
					return cv::Mat();
				}
				retries--;
				if(retries > 0)
				{
					usleep(10000);
				}
			}
		}
	}

	if(!success || frame.empty())
	{
		return cv::Mat();
	}
	
	// Store last captured frame for video recording (only for camera, not video file)
	// IMPORTANT: Save BEFORE resize to preserve original resolution
	if(!isVideoFile && !frame.empty())
	{
		std::lock_guard<std::mutex> frameLock(lastFrameMutex);
		lastCapturedFrame = frame.clone();  // Clone to keep a copy of original frame
	}

	// Resize if needed (for processing, but we keep original in lastCapturedFrame)
	if(frame.cols != widthOrg || frame.rows != heightOrg)
	{
		cv::Mat resized;
		cv::resize(frame, resized, cv::Size(widthOrg, heightOrg), 0, 0, cv::INTER_LINEAR);
		return resized;
	}

	return frame;
}

// Get raw image (no undistortion, no vignette, only photometric calibration)
ImageAndExposure* CameraReader::getImageRaw(int id)
{
	if(!enableDualMode || undistort == nullptr)
	{
		return nullptr;
	}

	cv::Mat frame = captureFrame();
	if(frame.empty())
	{
		return nullptr;
	}

	try
	{
		// Convert to grayscale
		cv::Mat gray;
		if(frame.channels() == 3)
		{
			cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		}
		else
		{
			gray = frame;
		}

		// Validate dimensions
		if(gray.cols <= 0 || gray.rows <= 0 || gray.cols > 10000 || gray.rows > 10000)
		{
			printf("ERROR: Invalid image dimensions: %d x %d\n", gray.cols, gray.rows);
			return nullptr;
		}

		// Create MinimalImageB
		MinimalImageB* minimg = new MinimalImageB(gray.cols, gray.rows);
		if(minimg == nullptr || minimg->data == nullptr)
		{
			printf("ERROR: Failed to allocate MinimalImageB\n");
			if(minimg != nullptr) delete minimg;
			return nullptr;
		}
		
		// Safe memory copy with bounds check
		size_t dataSize = (size_t)gray.rows * gray.cols;
		if(dataSize > 0 && dataSize <= (size_t)(gray.cols * gray.rows))
		{
			memcpy(minimg->data, gray.data, dataSize);
		}
		else
		{
			printf("ERROR: Invalid data size for memory copy\n");
			delete minimg;
			return nullptr;
		}

		// Apply only photometric calibration (no geometric undistortion, no vignette)
		if(undistort == nullptr || undistort->photometricUndist == nullptr)
		{
			// If no photometric calibration, just return the image
			ImageAndExposure* result = new ImageAndExposure(gray.cols, gray.rows, frameCount * 0.033);
			if(result == nullptr || result->image == nullptr)
			{
				printf("ERROR: Failed to allocate ImageAndExposure\n");
				delete minimg;
				if(result != nullptr) delete result;
				return nullptr;
			}
			
			size_t imgSize = (size_t)gray.rows * gray.cols;
			for(size_t i = 0; i < imgSize; i++)
			{
				if(i < imgSize && minimg->data != nullptr)
			{
				result->image[i] = (float)minimg->data[i];
				}
			}
			result->exposure_time = 1.0f;
			delete minimg;
			frameCount++;
			return result;
		}

		// Apply photometric calibration only (skip geometric undistortion)
		undistort->photometricUndist->processFrame<unsigned char>(minimg->data, 1.0f, 1.0f);
		
		// Create result with same size as photometricUndistorter output
		// photometricUndistorter output has same size as input (w x h)
		if(undistort->photometricUndist->output == nullptr || 
		   undistort->photometricUndist->output->image == nullptr)
		{
			printf("ERROR: PhotometricUndistorter output is null\n");
			delete minimg;
			return nullptr;
		}
		
		int outW = undistort->photometricUndist->output->w;
		int outH = undistort->photometricUndist->output->h;
		
		// Validate output dimensions
		if(outW <= 0 || outH <= 0 || outW > 10000 || outH > 10000)
		{
			printf("ERROR: Invalid output dimensions: %d x %d\n", outW, outH);
			delete minimg;
			return nullptr;
		}
		
		ImageAndExposure* result = new ImageAndExposure(outW, outH, frameCount * 0.033);
		if(result == nullptr || result->image == nullptr)
		{
			printf("ERROR: Failed to allocate ImageAndExposure\n");
			delete minimg;
			if(result != nullptr) delete result;
			return nullptr;
		}
		result->exposure_time = 1.0f;
		
		// Copy image data directly from photometricUndistorter output with bounds check
		float* src = undistort->photometricUndist->output->image;
		float* dst = result->image;
		size_t size = (size_t)outW * outH;
		if(size > 0 && size <= (size_t)(outW * outH))
		{
		memcpy(dst, src, sizeof(float) * size);
		}
		else
		{
			printf("ERROR: Invalid size for image data copy\n");
			delete minimg;
			delete result;
			return nullptr;
		}

		delete minimg;
		frameCount++;
		return result;
	}
	catch(const std::exception& e)
	{
		printf("ERROR: Exception in getImageRaw: %s\n", e.what());
		return nullptr;
	}
	catch(...)
	{
		printf("ERROR: Unknown exception in getImageRaw\n");
		return nullptr;
	}
}

// Get pipeline processed image
ImageAndExposure* CameraReader::getImagePipeline(int id)
{
	if(!enableDualMode || pipelineProcessor == nullptr || undistort_pipeline == nullptr)
	{
		return nullptr;
	}

	cv::Mat frame = captureFrame();
	if(frame.empty())
	{
		return nullptr;
	}

	try
	{
		// Validate input frame
		if(frame.empty() || frame.cols <= 0 || frame.rows <= 0)
		{
			printf("ERROR: Invalid input frame in getImagePipeline\n");
			return nullptr;
		}
		
		// Apply pipeline processing
		cv::Mat processed = pipelineProcessor->processFrame(frame);
		if(processed.empty())
		{
			printf("ERROR: Pipeline processing returned empty frame\n");
			return nullptr;
		}
		
		// Convert to grayscale
		cv::Mat gray;
		if(processed.channels() == 3)
		{
			cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
		}
		else
		{
			gray = processed;
		}

		// Validate grayscale image
		if(gray.empty() || gray.cols <= 0 || gray.rows <= 0 || gray.cols > 10000 || gray.rows > 10000)
		{
			printf("ERROR: Invalid grayscale image dimensions: %d x %d\n", gray.cols, gray.rows);
			return nullptr;
		}

		// Create MinimalImageB with validation
		MinimalImageB* minimg = new MinimalImageB(gray.cols, gray.rows);
		if(minimg == nullptr || minimg->data == nullptr)
		{
			printf("ERROR: Failed to allocate MinimalImageB in getImagePipeline\n");
			if(minimg != nullptr) delete minimg;
			return nullptr;
		}
		
		// Safe memory copy with bounds check
		size_t dataSize = (size_t)gray.rows * gray.cols;
		if(dataSize > 0 && dataSize <= (size_t)(gray.cols * gray.rows))
		{
			memcpy(minimg->data, gray.data, dataSize);
		}
		else
		{
			printf("ERROR: Invalid data size for memory copy in getImagePipeline\n");
			delete minimg;
			return nullptr;
		}

		// Apply full undistortion (geometric + photometric)
		ImageAndExposure* result = undistort_pipeline->undistort<unsigned char>(
			minimg,
			1.0f,
			frameCount * 0.033
		);

		delete minimg;
		frameCount++;
		return result;
	}
	catch(const std::exception& e)
	{
		printf("ERROR: Exception in getImagePipeline: %s\n", e.what());
		return nullptr;
	}
	catch(...)
	{
		printf("ERROR: Unknown exception in getImagePipeline\n");
		return nullptr;
	}
}

// Configure camera for DSO: disable auto functions and set fixed parameters
bool CameraReader::configureCameraForDSO()
{
	if(isVideoFile) 
	{
		printf("Video file input: hardware control not applicable\n");
		return false;  // Video files cannot control hardware
	}
	
	bool success = false;
	
	// Try to disable auto exposure
	capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // 0.25 = manual mode (V4L2)
	double aeValue = capture.get(cv::CAP_PROP_AUTO_EXPOSURE);
	bool aeDisabled = (std::abs(aeValue - 0.25) < 0.1);
	
	// Try to set fixed exposure time
	capture.set(cv::CAP_PROP_EXPOSURE, -6);  // -6 = 1/64s (log scale)
	double expValue = capture.get(cv::CAP_PROP_EXPOSURE);
	bool expSet = (std::abs(expValue - (-6)) < 0.5);
	
	// Try to disable auto white balance
	capture.set(cv::CAP_PROP_AUTO_WB, 0);
	double wbValue = capture.get(cv::CAP_PROP_AUTO_WB);
	bool wbDisabled = (wbValue < 0.1);
	
	// Try to disable auto focus (if supported)
	capture.set(cv::CAP_PROP_AUTOFOCUS, 0);
	
	success = aeDisabled && expSet && wbDisabled;
	
	if(success) 
	{
		printf("Camera hardware control: SUCCESS\n");
		printf("  Auto Exposure: DISABLED\n");
		printf("  Auto White Balance: DISABLED\n");
		printf("  Fixed Exposure: %.2f\n", expValue);
	} 
	else 
	{
		printf("Camera hardware control: FAILED (falling back to software compensation)\n");
		printf("  Auto Exposure: %s\n", aeDisabled ? "DISABLED" : "FAILED");
		printf("  Exposure Setting: %s (value=%.2f)\n", expSet ? "SET" : "FAILED", expValue);
		printf("  Auto White Balance: %s\n", wbDisabled ? "DISABLED" : "FAILED");
	}
	
	return success;
}

// Set fixed exposure time (microseconds)
void CameraReader::setExposureTime(double exposureUs)
{
	if(isVideoFile) return;  // Video files cannot control hardware
	
	// Try to set exposure time
	// Note: OpenCV uses log scale for some backends, absolute microseconds for others
	capture.set(cv::CAP_PROP_EXPOSURE, exposureUs);
	
	// Verify
	double actualValue = capture.get(cv::CAP_PROP_EXPOSURE);
	printf("Set exposure time: requested=%.0f us, actual=%.2f\n", exposureUs, actualValue);
}

// Set fixed gain/ISO
void CameraReader::setGain(double gain)
{
	if(isVideoFile) return;  // Video files cannot control hardware
	
	capture.set(cv::CAP_PROP_GAIN, gain);
	
	// Verify
	double actualValue = capture.get(cv::CAP_PROP_GAIN);
	printf("Set gain: requested=%.2f, actual=%.2f\n", gain, actualValue);
}

// Verify camera settings
bool CameraReader::verifyCameraSettings()
{
	if(isVideoFile) return false;  // Video files cannot control hardware
	
	bool allGood = true;
	
	// Check auto exposure
	double aeValue = capture.get(cv::CAP_PROP_AUTO_EXPOSURE);
	if(std::abs(aeValue - 0.25) >= 0.1)
	{
		printf("WARNING: Auto Exposure verification failed (value=%.2f)\n", aeValue);
		allGood = false;
	}
	
	// Check auto white balance
	double wbValue = capture.get(cv::CAP_PROP_AUTO_WB);
	if(wbValue >= 0.1)
	{
		printf("WARNING: Auto White Balance verification failed (value=%.2f)\n", wbValue);
		allGood = false;
	}
	
	return allGood;
}


