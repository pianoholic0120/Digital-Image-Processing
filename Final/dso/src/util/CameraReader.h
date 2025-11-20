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
	CameraReader(int cameraIndex, std::string calibFile, std::string gammaFile, std::string vignetteFile)
	{
		this->cameraIndex = cameraIndex;
		this->calibfile = calibFile;

		// Initialize camera
		capture.open(cameraIndex);
		if(!capture.isOpened())
		{
			printf("ERROR: Cannot open camera with index %d!\n", cameraIndex);
			exit(1);
		}

		// Set camera properties (optional, adjust as needed)
		// Don't force resolution, let camera use its native resolution
		// capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		// capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
		capture.set(cv::CAP_PROP_FPS, 30);

		// Get actual camera resolution
		int camWidth = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
		int camHeight = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
		printf("Camera opened: resolution %d x %d\n", camWidth, camHeight);

		// Initialize undistorter
		undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);

		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		width = undistort->getSize()[0];
		height = undistort->getSize()[1];

		printf("CameraReader initialized with calibration from %s\n", calibFile.c_str());
		printf("Expected resolution: %d x %d\n", widthOrg, heightOrg);

		// Check if camera resolution matches calibration
		if(camWidth != widthOrg || camHeight != heightOrg)
		{
			printf("WARNING: Camera resolution (%d x %d) doesn't match calibration (%d x %d)\n", 
				camWidth, camHeight, widthOrg, heightOrg);
			printf("Images will be resized to match calibration.\n");
		}

		frameCount = 0;
		running = true;
	}

	~CameraReader()
	{
		running = false;
		// Thread-safe cleanup
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
	}

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
		// For camera, return a large number to indicate continuous stream
		return 1000000;
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

	inline float* getPhotometricGamma()
	{
		if(undistort==0 || undistort->photometricUndist==0) return 0;
		return undistort->photometricUndist->getG();
	}

	Undistort* undistort;
	int width, height;
	int widthOrg, heightOrg;
	int frameCount;
	bool running;

private:
	int cameraIndex;
	std::string calibfile;
	cv::VideoCapture capture;
	std::mutex captureMutex;  // Mutex for thread-safe camera access

	ImageAndExposure* getImage_internal(int unused)
	{
		if(!running || !capture.isOpened())
		{
			return nullptr;
		}

		cv::Mat frame;
		bool success = false;
		
		// Thread-safe camera access
		{
			std::lock_guard<std::mutex> lock(captureMutex);
			
			// Capture frame from camera with retry mechanism
			int retries = 3;
			while(retries > 0 && !success)
			{
				if(capture.read(frame))
				{
					success = true;
				}
				else
				{
					retries--;
					if(retries > 0)
					{
						usleep(10000); // Wait 10ms before retry
					}
				}
			}
		}

		if(!success || frame.empty())
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

			// Resize if needed to match calibration
			if(gray.cols != widthOrg || gray.rows != heightOrg)
			{
				cv::Mat resized;
				cv::resize(gray, resized, cv::Size(widthOrg, heightOrg), 0, 0, cv::INTER_LINEAR);
				gray = resized;
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
};

