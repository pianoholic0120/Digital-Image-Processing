/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#include "IOWrapper/ImageDisplay.h"

#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <unordered_set>
#include <queue>
#include <memory>

#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>

#ifdef __APPLE__
#include <pthread.h>
#endif

#include "util/settings.h"

namespace dso
{


namespace IOWrap
{

std::unordered_set<std::string> openWindows;
boost::mutex openCVdisplayMutex;

// Thread-safe queue for OpenCV operations on macOS
#ifdef __APPLE__
struct OpenCVOperation {
    enum Type { IMSHOW, WAITKEY };
    Type type;
    std::string windowName;
    cv::Mat image;
    int milliseconds;
    
    OpenCVOperation(Type t, const std::string& wn, const cv::Mat& img) 
        : type(t), windowName(wn), image(img.clone()), milliseconds(0) {}
    OpenCVOperation(Type t, int ms) 
        : type(t), milliseconds(ms) {}
};

std::queue<OpenCVOperation> openCVOperationQueue;
boost::mutex openCVQueueMutex;
bool processOpenCVQueue = true;  // Enable queue processing
#endif



void displayImage(const char* windowName, const cv::Mat& image, bool autoSize)
{
	if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lock(openCVdisplayMutex);
	if(!autoSize)
	{
		if(openWindows.find(windowName) == openWindows.end())
		{
			// macOS requires window creation on main thread
			#ifdef __APPLE__
			// Check if we're on main thread
			if(pthread_main_np() != 0) {
				// On main thread, safe to create window
				cv::namedWindow(windowName, cv::WINDOW_NORMAL);
				if(image.cols > 0 && image.rows > 0) {
					cv::resizeWindow(windowName, image.cols, image.rows);
				}
				openWindows.insert(windowName);
			} else {
				// Not on main thread, skip window creation to avoid crash
				// Just mark as "opened" but don't actually create
				openWindows.insert(windowName);
				lock.unlock();
				return; // Skip imshow if window wasn't created
			}
			#else
			cv::namedWindow(windowName, cv::WINDOW_NORMAL);
			if(image.cols > 0 && image.rows > 0) {
				cv::resizeWindow(windowName, image.cols, image.rows);
			}
			openWindows.insert(windowName);
			#endif
		}
	}
	// Only show if window was successfully created
	if(openWindows.find(windowName) != openWindows.end()) {
		#ifdef __APPLE__
		// On macOS, queue imshow operations to be processed on main thread
		if(pthread_main_np() != 0) {
			// On main thread, execute directly
			cv::imshow(windowName, image);
		} else {
			// Not on main thread, queue the operation
			boost::unique_lock<boost::mutex> queueLock(openCVQueueMutex);
			openCVOperationQueue.push(OpenCVOperation(OpenCVOperation::IMSHOW, std::string(windowName), image));
		}
		#else
		cv::imshow(windowName, image);
		#endif
	}
}


void displayImageStitch(const char* windowName, const std::vector<cv::Mat*> images, int cc, int rc)
{
	if(disableAllDisplay) return;
	if(images.size() == 0) return;

	// get dimensions.
	int w = images[0]->cols;
	int h = images[0]->rows;

	int num = std::max((int)setting_maxFrames, (int)images.size());

	// get optimal dimensions.
	int bestCC = 0;
	float bestLoss = 1e10;
	for(int cc=1;cc<10;cc++)
	{
		int ww = w * cc;
		int hh = h * ((num+cc-1)/cc);


		float wLoss = ww/16.0f;
		float hLoss = hh/10.0f;
		float loss = std::max(wLoss, hLoss);

		if(loss < bestLoss)
		{
			bestLoss = loss;
			bestCC = cc;
		}
	}

	int bestRC = ((num+bestCC-1)/bestCC);
	if(cc != 0)
	{
		bestCC = cc;
		bestRC= rc;
	}
	cv::Mat stitch = cv::Mat(bestRC*h, bestCC*w, images[0]->type());
	stitch.setTo(0);
	for(int i=0;i<(int)images.size() && i < bestCC*bestRC;i++)
	{
		int c = i%bestCC;
		int r = i/bestCC;

		cv::Mat roi = stitch(cv::Rect(c*w, r*h, w,h));
		images[i]->copyTo(roi);
	}
	displayImage(windowName, stitch, false);
}



void displayImage(const char* windowName, const MinimalImageB* img, bool autoSize)
{
	displayImage(windowName, cv::Mat(img->h, img->w, CV_8U, img->data), autoSize);
}
void displayImage(const char* windowName, const MinimalImageB3* img, bool autoSize)
{
	displayImage(windowName, cv::Mat(img->h, img->w, CV_8UC3, img->data), autoSize);
}
void displayImage(const char* windowName, const MinimalImageF* img, bool autoSize)
{
	displayImage(windowName, cv::Mat(img->h, img->w, CV_32F, img->data)*(1/254.0f), autoSize);
}
void displayImage(const char* windowName, const MinimalImageF3* img, bool autoSize)
{
	displayImage(windowName, cv::Mat(img->h, img->w, CV_32FC3, img->data)*(1/254.0f), autoSize);
}
void displayImage(const char* windowName, const MinimalImageB16* img, bool autoSize)
{
	displayImage(windowName, cv::Mat(img->h, img->w, CV_16U, img->data), autoSize);
}


void displayImageStitch(const char* windowName, const std::vector<MinimalImageB*> images, int cc, int rc)
{
	std::vector<cv::Mat*> imagesCV;
    for(size_t i=0; i < images.size();i++)
		imagesCV.push_back(new cv::Mat(images[i]->h, images[i]->w, CV_8U, images[i]->data));
	displayImageStitch(windowName, imagesCV, cc, rc);
    for(size_t i=0; i < images.size();i++)
		delete imagesCV[i];
}
void displayImageStitch(const char* windowName, const std::vector<MinimalImageB3*> images, int cc, int rc)
{
	std::vector<cv::Mat*> imagesCV;
    for(size_t i=0; i < images.size();i++)
		imagesCV.push_back(new cv::Mat(images[i]->h, images[i]->w, CV_8UC3, images[i]->data));
	displayImageStitch(windowName, imagesCV, cc, rc);
    for(size_t i=0; i < images.size();i++)
		delete imagesCV[i];
}
void displayImageStitch(const char* windowName, const std::vector<MinimalImageF*> images, int cc, int rc)
{
	std::vector<cv::Mat*> imagesCV;
    for(size_t i=0; i < images.size();i++)
		imagesCV.push_back(new cv::Mat(images[i]->h, images[i]->w, CV_32F, images[i]->data));
	displayImageStitch(windowName, imagesCV, cc, rc);
    for(size_t i=0; i < images.size();i++)
		delete imagesCV[i];
}
void displayImageStitch(const char* windowName, const std::vector<MinimalImageF3*> images, int cc, int rc)
{
	std::vector<cv::Mat*> imagesCV;
    for(size_t i=0; i < images.size();i++)
		imagesCV.push_back(new cv::Mat(images[i]->h, images[i]->w, CV_32FC3, images[i]->data));
	displayImageStitch(windowName, imagesCV, cc, rc);
    for(size_t i=0; i < images.size();i++)
		delete imagesCV[i];
}



int waitKey(int milliseconds)
{
	if(disableAllDisplay) return 0;

	#ifdef __APPLE__
	// macOS requires waitKey to be called from main thread
	if(pthread_main_np() != 0) {
		// On main thread, execute directly
		boost::unique_lock<boost::mutex> lock(openCVdisplayMutex);
		return cv::waitKey(milliseconds);
	} else {
		// Not on main thread, return immediately without waiting
		// (can't queue waitKey as it's blocking)
		return 0;
	}
	#else
	boost::unique_lock<boost::mutex> lock(openCVdisplayMutex);
	return cv::waitKey(milliseconds);
	#endif
}

#ifdef __APPLE__
// Process queued OpenCV operations (must be called from main thread)
void processOpenCVOperations()
{
	if(!processOpenCVQueue) return;
	
	boost::unique_lock<boost::mutex> queueLock(openCVQueueMutex);
	while(!openCVOperationQueue.empty()) {
		OpenCVOperation op = openCVOperationQueue.front();
		openCVOperationQueue.pop();
		queueLock.unlock();
		
		if(op.type == OpenCVOperation::IMSHOW) {
			boost::unique_lock<boost::mutex> displayLock(openCVdisplayMutex);
			if(openWindows.find(op.windowName) != openWindows.end()) {
				cv::imshow(op.windowName, op.image);
			}
			displayLock.unlock();
		}
		
		queueLock.lock();
	}
}
#endif

void closeAllWindows()
{
	if(disableAllDisplay) return;
	boost::unique_lock<boost::mutex> lock(openCVdisplayMutex);
	cv::destroyAllWindows();
	openWindows.clear();
}
}

}

