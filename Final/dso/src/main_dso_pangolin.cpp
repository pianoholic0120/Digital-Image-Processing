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



#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/CameraReader.h"
#include "util/globalCalib.h"
#include "util/DataExporter.h"
#include <mutex>
#include <atomic>

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"



#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#ifdef __APPLE__
#include <opencv2/opencv.hpp>
#include <pthread.h>
#endif


std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;
int cameraIndex = -1;  // -1 means use image folder, >= 0 means use camera


int mode=0;

bool firstRosSpin=false;

using namespace dso;


void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();
}



void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations=6;
		setting_minOptIterations=1;

		setting_logStuff = false;
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}






void parseArgument(char* arg)
{
	int option;
	float foption;
	char buf[1000];


    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

	if(1==sscanf(arg,"preset=%d",&option))
	{
		settingsDefault(option);
		return;
	}


	if(1==sscanf(arg,"rec=%d",&option))
	{
		if(option==0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
		return;
	}



	if(1==sscanf(arg,"noros=%d",&option))
	{
		if(option==1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}
	if(1==sscanf(arg,"reverse=%d",&option))
	{
		if(option==1)
		{
			reverse = true;
			printf("REVERSE!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"prefetch=%d",&option))
	{
		if(option==1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
		return;
	}
	if(1==sscanf(arg,"start=%d",&option))
	{
		start = option;
		printf("START AT %d!\n",start);
		return;
	}
	if(1==sscanf(arg,"end=%d",&option))
	{
		end = option;
		printf("END AT %d!\n",start);
		return;
	}

	if(1==sscanf(arg,"files=%s",buf))
	{
		source = buf;
		printf("loading data from %s!\n", source.c_str());
		return;
	}

	if(1==sscanf(arg,"camera=%d",&option))
	{
		cameraIndex = option;
		printf("Using camera with index %d!\n", cameraIndex);
		return;
	}

	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}

	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignette = buf;
		printf("loading vignette from %s!\n", vignette.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaCalib = buf;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
		return;
	}

	if(1==sscanf(arg,"rescale=%f",&foption))
	{
		rescale = foption;
		printf("RESCALE %f!\n", rescale);
		return;
	}

	if(1==sscanf(arg,"speed=%f",&foption))
	{
		playbackSpeed = foption;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
		return;
	}

	if(1==sscanf(arg,"save=%d",&option))
	{
		if(option==1)
		{
			debugSaveImages = true;
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

	if(1==sscanf(arg,"mode=%d",&option))
	{

		mode = option;
		if(option==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(option==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		}
		if(option==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
		}
		return;
	}

	printf("could not parse argument \"%s\"!!!!\n", arg);
}



int main( int argc, char** argv )
{
	//setlocale(LC_ALL, "");
	for(int i=1; i<argc;i++)
		parseArgument(argv[i]);

	// hook crtl+C.
	boost::thread exThread = boost::thread(exitThread);


	ImageFolderReader* reader = nullptr;
	CameraReader* cameraReader = nullptr;
	
	if(cameraIndex >= 0)
	{
		// Use camera input
		printf("Initializing camera input...\n");
		cameraReader = new CameraReader(cameraIndex, calib, gammaCalib, vignette);
		cameraReader->setGlobalCalibration();
	}
	else
	{
		// Use image folder input
		reader = new ImageFolderReader(source,calib, gammaCalib, vignette);
		reader->setGlobalCalibration();
	}



	if(setting_photometricCalibration > 0)
	{
		float* gamma = (cameraReader != nullptr) ? cameraReader->getPhotometricGamma() : reader->getPhotometricGamma();
		if(gamma == 0)
		{
			printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
			exit(1);
		}
	}




	int lstart=start;
	int lend = end;
	int linc = 1;
	if(reverse)
	{
		printf("REVERSE!!!!");
		int numImages = (cameraReader != nullptr) ? cameraReader->getNumImages() : (reader != nullptr ? reader->getNumImages() : 0);
		lstart=end-1;
		if(lstart >= numImages)
			lstart = numImages-1;
		lend = start;
		linc = -1;
	}



	FullSystem* fullSystem = new FullSystem();
	float* gamma = (cameraReader != nullptr) ? cameraReader->getPhotometricGamma() : reader->getPhotometricGamma();
	fullSystem->setGammaFunction(gamma);
	fullSystem->linearizeOperation = (playbackSpeed==0);







    IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        #ifdef __APPLE__
        // Verify we're on main thread before GUI initialization (macOS requirement)
        if(pthread_main_np() == 0) {
            printf("ERROR: GUI initialization must be on main thread on macOS!\n");
            printf("Exiting to prevent crashes...\n");
            return -1;
        }
        printf("Verified: GUI initialization on main thread (macOS requirement satisfied)\n");
        #endif
        
        // Create Pangolin viewer on main thread (required for macOS)
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
        
        // Pre-create OpenCV windows on main thread (required for macOS)
        // This prevents crashes when displayImage is called from tracking thread
        #ifdef __APPLE__
        {
            cv::namedWindow("frameToTrack", cv::WINDOW_NORMAL);
            cv::namedWindow("RES", cv::WINDOW_NORMAL);
            cv::namedWindow("Selector Image", cv::WINDOW_NORMAL);
            cv::namedWindow("Selector Pixels", cv::WINDOW_NORMAL);
            printf("OpenCV windows created on main thread\n");
        }
        #endif
    }



    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());




    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    // Capture reader pointers for lambda
    ImageFolderReader* readerPtr = reader;
    CameraReader* cameraReaderPtr = cameraReader;
    
    // Mutex to protect fullSystem access during reset
    std::mutex fullSystemMutex;
    std::atomic<bool> resetting(false);
    
    // Keyboard control for camera input
    std::atomic<bool> startProcessing(false);
    std::atomic<bool> stopProcessing(false);
    
    // Storage for captured frames (for video export)
    std::vector<cv::Mat> capturedFrames;
    std::mutex framesMutex;
    
    // For camera input, wait for 's' to start
    if(cameraReaderPtr != nullptr)
    {
        printf("\n========================================\n");
        printf("USB Camera Mode\n");
        printf("Press 's' to START processing\n");
        printf("Press 'e' to END and save files\n");
        printf("========================================\n\n");
        
        // Set terminal to non-blocking mode for keyboard input
        struct termios oldt, newt;
        int oldf;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
        
        // Wait for 's' to start
        char ch;
        while(!startProcessing && !stopProcessing)
        {
            ch = getchar();
            if(ch == 's' || ch == 'S')
            {
                startProcessing = true;
                printf(">>> STARTED! Processing frames...\n");
                printf(">>> Press 'e' to END and save files\n");
            }
            else if(ch == 'e' || ch == 'E')
            {
                stopProcessing = true;
                printf(">>> STOPPED! Saving files...\n");
            }
            usleep(10000); // Sleep 10ms to avoid busy waiting
        }
        
        // Restore terminal settings
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldf);
    }
    else
    {
        // For image folder input, start immediately
        startProcessing = true;
    }
    
    std::thread runthread([&, readerPtr, cameraReaderPtr]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        int numImages = (cameraReaderPtr != nullptr) ? cameraReaderPtr->getNumImages() : (readerPtr != nullptr ? readerPtr->getNumImages() : 0);
        
        // For camera input, we'll process frames continuously
        // For image folder, build the list of frames to process
        if(cameraReaderPtr == nullptr)
        {
            // Image folder mode: build list of frames
            for(int i=lstart;i>= 0 && i< numImages && linc*i < linc*lend;i+=linc)
            {
                idsToPlay.push_back(i);
                if(timesToPlayAt.size() == 0)
                {
                    timesToPlayAt.push_back((double)0);
                }
                else
                {
                    double tsThis = readerPtr->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                    double tsPrev = readerPtr->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                    timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
                }
            }
        }
        else
        {
            // Camera mode: start with frame 0, will process continuously
            idsToPlay.push_back(0);
            timesToPlayAt.push_back(0.0);
        }


        std::vector<ImageAndExposure*> preloadedImages;
        if(preload && cameraReaderPtr == nullptr)  // Don't preload for camera (too many frames)
        {
            printf("LOADING ALL IMAGES!\n");
            for(int ii=0;ii<(int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                if(readerPtr != nullptr)
                {
                    preloadedImages.push_back(readerPtr->getImage(i));
                }
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0;


        // For camera mode, use continuous loop; for image folder, use fixed list
        int frameIndex = 0;
        bool continueLoop = true;
        
        while(continueLoop)
        {
            // For camera input, wait for start signal
            if(cameraReaderPtr != nullptr && !startProcessing)
            {
                usleep(10000); // Wait 10ms
                continue;
            }
            
            // Check if stop signal received
            if(stopProcessing)
            {
                printf("Stop signal received, breaking loop...\n");
                break;
            }
            
            // Determine which frame to process
            int i;
            if(cameraReaderPtr != nullptr)
            {
                // Camera mode: use frame index (continuous)
                i = frameIndex;
                frameIndex++;
            }
            else
            {
                // Image folder mode: use pre-built list
                if(frameIndex >= (int)idsToPlay.size())
                {
                    break; // All frames processed
                }
                i = idsToPlay[frameIndex];
                frameIndex++;
            }
            
            // Check initialization status (with mutex protection)
            {
                std::lock_guard<std::mutex> lock(fullSystemMutex);
                if(fullSystem != nullptr && !fullSystem->initialized)	// if not initialized: reset start time.
                {
                    gettimeofday(&tv_start, NULL);
                    started = clock();
                    if(cameraReaderPtr == nullptr && frameIndex > 0 && frameIndex <= (int)timesToPlayAt.size())
                    {
                        sInitializerOffset = timesToPlayAt[frameIndex-1];
                    }
                    else
                    {
                        sInitializerOffset = 0.0;
                    }
                }
            }


            ImageAndExposure* img = nullptr;
            if(preload && cameraReaderPtr == nullptr && frameIndex-1 < (int)preloadedImages.size())
            {
                img = preloadedImages[frameIndex-1];
            }
            else
            {
                if(cameraReaderPtr != nullptr)
                {
                    img = cameraReaderPtr->getImage(i);
                }
                else if(readerPtr != nullptr)
                {
                    img = readerPtr->getImage(i);
                }
            }

            if(img == nullptr)
            {
                printf("WARNING: Failed to get image %d, skipping frame.\n", i);
                // For camera input, if we can't get a frame, wait a bit and try again
                if(cameraReaderPtr != nullptr)
                {
                    usleep(33000); // Wait ~33ms (30 FPS) before retry
                }
                continue;
            }



            bool skipFrame=false;
            if(playbackSpeed!=0 && cameraReaderPtr == nullptr)  // Only use playback speed for image folder
            {
                int currentIdx = frameIndex - 1;
                if(currentIdx >= 0 && currentIdx < (int)timesToPlayAt.size())
                {
                    struct timeval tv_now; gettimeofday(&tv_now, NULL);
                    double sSinceStart = sInitializerOffset + ((tv_now.tv_sec-tv_start.tv_sec) + (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));

                    if(sSinceStart < timesToPlayAt[currentIdx])
                        usleep((int)((timesToPlayAt[currentIdx]-sSinceStart)*1000*1000));
                    else if(sSinceStart > timesToPlayAt[currentIdx]+0.5+0.1*(currentIdx%2))
                    {
                        printf("SKIPFRAME %d (play at %f, now it is %f)!\n", currentIdx, timesToPlayAt[currentIdx], sSinceStart);
                        skipFrame=true;
                    }
                }
            }



            if(!skipFrame && img != nullptr) 
            {
                // Check if resetting before accessing fullSystem (quick check without lock)
                if(resetting)
                {
                    delete img;
                    img = nullptr;
                    usleep(10000); // Wait 10ms if resetting
                    continue;
                }
                
                // Lock mutex when accessing fullSystem
                std::lock_guard<std::mutex> lock(fullSystemMutex);
                
                // Check again after acquiring lock (another thread might have started reset)
                if(resetting || fullSystem == nullptr)
                {
                    delete img;
                    img = nullptr;
                    continue;
                }
                
                try {
                    fullSystem->addActiveFrame(img, i);
                } catch(const std::exception& e) {
                    printf("ERROR: Exception in addActiveFrame: %s\n", e.what());
                    delete img;
                    img = nullptr;
                    continue;
                } catch(...) {
                    printf("ERROR: Unknown exception in addActiveFrame\n");
                    delete img;
                    img = nullptr;
                    continue;
                }
            }




            if(img != nullptr)
            {
                delete img;
                img = nullptr;
            }

            // Check if reset is needed (with mutex protection)
            bool needReset = false;
            {
                std::lock_guard<std::mutex> lock(fullSystemMutex);
                if(fullSystem != nullptr && (fullSystem->initFailed || setting_fullResetRequested))
                {
                    needReset = true;
                }
            }
            
            if(needReset)
            {
                if(frameIndex < 250 || setting_fullResetRequested)
                {
                    printf("RESETTING!\n");
                    
                    // Set resetting flag first (without lock, so other threads can see it)
                    resetting = true;
                    
                    // Wait a bit for other threads to finish current operations
                    usleep(200000); // Wait 200ms for other threads to finish
                    
                    // Lock mutex before modifying fullSystem
                    std::lock_guard<std::mutex> lock(fullSystemMutex);
                    
                    // Double-check that fullSystem still exists and needs reset
                    if(fullSystem == nullptr)
                    {
                        resetting = false;
                        continue;
                    }
                    
                    // Block until mapping is finished to ensure no active operations
                    try {
                        fullSystem->blockUntilMappingIsFinished();
                    } catch(const std::exception& e) {
                        printf("WARNING: Exception in blockUntilMappingIsFinished during reset: %s\n", e.what());
                    } catch(...) {
                        printf("WARNING: Unknown exception in blockUntilMappingIsFinished during reset\n");
                    }

                    // Save output wrappers before deleting fullSystem
                    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                    
                    // Delete old fullSystem
                    try {
                        delete fullSystem;
                        fullSystem = nullptr;
                    } catch(const std::exception& e) {
                        printf("WARNING: Exception deleting fullSystem during reset: %s\n", e.what());
                        fullSystem = nullptr; // Ensure it's null even if delete throws
                    } catch(...) {
                        printf("WARNING: Unknown exception deleting fullSystem during reset\n");
                        fullSystem = nullptr;
                    }

                    // Reset output wrappers
                    for(IOWrap::Output3DWrapper* ow : wraps) 
                    {
                        if(ow != nullptr)
                        {
                            try {
                                ow->reset();
                            } catch(const std::exception& e) {
                                printf("WARNING: Exception resetting output wrapper: %s\n", e.what());
                            } catch(...) {
                                printf("WARNING: Unknown exception resetting output wrapper\n");
                            }
                        }
                    }

                    // Create new fullSystem
                    try {
                        fullSystem = new FullSystem();
                        if(fullSystem == nullptr)
                        {
                            printf("ERROR: Failed to allocate new fullSystem during reset!\n");
                            resetting = false;
                            break; // Exit loop if we can't create new system
                        }
                        
                        float* gamma = (cameraReaderPtr != nullptr) ? cameraReaderPtr->getPhotometricGamma() : (readerPtr != nullptr ? readerPtr->getPhotometricGamma() : nullptr);
                        fullSystem->setGammaFunction(gamma);
                        fullSystem->linearizeOperation = (playbackSpeed==0);
                        fullSystem->outputWrapper = wraps;
                    } catch(const std::exception& e) {
                        printf("ERROR: Exception creating new fullSystem during reset: %s\n", e.what());
                        resetting = false;
                        break; // Exit loop if we can't create new system
                    } catch(...) {
                        printf("ERROR: Unknown exception creating new fullSystem during reset!\n");
                        resetting = false;
                        break; // Exit loop if we can't create new system
                    }

                    setting_fullResetRequested=false;
                    resetting = false;
                    printf("RESET COMPLETE!\n");
                }
                else
                {
                    // If reset is needed but we're past frame 250, just mark as lost
                    std::lock_guard<std::mutex> lock(fullSystemMutex);
                    if(fullSystem != nullptr)
                    {
                        fullSystem->isLost = true;
                    }
                }
            }

            // Check if system is lost (with mutex protection)
            {
                std::lock_guard<std::mutex> lock(fullSystemMutex);
                if(fullSystem != nullptr && fullSystem->isLost)
                {
                    printf("LOST!!\n");
                    break;
                }
            }

        }
        
        // Block until mapping is finished (with mutex protection)
        // For camera mode with stopProcessing, skip to exit quickly
        if(!stopProcessing || cameraReaderPtr == nullptr)
        {
            std::lock_guard<std::mutex> lock(fullSystemMutex);
            if(fullSystem != nullptr)
            {
                try {
                    fullSystem->blockUntilMappingIsFinished();
                } catch(...) {
                    printf("WARNING: Exception in blockUntilMappingIsFinished at end\n");
                }
            }
        }
        else
        {
            printf("Skipping mapping wait for quick exit...\n");
        }
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);


        // Print result with mutex protection
        {
            std::lock_guard<std::mutex> lock(fullSystemMutex);
            if(fullSystem != nullptr)
            {
                try {
                    fullSystem->printResult("result.txt");
                } catch(...) {
                    printf("WARNING: Exception in printResult\n");
                }
            }
        }


        int numFramesProcessed = frameIndex;
        double numSecondsProcessed = 0.0;
        if(cameraReaderPtr != nullptr)
        {
            // For camera, estimate based on frame count and assumed FPS
            numSecondsProcessed = frameIndex * 0.033; // Assuming ~30 FPS
        }
        else if(!idsToPlay.empty())
        {
            double ts0 = readerPtr->getTimestamp(idsToPlay[0]);
            double ts1 = readerPtr->getTimestamp(idsToPlay.back());
            numSecondsProcessed = fabs(ts0 - ts1);
        }
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            int totalImages = (cameraReaderPtr != nullptr) ? cameraReaderPtr->getNumImages() : (readerPtr != nullptr ? readerPtr->getNumImages() : 0);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*totalImages) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)totalImages << "\n";
            tmlog.flush();
            tmlog.close();
        }

    });


    // Thread structure for macOS:
    // - Main thread: Runs GUI (viewer->run()) - REQUIRED for macOS
    // - Tracking thread (runthread): Processes images in parallel
    // - Keyboard listener thread (for camera input): Listens for 'e' to stop
    // 
    // On macOS, all GUI operations MUST be on main thread.
    // The GUI loop will run until window is closed or program exits.
    
    // Start keyboard listener thread for camera input
    std::thread keyboardThread;
    if(cameraReaderPtr != nullptr)
    {
        keyboardThread = std::thread([&]() {
            // Set terminal to non-blocking mode for keyboard input
            struct termios oldt, newt;
            int oldf;
            tcgetattr(STDIN_FILENO, &oldt);
            newt = oldt;
            newt.c_lflag &= ~(ICANON | ECHO);
            tcsetattr(STDIN_FILENO, TCSANOW, &newt);
            oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
            fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
            
            // Listen for 'e' to stop
            char ch;
            while(!stopProcessing && startProcessing)
            {
                ch = getchar();
                if(ch == 'e' || ch == 'E')
                {
                    stopProcessing = true;
                    printf("\n>>> 'e' pressed! Stopping processing and saving files...\n");
                    // Close viewer to exit GUI loop quickly (for camera mode)
                    if(cameraReaderPtr != nullptr && viewer != nullptr)
                    {
                        printf(">>> Closing viewer...\n");
                        viewer->close();
                    }
                    break;
                }
                usleep(50000); // Sleep 50ms to avoid busy waiting
            }
            
            // Restore terminal settings
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
            fcntl(STDIN_FILENO, F_SETFL, oldf);
        });
    }
    
    if(viewer != 0)
    {
        // Start tracking thread first (it runs in parallel)
        // Note: runthread is already started above with std::thread constructor
        
        // Run GUI on main thread (blocking call - this is the main event loop)
        // This ensures all GUI operations are on main thread for macOS
        // The GUI will continue running until window is closed
        viewer->run();
        
        // After GUI exits, wait for tracking thread to finish
        if(runthread.joinable())
        {
            runthread.join();
        }
    }
    else
    {
        // If no viewer, just wait for tracking thread
        runthread.join();
    }
    
    // Wait for keyboard thread to finish (if started)
    if(keyboardThread.joinable())
    {
        keyboardThread.join();
    }

	// Export data before cleanup
	printf("\n==================== EXPORTING DATA ====================\n");
	printf("Saving point cloud, camera poses, and video...\n");
	std::string outputDir = "dso_output";
	
	{
		std::lock_guard<std::mutex> lock(fullSystemMutex);
		if(fullSystem != nullptr)
		{
			try {
				IOWrap::DataExporter::exportAll(fullSystem, viewer, capturedFrames, outputDir, 30.0);
				printf("Data export completed successfully!\n");
			} catch(const std::exception& e) {
				printf("ERROR: Exception during data export: %s\n", e.what());
			} catch(...) {
				printf("ERROR: Unknown exception during data export\n");
			}
		}
		else
		{
			printf("WARNING: fullSystem is null, skipping data export\n");
		}
	}
	printf("=======================================================\n\n");
	printf("All done! Files saved to: %s\n", outputDir.c_str());

	// Clean up with mutex protection
	{
		std::lock_guard<std::mutex> lock(fullSystemMutex);
		if(fullSystem != nullptr)
		{
			for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
			{
				if(ow != nullptr)
				{
					try {
						ow->join();
						delete ow;
					} catch(...) {
						printf("WARNING: Exception cleaning up output wrapper\n");
					}
				}
			}

			printf("DELETE FULLSYSTEM!\n");
			try {
				delete fullSystem;
				fullSystem = nullptr;
			} catch(...) {
				printf("WARNING: Exception deleting fullSystem at cleanup\n");
			}
		}
	}

	printf("DELETE READER!\n");
	if(cameraReader != nullptr)
	{
		delete cameraReader;
	}
	else
	{
		delete reader;
	}

	printf("EXIT NOW!\n");
	return 0;
}
