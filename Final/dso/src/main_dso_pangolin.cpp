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
#include <chrono>
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
#include "IOWrapper/Pangolin/DualPangolinDSOViewer.h"
#include <pangolin/pangolin.h>
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#ifdef __APPLE__
#include <opencv2/opencv.hpp>
#include <pthread.h>
#endif
#include <pthread.h>
#include <exception>
#include <cstdlib>


std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
std::string videoFile = "";  // Video file path (if using video input)
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
int dualMode = 0;  // Dual mode: 0=raw only, 1=both (raw+pipeline), 2=pipeline only
bool enableCLAHE = false;  // Enable CLAHE for pipeline path
bool saveVideo = false;  // Save video when using camera=0 (0=disabled, 1=enabled)


int mode=0;

bool firstRosSpin=false;

using namespace dso;


bool shouldExit = false;
bool trackingThreadCrashed = false;
std::string crashMessage = "";
struct timeval tv_start_global;
int frameIndex_global = 0;
// Separate crash flags for dual mode
bool rawSystemCrashed = false;
bool pipelineSystemCrashed = false;
std::string rawCrashMessage = "";
std::string pipelineCrashMessage = "";
std::atomic<bool> exportRequested(false);  // Flag to trigger export when 'e' is pressed

void my_exit_handler(int s)
{
	printf("Caught signal %d - setting exit flag (GUI will remain open)\n", s);
	shouldExit = true;
	trackingThreadCrashed = true;
	crashMessage = "Signal " + std::to_string(s) + " received";
	// Don't exit immediately - let the main loop handle cleanup gracefully
}

// Signal handler for segmentation faults and other critical errors
void crash_handler(int sig, siginfo_t* info, void* context)
{
	// Only handle signals in tracking thread, not in GUI thread
	// Check if we're in the tracking thread by checking if it's set
	static bool handlerCalled = false;
	if(handlerCalled) {
		// Prevent recursive calls - if we're called again, just exit
		// This means the crash is too severe to recover
		_exit(1);
	}
	handlerCalled = true;
	
	printf("\n========================================\n");
	printf("CRITICAL ERROR: Signal %d received\n", sig);
	printf("Tracking thread has crashed - reconstruction stopped\n");
	printf("Pangolin viewer will remain open for inspection\n");
	printf("========================================\n\n");
	
	trackingThreadCrashed = true;
	shouldExit = true;
	
	switch(sig) {
		case SIGSEGV:
			crashMessage = "Segmentation fault (memory access violation)";
			break;
		case SIGABRT:
			crashMessage = "Abort signal (assertion failed or abort called)";
			// SIGABRT is tricky - abort() will terminate the process
			// We can't prevent it easily, but we can try to delay it
			// by not returning from the handler immediately
			// Instead, we'll use a different approach: catch it in the thread
			break;
		case SIGBUS:
			crashMessage = "Bus error (invalid memory access)";
			break;
		case SIGFPE:
			crashMessage = "Floating point exception";
			break;
		default:
			crashMessage = "Signal " + std::to_string(sig);
	}
	
	// For SIGABRT, the process will be terminated by abort()
	// We can't prevent this, but we've set the flags so if the process
	// somehow continues, the GUI will know what happened
	// The real solution is to prevent the crash in the first place
	
	// Don't return immediately - let the flags be set and let the tracking thread
	// detect them and stop processing gracefully
}

void exitThread()
{
	// Set up signal handlers for graceful error handling
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);
	
	// Set up crash handlers for critical errors (SIGSEGV, SIGABRT, etc.)
	struct sigaction crashHandler;
	crashHandler.sa_sigaction = crash_handler;
	sigemptyset(&crashHandler.sa_mask);
	crashHandler.sa_flags = SA_SIGINFO;
	sigaction(SIGSEGV, &crashHandler, NULL);
	sigaction(SIGABRT, &crashHandler, NULL);
	sigaction(SIGBUS, &crashHandler, NULL);
	sigaction(SIGFPE, &crashHandler, NULL);

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

	if(1==sscanf(arg,"video=%s",buf))
	{
		videoFile = buf;
		printf("Using video file: %s!\n", videoFile.c_str());
		return;
	}

	if(1==sscanf(arg,"camera=%d",&option))
	{
		cameraIndex = option;
		printf("Using camera with index %d!\n", cameraIndex);
		return;
	}

	if(1==sscanf(arg,"dual=%d",&option))
	{
		dualMode = option;
		if(dualMode == 0)
			printf("Dual mode: 0 (Raw only)\n");
		else if(dualMode == 1)
			printf("Dual mode: 1 (Both - Raw + Pipeline)\n");
		else if(dualMode == 2)
			printf("Dual mode: 2 (Pipeline only)\n");
		else
		{
			printf("WARNING: Invalid dual mode %d, using 0 (Raw only)\n", option);
			dualMode = 0;
		}
		return;
	}
	
	// Also try parsing without = sign for compatibility
	if(1==sscanf(arg,"dual%d",&option))
	{
		dualMode = option;
		if(dualMode == 0)
			printf("Dual mode: 0 (Raw only)\n");
		else if(dualMode == 1)
			printf("Dual mode: 1 (Both - Raw + Pipeline)\n");
		else if(dualMode == 2)
			printf("Dual mode: 2 (Pipeline only)\n");
		else
		{
			printf("WARNING: Invalid dual mode %d, using 0 (Raw only)\n", option);
			dualMode = 0;
		}
		return;
	}

	if(1==sscanf(arg,"clahe=%d",&option))
	{
		enableCLAHE = (option == 1);
		printf("CLAHE: %s\n", enableCLAHE ? "ENABLED" : "DISABLED");
		return;
	}

	if(1==sscanf(arg,"save_video=%d",&option))
	{
		saveVideo = (option == 1);
		printf("Save video (camera mode): %s\n", saveVideo ? "ENABLED" : "DISABLED");
		return;
	}
	
	// Also try parsing without = sign for compatibility
	if(1==sscanf(arg,"save_video%d",&option))
	{
		saveVideo = (option == 1);
		printf("Save video (camera mode): %s\n", saveVideo ? "ENABLED" : "DISABLED");
		return;
	}
	
	// Also try parsing without = sign for compatibility
	if(1==sscanf(arg,"clahe%d",&option))
	{
		enableCLAHE = (option == 1);
		printf("CLAHE: %s\n", enableCLAHE ? "ENABLED" : "DISABLED");
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
	
	if(!videoFile.empty())
	{
		// Use video file input
		printf("Initializing video file input...\n");
		if(dualMode == 1)
		{
			printf("Dual mode enabled: Raw path (photometric only) + Pipeline path (full processing)\n");
		}
		else if(dualMode == 0)
		{
			printf("Raw mode enabled: Raw path only (photometric only)\n");
		}
		else if(dualMode == 2)
		{
			printf("Pipeline mode enabled: Pipeline path only (full processing)\n");
		}
		// Always enable dual mode in CameraReader if dualMode > 0, so it can provide both paths
		// But we'll only use the paths we need based on dualMode
		bool enableDualModeForReader = (dualMode == 1 || dualMode == 0 || dualMode == 2);
		cameraReader = new CameraReader(videoFile, calib, gammaCalib, vignette, enableDualModeForReader, enableCLAHE);
		cameraReader->setGlobalCalibration();
	}
	else if(cameraIndex >= 0)
	{
		// Use camera input
		printf("Initializing camera input...\n");
		if(dualMode == 1)
		{
			printf("Dual mode enabled: Raw path (photometric only) + Pipeline path (full processing)\n");
		}
		else if(dualMode == 0)
		{
			printf("Raw mode enabled: Raw path only (photometric only)\n");
		}
		else if(dualMode == 2)
		{
			printf("Pipeline mode enabled: Pipeline path only (full processing)\n");
		}
		// Always enable dual mode in CameraReader if dualMode > 0, so it can provide both paths
		// But we'll only use the paths we need based on dualMode
		bool enableDualModeForReader = (dualMode == 1 || dualMode == 0 || dualMode == 2);
		cameraReader = new CameraReader(cameraIndex, calib, gammaCalib, vignette, enableDualModeForReader, enableCLAHE);
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
		printf("WARNING: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2\n");
		printf("Continuing without photometric calibration...\n");
		// Don't exit - continue without photometric calibration
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



	FullSystem* fullSystem = nullptr;
	FullSystem* fullSystem_raw = nullptr;
	FullSystem* fullSystem_pipeline = nullptr;
	
	// Create FullSystem instances based on dualMode
	if(cameraReader != nullptr)
	{
		float* gamma = cameraReader->getPhotometricGamma();
		
		if(dualMode == 1)
		{
			// Dual mode: create two FullSystem instances
		fullSystem_raw = new FullSystem();
		fullSystem_raw->setGammaFunction(gamma);
		fullSystem_raw->linearizeOperation = (playbackSpeed==0);
		
		fullSystem_pipeline = new FullSystem();
		fullSystem_pipeline->setGammaFunction(gamma);
		fullSystem_pipeline->linearizeOperation = (playbackSpeed==0);
		
			printf("Created two FullSystem instances for dual mode (both paths)\n");
		}
		else if(dualMode == 0)
		{
			// Raw only mode: create only raw FullSystem
			fullSystem_raw = new FullSystem();
			fullSystem_raw->setGammaFunction(gamma);
			fullSystem_raw->linearizeOperation = (playbackSpeed==0);
			
			printf("Created FullSystem for raw path only\n");
		}
		else if(dualMode == 2)
		{
			// Pipeline only mode: create only pipeline FullSystem
			fullSystem_pipeline = new FullSystem();
			fullSystem_pipeline->setGammaFunction(gamma);
			fullSystem_pipeline->linearizeOperation = (playbackSpeed==0);
			
			printf("Created FullSystem for pipeline path only\n");
		}
	}
	else
	{
		// Single mode: original behavior (for ImageFolderReader)
		fullSystem = new FullSystem();
		float* gamma = reader->getPhotometricGamma();
		fullSystem->setGammaFunction(gamma);
		fullSystem->linearizeOperation = (playbackSpeed==0);
	}







    IOWrap::PangolinDSOViewer* viewer = 0;
    IOWrap::DualPangolinDSOViewer* dualViewer = 0;
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
        
        if(dualMode == 1 && cameraReader != nullptr)
        {
            // Dual mode: create single dual viewer that handles both systems
            printf("Creating dual viewer with dimensions: wG[0]=%d, hG[0]=%d\n", wG[0], hG[0]);
            if(wG[0] <= 0 || hG[0] <= 0)
            {
                printf("ERROR: Invalid global calibration dimensions! wG[0]=%d, hG[0]=%d\n", wG[0], hG[0]);
                printf("Make sure setGlobalCalibration() was called before creating viewer!\n");
                printf("Using default dimensions 640x480 for GUI\n");
                // Use default dimensions instead of disabling
                wG[0] = 640;
                hG[0] = 480;
                setGlobalCalib(wG[0], hG[0], Eigen::Matrix3f::Identity());
            }
            dualViewer = new IOWrap::DualPangolinDSOViewer(wG[0], hG[0], false);
            
            // Create wrapper for raw system - this collects data and forwards to main viewer
            IOWrap::DualPangolinDSOViewer* viewerRawWrapper = new IOWrap::DualPangolinDSOViewer(wG[0], hG[0], false);
            viewerRawWrapper->setSystemType(true);  // true for raw
            viewerRawWrapper->setMainViewer(dualViewer);  // Forward data to main viewer
            fullSystem_raw->outputWrapper.push_back(viewerRawWrapper);
            
            // Create wrapper for pipeline system - this collects data and forwards to main viewer
            IOWrap::DualPangolinDSOViewer* viewerPipelineWrapper = new IOWrap::DualPangolinDSOViewer(wG[0], hG[0], false);
            viewerPipelineWrapper->setSystemType(false);  // false for pipeline
            viewerPipelineWrapper->setMainViewer(dualViewer);  // Forward data to main viewer
            fullSystem_pipeline->outputWrapper.push_back(viewerPipelineWrapper);
            
            printf("Created dual viewer for side-by-side display (raw left, pipeline right)\n");
        }
        else if((dualMode == 0 || dualMode == 2) && cameraReader != nullptr)
        {
            // Single path mode: create single viewer
            printf("Creating single viewer with dimensions: wG[0]=%d, hG[0]=%d\n", wG[0], hG[0]);
            if(wG[0] <= 0 || hG[0] <= 0)
            {
                printf("ERROR: Invalid global calibration dimensions! wG[0]=%d, hG[0]=%d\n", wG[0], hG[0]);
                printf("Make sure setGlobalCalibration() was called before creating viewer!\n");
                printf("Using default dimensions 640x480 for GUI\n");
                wG[0] = 640;
                hG[0] = 480;
                setGlobalCalib(wG[0], hG[0], Eigen::Matrix3f::Identity());
            }
            viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
            
            if(dualMode == 0 && fullSystem_raw != nullptr)
            {
                fullSystem_raw->outputWrapper.push_back(viewer);
                printf("Created single viewer for raw path only\n");
            }
            else if(dualMode == 2 && fullSystem_pipeline != nullptr)
            {
                fullSystem_pipeline->outputWrapper.push_back(viewer);
                printf("Created single viewer for pipeline path only\n");
            }
        }
        else
        {
            // Single mode: create normal viewer
            viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
            fullSystem->outputWrapper.push_back(viewer);
        }
        
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
    
    // Capture dual mode value for lambda
    int dualModeFlag = dualMode;
    
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
    
    // IMPORTANT: Start GUI first to ensure window is open before tracking thread starts
    // This way, even if tracking thread crashes immediately, GUI will be visible
    // For dual mode, we need to start the viewer's run() method which is blocking
    // So we'll start tracking thread first but add a delay, then start GUI
    
    // Create a flag to indicate GUI is ready
    std::atomic<bool> guiReady(false);
    
    std::thread runthread([&, readerPtr, cameraReaderPtr, dualModeFlag]() {
        // Wait a bit for GUI to initialize before starting processing
        // This gives GUI time to open the window
        int waitCount = 0;
        while(!guiReady && waitCount < 100) {
            usleep(100000);  // Wait 100ms
            waitCount++;
        }
        
        if(!guiReady) {
            printf("WARNING: GUI not ready after 10 seconds, starting anyway...\n");
        } else {
            printf("GUI is ready, starting frame processing...\n");
        }
        // Set up thread-specific signal handling
        // Ignore signals in this thread - let them be handled by the main thread
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, SIGSEGV);
        sigaddset(&set, SIGABRT);
        sigaddset(&set, SIGBUS);
        sigaddset(&set, SIGFPE);
        pthread_sigmask(SIG_BLOCK, &set, NULL);
        
        // Set custom terminate handler for this thread
        std::set_terminate([]() {
            printf("\n========================================\n");
            printf("TERMINATE HANDLER: Unhandled exception in tracking thread\n");
            printf("Tracking thread will stop, but GUI will remain open\n");
            printf("========================================\n\n");
            trackingThreadCrashed = true;
            crashMessage = "Unhandled exception - terminate called";
            std::abort();  // This will be caught by signal handler
        });
        
        // Wrap entire tracking thread in try-catch to prevent crashes from closing GUI
        try {
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
        tv_start_global = tv_start;  // Store globally for later use
        clock_t started = clock();
        double sInitializerOffset=0;


        // For camera mode, use continuous loop; for image folder, use fixed list
        int frameIndex = 0;
        frameIndex_global = 0;  // Initialize global counter
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
                frameIndex_global = frameIndex;  // Update global counter
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
                frameIndex_global = frameIndex;  // Update global counter
            }
            
            // Check if we should exit (from signal handler) or if both systems crashed
            if(dualModeFlag == 0 || dualModeFlag == 1 || dualModeFlag == 2) {
                if(shouldExit || (rawSystemCrashed && pipelineSystemCrashed)) {
                    if(shouldExit) {
                        printf("Exit flag set - stopping frame processing but keeping GUI open...\n");
                    } else {
                        printf("Both systems crashed - stopping frame processing but keeping GUI open...\n");
                        printf("Reconstruction has stopped. Pangolin viewer will remain open.\n");
                        printf("Press 'e' to export current data and metrics\n");
                    }
                    // Stop processing frames but keep GUI running
                    break;
                } else if(rawSystemCrashed || pipelineSystemCrashed) {
                    // One system crashed, but continue with the other
                    // Don't break - continue processing
                }
            } else {
                if(shouldExit || trackingThreadCrashed) {
                    if(trackingThreadCrashed) {
                        printf("Crash detected - stopping frame processing but keeping GUI open...\n");
                        printf("Reconstruction has stopped. Pangolin viewer will remain open.\n");
                        printf("Press 'e' to export current data and metrics\n");
                    } else {
                        printf("Exit flag set - stopping frame processing but keeping GUI open...\n");
                    }
                    // Stop processing frames but keep GUI running
                    break;
                }
            }
            
            // Check initialization status (with mutex protection)
        {
                std::lock_guard<std::mutex> lock(fullSystemMutex);
                if(dualModeFlag == 1)
                {
                    if((fullSystem_raw != nullptr && !fullSystem_raw->initialized) ||
                       (fullSystem_pipeline != nullptr && !fullSystem_pipeline->initialized))
                    {
                        gettimeofday(&tv_start, NULL);
                        started = clock();
                        sInitializerOffset = 0.0;
                    }
                }
                else if((dualModeFlag == 0 && fullSystem_raw != nullptr && !fullSystem_raw->initialized) ||
                        (dualModeFlag == 2 && fullSystem_pipeline != nullptr && !fullSystem_pipeline->initialized) ||
                        (fullSystem != nullptr && !fullSystem->initialized))	// if not initialized: reset start time.
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
            ImageAndExposure* img_raw = nullptr;
            ImageAndExposure* img_pipeline = nullptr;
            
            if((dualModeFlag == 0 || dualModeFlag == 1 || dualModeFlag == 2) && cameraReaderPtr != nullptr)
            {
                // Get images based on dualMode
                if(dualModeFlag == 0)
                {
                    // Raw only mode: get raw image
                    img_raw = cameraReaderPtr->getImageRaw(i);
                    img_pipeline = nullptr;
                }
                else if(dualModeFlag == 1)
            {
                // Dual mode: get both raw and pipeline images
                img_raw = cameraReaderPtr->getImageRaw(i);
                img_pipeline = cameraReaderPtr->getImagePipeline(i);
                }
                else if(dualModeFlag == 2)
                {
                    // Pipeline only mode: get pipeline image
                    img_raw = nullptr;
                    img_pipeline = cameraReaderPtr->getImagePipeline(i);
                }
                
                // Record original BGR frame for video export (camera mode only, if saveVideo is enabled)
                // Note: In dual mode, getImageRaw() and getImagePipeline() both call captureFrame(),
                // so lastCapturedFrame will be updated twice. We record after both calls to get
                // the frame that was captured (which will be from getImagePipeline, the last one).
                // Both paths should use the same original frame, but due to timing they may differ.
                // For video recording, we record the frame that was actually captured.
                if(saveVideo && cameraReaderPtr != nullptr && !cameraReaderPtr->isVideoFile)
                {
                    cv::Mat originalFrame = cameraReaderPtr->getOriginalBGRFrame();
                    if(!originalFrame.empty())
                    {
                        std::lock_guard<std::mutex> framesLock(framesMutex);
                        capturedFrames.push_back(originalFrame.clone());
                    }
                    else
                    {
                        // Debug: check if lastCapturedFrame is empty
                        printf("DEBUG: getOriginalBGRFrame() returned empty frame at iteration %d\n", i);
                    }
                }
                
                // Validate images based on mode
                if(dualModeFlag == 0 && img_raw == nullptr)
                {
                    printf("WARNING: Failed to get raw image for frame %d, skipping.\n", i);
                    usleep(33000);
                    continue;
                }
                else if(dualModeFlag == 1 && (img_raw == nullptr || img_pipeline == nullptr))
                {
                    printf("WARNING: Failed to get images for frame %d, skipping.\n", i);
                    if(img_raw != nullptr) delete img_raw;
                    if(img_pipeline != nullptr) delete img_pipeline;
                    usleep(33000);
                    continue;
                }
                else if(dualModeFlag == 2 && img_pipeline == nullptr)
                {
                    printf("WARNING: Failed to get pipeline image for frame %d, skipping.\n", i);
                    usleep(33000);
                    continue;
                }
            }
            else
            {
                // Single mode: original behavior (for ImageFolderReader)
                if(preload && cameraReaderPtr == nullptr && frameIndex-1 < (int)preloadedImages.size())
                {
                    img = preloadedImages[frameIndex-1];
                }
                else
                {
                    if(cameraReaderPtr != nullptr)
                    {
                        img = cameraReaderPtr->getImage(i);
                        
                        // Record original BGR frame for video export (camera mode only, if saveVideo is enabled)
                        if(saveVideo && !cameraReaderPtr->isVideoFile)
                        {
                            cv::Mat originalFrame = cameraReaderPtr->getOriginalBGRFrame();
                            if(!originalFrame.empty())
                            {
                                std::lock_guard<std::mutex> framesLock(framesMutex);
                                capturedFrames.push_back(originalFrame.clone());
                            }
                        }
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



            if((dualModeFlag == 0 || dualModeFlag == 1 || dualModeFlag == 2) && cameraReaderPtr != nullptr)
            {
                // Check if we should stop before processing frames
                if(stopProcessing || shouldExit)
                {
                    if(img_raw != nullptr) {
                        try { delete img_raw; } catch(...) {}
                        img_raw = nullptr;
                    }
                    if(img_pipeline != nullptr) {
                        try { delete img_pipeline; } catch(...) {}
                        img_pipeline = nullptr;
                    }
                    printf("Stop signal received - exiting frame processing loop\n");
                    break;
                }
                
                // Process images based on mode
                if(dualModeFlag == 0)
                {
                    // Raw only mode: process raw image
                    if(!skipFrame && img_raw != nullptr)
                    {
                        if(resetting)
                        {
                            delete img_raw;
                            img_raw = nullptr;
                            usleep(10000);
                            continue;
                        }
                        
                        std::lock_guard<std::mutex> lock(fullSystemMutex);
                        
                        if(resetting || fullSystem_raw == nullptr)
                        {
                            delete img_raw;
                            img_raw = nullptr;
                            continue;
                        }
                        
                        // Validate image before processing
                        if(img_raw == nullptr || img_raw->image == nullptr)
                        {
                            printf("WARNING: Invalid raw image data, skipping frame %d\n", i);
                            if(img_raw != nullptr) {
                                try { delete img_raw; } catch(...) {}
                                img_raw = nullptr;
                            }
                            continue;
                        }
                        
                        // Check if we should stop due to crash or exit
                        if(rawSystemCrashed || shouldExit) {
                            if(shouldExit) {
                                printf("Exit requested - stopping frame processing\n");
                            } else {
                                printf("Raw system has crashed - stopping frame processing\n");
                                printf("Reconstruction has stopped. Pangolin viewer will remain open for inspection.\n");
                            }
                            if(img_raw != nullptr) {
                                try { delete img_raw; } catch(...) {}
                                img_raw = nullptr;
                            }
                            break;
                        }
                        
                        // Process raw path
                        try {
                            if(fullSystem_raw != nullptr && !rawSystemCrashed && !shouldExit) {
                                fullSystem_raw->addActiveFrame(img_raw, i);
                                img_raw = nullptr;  // Transfer ownership, don't delete
                            } else {
                                if(img_raw != nullptr) {
                                    try { delete img_raw; } catch(...) {}
                                    img_raw = nullptr;
                                }
                                if(shouldExit) break;
                            }
                        } catch(const std::exception& e) {
                            printf("ERROR: Exception in addActiveFrame (raw): %s\n", e.what());
                            rawSystemCrashed = true;
                            rawCrashMessage = std::string("Exception in raw path: ") + e.what();
                            if(img_raw != nullptr) {
                                try { delete img_raw; } catch(...) {}
                                img_raw = nullptr;
                            }
                            break;
                        } catch(...) {
                            printf("ERROR: Unknown exception in addActiveFrame (raw)\n");
                            rawSystemCrashed = true;
                            rawCrashMessage = "Unknown exception in raw path";
                            if(img_raw != nullptr) {
                                try { delete img_raw; } catch(...) {}
                                img_raw = nullptr;
                            }
                            break;
                        }
                    }
                }
                else if(dualModeFlag == 2)
                {
                    // Pipeline only mode: process pipeline image
                    if(!skipFrame && img_pipeline != nullptr)
                    {
                        if(resetting)
                        {
                            delete img_pipeline;
                            img_pipeline = nullptr;
                            usleep(10000);
                            continue;
                        }
                        
                        std::lock_guard<std::mutex> lock(fullSystemMutex);
                        
                        if(resetting || fullSystem_pipeline == nullptr)
                        {
                            delete img_pipeline;
                            img_pipeline = nullptr;
                            continue;
                        }
                        
                        // Validate image before processing
                        if(img_pipeline == nullptr || img_pipeline->image == nullptr)
                        {
                            printf("WARNING: Invalid pipeline image data, skipping frame %d\n", i);
                            if(img_pipeline != nullptr) {
                                try { delete img_pipeline; } catch(...) {}
                                img_pipeline = nullptr;
                            }
                            continue;
                        }
                        
                        // Check if we should stop due to crash or exit
                        if(pipelineSystemCrashed || shouldExit) {
                            if(shouldExit) {
                                printf("Exit requested - stopping frame processing\n");
                            } else {
                                printf("Pipeline system has crashed - stopping frame processing\n");
                                printf("Reconstruction has stopped. Pangolin viewer will remain open for inspection.\n");
                            }
                            if(img_pipeline != nullptr) {
                                try { delete img_pipeline; } catch(...) {}
                                img_pipeline = nullptr;
                            }
                            break;
                        }
                        
                        // Process pipeline path
                        try {
                            if(fullSystem_pipeline != nullptr && !pipelineSystemCrashed && !shouldExit) {
                                fullSystem_pipeline->addActiveFrame(img_pipeline, i);
                                img_pipeline = nullptr;  // Transfer ownership, don't delete
                            } else {
                                if(img_pipeline != nullptr) {
                                    try { delete img_pipeline; } catch(...) {}
                                    img_pipeline = nullptr;
                                }
                                if(shouldExit) break;
                            }
                        } catch(const std::exception& e) {
                            printf("ERROR: Exception in addActiveFrame (pipeline): %s\n", e.what());
                            pipelineSystemCrashed = true;
                            pipelineCrashMessage = std::string("Exception in pipeline path: ") + e.what();
                            if(img_pipeline != nullptr) {
                                try { delete img_pipeline; } catch(...) {}
                                img_pipeline = nullptr;
                            }
                            break;
                        } catch(...) {
                            printf("ERROR: Unknown exception in addActiveFrame (pipeline)\n");
                            pipelineSystemCrashed = true;
                            pipelineCrashMessage = "Unknown exception in pipeline path";
                            if(img_pipeline != nullptr) {
                                try { delete img_pipeline; } catch(...) {}
                                img_pipeline = nullptr;
                            }
                            break;
                        }
                    }
                }
                else if(dualModeFlag == 1)
            {
                // Dual mode: process both images
                if(!skipFrame && img_raw != nullptr && img_pipeline != nullptr)
                {
                    if(resetting)
                    {
                        delete img_raw;
                        delete img_pipeline;
                        img_raw = nullptr;
                        img_pipeline = nullptr;
                        usleep(10000);
                        continue;
                    }
                    
                    std::lock_guard<std::mutex> lock(fullSystemMutex);
                    
                    if(resetting || fullSystem_raw == nullptr || fullSystem_pipeline == nullptr)
                    {
                        delete img_raw;
                        delete img_pipeline;
                        img_raw = nullptr;
                        img_pipeline = nullptr;
                        continue;
                    }
                    
                        // Validate images before processing
                        if(img_raw == nullptr || img_pipeline == nullptr || 
                           img_raw->image == nullptr || img_pipeline->image == nullptr)
                        {
                            printf("WARNING: Invalid image data, skipping frame %d\n", i);
                            if(img_raw != nullptr) {
                                try { delete img_raw; } catch(...) {}
                                img_raw = nullptr;
                            }
                            if(img_pipeline != nullptr) {
                                try { delete img_pipeline; } catch(...) {}
                                img_pipeline = nullptr;
                            }
                            continue;
                        }
                        
                        // Check if we should stop due to crash (both systems) or exit
                        if((rawSystemCrashed && pipelineSystemCrashed) || shouldExit) {
                            if(shouldExit) {
                                printf("Exit requested - stopping frame processing\n");
                            } else {
                                printf("Both systems have crashed - stopping frame processing\n");
                                printf("Reconstruction has stopped. Pangolin viewer will remain open for inspection.\n");
                            }
                            if(img_raw != nullptr) {
                                try { delete img_raw; } catch(...) {}
                                img_raw = nullptr;
                            }
                            if(img_pipeline != nullptr) {
                                try { delete img_pipeline; } catch(...) {}
                                img_pipeline = nullptr;
                            }
                            break;  // Exit the loop, but keep thread alive
                        }
                        // If only one system crashed, continue with the other
                        if(rawSystemCrashed && !pipelineSystemCrashed) {
                            // Skip raw processing, continue with pipeline
                            if(img_raw != nullptr) {
                                try { delete img_raw; } catch(...) {}
                                img_raw = nullptr;
                            }
                        } else if(pipelineSystemCrashed && !rawSystemCrashed) {
                            // Skip pipeline processing, continue with raw
                            if(img_pipeline != nullptr) {
                                try { delete img_pipeline; } catch(...) {}
                                img_pipeline = nullptr;
                            }
                        }
                        
                        // Process raw path with comprehensive error handling
                        try {
                            if(fullSystem_raw != nullptr && !rawSystemCrashed && !shouldExit) {
                        fullSystem_raw->addActiveFrame(img_raw, i);
                                img_raw = nullptr;  // Transfer ownership, don't delete
                            } else {
                                // System is null or crashed, clean up
                                if(img_raw != nullptr) {
                                    try { delete img_raw; } catch(...) {}
                                    img_raw = nullptr;
                                }
                                // If raw crashed but pipeline is still running, continue with pipeline only
                                if(rawSystemCrashed && !pipelineSystemCrashed) {
                                    // Continue processing pipeline path
                                } else if(shouldExit) {
                                    // Exit requested, stop both
                                    break;
                                }
                            }
                    } catch(const std::exception& e) {
                            printf("ERROR: Exception in addActiveFrame (raw): %s\n", e.what());
                            rawSystemCrashed = true;
                            rawCrashMessage = std::string("Exception in raw path: ") + e.what();
                            printf("Raw system has crashed, but pipeline will continue if possible\n");
                            if(img_raw != nullptr) {
                                try { delete img_raw; } catch(...) {}
                        img_raw = nullptr;
                            }
                            // Don't break - let pipeline continue if it's still working
                            // Only break if both systems crashed or exit requested
                            if(pipelineSystemCrashed || shouldExit) {
                                break;
                            }
                        } catch(...) {
                            printf("ERROR: Unknown exception in addActiveFrame (raw)\n");
                            rawSystemCrashed = true;
                            rawCrashMessage = "Unknown exception in raw path";
                            printf("Raw system has crashed, but pipeline will continue if possible\n");
                            if(img_raw != nullptr) {
                                try { delete img_raw; } catch(...) {}
                                img_raw = nullptr;
                            }
                            if(pipelineSystemCrashed || shouldExit) {
                                break;
                            }
                        }
                        
                        // Process pipeline path with comprehensive error handling
                        try {
                            if(fullSystem_pipeline != nullptr && !pipelineSystemCrashed && !shouldExit) {
                                fullSystem_pipeline->addActiveFrame(img_pipeline, i);
                                img_pipeline = nullptr;  // Transfer ownership, don't delete
                            } else {
                                // System is null or crashed, clean up
                                if(img_pipeline != nullptr) {
                                    try { delete img_pipeline; } catch(...) {}
                        img_pipeline = nullptr;
                                }
                                // If pipeline crashed but raw is still running, continue with raw only
                                if(pipelineSystemCrashed && !rawSystemCrashed) {
                                    // Continue processing raw path
                                } else if(shouldExit) {
                                    // Exit requested, stop both
                                    break;
                                }
                            }
                        } catch(const std::exception& e) {
                            printf("ERROR: Exception in addActiveFrame (pipeline): %s\n", e.what());
                            pipelineSystemCrashed = true;
                            pipelineCrashMessage = std::string("Exception in pipeline path: ") + e.what();
                            printf("Pipeline system has crashed, but raw will continue if possible\n");
                            if(img_pipeline != nullptr) {
                                try { delete img_pipeline; } catch(...) {}
                                img_pipeline = nullptr;
                            }
                            // Don't break - let raw continue if it's still working
                            // Only break if both systems crashed or exit requested
                            if(rawSystemCrashed || shouldExit) {
                                break;
                            }
                    } catch(...) {
                            printf("ERROR: Unknown exception in addActiveFrame (pipeline)\n");
                            pipelineSystemCrashed = true;
                            pipelineCrashMessage = "Unknown exception in pipeline path";
                            printf("Pipeline system has crashed, but raw will continue if possible\n");
                            if(img_pipeline != nullptr) {
                                try { delete img_pipeline; } catch(...) {}
                                img_pipeline = nullptr;
                            }
                            // Don't break - let raw continue if it's still working
                            // Only break if both systems crashed or exit requested
                            if(rawSystemCrashed || shouldExit) {
                                break;
                            }
                        }
                        
                        // Clean up any remaining images (shouldn't happen if both succeeded)
                        if(img_raw != nullptr) {
                            try { delete img_raw; } catch(...) {}
                        img_raw = nullptr;
                        }
                        if(img_pipeline != nullptr) {
                            try { delete img_pipeline; } catch(...) {}
                        img_pipeline = nullptr;
                        }
                    }
                }
            }
            else if(!skipFrame && img != nullptr) 
            {
                // Single mode: original behavior
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
                
                // Validate image before processing
                if(img == nullptr || img->image == nullptr)
                {
                    printf("WARNING: Invalid image data, skipping frame %d\n", i);
                    if(img != nullptr) {
                        try { delete img; } catch(...) {}
                        img = nullptr;
                    }
                    continue;
                }
                
                // Check crash flag before processing
                if(trackingThreadCrashed || shouldExit) {
                    if(img != nullptr) {
                        try { delete img; } catch(...) {}
                        img = nullptr;
                    }
                    break;  // Exit loop immediately
                }
                
                try {
                    fullSystem->addActiveFrame(img, i);
                    img = nullptr;  // Transfer ownership, don't delete
                } catch(const std::exception& e) {
                    printf("ERROR: Exception in addActiveFrame: %s\n", e.what());
                    trackingThreadCrashed = true;
                    crashMessage = std::string("Exception in addActiveFrame: ") + e.what();
                    if(img != nullptr) {
                        try { delete img; } catch(...) {}
                    img = nullptr;
                    }
                    break;  // Exit loop on crash
                } catch(...) {
                    printf("ERROR: Unknown exception in addActiveFrame\n");
                    trackingThreadCrashed = true;
                    crashMessage = "Unknown exception in addActiveFrame";
                    if(img != nullptr) {
                        try { delete img; } catch(...) {}
                    img = nullptr;
                    }
                    break;  // Exit loop on crash
                }
            }




            if(img != nullptr)
            {
                delete img;
                img = nullptr;
            }
            if(img_raw != nullptr)
            {
                delete img_raw;
                img_raw = nullptr;
            }
            if(img_pipeline != nullptr)
            {
                delete img_pipeline;
                img_pipeline = nullptr;
            }

            // Check if reset is needed (with mutex protection)
            bool needReset = false;
            {
                std::lock_guard<std::mutex> lock(fullSystemMutex);
                if(dualModeFlag == 1)
                {
                    if((fullSystem_raw != nullptr && fullSystem_raw->initFailed) ||
                       (fullSystem_pipeline != nullptr && fullSystem_pipeline->initFailed) ||
                       setting_fullResetRequested)
                    {
                        needReset = true;
                    }
                }
                else if(dualModeFlag == 0 && fullSystem_raw != nullptr && (fullSystem_raw->initFailed || setting_fullResetRequested))
                {
                    needReset = true;
                }
                else if(dualModeFlag == 2 && fullSystem_pipeline != nullptr && (fullSystem_pipeline->initFailed || setting_fullResetRequested))
                {
                    needReset = true;
                }
                else if(fullSystem != nullptr && (fullSystem->initFailed || setting_fullResetRequested))
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
                    
                    if(dualModeFlag == 1)
                    {
                        // Dual mode reset
                        if(fullSystem_raw == nullptr || fullSystem_pipeline == nullptr)
                        {
                            resetting = false;
                            continue;
                        }
                        
                        // Block until mapping is finished
                        try {
                            fullSystem_raw->blockUntilMappingIsFinished();
                            fullSystem_pipeline->blockUntilMappingIsFinished();
                        } catch(const std::exception& e) {
                            printf("WARNING: Exception in blockUntilMappingIsFinished during reset: %s\n", e.what());
                        } catch(...) {
                            printf("WARNING: Unknown exception in blockUntilMappingIsFinished during reset\n");
                        }

                        // Save output wrappers
                        std::vector<IOWrap::Output3DWrapper*> wrapsRaw = fullSystem_raw->outputWrapper;
                        std::vector<IOWrap::Output3DWrapper*> wrapsPipeline = fullSystem_pipeline->outputWrapper;
                        
                        // Delete old fullSystems
                        try {
                            delete fullSystem_raw;
                            fullSystem_raw = nullptr;
                        } catch(...) {
                            fullSystem_raw = nullptr;
                        }
                        
                        try {
                            delete fullSystem_pipeline;
                            fullSystem_pipeline = nullptr;
                        } catch(...) {
                            fullSystem_pipeline = nullptr;
                        }

                        // Reset output wrappers
                        for(IOWrap::Output3DWrapper* ow : wrapsRaw) {
                            if(ow != nullptr) {
                                try { ow->reset(); } catch(...) {}
                            }
                        }
                        for(IOWrap::Output3DWrapper* ow : wrapsPipeline) {
                            if(ow != nullptr) {
                                try { ow->reset(); } catch(...) {}
                            }
                        }

                        // Create new fullSystems
                        try {
                            float* gamma = cameraReaderPtr != nullptr ? cameraReaderPtr->getPhotometricGamma() : nullptr;
                            
                            fullSystem_raw = new FullSystem();
                            fullSystem_raw->setGammaFunction(gamma);
                            fullSystem_raw->linearizeOperation = (playbackSpeed==0);
                            fullSystem_raw->outputWrapper = wrapsRaw;
                            
                            fullSystem_pipeline = new FullSystem();
                            fullSystem_pipeline->setGammaFunction(gamma);
                            fullSystem_pipeline->linearizeOperation = (playbackSpeed==0);
                            fullSystem_pipeline->outputWrapper = wrapsPipeline;
                        } catch(...) {
                            printf("ERROR: Exception creating new fullSystems during reset!\n");
                            resetting = false;
                            break;
                        }
                    }
                    else if(dualModeFlag == 0)
                    {
                        // Raw only mode reset
                        if(fullSystem_raw == nullptr)
                        {
                            resetting = false;
                            continue;
                        }
                        
                        // Block until mapping is finished
                        try {
                            fullSystem_raw->blockUntilMappingIsFinished();
                        } catch(const std::exception& e) {
                            printf("WARNING: Exception in blockUntilMappingIsFinished during reset: %s\n", e.what());
                        } catch(...) {
                            printf("WARNING: Unknown exception in blockUntilMappingIsFinished during reset\n");
                        }

                        // Save output wrappers
                        std::vector<IOWrap::Output3DWrapper*> wrapsRaw = fullSystem_raw->outputWrapper;
                        
                        // Delete old fullSystem
                        try {
                            delete fullSystem_raw;
                            fullSystem_raw = nullptr;
                        } catch(...) {
                            fullSystem_raw = nullptr;
                        }

                        // Reset output wrappers
                        for(IOWrap::Output3DWrapper* ow : wrapsRaw) {
                            if(ow != nullptr) {
                                try { ow->reset(); } catch(...) {}
                            }
                        }

                        // Create new fullSystem
                        try {
                            float* gamma = cameraReaderPtr != nullptr ? cameraReaderPtr->getPhotometricGamma() : nullptr;
                            
                            fullSystem_raw = new FullSystem();
                            fullSystem_raw->setGammaFunction(gamma);
                            fullSystem_raw->linearizeOperation = (playbackSpeed==0);
                            fullSystem_raw->outputWrapper = wrapsRaw;
                        } catch(...) {
                            printf("ERROR: Exception creating new fullSystem_raw during reset!\n");
                            resetting = false;
                            break;
                        }
                    }
                    else if(dualModeFlag == 2)
                    {
                        // Pipeline only mode reset
                        if(fullSystem_pipeline == nullptr)
                        {
                            resetting = false;
                            continue;
                        }
                        
                        // Block until mapping is finished
                        try {
                            fullSystem_pipeline->blockUntilMappingIsFinished();
                        } catch(const std::exception& e) {
                            printf("WARNING: Exception in blockUntilMappingIsFinished during reset: %s\n", e.what());
                        } catch(...) {
                            printf("WARNING: Unknown exception in blockUntilMappingIsFinished during reset\n");
                        }

                        // Save output wrappers
                        std::vector<IOWrap::Output3DWrapper*> wrapsPipeline = fullSystem_pipeline->outputWrapper;
                        
                        // Delete old fullSystem
                        try {
                            delete fullSystem_pipeline;
                            fullSystem_pipeline = nullptr;
                        } catch(...) {
                            fullSystem_pipeline = nullptr;
                        }

                        // Reset output wrappers
                        for(IOWrap::Output3DWrapper* ow : wrapsPipeline) {
                            if(ow != nullptr) {
                                try { ow->reset(); } catch(...) {}
                            }
                        }

                        // Create new fullSystem
                        try {
                            float* gamma = cameraReaderPtr != nullptr ? cameraReaderPtr->getPhotometricGamma() : nullptr;
                            
                            fullSystem_pipeline = new FullSystem();
                            fullSystem_pipeline->setGammaFunction(gamma);
                            fullSystem_pipeline->linearizeOperation = (playbackSpeed==0);
                            fullSystem_pipeline->outputWrapper = wrapsPipeline;
                        } catch(...) {
                            printf("ERROR: Exception creating new fullSystem_pipeline during reset!\n");
                            resetting = false;
                            break;
                        }
                    }
                    else
                    {
                        // Single mode reset (original behavior for ImageFolderReader)
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
                    }

                    setting_fullResetRequested=false;
                    resetting = false;
                    printf("RESET COMPLETE!\n");
                }
                else
                {
                    // If reset is needed but we're past frame 250, just mark as lost
                    std::lock_guard<std::mutex> lock(fullSystemMutex);
                    if(dualModeFlag == 1)
                    {
                        if(fullSystem_raw != nullptr)
                        {
                            fullSystem_raw->isLost = true;
                        }
                        if(fullSystem_pipeline != nullptr)
                        {
                            fullSystem_pipeline->isLost = true;
                        }
                    }
                    else if(dualModeFlag == 0 && fullSystem_raw != nullptr)
                    {
                        fullSystem_raw->isLost = true;
                    }
                    else if(dualModeFlag == 2 && fullSystem_pipeline != nullptr)
                    {
                        fullSystem_pipeline->isLost = true;
                    }
                    else if(fullSystem != nullptr)
                    {
                        fullSystem->isLost = true;
                    }
                }
            }

            // Check if system is lost (with mutex protection)
            {
                std::lock_guard<std::mutex> lock(fullSystemMutex);
                if(dualModeFlag == 1)
                {
                    if((fullSystem_raw != nullptr && fullSystem_raw->isLost) ||
                       (fullSystem_pipeline != nullptr && fullSystem_pipeline->isLost))
                    {
                        printf("LOST!!\n");
                        break;
                    }
                }
                else if(fullSystem != nullptr && fullSystem->isLost)
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
        
        } catch(const std::exception& e) {
            printf("\n========================================\n");
            printf("EXCEPTION in tracking thread: %s\n", e.what());
            printf("Tracking thread has stopped, but GUI will remain open\n");
            printf("You can still inspect the current state in the viewer\n");
            printf("========================================\n\n");
            trackingThreadCrashed = true;
            crashMessage = std::string("Exception: ") + e.what();
        } catch(...) {
            printf("\n========================================\n");
            printf("UNKNOWN EXCEPTION in tracking thread\n");
            printf("Tracking thread has stopped, but GUI will remain open\n");
            printf("You can still inspect the current state in the viewer\n");
            printf("========================================\n\n");
            trackingThreadCrashed = true;
            crashMessage = "Unknown exception in tracking thread";
        }
        
        printf("Tracking thread has finished (normal or error)\n");

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
                    shouldExit = true;
                    exportRequested = true;  // Set flag to trigger export
                    printf("\n>>> 'e' pressed! Stopping processing and saving files...\n");
                    printf(">>> Setting stop flags - tracking thread will exit gracefully...\n");
                    // Give tracking thread a moment to see the stop flag
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    // Close viewer to exit GUI loop (for camera mode)
                    if(cameraReaderPtr != nullptr)
                    {
                        if(dualModeFlag == 1)
                        {
                            printf(">>> Closing dual viewer...\n");
                            if(dualViewer != nullptr)
                            {
                                dualViewer->close();
                            }
                        }
                        else if(viewer != nullptr)
                        {
                            printf(">>> Closing viewer...\n");
                            viewer->close();
                        }
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

    if(dualMode == 1 && dualViewer != 0)
    {
        // Dual mode: run dual viewer
        // Start tracking thread first (it runs in parallel)
        // Note: runthread is already started above with std::thread constructor
        
        // Run GUI on main thread (blocking call - this is the main event loop)
        // This ensures all GUI operations are on main thread for macOS
        // The GUI will continue running until window is closed
        // Even if tracking thread crashes, GUI will remain open
        try {
            // Check if tracking thread crashed and display message
            if(trackingThreadCrashed) {
                printf("\n========================================\n");
                printf("WARNING: Tracking thread has crashed!\n");
                printf("Error: %s\n", crashMessage.c_str());
                printf("GUI will remain open for inspection\n");
                printf("You can view the current reconstruction state\n");
                printf("Close the window when done\n");
                printf("========================================\n\n");
            }
            
            // Run GUI on main thread (REQUIRED for macOS)
            dualViewer->run();
        } catch(const std::exception& e) {
            printf("ERROR: Exception in dual viewer: %s\n", e.what());
            printf("Keeping viewer window open for inspection...\n");
            printf("Press any key or close window to continue...\n");
            // Keep window open - wait for user to close manually
            // The viewer's run loop will handle window closing
            usleep(1000000);  // Wait 1 second to allow user to see the error
        } catch(...) {
            printf("ERROR: Unknown exception in dual viewer\n");
            printf("Keeping viewer window open for inspection...\n");
            printf("Press any key or close window to continue...\n");
            // Keep window open - wait for user to close manually
            usleep(1000000);  // Wait 1 second to allow user to see the error
        }

        // After GUI exits, wait for tracking thread to finish
        // For camera mode with stopProcessing, give thread time to exit gracefully
        if(runthread.joinable())
        {
            if(stopProcessing || shouldExit) {
                // User requested stop - give tracking thread a moment to exit gracefully
                printf("Waiting for tracking thread to finish (stop requested)...\n");
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if(runthread.joinable()) {
                    // Thread should have exited by now, but if not, detach to continue
                    printf("Tracking thread still running, detaching to continue export...\n");
                    runthread.detach();
                }
            } else if(trackingThreadCrashed) {
                // Thread crashed - detach immediately
                printf("Tracking thread has crashed, detaching...\n");
                runthread.detach();
            } else {
                // Normal case - wait for thread to finish
                printf("Waiting for tracking thread to finish...\n");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if(runthread.joinable()) {
                    runthread.detach();
                }
            }
        }
    }
    else if(viewer != nullptr)
    {
        printf("Starting single viewer GUI...\n");
        // Single mode: run normal viewer
        // Note: runthread is already started above, but it will wait for guiReady
        
        // Run GUI on main thread (blocking call - this is the main event loop)
        // This ensures all GUI operations are on main thread for macOS
        // The GUI will continue running until window is closed
        // Even if tracking thread crashes, GUI will remain open
        try {
            // Check if tracking thread crashed and display message
            if(trackingThreadCrashed) {
                printf("\n========================================\n");
                printf("WARNING: Tracking thread has crashed!\n");
                printf("Error: %s\n", crashMessage.c_str());
                printf("GUI will remain open for inspection\n");
                printf("You can view the current reconstruction state\n");
                printf("Close the window when done\n");
                printf("========================================\n\n");
            }
            
            printf("Launching Pangolin GUI window...\n");
            guiReady = true;  // Signal that we're about to start GUI
            
            // Small delay to ensure signal is received by tracking thread
            usleep(50000);  // 50ms
            
            viewer->run();
            printf("GUI window closed (viewer->run() returned).\n");
        } catch(const std::exception& e) {
            printf("ERROR: Exception in viewer: %s\n", e.what());
            printf("Keeping viewer window open for inspection...\n");
            printf("Press any key or close window to continue...\n");
            // The viewer's run loop will handle window closing
            usleep(1000000);  // Wait 1 second to allow user to see the error
        } catch(...) {
            printf("ERROR: Unknown exception in viewer\n");
            printf("Keeping viewer window open for inspection...\n");
            printf("Press any key or close window to continue...\n");
            // The viewer's run loop will handle window closing
            usleep(1000000);  // Wait 1 second to allow user to see the error
        }

        // After GUI exits, wait for tracking thread to finish
        // For camera mode with stopProcessing, give thread time to exit gracefully
        if(runthread.joinable())
        {
            if(stopProcessing || shouldExit) {
                // User requested stop - give tracking thread a moment to exit gracefully
                printf("Waiting for tracking thread to finish (stop requested)...\n");
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if(runthread.joinable()) {
                    // Thread should have exited by now, but if not, detach to continue
                    printf("Tracking thread still running, detaching to continue export...\n");
                    runthread.detach();
                }
            } else if(trackingThreadCrashed) {
                // Thread crashed - detach immediately
                printf("Tracking thread has crashed, detaching...\n");
                runthread.detach();
            } else {
                // Normal case - wait for thread to finish
            runthread.join();
            }
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

	// Calculate total processing time
	struct timeval tv_end;
	gettimeofday(&tv_end, NULL);
	double totalTime = (tv_end.tv_sec - tv_start_global.tv_sec) + (tv_end.tv_usec - tv_start_global.tv_usec) / 1000000.0;
	int totalFramesProcessed = frameIndex_global;

	// Export data before cleanup
	printf("\n==================== EXPORTING DATA ====================\n");
	printf("Saving point cloud, camera poses, and video...\n");
	
	// Get camera FPS for video export (if using camera)
	double cameraFPS = 30.0;
	if(cameraReaderPtr != nullptr && !cameraReaderPtr->isVideoFile)
	{
		cameraFPS = cameraReaderPtr->getFPS();
		printf("Using camera FPS: %.2f for video export\n", cameraFPS);
	}
	
	// Save complete recorded video if in camera mode and saveVideo is enabled
	if(saveVideo && cameraReaderPtr != nullptr && !cameraReaderPtr->isVideoFile)
	{
		std::lock_guard<std::mutex> framesLock(framesMutex);
		if(!capturedFrames.empty())
		{
			printf("\nSaving complete recorded video from camera...\n");
			std::string videoOutputPath = "dso_output/recorded_camera_video.mp4";
			try {
				IOWrap::DataExporter::exportVideo(capturedFrames, videoOutputPath, cameraFPS);
				printf("Complete camera video saved to: %s (%zu frames, %.2f fps)\n", 
				       videoOutputPath.c_str(), capturedFrames.size(), cameraFPS);
			} catch(const std::exception& e) {
				printf("ERROR: Exception saving camera video: %s\n", e.what());
			} catch(...) {
				printf("ERROR: Unknown exception saving camera video\n");
			}
		} else {
			printf("WARNING: No frames were recorded for video export\n");
		}
	}
	else if(cameraReaderPtr != nullptr && !cameraReaderPtr->isVideoFile && !saveVideo)
	{
		printf("Video recording disabled (save_video=0). Skipping video export.\n");
	}
	
	{
		std::lock_guard<std::mutex> lock(fullSystemMutex);
		if(dualMode == 1 && fullSystem_raw != nullptr && fullSystem_pipeline != nullptr)
		{
			// Dual mode: export both results separately
			std::string outputDirRaw = "dso_output/raw";
			std::string outputDirPipeline = "dso_output/pipeline";
			
			try {
				// Export raw even if it crashed - save what we have
				if(rawSystemCrashed) {
					printf("WARNING: Raw system crashed, but exporting available data...\n");
				}
				printf("Exporting raw reconstruction to: %s\n", outputDirRaw.c_str());
				if(fullSystem_raw != nullptr) {
					if(dualViewer != nullptr) {
						IOWrap::DataExporter::exportAllDual(fullSystem_raw, dualViewer, capturedFrames, outputDirRaw, cameraFPS, true);
					} else {
						IOWrap::DataExporter::exportAll(fullSystem_raw, nullptr, capturedFrames, outputDirRaw, cameraFPS);
					}
				printf("Raw reconstruction export completed successfully!\n");
					
					// Export quantitative metrics for raw
					printf("Exporting quantitative metrics for raw reconstruction...\n");
					std::string metricsFileRaw = outputDirRaw + "/quantitative_metrics.txt";
					IOWrap::DataExporter::exportQuantitativeMetrics(fullSystem_raw, dualViewer, metricsFileRaw, true, totalFramesProcessed, totalTime);
				} else {
					printf("WARNING: fullSystem_raw is null, skipping raw export\n");
				}
			} catch(const std::exception& e) {
				printf("ERROR: Exception during raw data export: %s\n", e.what());
			} catch(...) {
				printf("ERROR: Unknown exception during raw data export\n");
			}
			
			try {
				// Export pipeline even if it crashed - save what we have
				if(pipelineSystemCrashed) {
					printf("WARNING: Pipeline system crashed, but exporting available data...\n");
				}
				printf("Exporting pipeline reconstruction to: %s\n", outputDirPipeline.c_str());
				if(fullSystem_pipeline != nullptr) {
					if(dualViewer != nullptr) {
						IOWrap::DataExporter::exportAllDual(fullSystem_pipeline, dualViewer, capturedFrames, outputDirPipeline, cameraFPS, false);
					} else {
						IOWrap::DataExporter::exportAll(fullSystem_pipeline, nullptr, capturedFrames, outputDirPipeline, cameraFPS);
					}
				printf("Pipeline reconstruction export completed successfully!\n");
					
					// Export quantitative metrics for pipeline
					printf("Exporting quantitative metrics for pipeline reconstruction...\n");
					std::string metricsFilePipeline = outputDirPipeline + "/quantitative_metrics.txt";
					IOWrap::DataExporter::exportQuantitativeMetrics(fullSystem_pipeline, dualViewer, metricsFilePipeline, false, totalFramesProcessed, totalTime);
				} else {
					printf("WARNING: fullSystem_pipeline is null, skipping pipeline export\n");
				}
			} catch(const std::exception& e) {
				printf("ERROR: Exception during pipeline data export: %s\n", e.what());
			} catch(...) {
				printf("ERROR: Unknown exception during pipeline data export\n");
			}
			
			printf("=======================================================\n\n");
			printf("All done! Files saved to: %s and %s\n", outputDirRaw.c_str(), outputDirPipeline.c_str());
		}
		else if(dualMode == 0 && fullSystem_raw != nullptr)
		{
			// Raw only mode: export raw result
			std::string outputDir = "dso_output/raw";
			try {
				// Export even if crashed - save what we have
				if(rawSystemCrashed) {
					printf("WARNING: Raw system crashed, but exporting available data...\n");
				}
				printf("Exporting raw reconstruction to: %s\n", outputDir.c_str());
				if(viewer != nullptr) {
					IOWrap::DataExporter::exportAll(fullSystem_raw, viewer, capturedFrames, outputDir, cameraFPS);
				} else {
					IOWrap::DataExporter::exportAll(fullSystem_raw, nullptr, capturedFrames, outputDir, cameraFPS);
				}
				printf("Raw reconstruction export completed successfully!\n");
				
				// Export quantitative metrics
				printf("Exporting quantitative metrics...\n");
				std::string metricsFile = outputDir + "/quantitative_metrics.txt";
				IOWrap::DataExporter::exportQuantitativeMetrics(fullSystem_raw, nullptr, metricsFile, true, totalFramesProcessed, totalTime);
			} catch(const std::exception& e) {
				printf("ERROR: Exception during raw data export: %s\n", e.what());
			} catch(...) {
				printf("ERROR: Unknown exception during raw data export\n");
			}
			printf("=======================================================\n\n");
			printf("All done! Files saved to: %s\n", outputDir.c_str());
		}
		else if(dualMode == 2 && fullSystem_pipeline != nullptr)
		{
			// Pipeline only mode: export pipeline result
			std::string outputDir = "dso_output/pipeline";
			try {
				// Export even if crashed - save what we have
				if(pipelineSystemCrashed) {
					printf("WARNING: Pipeline system crashed, but exporting available data...\n");
				}
				printf("Exporting pipeline reconstruction to: %s\n", outputDir.c_str());
				if(viewer != nullptr) {
					IOWrap::DataExporter::exportAll(fullSystem_pipeline, viewer, capturedFrames, outputDir, cameraFPS);
				} else {
					IOWrap::DataExporter::exportAll(fullSystem_pipeline, nullptr, capturedFrames, outputDir, cameraFPS);
				}
				printf("Pipeline reconstruction export completed successfully!\n");
				
				// Export quantitative metrics
				printf("Exporting quantitative metrics...\n");
				std::string metricsFile = outputDir + "/quantitative_metrics.txt";
				IOWrap::DataExporter::exportQuantitativeMetrics(fullSystem_pipeline, nullptr, metricsFile, false, totalFramesProcessed, totalTime);
			} catch(const std::exception& e) {
				printf("ERROR: Exception during pipeline data export: %s\n", e.what());
			} catch(...) {
				printf("ERROR: Unknown exception during pipeline data export\n");
			}
			printf("=======================================================\n\n");
			printf("All done! Files saved to: %s\n", outputDir.c_str());
		}
		else if(fullSystem != nullptr)
		{
			// Single mode: export single result
			std::string outputDir = "dso_output";
			try {
				// Export even if crashed - save what we have
				if(trackingThreadCrashed) {
					printf("WARNING: System crashed, but exporting available data...\n");
				}
				printf("Exporting reconstruction to: %s\n", outputDir.c_str());
				IOWrap::DataExporter::exportAll(fullSystem, viewer, capturedFrames, outputDir, cameraFPS);
				printf("Reconstruction export completed successfully!\n");
				
				// Export quantitative metrics
				printf("Exporting quantitative metrics...\n");
				std::string metricsFile = outputDir + "/quantitative_metrics.txt";
				IOWrap::DataExporter::exportQuantitativeMetrics(fullSystem, nullptr, metricsFile, false, totalFramesProcessed, totalTime);
			} catch(const std::exception& e) {
				printf("ERROR: Exception during data export: %s\n", e.what());
			} catch(...) {
				printf("ERROR: Unknown exception during data export\n");
			}
			printf("=======================================================\n\n");
			printf("All done! Files saved to: %s\n", outputDir.c_str());
		}
		else
		{
			printf("WARNING: fullSystem is null, skipping data export\n");
		}
	}

	// Close Pangolin viewers properly BEFORE calculating detailed metrics
	printf("\n==================== CLOSING PANGOLIN ====================\n");
	if(dualMode == 1 && dualViewer != nullptr)
	{
		printf("Closing dual Pangolin viewer...\n");
		try {
			// First, force quit Pangolin to exit the run() loop immediately
			// This must be called before close() to ensure the loop exits
			pangolin::Quit();
			// Then mark viewer as closed
			dualViewer->close();
			// Wait a bit for viewer to finish
			usleep(50000); // 50ms
			printf("Dual Pangolin viewer closed successfully.\n");
		} catch(const std::exception& e) {
			printf("WARNING: Exception closing dual viewer: %s\n", e.what());
			// Force quit anyway
			try {
				pangolin::Quit();
			} catch(...) {}
		} catch(...) {
			printf("WARNING: Unknown exception closing dual viewer\n");
			// Force quit anyway
			try {
				pangolin::Quit();
			} catch(...) {}
		}
	}
	else if(viewer != nullptr)
	{
		printf("Closing Pangolin viewer...\n");
		try {
			// First, force quit Pangolin to exit the run() loop immediately
			// This must be called before close() to ensure the loop exits
			pangolin::Quit();
			// Then mark viewer as closed
			viewer->close();
			// Wait a bit for viewer to finish
			usleep(50000); // 50ms
			printf("Pangolin viewer closed successfully.\n");
		} catch(const std::exception& e) {
			printf("WARNING: Exception closing viewer: %s\n", e.what());
			// Force quit anyway
			try {
				pangolin::Quit();
			} catch(...) {}
		} catch(...) {
			printf("WARNING: Unknown exception closing viewer\n");
			// Force quit anyway
			try {
				pangolin::Quit();
			} catch(...) {}
		}
	}
	
	// Note: pangolin::Quit() has already been called above, which should clean up all windows
	// We don't need to explicitly destroy windows - pangolin::Quit() handles that
	// Additional cleanup is idempotent and safe to skip
	printf("Pangolin cleanup completed.\n");
	printf("=======================================================\n\n");
	
	// Now calculate detailed metrics from exported files (after Pangolin is closed)
	// Always calculate metrics, even if crashed (to save what we have)
	printf("\n==================== CALCULATING DETAILED METRICS ====================\n");
	{
		std::lock_guard<std::mutex> lock(fullSystemMutex);
		if(dualMode == 1)
		{
		// Calculate detailed metrics for raw (even if crashed)
		// Always calculate metrics to save what we have, since export happens automatically
		try {
			printf("Calculating detailed metrics for raw reconstruction...\n");
			std::string outputDirRaw = "dso_output/raw";
			std::string metricsFileRaw = outputDirRaw + "/quantitative_metrics.txt";
			IOWrap::DataExporter::calculateAndUpdateMetricsFromFiles(outputDirRaw, metricsFileRaw, true);
		} catch(const std::exception& e) {
			printf("ERROR: Exception calculating detailed metrics for raw: %s\n", e.what());
		} catch(...) {
			printf("ERROR: Unknown exception calculating detailed metrics for raw\n");
		}
			
		// Calculate detailed metrics for pipeline (even if crashed)
		// Always calculate metrics to save what we have, since export happens automatically
		try {
			printf("Calculating detailed metrics for pipeline reconstruction...\n");
			std::string outputDirPipeline = "dso_output/pipeline";
			std::string metricsFilePipeline = outputDirPipeline + "/quantitative_metrics.txt";
			IOWrap::DataExporter::calculateAndUpdateMetricsFromFiles(outputDirPipeline, metricsFilePipeline, false);
		} catch(const std::exception& e) {
			printf("ERROR: Exception calculating detailed metrics for pipeline: %s\n", e.what());
		} catch(...) {
			printf("ERROR: Unknown exception calculating detailed metrics for pipeline\n");
		}
		}
		else if(fullSystem != nullptr)
		{
		// Single mode: calculate detailed metrics (even if crashed)
		// Always calculate metrics to save what we have, since export happens automatically
		try {
			printf("Calculating detailed metrics...\n");
			std::string outputDir = "dso_output";
			std::string metricsFile = outputDir + "/quantitative_metrics.txt";
			IOWrap::DataExporter::calculateAndUpdateMetricsFromFiles(outputDir, metricsFile, false);
		} catch(const std::exception& e) {
			printf("ERROR: Exception calculating detailed metrics: %s\n", e.what());
		} catch(...) {
			printf("ERROR: Unknown exception calculating detailed metrics\n");
		}
		}
	}
	printf("=======================================================\n\n");

	// Clean up with mutex protection
	{
		std::lock_guard<std::mutex> lock(fullSystemMutex);
		if(dualMode == 1 && fullSystem_raw != nullptr && fullSystem_pipeline != nullptr)
		{
			// Clean up dual mode
			if(fullSystem_raw != nullptr)
			{
				for(IOWrap::Output3DWrapper* ow : fullSystem_raw->outputWrapper)
				{
					if(ow != nullptr)
					{
						try {
							ow->join();
							delete ow;
						} catch(...) {
							printf("WARNING: Exception cleaning up output wrapper (raw)\n");
						}
					}
				}
				try {
					delete fullSystem_raw;
					fullSystem_raw = nullptr;
				} catch(...) {
					fullSystem_raw = nullptr;
				}
			}
			
			if(fullSystem_pipeline != nullptr)
			{
				for(IOWrap::Output3DWrapper* ow : fullSystem_pipeline->outputWrapper)
				{
					if(ow != nullptr)
					{
						try {
							ow->join();
							delete ow;
						} catch(...) {
							printf("WARNING: Exception cleaning up output wrapper (pipeline)\n");
						}
					}
				}
				try {
					delete fullSystem_pipeline;
					fullSystem_pipeline = nullptr;
				} catch(...) {
					fullSystem_pipeline = nullptr;
				}
			}
		}
		else if(fullSystem != nullptr)
		{
			// Clean up single mode
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
