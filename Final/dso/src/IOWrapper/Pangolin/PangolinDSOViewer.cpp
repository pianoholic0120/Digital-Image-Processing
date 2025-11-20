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



#include "PangolinDSOViewer.h"
#include "KeyFrameDisplay.h"

#include "util/settings.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"
#include "IOWrapper/ImageDisplay.h"
#include <set>
#ifdef __APPLE__
#include <pthread.h>
#endif

namespace dso
{
namespace IOWrap
{



PangolinDSOViewer::PangolinDSOViewer(int w, int h, bool startRunThread)
{
	this->w = w;
	this->h = h;
	running=true;


	{
		boost::unique_lock<boost::mutex> lk(openImagesMutex);
		internalVideoImg = new MinimalImageB3(w,h);
		internalKFImg = new MinimalImageB3(w,h);
		internalResImg = new MinimalImageB3(w,h);
		videoImgChanged=kfImgChanged=resImgChanged=true;

		internalVideoImg->setBlack();
		internalKFImg->setBlack();
		internalResImg->setBlack();
	}


	{
		currentCam = new KeyFrameDisplay();
	}

	needReset = false;


    if(startRunThread)
        runThread = boost::thread(&PangolinDSOViewer::run, this);

}


PangolinDSOViewer::~PangolinDSOViewer()
{
	close();
	runThread.join();
}


void PangolinDSOViewer::run()
{
	printf("START PANGOLIN!\n");

	#ifdef __APPLE__
	// Ensure we're on the main thread for GUI initialization (macOS requirement)
	if(pthread_main_np() == 0) {
		printf("ERROR: Pangolin GUI initialization must be on main thread!\n");
		printf("This will cause crashes on macOS. Exiting...\n");
		return;
	}
	#endif

	pangolin::CreateWindowAndBind("Main",2*w,2*h);
	const int UI_WIDTH = 180;

	glEnable(GL_DEPTH_TEST);

	// 3D visualization
	pangolin::OpenGlRenderState Visualization3D_camera(
		pangolin::ProjectionMatrix(w,h,400,400,w/2,h/2,0.1,1000),
		pangolin::ModelViewLookAt(-0,-5,-10, 0,0,0, pangolin::AxisNegY)
		);

	pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w/(float)h)
		.SetHandler(new pangolin::Handler3D(Visualization3D_camera));


	// 3 images
	pangolin::View& d_kfDepth = pangolin::Display("imgKFDepth")
	    .SetAspect(w/(float)h);

	pangolin::View& d_video = pangolin::Display("imgVideo")
	    .SetAspect(w/(float)h);

	pangolin::View& d_residual = pangolin::Display("imgResidual")
	    .SetAspect(w/(float)h);

	pangolin::GlTexture texKFDepth(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	pangolin::GlTexture texVideo(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	pangolin::GlTexture texResidual(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);


    pangolin::CreateDisplay()
		  .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 1.0)
		  .SetLayout(pangolin::LayoutEqual)
		  .AddDisplay(d_kfDepth)
		  .AddDisplay(d_video)
		  .AddDisplay(d_residual);

	// parameter reconfigure gui
	pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

	pangolin::Var<int> settings_pointCloudMode("ui.PC_mode",1,1,4,false);

	pangolin::Var<bool> settings_showKFCameras("ui.KFCam",false,true);
	pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam",true,true);
	pangolin::Var<bool> settings_showTrajectory("ui.Trajectory",true,true);
	pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory",false,true);
	pangolin::Var<bool> settings_showActiveConstraints("ui.ActiveConst",true,true);
	pangolin::Var<bool> settings_showAllConstraints("ui.AllConst",false,true);


	pangolin::Var<bool> settings_show3D("ui.show3D",true,true);
	pangolin::Var<bool> settings_showLiveDepth("ui.showDepth",true,true);
	pangolin::Var<bool> settings_showLiveVideo("ui.showVideo",true,true);
    pangolin::Var<bool> settings_showLiveResidual("ui.showResidual",false,true);

	pangolin::Var<bool> settings_showFramesWindow("ui.showFramesWindow",false,true);
	pangolin::Var<bool> settings_showFullTracking("ui.showFullTracking",false,true);
	pangolin::Var<bool> settings_showCoarseTracking("ui.showCoarseTracking",false,true);


	pangolin::Var<int> settings_sparsity("ui.sparsity",1,1,20,false);
	pangolin::Var<double> settings_scaledVarTH("ui.relVarTH",0.001,1e-10,1e10, true);
	pangolin::Var<double> settings_absVarTH("ui.absVarTH",0.001,1e-10,1e10, true);
	pangolin::Var<double> settings_minRelBS("ui.minRelativeBS",0.1,0,1, false);


	pangolin::Var<bool> settings_resetButton("ui.Reset",false,false);


	pangolin::Var<int> settings_nPts("ui.activePoints",setting_desiredPointDensity, 50,5000, false);
	pangolin::Var<int> settings_nCandidates("ui.pointCandidates",setting_desiredImmatureDensity, 50,5000, false);
	pangolin::Var<int> settings_nMaxFrames("ui.maxFrames",setting_maxFrames, 4,10, false);
	pangolin::Var<double> settings_kfFrequency("ui.kfFrequency",setting_kfGlobalWeight,0.1,3, false);
	pangolin::Var<double> settings_gradHistAdd("ui.minGradAdd",setting_minGradHistAdd,0,15, false);

	pangolin::Var<double> settings_trackFps("ui.Track fps",0,0,0,false);
	pangolin::Var<double> settings_mapFps("ui.KF fps",0,0,0,false);


	// Default hooks for exiting (Esc) and fullscreen (tab).
	// Keep running GUI loop on main thread (macOS requirement)
	// Only exit when window is closed or explicitly stopped
	while( !pangolin::ShouldQuit() && running )
	{
		// Clear entire screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if(setting_render_display3D)
		{
			// Activate efficiently by object
			Visualization3D_display.Activate(Visualization3D_camera);
			boost::unique_lock<boost::mutex> lk3d(model3DMutex);
			//pangolin::glDrawColouredCube();
			int refreshed=0;
			
			// Create a copy of keyframes list and build valid set to avoid issues if list is modified during iteration
			std::vector<KeyFrameDisplay*> keyframesCopy = keyframes;
			std::set<KeyFrameDisplay*> validKeyframesSet;
			for(auto& kf : keyframes) {
				if(kf != nullptr) {
					validKeyframesSet.insert(kf);
				}
			}
			lk3d.unlock(); // Unlock early to avoid holding lock during rendering
			
			for(KeyFrameDisplay* fh : keyframesCopy)
			{
				if(fh == nullptr) continue;
				
				// Quick check if pointer is still in valid set
				if(validKeyframesSet.find(fh) == validKeyframesSet.end()) continue;
				
				try {
					float blue[3] = {0,0,1};
					if(this->settings_showKFCameras) fh->drawCam(1,blue,0.1);

					refreshed += (int)(fh->refreshPC(refreshed < 10, this->settings_scaledVarTH, this->settings_absVarTH,
							this->settings_pointCloudMode, this->settings_minRelBS, this->settings_sparsity));
					fh->drawPC(1);
				} catch (...) {
					// Skip invalid keyframes
					continue;
				}
			}
			
			// Re-lock for currentCam and drawConstraints
			lk3d.lock();
			if(this->settings_showCurrentCamera && currentCam != nullptr) {
				try {
					currentCam->drawCam(2,0,0.2);
				} catch (...) {
					// Skip if currentCam is invalid
				}
			}
			drawConstraints();
			lk3d.unlock();
		}



		openImagesMutex.lock();
		// Safety checks before uploading textures
		if(videoImgChanged && internalVideoImg != nullptr) {
			try {
				texVideo.Upload(internalVideoImg->data,GL_BGR,GL_UNSIGNED_BYTE);
			} catch (...) {
				// Skip if upload fails
			}
		}
		if(kfImgChanged && internalKFImg != nullptr) {
			try {
				texKFDepth.Upload(internalKFImg->data,GL_BGR,GL_UNSIGNED_BYTE);
			} catch (...) {
				// Skip if upload fails
			}
		}
		if(resImgChanged && internalResImg != nullptr) {
			try {
				texResidual.Upload(internalResImg->data,GL_BGR,GL_UNSIGNED_BYTE);
			} catch (...) {
				// Skip if upload fails
			}
		}
		videoImgChanged=kfImgChanged=resImgChanged=false;
		openImagesMutex.unlock();




		// update fps counters
		{
			openImagesMutex.lock();
			float sd=0;
			for(float d : lastNMappingMs) sd+=d;
			settings_mapFps=lastNMappingMs.size()*1000.0f / sd;
			openImagesMutex.unlock();
		}
		{
			model3DMutex.lock();
			float sd=0;
			for(float d : lastNTrackingMs) sd+=d;
			settings_trackFps = lastNTrackingMs.size()*1000.0f / sd;
			model3DMutex.unlock();
		}


		if(setting_render_displayVideo)
		{
			try {
				d_video.Activate();
				glColor4f(1.0f,1.0f,1.0f,1.0f);
				texVideo.RenderToViewportFlipY();
			} catch (...) {
				// Skip if OpenGL context error
			}
		}

		if(setting_render_displayDepth)
		{
			try {
				d_kfDepth.Activate();
				glColor4f(1.0f,1.0f,1.0f,1.0f);
				texKFDepth.RenderToViewportFlipY();
			} catch (...) {
				// Skip if OpenGL context error
			}
		}

		if(setting_render_displayResidual)
		{
			try {
				d_residual.Activate();
				glColor4f(1.0f,1.0f,1.0f,1.0f);
				texResidual.RenderToViewportFlipY();
			} catch (...) {
				// Skip if OpenGL context error
			}
		}


	    // update parameters
	    this->settings_pointCloudMode = settings_pointCloudMode.Get();

	    this->settings_showActiveConstraints = settings_showActiveConstraints.Get();
	    this->settings_showAllConstraints = settings_showAllConstraints.Get();
	    this->settings_showCurrentCamera = settings_showCurrentCamera.Get();
	    this->settings_showKFCameras = settings_showKFCameras.Get();
	    this->settings_showTrajectory = settings_showTrajectory.Get();
	    this->settings_showFullTrajectory = settings_showFullTrajectory.Get();

		setting_render_display3D = settings_show3D.Get();
		setting_render_displayDepth = settings_showLiveDepth.Get();
		setting_render_displayVideo =  settings_showLiveVideo.Get();
		setting_render_displayResidual = settings_showLiveResidual.Get();

		setting_render_renderWindowFrames = settings_showFramesWindow.Get();
		setting_render_plotTrackingFull = settings_showFullTracking.Get();
		setting_render_displayCoarseTrackingFull = settings_showCoarseTracking.Get();


	    this->settings_absVarTH = settings_absVarTH.Get();
	    this->settings_scaledVarTH = settings_scaledVarTH.Get();
	    this->settings_minRelBS = settings_minRelBS.Get();
	    this->settings_sparsity = settings_sparsity.Get();

	    setting_desiredPointDensity = settings_nPts.Get();
	    setting_desiredImmatureDensity = settings_nCandidates.Get();
	    setting_maxFrames = settings_nMaxFrames.Get();
	    setting_kfGlobalWeight = settings_kfFrequency.Get();
	    setting_minGradHistAdd = settings_gradHistAdd.Get();


	    if(settings_resetButton.Get())
	    {
	    	printf("RESET!\n");
	    	settings_resetButton.Reset();
	    	setting_fullResetRequested = true;
	    }

		// Process queued OpenCV operations on main thread (macOS requirement)
		#ifdef __APPLE__
		// Verify we're still on main thread (safety check)
		if(pthread_main_np() != 0) {
			IOWrap::processOpenCVOperations();
		}
		#endif
		
		// Swap frames and Process Events (must be on main thread for macOS)
		#ifdef __APPLE__
		if(pthread_main_np() != 0) {
			pangolin::FinishFrame();
		} else {
			printf("WARNING: pangolin::FinishFrame() called from non-main thread!\n");
		}
		#else
		pangolin::FinishFrame();
		#endif


	    if(needReset) reset_internal();
	}


	printf("QUIT Pangolin GUI loop!\n");
	// Don't exit here - let main thread handle cleanup
	// exit(1);  // Removed - let main thread handle program termination
}


void PangolinDSOViewer::close()
{
	running = false;
}

void PangolinDSOViewer::join()
{
	runThread.join();
	printf("JOINED Pangolin thread!\n");
}

void PangolinDSOViewer::reset()
{
	needReset = true;
}

void PangolinDSOViewer::reset_internal()
{
	model3DMutex.lock();
	for(size_t i=0; i<keyframes.size();i++) delete keyframes[i];
	keyframes.clear();
	allFramePoses.clear();
	keyframesByKFID.clear();
	connections.clear();
	model3DMutex.unlock();


	openImagesMutex.lock();
	internalVideoImg->setBlack();
	internalKFImg->setBlack();
	internalResImg->setBlack();
	videoImgChanged= kfImgChanged= resImgChanged=true;
	openImagesMutex.unlock();

	needReset = false;
}


void PangolinDSOViewer::drawConstraints()
{
	// Quick validation: check if we have any valid connections
	if(connections.empty()) return;
	
	// Build a set of valid keyframe pointers for fast lookup
	// Note: This function is called with model3DMutex already locked
	std::set<KeyFrameDisplay*> validKeyframes;
	for(auto& kf : keyframes) {
		if(kf != nullptr) {
			validKeyframes.insert(kf);
		}
	}
	
	// Early return if no valid keyframes
	if(validKeyframes.empty()) return;
	
	if(settings_showAllConstraints)
	{
		// draw constraints
		glLineWidth(1);
		glColor3f(0,1,0);
		glBegin(GL_LINES);
		for(unsigned int i=0;i<connections.size();i++)
		{
			// Safety checks: verify pointers are valid and in validKeyframes set
			if(connections[i].to == nullptr || connections[i].from == nullptr) continue;
			if(connections[i].to == 0 || connections[i].from == 0) continue;
			
			// Fast lookup in validKeyframes set
			if(validKeyframes.find(connections[i].from) == validKeyframes.end()) continue;
			if(validKeyframes.find(connections[i].to) == validKeyframes.end()) continue;
			
			int nAct = connections[i].bwdAct + connections[i].fwdAct;
			int nMarg = connections[i].bwdMarg + connections[i].fwdMarg;
			if(nAct==0 && nMarg>0  )
			{
				try {
					Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
					glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
					t = connections[i].to->camToWorld.translation().cast<float>();
					glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
				} catch (...) {
					// Skip invalid connections
					continue;
				}
			}
		}
		glEnd();
	}

	if(settings_showActiveConstraints)
	{
		glLineWidth(3);
		glColor3f(0,0,1);
		glBegin(GL_LINES);
		for(unsigned int i=0;i<connections.size();i++)
		{
			// Safety checks: verify pointers are valid and in validKeyframes set
			if(connections[i].to == nullptr || connections[i].from == nullptr) continue;
			if(connections[i].to == 0 || connections[i].from == 0) continue;
			
			// Fast lookup in validKeyframes set
			if(validKeyframes.find(connections[i].from) == validKeyframes.end()) continue;
			if(validKeyframes.find(connections[i].to) == validKeyframes.end()) continue;
			
			int nAct = connections[i].bwdAct + connections[i].fwdAct;

			if(nAct>0)
			{
				try {
					Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
					glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
					t = connections[i].to->camToWorld.translation().cast<float>();
					glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
				} catch (...) {
					// Skip invalid connections
					continue;
				}
			}
		}
		glEnd();
	}

	if(settings_showTrajectory)
	{
		float colorRed[3] = {1,0,0};
		glColor3f(colorRed[0],colorRed[1],colorRed[2]);
		glLineWidth(3);

		glBegin(GL_LINE_STRIP);
		for(unsigned int i=0;i<keyframes.size();i++)
		{
			if(keyframes[i] == nullptr) continue;
			try {
				glVertex3f((float)keyframes[i]->camToWorld.translation()[0],
						(float)keyframes[i]->camToWorld.translation()[1],
						(float)keyframes[i]->camToWorld.translation()[2]);
			} catch (...) {
				// Skip invalid keyframes
				continue;
			}
		}
		glEnd();
	}

	if(settings_showFullTrajectory)
	{
		float colorGreen[3] = {0,1,0};
		glColor3f(colorGreen[0],colorGreen[1],colorGreen[2]);
		glLineWidth(3);

		glBegin(GL_LINE_STRIP);
		for(unsigned int i=0;i<allFramePoses.size();i++)
		{
			if(i >= allFramePoses.size()) break; // Safety check
			try {
				glVertex3f((float)allFramePoses[i][0],
						(float)allFramePoses[i][1],
						(float)allFramePoses[i][2]);
			} catch (...) {
				// Skip invalid poses
				continue;
			}
		}
		glEnd();
	}
}






void PangolinDSOViewer::publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	model3DMutex.lock();
    // First pass: count valid connections (excluding self-connections and duplicates)
    int validConnections = 0;
    for(std::pair<uint64_t,Eigen::Vector2i> p : connectivity)
	{
		int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);
		if(host == target) continue;
		if(host > target) continue;
		validConnections++;
	}
    connections.resize(validConnections);
    
	int runningID=0;
	int totalActFwd=0, totalActBwd=0, totalMargFwd=0, totalMargBwd=0;
    for(std::pair<uint64_t,Eigen::Vector2i> p : connectivity)
	{
		int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);

		assert(host >= 0 && target >= 0);
		if(host == target)
		{
			// Skip self-connections (host == target)
			// Note: p.second may not always be zero in some edge cases
			continue;
		}

		if(host > target) continue;

		// Only set pointers if keyframes exist
		if(keyframesByKFID.count(host) > 0 && keyframesByKFID.count(target) > 0) {
			connections[runningID].from = keyframesByKFID[host];
			connections[runningID].to = keyframesByKFID[target];
		} else {
			connections[runningID].from = 0;
			connections[runningID].to = 0;
		}
		connections[runningID].fwdAct = p.second[0];
		connections[runningID].fwdMarg = p.second[1];
		totalActFwd += p.second[0];
		totalMargFwd += p.second[1];

        uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
		// Check if inverse key exists before accessing
		if(connectivity.count(inverseKey) > 0)
		{
			Eigen::Vector2i st = connectivity.at(inverseKey);
			connections[runningID].bwdAct = st[0];
			connections[runningID].bwdMarg = st[1];
			totalActBwd += st[0];
			totalMargBwd += st[1];
		}
		else
		{
			connections[runningID].bwdAct = 0;
			connections[runningID].bwdMarg = 0;
		}

		runningID++;
	}


	model3DMutex.unlock();
}
void PangolinDSOViewer::publishKeyframes(
		std::vector<FrameHessian*> &frames,
		bool final,
		CalibHessian* HCalib)
{
	if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	for(FrameHessian* fh : frames)
	{
		if(fh == nullptr) continue; // Safety check
		if(HCalib == nullptr) continue; // Safety check
		
		if(keyframesByKFID.find(fh->frameID) == keyframesByKFID.end())
		{
			KeyFrameDisplay* kfd = new KeyFrameDisplay();
			if(kfd == nullptr) continue; // Safety check for allocation failure
			keyframesByKFID[fh->frameID] = kfd;
			keyframes.push_back(kfd);
		}
		
		// Safety check: ensure keyframe exists before setting
		if(keyframesByKFID.find(fh->frameID) != keyframesByKFID.end() && 
		   keyframesByKFID[fh->frameID] != nullptr)
		{
			try {
				keyframesByKFID[fh->frameID]->setFromKF(fh, HCalib);
			} catch (...) {
				// Skip if setFromKF fails
				continue;
			}
		}
	}
}
void PangolinDSOViewer::publishCamPose(FrameShell* frame,
		CalibHessian* HCalib)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;
	if(frame == nullptr || HCalib == nullptr) return; // Safety check

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNTrackingMs.push_back(((time_now.tv_sec-last_track.tv_sec)*1000.0f + (time_now.tv_usec-last_track.tv_usec)/1000.0f));
	if(lastNTrackingMs.size() > 10) lastNTrackingMs.pop_front();
	last_track = time_now;

	if(!setting_render_display3D) return;

	if(currentCam != nullptr) {
		try {
			currentCam->setFromF(frame, HCalib);
			allFramePoses.push_back(frame->camToWorld.translation().cast<float>());
		} catch (...) {
			// Skip if setFromF fails
		}
	}
}


void PangolinDSOViewer::pushLiveFrame(FrameHessian* image)
{
	if(!setting_render_displayVideo) return;
    if(disableAllDisplay) return;
	if(image == nullptr) return; // Safety check

	boost::unique_lock<boost::mutex> lk(openImagesMutex);
	
	if(internalVideoImg == nullptr) return; // Safety check
	
	try {
		if(image->dI != nullptr) {
			for(int i=0;i<w*h;i++) // Safety: ensure we don't go out of bounds
			{
				internalVideoImg->data[i][0] =
				internalVideoImg->data[i][1] =
				internalVideoImg->data[i][2] =
					image->dI[i][0]*0.8 > 255.0f ? 255.0 : image->dI[i][0]*0.8;
			}
			videoImgChanged=true;
		}
	} catch (...) {
		// Skip if copy fails
	}
}

bool PangolinDSOViewer::needPushDepthImage()
{
    return setting_render_displayDepth;
}
void PangolinDSOViewer::pushDepthImage(MinimalImageB3* image)
{

    if(!setting_render_displayDepth) return;
    if(disableAllDisplay) return;
	if(image == nullptr) return; // Safety check

	boost::unique_lock<boost::mutex> lk(openImagesMutex);
	
	if(internalKFImg == nullptr) return; // Safety check

	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNMappingMs.push_back(((time_now.tv_sec-last_map.tv_sec)*1000.0f + (time_now.tv_usec-last_map.tv_usec)/1000.0f));
	if(lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
	last_map = time_now;

	try {
		if(image->data != nullptr && internalKFImg->data != nullptr) {
			memcpy(internalKFImg->data, image->data, w*h*3);
			kfImgChanged=true;
		}
	} catch (...) {
		// Skip if memcpy fails
	}
}

void PangolinDSOViewer::exportPointCloud(std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector3f>& colors)
{
	points.clear();
	colors.clear();

	boost::unique_lock<boost::mutex> lock(model3DMutex);

	for(KeyFrameDisplay* fh : keyframes)
	{
		if(fh == nullptr || !fh->active) continue;

		std::vector<Eigen::Vector3f> kfPoints;
		std::vector<Eigen::Vector3f> kfColors;

		// Export points from this keyframe
		fh->exportPointCloud(kfPoints, kfColors, 
		                     settings_scaledVarTH, settings_absVarTH, 
		                     settings_pointCloudMode, settings_minRelBS, settings_sparsity);

		// Append to global point cloud
		points.insert(points.end(), kfPoints.begin(), kfPoints.end());
		colors.insert(colors.end(), kfColors.begin(), kfColors.end());
	}
}

}
}
