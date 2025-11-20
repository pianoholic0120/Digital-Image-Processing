CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
SET(PROJECT_NAME DSO)

PROJECT(${PROJECT_NAME})
#set(CMAKE_VERBOSE_MAKEFILE ON)


set(BUILD_TYPE Release)
#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
# find_package(SuiteSparse REQUIRED)
set(CHOLMOD_INCLUDE_DIR /opt/homebrew/include/suitesparse)
set(CSPARSE_INCLUDE_DIR /opt/homebrew/include/suitesparse)

set(CHOLMOD_LIBRARY /opt/homebrew/lib/libcholmod.dylib)
set(CSPARSE_LIBRARY /opt/homebrew/lib/libcxsparse.dylib)

include_directories(${CHOLMOD_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})
set(CHOLMOD_INCLUDE_DIR /opt/homebrew/include/suitesparse)
set(CSPARSE_INCLUDE_DIR /opt/homebrew/include/suitesparse)
set(CHOLMOD_LIBRARY /opt/homebrew/lib/libcholmod.dylib)
set(CSPARSE_LIBRARY /opt/homebrew/lib/libcxsparse.dylib)

# 先找到 Eigen3 和 Threads（Pangolin 需要這些）
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# 為 Eigen3 創建 IMPORTED 目標（如果不存在）
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
    )
endif()
# Apple Homebrew Boost 路徑
if(APPLE)
    set(BOOST_ROOT "/opt/homebrew/opt/boost")
    set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include")
    set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
    # 使用傳統的 FindBoost 模組而不是新的 BoostConfig
    set(Boost_NO_BOOST_CMAKE ON)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
endif()

# 尋找 Boost（需要 thread 和 filesystem 組件）
find_package(Boost 1.89 QUIET COMPONENTS thread filesystem)
if(NOT Boost_FOUND)
    # 如果找不到，嘗試不指定組件
    find_package(Boost 1.89 QUIET)
endif()

# 手動設置庫路徑（如果 FindBoost 沒有正確設置）
if(Boost_FOUND AND APPLE)
    if(EXISTS "${BOOST_LIBRARYDIR}/libboost_thread.dylib")
        set(Boost_THREAD_LIBRARY "${BOOST_LIBRARYDIR}/libboost_thread.dylib")
    endif()
    if(EXISTS "${BOOST_LIBRARYDIR}/libboost_filesystem.dylib")
        set(Boost_FILESYSTEM_LIBRARY "${BOOST_LIBRARYDIR}/libboost_filesystem.dylib")
    endif()
    # 如果 FindBoost 沒有設置 Boost_LIBRARIES，手動設置
    if(NOT Boost_LIBRARIES)
        set(Boost_LIBRARIES "")
        if(Boost_THREAD_LIBRARY)
            list(APPEND Boost_LIBRARIES ${Boost_THREAD_LIBRARY})
        endif()
        if(Boost_FILESYSTEM_LIBRARY)
            list(APPEND Boost_LIBRARIES ${Boost_FILESYSTEM_LIBRARY})
        endif()
    else()
        # FindBoost 已經設置了，但確保 filesystem 也被包含
        if(Boost_FILESYSTEM_LIBRARY)
            list(APPEND Boost_LIBRARIES ${Boost_FILESYSTEM_LIBRARY})
        endif()
    endif()
endif()

if(NOT Boost_FOUND)
    message(FATAL_ERROR "Could not find Boost")
endif()

if(Boost_FOUND)
    message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
endif()


# optional libraries
find_package(LibZip QUIET)

# 手動設定 Pangolin 路徑（用於視覺化）
# 注意：必須在找到 Eigen3 和 Threads 之後載入
set(PANGOLIN_ROOT "/Users/arthurlin/Desktop/DIP/Final/Pangolin")
set(Pangolin_DIR "${PANGOLIN_ROOT}/build")
if(EXISTS "${Pangolin_DIR}/PangolinConfig.cmake")
    # 確保 Eigen3 和 Threads 目標存在
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    endif()
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    
    include("${Pangolin_DIR}/PangolinConfig.cmake")
    set(Pangolin_FOUND TRUE)
    # 設定 include 目錄（Pangolin 使用統一的 include 目錄）
    if(NOT Pangolin_INCLUDE_DIRS)
        set(Pangolin_INCLUDE_DIRS "${PANGOLIN_ROOT}/components/pango_core/include;${PANGOLIN_ROOT}/components/pango_display/include;${PANGOLIN_ROOT}/components/pango_opengl/include;${PANGOLIN_ROOT}/components/pango_windowing/include")
    endif()
    # 設定庫目錄
    if(NOT Pangolin_LIBRARY_DIRS)
        set(Pangolin_LIBRARY_DIRS "${Pangolin_DIR}")
    endif()
    message(STATUS "=== 手動載入 Pangolin ===")
    message(STATUS "Pangolin_DIR: ${Pangolin_DIR}")
    message(STATUS "Pangolin_INCLUDE_DIRS: ${Pangolin_INCLUDE_DIRS}")
    message(STATUS "Pangolin_LIBRARIES: ${Pangolin_LIBRARIES}")
else()
    find_package(Pangolin 0.2 QUIET)
endif()

find_package(OpenCV QUIET)

# flags
add_definitions("-DENABLE_SSE")
# set(CMAKE_CXX_FLAGS
   # "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
#   # "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
# )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# Sources files
set(dso_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector2.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
)



include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
  ${EIGEN3_INCLUDE_DIR}
) 


# decide if we have pangolin
if (Pangolin_FOUND)
    message("--- found PANGOLIN, compiling dso_pangolin library.")
    # 確保使用正確的 Pangolin include 目錄（優先使用本地編譯的）
    if(PANGOLIN_ROOT)
        include_directories(${PANGOLIN_ROOT}/components/pango_core/include)
        include_directories(${PANGOLIN_ROOT}/components/pango_display/include)
        include_directories(${PANGOLIN_ROOT}/components/pango_opengl/include)
        include_directories(${PANGOLIN_ROOT}/components/pango_windowing/include)
        include_directories(${PANGOLIN_ROOT}/components/pango_geometry/include)
    endif()
    include_directories( ${Pangolin_INCLUDE_DIRS} ) 
    # macOS 需要 OpenGL framework 和定義宏
    if(APPLE)
        find_library(OPENGL_LIBRARY OpenGL REQUIRED)
        find_library(COCOA_LIBRARY Cocoa REQUIRED)
        find_library(IOKIT_LIBRARY IOKit REQUIRED)
        find_library(COREVIDEO_LIBRARY CoreVideo REQUIRED)
        # 確保定義 macOS 相關宏，讓 Pangolin 知道這是 macOS
        add_definitions(-D__APPLE__ -D_OSX_)
        # macOS OpenGL 使用 GLuint 而不是 GLhandleARB（類似 GLES）
        # 在 glsl.h 中，只有在 HAVE_GLES 時才定義 GLhandleARB 為 GLuint
        # 我們需要在 macOS 上也這樣做
        add_definitions(-DGL_SILENCE_DEPRECATION)
        # 添加 OpenGL framework 的 include 路徑
        include_directories(/System/Library/Frameworks/OpenGL.framework/Headers)
        # 為編譯目標添加編譯定義
        if(TARGET dso)
            target_compile_definitions(dso PRIVATE __APPLE__ _OSX_ GL_SILENCE_DEPRECATION)
        endif()
    endif()
    set(dso_pangolin_SOURCE_FILES 
      ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
      ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp)
    set(HAS_PANGOLIN 1)
else ()
    message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
    message("    this means there will be no 3D display / GUI available for dso_dataset.")
    set(dso_pangolin_SOURCE_FILES )
    set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
if (OpenCV_FOUND)
	message("--- found OpenCV, compiling dso_opencv library.")
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV, not compiling dso_opencv library.")
	message("    this means there will be no image display, and image read / load functionality.")
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay_dummy.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW_dummy.cpp)
	set(HAS_OPENCV 0)
endif ()

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()


# compile main library.
include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 
add_library(dso ${dso_SOURCE_FILES} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})
target_link_libraries(dso ${Boost_LIBRARIES})
# macOS 上為 dso 庫也添加 OpenGL framework 和編譯定義（如果編譯了 Pangolin）
if(APPLE AND Pangolin_FOUND)
    target_compile_definitions(dso PRIVATE __APPLE__ _OSX_)
    if(OPENGL_LIBRARY)
        target_link_libraries(dso ${OPENGL_LIBRARY})
    endif()
    if(COCOA_LIBRARY)
        target_link_libraries(dso ${COCOA_LIBRARY})
    endif()
    if(IOKIT_LIBRARY)
        target_link_libraries(dso ${IOKIT_LIBRARY})
    endif()
    if(COREVIDEO_LIBRARY)
        target_link_libraries(dso ${COREVIDEO_LIBRARY})
    endif()
endif()

#set_property( TARGET dso APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )


#if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
#    set(BOOST_THREAD_LIBRARY boost_thread-mt)
#else()
#    set(BOOST_THREAD_LIBRARY boost_thread)
#endif()

# build main executable (only if we have both OpenCV and Pangolin)
if (OpenCV_FOUND AND Pangolin_FOUND)
	message("--- compiling dso_dataset.")
	add_executable(dso_dataset 
		${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp 
		${PROJECT_SOURCE_DIR}/src/util/DataExporter.cpp
	)
	# 連結庫
	set(DATASET_LIBS dso ${Boost_LIBRARIES} ${CSPARSE_LIBRARY} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})
	# macOS 需要額外的框架
	if(APPLE)
		if(OPENGL_LIBRARY)
			list(APPEND DATASET_LIBS ${OPENGL_LIBRARY})
		endif()
		if(COCOA_LIBRARY)
			list(APPEND DATASET_LIBS ${COCOA_LIBRARY})
		endif()
		if(IOKIT_LIBRARY)
			list(APPEND DATASET_LIBS ${IOKIT_LIBRARY})
		endif()
		if(COREVIDEO_LIBRARY)
			list(APPEND DATASET_LIBS ${COREVIDEO_LIBRARY})
		endif()
		# 連結到 Pangolin 庫目錄
		if(Pangolin_LIBRARY_DIRS)
			target_link_directories(dso_dataset PRIVATE ${Pangolin_LIBRARY_DIRS})
		endif()
	endif()
    target_link_libraries(dso_dataset ${DATASET_LIBS})
else()
	message("--- not building dso_dataset, since either don't have openCV or Pangolin.")
endif()

