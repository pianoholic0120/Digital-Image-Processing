#!/usr/bin/env python3
"""
Visual SLAM Node - Complete SLAM implementation with map building
Uses feature-based visual SLAM with keyframe management and loop closure
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from collections import deque

# tf2_ros is optional
try:
    from tf2_ros import TransformBroadcaster
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False


class KeyFrame:
    """Keyframe for SLAM"""
    def __init__(self, image, keypoints, descriptors, pose, timestamp):
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.pose = pose.copy()  # [x, y, z, roll, pitch, yaw] - 3D pose
        self.timestamp = timestamp
        self.id = None


class VisualSLAMNode(Node):
    """Complete Visual SLAM Node with map building"""
    
    def __init__(self):
        super().__init__('visual_slam_node')
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/slam/pose', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/slam/map', 10)
        self.path_pub = self.create_publisher(Path, '/slam/path', 10)
        self.features_image_pub = self.create_publisher(Image, '/slam/features_image', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/slam/pointcloud', 10)
        
        # TF broadcaster
        if TF2_AVAILABLE:
            self.tf_broadcaster = TransformBroadcaster(self)
        else:
            self.tf_broadcaster = None
            self.get_logger().warn('tf2_ros unavailable, TF publishing disabled')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # SLAM state (3D pose: x, y, z, roll, pitch, yaw)
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, z, roll, pitch, yaw]
        self.pose_history = deque(maxlen=1000)  # Store pose history
        
        # Keyframe management - 改进的关键帧策略
        self.keyframes = []
        self.keyframe_threshold = 0.3  # 降低到0.3m，更频繁创建关键帧，提升点云密度
        self.last_keyframe_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.keyframe_angle_threshold = 0.2  # 角度阈值（弧度），约11度
        
        # Feature detector (ORB) - 使用更多特征点提升精度
        self.declare_parameter('n_features', 3000)  # 增加到3000个特征点
        self.declare_parameter('scale_factor', 1.2)
        self.declare_parameter('n_levels', 8)
        n_features = self.get_parameter('n_features').get_parameter_value().integer_value
        scale_factor = self.get_parameter('scale_factor').get_parameter_value().double_value
        n_levels = self.get_parameter('n_levels').get_parameter_value().integer_value
        
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=31,  # 边缘阈值，避免在图像边缘检测特征
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,  # 使用Harris角点评分
            patchSize=31,
            fastThreshold=20
        )
        
        # Feature matcher - 使用更严格的匹配策略
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # 网格特征检测 - 确保特征点均匀分布
        self.use_grid_detection = True
        self.grid_rows = 4
        self.grid_cols = 4
        
        # Map building
        self.map_points = []  # 3D points in map
        self.map_size = 50.0  # Map size in meters
        self.map_resolution = 0.05  # 5cm per pixel
        self.map_origin_x = -self.map_size / 2
        self.map_origin_y = -self.map_size / 2
        
        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_timestamp = None
        self.current_image = None
        self.current_keypoints = None
        
        # 3D map points (for point cloud) - 改进的点云管理
        self.map_points_3d = []  # List of [x, y, z] points in map frame
        self.map_points_3d_with_observations = []  # 存储每个点的观测次数，用于质量评估
        self.map_points_3d_colors = []  # 存储每个点的RGB颜色 [r, g, b]
        self.max_map_points = 10000  # 增加到10000个点，保留更多细节
        
        # Motion estimation
        self.scale_factor = 1.0  # Will be calibrated from first motion
        self.min_matches = 100  # 增加到100个匹配点，提升精度
        self.motion_threshold = 0.02  # Minimum motion to update pose (2cm) - prevents drift when stationary
        self.rotation_threshold = 0.02  # Minimum rotation to update pose (radians, ~1.15 degrees)
        self.min_inlier_ratio = 0.6  # 提高到60% inlier比例，更严格的质量要求
        
        # For scale recovery
        self.scale_initialized = False
        self.reference_depth = 1.0  # Reference depth for scale recovery
        self.last_keyframe_pose_3d = None  # Store 3D pose for scale recovery
        
        # Loop closure
        self.loop_closure_threshold = 0.3  # Distance threshold for loop closure
        self.loop_candidates = []
        
        # Path for visualization
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'
        
        # Timer for map publishing
        self.map_timer = self.create_timer(1.0, self.publish_map)  # 1 Hz
        
        # Timer for point cloud publishing - 提高发布频率确保点云及时显示
        self.pointcloud_timer = self.create_timer(0.5, self.publish_pointcloud)  # 2 Hz (更频繁)
        
        self.get_logger().info('Visual SLAM node started (Enhanced version for high-quality 3D reconstruction)')
        self.get_logger().info(f'Map size: {self.map_size}m, Resolution: {self.map_resolution}m')
        self.get_logger().info(f'Features: {n_features}, Min matches: {self.min_matches}, Max map points: {self.max_map_points}')
    
    def detect_features_grid(self, gray):
        """Detect features using grid-based approach for better distribution"""
        h, w = gray.shape
        grid_h = h // self.grid_rows
        grid_w = w // self.grid_cols
        
        all_keypoints = []
        all_descriptors = []
        
        # Detect features in each grid cell
        # Get nfeatures from ORB parameters (default 3000)
        n_features = 3000  # Default, will be overridden by parameter
        try:
            n_features = self.get_parameter('n_features').get_parameter_value().integer_value
        except:
            pass
        features_per_cell = n_features // (self.grid_rows * self.grid_cols)
        
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                y1 = i * grid_h
                y2 = (i + 1) * grid_h if i < self.grid_rows - 1 else h
                x1 = j * grid_w
                x2 = (j + 1) * grid_w if j < self.grid_cols - 1 else w
                
                # Extract grid cell
                cell = gray[y1:y2, x1:x2]
                
                if cell.size > 0:
                    # Create a temporary ORB detector with adjusted features
                    orb_cell = cv2.ORB_create(
                        nfeatures=features_per_cell,
                        scaleFactor=1.2,  # Same as main ORB
                        nlevels=8  # Same as main ORB
                    )
                    
                    # Detect in this cell
                    kp_cell, desc_cell = orb_cell.detectAndCompute(cell, None)
                    
                    # Adjust keypoint coordinates to full image
                    if kp_cell is not None:
                        for kp in kp_cell:
                            kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
                        all_keypoints.extend(kp_cell)
                        if desc_cell is not None:
                            if len(all_descriptors) == 0:
                                all_descriptors = desc_cell
                            else:
                                all_descriptors = np.vstack([all_descriptors, desc_cell])
        
        # If grid detection didn't work well, fall back to regular detection
        if len(all_keypoints) < 100:
            all_keypoints, all_descriptors = self.orb.detectAndCompute(gray, None)
        
        return all_keypoints, all_descriptors
    
    def camera_info_callback(self, msg):
        """Store camera calibration parameters"""
        if not self.camera_info_received:
            # Extract camera matrix K
            k = msg.k
            self.camera_matrix = np.array([
                [k[0], k[1], k[2]],
                [k[3], k[4], k[5]],
                [k[6], k[7], k[8]]
            ], dtype=np.float32)
            
            # Extract distortion coefficients
            if len(msg.d) >= 4:
                self.dist_coeffs = np.array([msg.d[0], msg.d[1], msg.d[2], msg.d[3]], dtype=np.float32)
            else:
                self.dist_coeffs = np.zeros((4,), dtype=np.float32)
            
            self.camera_info_received = True
            self.get_logger().info(f'Camera info received: fx={k[0]:.2f}, fy={k[4]:.2f}')
    
    def image_callback(self, msg):
        """Process incoming images for SLAM"""
        try:
            # Convert image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
            timestamp = msg.header.stamp
            self.current_image = cv_image.copy()
            
            # Detect features with grid-based distribution for better coverage
            if self.use_grid_detection and self.camera_matrix is not None:
                keypoints, descriptors = self.detect_features_grid(gray)
            else:
                keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            # Always publish features image (even if no keypoints detected)
            # This ensures the image is always visible in RViz2
            matches_for_viz = None
            if self.prev_frame is not None and self.prev_keypoints is not None and keypoints is not None:
                # Try to get matches for visualization
                if descriptors is not None and self.prev_descriptors is not None:
                    try:
                        matches = self.matcher.knnMatch(descriptors, self.prev_descriptors, k=2)
                        good_matches = []
                        for match_pair in matches:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.7 * n.distance:
                                    good_matches.append(m)
                        matches_for_viz = good_matches if len(good_matches) > 0 else None
                    except Exception as e:
                        self.get_logger().debug(f'Match visualization error: {e}')
                        matches_for_viz = None
            
            # Always publish features image (even if keypoints is None or empty)
            # This ensures RViz2 always shows the camera image
            self.publish_features_image(cv_image, keypoints, matches_for_viz, self.prev_keypoints, timestamp)
            
            if descriptors is None or len(keypoints) < self.min_matches:
                self.get_logger().warn(f'Not enough features detected: {len(keypoints) if keypoints else 0} features')
                # Still update previous frame for next iteration
                self.prev_frame = gray
                self.prev_keypoints = keypoints
                self.prev_descriptors = descriptors
                self.prev_timestamp = timestamp
                self.current_keypoints = keypoints
                return
            
            # Track motion if we have previous frame
            if self.prev_frame is not None and self.prev_descriptors is not None:
                # Match features
                matches = self.matcher.knnMatch(descriptors, self.prev_descriptors, k=2)
                
                # Apply Lowe's ratio test with improved filtering
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        # 使用更严格的0.6比例，提升匹配质量
                        if m.distance < 0.6 * n.distance:
                            # 额外的距离阈值检查
                            if m.distance < 50:  # Hamming距离阈值
                                good_matches.append(m)
                
                if len(good_matches) >= self.min_matches:
                    # Estimate motion
                    motion = self.estimate_motion(keypoints, self.prev_keypoints, good_matches)
                    
                    if motion is not None:
                        # Motion has already been filtered in estimate_motion
                        # Additional check: verify motion is significant
                        motion_magnitude = math.sqrt(motion['dx']**2 + motion['dy']**2 + motion.get('dz', 0.0)**2)
                        rotation_magnitude = abs(motion.get('dyaw', 0.0)) + abs(motion.get('droll', 0.0)) + abs(motion.get('dpitch', 0.0))
                        
                        # Double-check thresholds (defense in depth)
                        if motion_magnitude >= self.motion_threshold or rotation_magnitude >= self.rotation_threshold:
                            # Update pose
                            self.update_pose(motion)
                            
                            # Check for new keyframe
                            if self.should_create_keyframe():
                                self.create_keyframe(gray, keypoints, descriptors, timestamp)
                            
                            # Publish pose
                            self.publish_pose(timestamp)
                            
                            # Publish TF
                            self.publish_tf(timestamp)
                            
                            # Update path (only when actually moving)
                            self.update_path(timestamp)
                            
                            # Update 3D map points using proper triangulation
                            self.update_map_points(keypoints, good_matches, self.prev_keypoints, motion)
                        else:
                            # Motion too small, likely stationary - don't update pose or path
                            # Just publish current pose (no change)
                            self.publish_pose(timestamp)
                            self.publish_tf(timestamp)
                            # DO NOT update path when stationary
                    else:
                        # No motion detected (camera likely stationary)
                        # Just publish current pose (no change)
                        self.publish_pose(timestamp)
                        self.publish_tf(timestamp)
                        # DO NOT update path when stationary
            
            # Update previous frame
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_timestamp = timestamp
            self.current_keypoints = keypoints
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def estimate_motion(self, kp1, kp2, matches):
        """Estimate camera motion using Essential Matrix (more accurate for 3D scenes)"""
        if len(matches) < self.min_matches or self.camera_matrix is None:
            return None
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # First check: if camera is stationary, point displacements should be very small
        # Calculate average displacement
        displacements = np.linalg.norm(src_pts - dst_pts, axis=2)
        avg_displacement = np.mean(displacements)
        
        # If average displacement is too small (< 2 pixels), likely stationary
        if avg_displacement < 2.0:
            return None
        
        # Estimate Essential Matrix with improved parameters for better accuracy
        E, mask = cv2.findEssentialMat(
            src_pts, dst_pts, 
            self.camera_matrix, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=0.5  # 降低阈值到0.5像素，更严格的要求
        )
        
        if E is None:
            return None
        
        # Check inlier ratio
        if mask is not None:
            inlier_count = np.sum(mask)
            inlier_ratio = inlier_count / len(matches)
            if inlier_ratio < self.min_inlier_ratio:
                # Too few inliers, motion estimate unreliable
                return None
        
        # Recover pose from Essential Matrix
        _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix, mask=mask)
        
        if R is None or t is None:
            return None
        
        # Check inlier ratio from pose recovery
        if mask_pose is not None:
            pose_inlier_count = np.sum(mask_pose)
            pose_inlier_ratio = pose_inlier_count / len(matches)
            if pose_inlier_ratio < self.min_inlier_ratio:
                # Too few inliers in pose recovery, motion estimate unreliable
                return None
        
        # Extract full 3D rotation (roll, pitch, yaw) from rotation matrix
        # R is 3x3 rotation matrix
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        
        # Translation vector (3D)
        tx, ty, tz = t[0, 0], t[1, 0], t[2, 0]
        
        # Check if rotation is too large (likely error if camera is stationary)
        rotation_magnitude = math.sqrt(roll**2 + pitch**2 + yaw**2)
        if rotation_magnitude > 0.1:  # More than ~5.7 degrees is suspicious for stationary camera
            # This might be noise, but we'll let the threshold filter it
            pass
        
        # Check translation magnitude before scaling
        translation_magnitude_raw = math.sqrt(tx**2 + ty**2 + tz**2)
        
        # Scale the translation (monocular SLAM has scale ambiguity)
        translation_scale = self.scale_factor
        
        # If scale not initialized, try to estimate from motion magnitude
        if not self.scale_initialized:
            # Use a reasonable initial scale based on typical camera motion
            # This will be refined as we track
            if translation_magnitude_raw > 0.01:  # If significant motion
                # Assume typical motion is a few cm per frame
                translation_scale = 0.05 / translation_magnitude_raw  # Scale to ~5cm
                self.scale_factor = translation_scale
                self.scale_initialized = True
                self.get_logger().info(f'Initialized scale factor: {translation_scale:.4f}')
        
        # Calculate scaled translation
        dx = tx * translation_scale
        dy = ty * translation_scale
        dz = tz * translation_scale
        
        # Final check: if scaled motion is too small, likely noise
        scaled_motion_magnitude = math.sqrt(dx**2 + dy**2 + dz**2)
        if scaled_motion_magnitude < self.motion_threshold and rotation_magnitude < self.rotation_threshold:
            # Motion is below threshold, likely stationary
            return None
        
        return {
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'droll': roll,
            'dpitch': pitch,
            'dyaw': yaw,
            'R': R,  # Full rotation matrix
            't': t   # Full translation vector
        }
    
    def update_pose(self, motion):
        """Update current pose based on motion estimate (3D SLAM)"""
        # Get current orientation
        roll, pitch, yaw = self.current_pose[3], self.current_pose[4], self.current_pose[5]
        
        # Create full 3D rotation matrix from current orientation
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        
        # Full 3D rotation matrix (ZYX Euler angles)
        R_full = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # Transform motion from camera frame to world coordinates
        motion_local = np.array([motion['dx'], motion['dy'], motion.get('dz', 0.0)])
        motion_world = R_full @ motion_local
        
        # Update position (3D)
        self.current_pose[0] += motion_world[0]
        self.current_pose[1] += motion_world[1]
        self.current_pose[2] += motion_world[2]
        
        # Update orientation (3D) - directly add rotation changes
        self.current_pose[3] += motion.get('droll', 0.0)
        self.current_pose[4] += motion.get('dpitch', 0.0)
        self.current_pose[5] += motion.get('dyaw', 0.0)
        
        # Normalize angles to [-pi, pi]
        self.current_pose[3] = math.atan2(math.sin(self.current_pose[3]), 
                                         math.cos(self.current_pose[3]))
        self.current_pose[4] = math.atan2(math.sin(self.current_pose[4]), 
                                         math.cos(self.current_pose[4]))
        self.current_pose[5] = math.atan2(math.sin(self.current_pose[5]), 
                                         math.cos(self.current_pose[5]))
        
        # Store pose history
        self.pose_history.append(self.current_pose.copy())
    
    def should_create_keyframe(self):
        """Check if we should create a new keyframe - improved strategy"""
        # Calculate 3D distance from last keyframe
        dx = self.current_pose[0] - self.last_keyframe_pose[0]
        dy = self.current_pose[1] - self.last_keyframe_pose[1]
        dz = self.current_pose[2] - self.last_keyframe_pose[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calculate angular change
        droll = abs(self.current_pose[3] - self.last_keyframe_pose[3])
        dpitch = abs(self.current_pose[4] - self.last_keyframe_pose[4])
        dyaw = abs(self.current_pose[5] - self.last_keyframe_pose[5])
        # Normalize angles
        droll = min(droll, 2*math.pi - droll)
        dpitch = min(dpitch, 2*math.pi - dpitch)
        dyaw = min(dyaw, 2*math.pi - dyaw)
        angle_change = math.sqrt(droll**2 + dpitch**2 + dyaw**2)
        
        # Create keyframe if moved enough distance OR rotated enough
        return distance >= self.keyframe_threshold or angle_change >= self.keyframe_angle_threshold
    
    def create_keyframe(self, image, keypoints, descriptors, timestamp):
        """Create a new keyframe"""
        keyframe = KeyFrame(
            image.copy(),
            keypoints,
            descriptors,
            self.current_pose.copy(),
            timestamp
        )
        keyframe.id = len(self.keyframes)
        self.keyframes.append(keyframe)
        self.last_keyframe_pose = self.current_pose.copy()
        
        self.get_logger().info(f'Created keyframe {keyframe.id} at pose: '
                              f'[{self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}, {self.current_pose[2]:.2f}, '
                              f'r={self.current_pose[3]:.2f}, p={self.current_pose[4]:.2f}, y={self.current_pose[5]:.2f}]')
    
    def publish_pose(self, stamp):
        """Publish current pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(self.current_pose[0])
        pose_msg.pose.position.y = float(self.current_pose[1])
        pose_msg.pose.position.z = float(self.current_pose[2])
        
        # Convert 3D orientation (roll, pitch, yaw) to quaternion
        roll, pitch, yaw = self.current_pose[3], self.current_pose[4], self.current_pose[5]
        
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)
        
        self.pose_pub.publish(pose_msg)
    
    def publish_tf(self, stamp):
        """Publish TF transform"""
        if not TF2_AVAILABLE or self.tf_broadcaster is None:
            return
        
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = float(self.current_pose[0])
        t.transform.translation.y = float(self.current_pose[1])
        t.transform.translation.z = float(self.current_pose[2])
        
        # Convert 3D orientation to quaternion
        roll, pitch, yaw = self.current_pose[3], self.current_pose[4], self.current_pose[5]
        
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        
        self.tf_broadcaster.sendTransform(t)
    
    def update_path(self, stamp):
        """Update path for visualization - only called when actually moving"""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose.position.x = float(self.current_pose[0])
        pose_stamped.pose.position.y = float(self.current_pose[1])
        pose_stamped.pose.position.z = float(self.current_pose[2])
        
        # Convert 3D orientation to quaternion
        roll, pitch, yaw = self.current_pose[3], self.current_pose[4], self.current_pose[5]
        
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        pose_stamped.pose.orientation.x = float(qx)
        pose_stamped.pose.orientation.y = float(qy)
        pose_stamped.pose.orientation.z = float(qz)
        pose_stamped.pose.orientation.w = float(qw)
        
        # Only add new pose if it's different from the last one (avoid duplicates)
        if len(self.path_msg.poses) == 0:
            self.path_msg.poses.append(pose_stamped)
        else:
            last_pose = self.path_msg.poses[-1]
            dx = pose_stamped.pose.position.x - last_pose.pose.position.x
            dy = pose_stamped.pose.position.y - last_pose.pose.position.y
            dz = pose_stamped.pose.position.z - last_pose.pose.position.z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)  # 3D distance
            # Only add if moved at least 1cm
            if dist >= 0.01:
                self.path_msg.poses.append(pose_stamped)
        
        self.path_msg.header.stamp = stamp
        self.path_pub.publish(self.path_msg)
    
    def publish_map(self):
        """Publish occupancy grid map"""
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'map'
        
        # Map metadata
        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = int(self.map_size / self.map_resolution)
        map_msg.info.height = int(self.map_size / self.map_resolution)
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        
        # Create map from pose history
        map_data = np.full((map_msg.info.height, map_msg.info.width), -1, dtype=np.int8)
        
        # Draw path (known free space)
        for pose in self.pose_history:
            x_idx = int((pose[0] - self.map_origin_x) / self.map_resolution)
            y_idx = int((pose[1] - self.map_origin_y) / self.map_resolution)
            
            if 0 <= x_idx < map_msg.info.width and 0 <= y_idx < map_msg.info.height:
                # Mark as free (0)
                map_data[y_idx, x_idx] = 0
        
        # Draw keyframes
        for keyframe in self.keyframes:
            x_idx = int((keyframe.pose[0] - self.map_origin_x) / self.map_resolution)
            y_idx = int((keyframe.pose[1] - self.map_origin_y) / self.map_resolution)
            
            if 0 <= x_idx < map_msg.info.width and 0 <= y_idx < map_msg.info.height:
                # Draw keyframe as a small circle (free space)
                cv2.circle(map_data, (x_idx, y_idx), 3, 0, -1)
        
        # Flatten and convert to list
        map_msg.data = map_data.flatten().tolist()
        
        self.map_pub.publish(map_msg)
    
    def publish_features_image(self, image, keypoints, matches, prev_keypoints, stamp):
        """Publish image with feature points visualized"""
        if image is None:
            return
        
        # Draw keypoints on image
        vis_image = image.copy()
        
        # Draw all keypoints (always draw, even if no matches)
        feature_count = 0
        if keypoints is not None and len(keypoints) > 0:
            feature_count = len(keypoints)
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                # Draw larger circles for better visibility
                cv2.circle(vis_image, (x, y), 4, (0, 255, 0), -1)  # Green dots
                cv2.circle(vis_image, (x, y), 5, (0, 255, 0), 1)  # Green outline
        
        # Draw matches if available
        match_count = 0
        if matches is not None and len(matches) > 0 and prev_keypoints is not None and keypoints is not None:
            match_count = len(matches)
            for match in matches[:50]:  # Draw first 50 matches
                try:
                    pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
                    pt2 = tuple(map(int, prev_keypoints[match.trainIdx].pt))
                    cv2.line(vis_image, pt1, pt2, (255, 0, 0), 2)  # Blue lines (thicker)
                    cv2.circle(vis_image, pt1, 5, (0, 255, 0), -1)  # Green circles
                except (IndexError, AttributeError):
                    continue
        
        # Add text showing feature count and match count
        text_y = 30
        cv2.putText(vis_image, f'Features: {feature_count}', (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if match_count > 0:
            cv2.putText(vis_image, f'Matches: {match_count}', (10, text_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert to ROS message
        try:
            features_msg = self.bridge.cv2_to_imgmsg(vis_image, 'rgb8')
            features_msg.header.stamp = stamp
            features_msg.header.frame_id = 'camera_frame'
            self.features_image_pub.publish(features_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing features image: {e}')
    
    def triangulate_points(self, kp1, kp2, matches, R, t, image1=None):
        """Triangulate 3D points from two views using proper triangulation"""
        if self.camera_matrix is None or len(matches) < 10:
            return [], []
        
        # Prepare points for triangulation
        pts1 = []
        pts2 = []
        good_matches = []
        match_colors = []  # Store colors for matched points
        
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            pts1.append(pt1)
            pts2.append(pt2)
            good_matches.append(match)
            
            # Get color from image if available
            if image1 is not None:
                x, y = int(pt1[0]), int(pt1[1])
                if 0 <= x < image1.shape[1] and 0 <= y < image1.shape[0]:
                    # OpenCV uses BGR, but we want RGB
                    b, g, r = image1[y, x]
                    match_colors.append([r, g, b])
                else:
                    match_colors.append([128, 128, 128])  # Gray default
            else:
                match_colors.append([128, 128, 128])  # Gray default
        
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)
        
        # Undistort points (if needed)
        # dist_coeffs needs to be reshaped for cv2.undistortPoints
        dist_coeffs_reshaped = self.dist_coeffs.reshape(-1, 1) if len(self.dist_coeffs.shape) == 1 else self.dist_coeffs
        pts1_undist = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.camera_matrix, 
                                          dist_coeffs_reshaped, P=self.camera_matrix)
        pts2_undist = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.camera_matrix, 
                                          dist_coeffs_reshaped, P=self.camera_matrix)
        
        # Create projection matrices
        # First camera: identity
        P1 = self.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        # Second camera: R and t
        P2 = self.camera_matrix @ np.hstack([R, t])
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1_undist.reshape(2, -1), pts2_undist.reshape(2, -1))
        
        # Convert from homogeneous to 3D
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T  # Shape: (N, 3)
        
        # Filter points with improved quality checks
        valid_points = []
        valid_colors = []
        for i, pt in enumerate(points_3d):
            # Check if point is in front of both cameras
            # 使用更合理的深度范围：10cm 到 15m（室内场景，放宽范围以获取更多点）
            if pt[2] > 0.1 and pt[2] < 15.0:  # Depth between 10cm and 15m
                # 检查重投影误差（使用匹配点）- 放宽阈值以获取更多点
                skip_point = False
                if i < len(matches):
                    try:
                        # 计算重投影误差
                        pt1 = kp1[matches[i].queryIdx].pt
                        pt2 = kp2[matches[i].trainIdx].pt
                        
                        # 重投影到第一帧
                        pt_3d_homogeneous = np.array([pt[0], pt[1], pt[2], 1.0])
                        proj1 = self.camera_matrix @ pt_3d_homogeneous[:3]
                        if abs(proj1[2]) > 1e-6:  # Avoid division by zero
                            proj1 = proj1 / proj1[2]
                            reproj_error1 = np.linalg.norm([proj1[0] - pt1[0], proj1[1] - pt1[1]])
                            
                            # 重投影到第二帧
                            pt_3d_cam2 = R @ pt_3d_homogeneous[:3] + t.flatten()
                            proj2 = self.camera_matrix @ pt_3d_cam2
                            if abs(proj2[2]) > 1e-6:  # Avoid division by zero
                                proj2 = proj2 / proj2[2]
                                reproj_error2 = np.linalg.norm([proj2[0] - pt2[0], proj2[1] - pt2[1]])
                                
                                # 放宽重投影误差阈值到3像素（获取更多点，同时保持质量）
                                if reproj_error1 > 3.0 or reproj_error2 > 3.0:
                                    skip_point = True
                        else:
                            skip_point = True
                    except (IndexError, ValueError) as e:
                        # 如果计算失败，跳过这个点
                        skip_point = True
                
                if skip_point:
                    continue
                # Transform to map frame using current pose
                x, y, z = pt[0], pt[1], pt[2]
                
                # Apply scale factor
                x *= self.scale_factor
                y *= self.scale_factor
                z *= self.scale_factor
                
                # Transform to map frame (3D SLAM - preserve real z coordinate)
                # Get current orientation
                roll, pitch, yaw = self.current_pose[3], self.current_pose[4], self.current_pose[5]
                
                # Create full rotation matrix
                cy, sy = math.cos(yaw), math.sin(yaw)
                cp, sp = math.cos(pitch), math.sin(pitch)
                cr, sr = math.cos(roll), math.sin(roll)
                
                R_map = np.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [-sp, cp*sr, cp*cr]
                ])
                
                # Transform point from camera frame to map frame
                point_camera = np.array([x, y, z])
                point_map = R_map @ point_camera
                
                map_x = self.current_pose[0] + point_map[0]
                map_y = self.current_pose[1] + point_map[1]
                map_z = self.current_pose[2] + point_map[2]  # Real 3D z coordinate!
                
                valid_points.append([map_x, map_y, map_z])
                
                # Get color for this point
                if i < len(match_colors):
                    valid_colors.append(match_colors[i])
                else:
                    valid_colors.append([128, 128, 128])  # Gray default
        
        return valid_points, valid_colors
    
    def update_map_points(self, keypoints, matches, prev_keypoints, motion_result):
        """Update 3D map points using proper triangulation"""
        if matches is None or len(matches) < 10 or motion_result is None:
            return
        
        # Get R and t from motion estimation
        R = motion_result.get('R')
        t = motion_result.get('t')
        
        if R is None or t is None:
            return
        
        # Get current image for color extraction
        current_image = self.current_image if hasattr(self, 'current_image') and self.current_image is not None else None
        
        # Triangulate points with colors
        new_points, new_colors = self.triangulate_points(keypoints, prev_keypoints, matches, R, t, current_image)
        
        # Add valid points to map with improved duplicate detection
        for idx, point in enumerate(new_points):
            color = new_colors[idx] if idx < len(new_colors) else [128, 128, 128]
            
            # Avoid duplicates (3D distance check) - 只检查最近的点以提高效率
            is_duplicate = False
            min_dist = float('inf')
            closest_idx = -1
            
            # 只检查最后500个点以提高效率（对于大量点云）
            check_range = min(500, len(self.map_points_3d))
            start_idx = max(0, len(self.map_points_3d) - check_range)
            
            for existing_idx in range(start_idx, len(self.map_points_3d)):
                existing_point = self.map_points_3d[existing_idx]
                dist = math.sqrt((point[0] - existing_point[0])**2 + 
                               (point[1] - existing_point[1])**2 +
                               (point[2] - existing_point[2])**2)
                if dist < 0.05:  # 5cm threshold (放宽以允许更多点，同时避免重复)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = existing_idx
                    is_duplicate = True
            
            if not is_duplicate:
                self.map_points_3d.append(point)
                self.map_points_3d_with_observations.append(1)  # 初始观测次数为1
                self.map_points_3d_colors.append(color)  # 添加颜色
            elif closest_idx >= 0:
                # 如果是重复点，增加观测次数（表示这个点被多次观测到，质量更高）
                self.map_points_3d_with_observations[closest_idx] += 1
                # 可选：更新点的位置为加权平均（更准确）
                obs_count = self.map_points_3d_with_observations[closest_idx]
                old_point = self.map_points_3d[closest_idx]
                # 加权平均：新点权重1，旧点权重(obs_count-1)
                self.map_points_3d[closest_idx] = [
                    (old_point[0] * (obs_count - 1) + point[0]) / obs_count,
                    (old_point[1] * (obs_count - 1) + point[1]) / obs_count,
                    (old_point[2] * (obs_count - 1) + point[2]) / obs_count
                ]
                # 更新颜色为加权平均
                old_color = self.map_points_3d_colors[closest_idx]
                self.map_points_3d_colors[closest_idx] = [
                    int((old_color[0] * (obs_count - 1) + color[0]) / obs_count),
                    int((old_color[1] * (obs_count - 1) + color[1]) / obs_count),
                    int((old_color[2] * (obs_count - 1) + color[2]) / obs_count)
                ]
        
        # Keep only best points (by observation count) up to max_map_points
        if len(self.map_points_3d) > self.max_map_points:
            # 按观测次数排序，保留观测次数最多的点
            if len(self.map_points_3d_with_observations) == len(self.map_points_3d):
                # 创建索引和观测次数的列表
                indexed_points = list(zip(range(len(self.map_points_3d)), 
                                         self.map_points_3d_with_observations))
                # 按观测次数排序（降序）
                indexed_points.sort(key=lambda x: x[1], reverse=True)
                # 保留前 max_map_points 个
                keep_indices = [idx for idx, _ in indexed_points[:self.max_map_points]]
                self.map_points_3d = [self.map_points_3d[i] for i in keep_indices]
                self.map_points_3d_with_observations = [self.map_points_3d_with_observations[i] 
                                                        for i in keep_indices]
                self.map_points_3d_colors = [self.map_points_3d_colors[i] 
                                             for i in keep_indices]
            else:
                # 如果观测次数列表不匹配，简单截断
                self.map_points_3d = self.map_points_3d[-self.max_map_points:]
                self.map_points_3d_with_observations = []
                self.map_points_3d_colors = self.map_points_3d_colors[-self.max_map_points:]
    
    def publish_pointcloud(self):
        """Publish 3D point cloud with improved quality filtering and colors"""
        # Always publish point cloud, even if empty (for debugging)
        # Use keyframes to create point cloud (more stable)
        points_3d = []
        colors_3d = []
        
        # Add map points (from triangulation) with quality filtering
        # 显示所有点，不进行观测次数过滤（确保点云可见）
        if len(self.map_points_3d) > 0:
            points_3d.extend(self.map_points_3d)
            if len(self.map_points_3d_colors) == len(self.map_points_3d):
                colors_3d.extend(self.map_points_3d_colors)
            else:
                # 如果没有颜色信息，使用默认颜色
                colors_3d.extend([[128, 128, 128]] * len(self.map_points_3d))
        
        # Add keyframe positions as points (for visualization) - 3D with yellow color
        for keyframe in self.keyframes:
            # Add keyframe position (3D)
            if len(keyframe.pose) >= 3:
                points_3d.append([keyframe.pose[0], keyframe.pose[1], keyframe.pose[2] if len(keyframe.pose) > 2 else 0.0])
            else:
                points_3d.append([keyframe.pose[0], keyframe.pose[1], 0.0])
            colors_3d.append([255, 255, 0])  # Yellow for keyframes
        
        # Always add current pose as a point (for debugging) - 3D with red color
        points_3d.append([self.current_pose[0], self.current_pose[1], self.current_pose[2]])
        colors_3d.append([255, 0, 0])  # Red for current pose
        
        if len(points_3d) == 0:
            # Even if empty, publish at least one point to ensure topic is active
            points_3d.append([0.0, 0.0, 0.0])
            colors_3d.append([128, 128, 128])
        
        # Ensure colors list matches points list
        while len(colors_3d) < len(points_3d):
            colors_3d.append([128, 128, 128])  # Gray default
        
        # Create PointCloud2 message with RGB colors
        pc_msg = PointCloud2()
        pc_msg.header = Header()
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        pc_msg.header.frame_id = 'map'
        
        # Set point fields: x, y, z, r, g, b
        # Standard PCL format: x, y, z (float32), rgb (packed as uint32) or r, g, b (uint8)
        pc_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
            PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
            PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step = 16  # 3 floats (12 bytes) + 3 uint8 (3 bytes) + 1 padding byte = 16 bytes
        pc_msg.row_step = pc_msg.point_step * len(points_3d)
        pc_msg.is_dense = True  # Set to True if no invalid points
        
        # Convert to numpy arrays
        points_array = np.array(points_3d, dtype=np.float32)
        colors_array = np.array(colors_3d, dtype=np.uint8)
        
        # Create byte array manually for better compatibility
        point_data = bytearray()
        for i in range(len(points_3d)):
            # Pack x, y, z as float32 (4 bytes each)
            point_data.extend(points_array[i, 0].tobytes())
            point_data.extend(points_array[i, 1].tobytes())
            point_data.extend(points_array[i, 2].tobytes())
            # Pack r, g, b as uint8 (1 byte each)
            point_data.append(colors_array[i, 0])
            point_data.append(colors_array[i, 1])
            point_data.append(colors_array[i, 2])
            # Padding byte
            point_data.append(0)
        
        pc_msg.data = bytes(point_data)
        pc_msg.width = len(points_3d)
        pc_msg.height = 1
        
        self.pointcloud_pub.publish(pc_msg)
        
        # Log point cloud info more frequently for debugging
        if not hasattr(self, 'last_pc_log_time'):
            self.last_pc_log_time = 0
        current_time = self.get_clock().now().nanoseconds
        if current_time - self.last_pc_log_time > 2e9:  # Every 2 seconds (more frequent)
            self.get_logger().info(f'Published point cloud: {len(points_3d)} points (with colors), {len(self.keyframes)} keyframes, map_points={len(self.map_points_3d)}')
            if len(self.map_points_3d) > 0:
                self.get_logger().info(f'  Map points: {len(self.map_points_3d)}, with colors: {len(self.map_points_3d_colors)}, observations: {len(self.map_points_3d_with_observations)}')
            self.last_pc_log_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = VisualSLAMNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

