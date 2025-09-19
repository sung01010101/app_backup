#!/usr/bin/env python3

# Web Framework
from flask import Flask, render_template, jsonify, send_from_directory, Response, request
from flask_socketio import SocketIO, emit
import os
import yaml
import numpy as np
from PIL import Image
import threading
import time
import requests

# Directory containing map files
MAPS_DIR = 'static/maps'

# OpenCV for error frame generation
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    import tf2_ros
    from tf2_ros import TransformException
    from geometry_msgs.msg import TransformStamped, PoseStamped, PoseWithCovarianceStamped
    from nav_msgs.msg import Path
    from nav2_msgs.srv import SetInitialPose
    from nav2_msgs.action import FollowWaypoints
    from std_msgs.msg import Header
    from action_msgs.msg import GoalStatus
    ROS2_AVAILABLE = True
except ImportError as e:
    print(f"ROS2 packages not available: {e}")
    ROS2_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize ROS2 and TF listener
tf_listener = None
ros_thread = None
ros_running = False

def init_ros2():
    """Initialize ROS2 node in a separate thread"""
    global tf_listener, ros_running
    if not ROS2_AVAILABLE:
        print("ROS2 not available, TF functionality disabled")
        return
        
    try:
        rclpy.init()
        
        tf_listener = TFListener()
        ros_running = True
        
        # Spin the node
        while ros_running:
            rclpy.spin_once(tf_listener, timeout_sec=0.1)
            
    except Exception as e:
        print(f"Error initializing ROS2: {e}")
    finally:
        if tf_listener:
            tf_listener.destroy_node()
        rclpy.shutdown()

def start_ros2_thread():
    """Start ROS2 in a background thread"""
    global ros_thread
    if ROS2_AVAILABLE and ros_thread is None:
        ros_thread = threading.Thread(target=init_ros2, daemon=True)
        ros_thread.start()

def stop_ros2():
    """Stop ROS2 node"""
    global ros_running
    ros_running = False

class TFListener(Node):
    def __init__(self):
        super().__init__('web_tf_listener')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.latest_transforms = {}
        self.frame_tree = {}
        
        # Service client for setting initial pose
        self.set_initial_pose_client = self.create_client(SetInitialPose, '/set_initial_pose')
        
        # Publisher fallback for initial pose (standard AMCL topic)
        self.initial_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        
        # Action client for waypoint following
        self.waypoint_follower_client = ActionClient(self, FollowWaypoints, '/follow_waypoints')
        self.current_waypoint_goal_handle = None
        
        # Goal pose subscription
        self.goal_pose = None
        self.goal_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )
        
        # Plan subscription
        self.robot_plan = None
        self.plan_subscriber = self.create_subscription(
            Path,
            '/plan',
            self.plan_callback,
            10
        )
        
        self.get_logger().info('TF Listener initialized with goal pose, plan subscriptions, and waypoint follower')
        
    def goal_pose_callback(self, msg):
        """Callback for goal pose messages"""
        try:
            # Extract yaw from quaternion
            q = msg.pose.orientation
            yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
            self.goal_pose = {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z,
                'theta': yaw,
                'frame_id': msg.header.frame_id,
                'timestamp': {
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec
                }
            }
            
            self.get_logger().info(f'Goal pose received: x={self.goal_pose["x"]:.3f}, y={self.goal_pose["y"]:.3f}, θ={yaw:.3f}')
            
        except Exception as e:
            self.get_logger().error(f"Error processing goal pose: {e}")
            
    def plan_callback(self, msg):
        """Callback for plan messages"""
        try:
            # Convert Path message to list of points
            plan_points = []
            for pose_stamped in msg.poses:
                pose = pose_stamped.pose
                plan_points.append({
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'z': pose.position.z
                })
            
            self.robot_plan = {
                'points': plan_points,
                'frame_id': msg.header.frame_id,
                'timestamp': {
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec
                }
            }
            
            self.get_logger().info(f'Plan received with {len(plan_points)} points in frame {msg.header.frame_id}')
            
        except Exception as e:
            self.get_logger().error(f"Error processing plan: {e}")
            
    def get_robot_plan(self):
        """Get the latest robot plan"""
        return self.robot_plan
            
    def get_goal_pose(self):
        """Get the latest goal pose"""
        return self.goal_pose
        
    def get_transform_tree(self):
        """Get the current transform tree structure"""
        try:
            # Get all available frames
            frames = self.tf_buffer.all_frames_as_yaml()
            if frames:
                frame_data = yaml.safe_load(frames)
                return frame_data
            return {}
        except Exception as e:
            self.get_logger().error(f"Error getting frame tree: {e}")
            return {}
    
    def get_transform(self, target_frame, source_frame):
        """Get transform between two frames"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            return {
                'translation': {
                    'x': transform.transform.translation.x,
                    'y': transform.transform.translation.y,
                    'z': transform.transform.translation.z
                },
                'rotation': {
                    'x': transform.transform.rotation.x,
                    'y': transform.transform.rotation.y,
                    'z': transform.transform.rotation.z,
                    'w': transform.transform.rotation.w
                },
                'header': {
                    'stamp': {
                        'sec': transform.header.stamp.sec,
                        'nanosec': transform.header.stamp.nanosec
                    },
                    'frame_id': transform.header.frame_id
                },
                'child_frame_id': transform.child_frame_id
            }
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {source_frame} to {target_frame}: {ex}')
            return None
    
    def get_robot_pose(self, base_frame='base_link', map_frame='map'):
        """Get robot pose in map frame"""
        try:
            transform = self.get_transform(map_frame, base_frame)
            if transform:
                # Extract position and orientation
                pos = transform['translation']
                ori = transform['rotation']
                
                # Convert quaternion to yaw
                yaw = np.arctan2(2.0 * (ori['w'] * ori['z'] + ori['x'] * ori['y']),
                                1.0 - 2.0 * (ori['y'] * ori['y'] + ori['z'] * ori['z']))
                
                return {
                    'x': pos['x'],
                    'y': pos['y'],
                    'theta': yaw,
                    'frame_id': map_frame,
                    'child_frame_id': base_frame,
                    'timestamp': transform['header']['stamp']
                }
        except Exception as e:
            self.get_logger().error(f"Error getting robot pose: {e}")
        
        return None
    
    def get_all_current_transforms(self):
        """Get all current transforms"""
        transforms = {}
        try:
            frame_tree = self.get_transform_tree()
            if not frame_tree:
                return transforms
                
            # Extract frame relationships from the tree
            for frame_id, frame_info in frame_tree.items():
                if 'parent' in frame_info and frame_info['parent']:
                    parent_frame = frame_info['parent']
                    transform = self.get_transform(parent_frame, frame_id)
                    if transform:
                        transforms[f"{parent_frame}->{frame_id}"] = transform
        except Exception as e:
            self.get_logger().error(f"Error getting all transforms: {e}")
        
        return transforms
    
    def set_initial_pose_service(self, position, orientation, frame_id='map'):
        """Set initial pose.
        Tries Nav2 '/set_initial_pose' service first; if unavailable, falls back to publishing on '/initialpose'."""
        # Normalize inputs
        position = [float(p) for p in position]
        orientation = [float(o) for o in orientation]

        # Build PoseWithCovarianceStamped (used both for service request & fallback publish)
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = Header()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.pose.position.x = position[0]
        pose_msg.pose.pose.position.y = position[1]
        pose_msg.pose.pose.position.z = position[2] if len(position) > 2 else 0.0
        pose_msg.pose.pose.orientation.x = orientation[0]
        pose_msg.pose.pose.orientation.y = orientation[1]
        pose_msg.pose.pose.orientation.z = orientation[2]
        pose_msg.pose.pose.orientation.w = orientation[3]
        # Covariance (same as before)
        covariance = [0.0] * 36
        covariance[0] = 0.25   # x variance
        covariance[7] = 0.25   # y variance
        covariance[35] = 0.06853891909122467  # yaw variance
        pose_msg.pose.covariance = covariance

        # Try service first
        try:
            if self.set_initial_pose_client.wait_for_service(timeout_sec=2.0):
                request = SetInitialPose.Request()
                request.pose = pose_msg  # service expects PoseWithCovarianceStamped inside
                future = self.set_initial_pose_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if future.result() is not None:
                    self.get_logger().info(f"Initial pose set via service at ({position[0]:.3f}, {position[1]:.3f})")
                    return True
                else:
                    self.get_logger().warn('Service call returned no result, will fallback to topic publish')
            else:
                self.get_logger().warn("'/set_initial_pose' service not available within timeout, using topic fallback")
        except Exception as e:
            self.get_logger().warn(f"Service call failed ({e}), falling back to topic publish")

        # Fallback: publish to /initialpose (AMCL standard)
        try:
            self.initial_pose_publisher.publish(pose_msg)
            self.get_logger().info(f"Initial pose published to /initialpose at ({position[0]:.3f}, {position[1]:.3f})")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to publish initial pose: {e}")
            return False
    
    def publish_goal_pose(self, position, orientation, frame_id='map'):
        """Publish goal pose to the /goal_pose topic"""
        try:
            # Create goal pose publisher if it doesn't exist
            if not hasattr(self, 'goal_pose_publisher'):
                self.goal_pose_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
            
            # Create PoseStamped message
            goal_msg = PoseStamped()
            
            # Set header
            goal_msg.header.frame_id = frame_id
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Set position
            goal_msg.pose.position.x = float(position[0])
            goal_msg.pose.position.y = float(position[1])
            goal_msg.pose.position.z = float(position[2])
            
            # Set orientation
            goal_msg.pose.orientation.x = float(orientation[0])
            goal_msg.pose.orientation.y = float(orientation[1])
            goal_msg.pose.orientation.z = float(orientation[2])
            goal_msg.pose.orientation.w = float(orientation[3])
            
            # Publish the goal pose
            self.goal_pose_publisher.publish(goal_msg)
            
            self.get_logger().info(f'Goal pose published at ({position[0]:.3f}, {position[1]:.3f}) in frame {frame_id}')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error publishing goal pose: {e}')
            return False
    
    def add_waypoint(self, position, orientation, name='waypoint'):
        """Add a waypoint to the navigation plan"""
        try:
            if not hasattr(self, 'waypoints'):
                self.waypoints = []
            
            waypoint = {
                'name': name,
                'position': position,
                'orientation': orientation,
                'timestamp': self.get_clock().now().to_msg()
            }
            
            self.waypoints.append(waypoint)
            self.get_logger().info(f'Added waypoint "{name}" at ({position[0]:.3f}, {position[1]:.3f}). Total waypoints: {len(self.waypoints)}')
            
            # Emit waypoint update to frontend via SocketIO
            try:
                global socketio
                socketio.emit('waypoint_update', {
                    'waypoint_count': len(self.waypoints),
                    'waypoints': self.waypoints,
                    'message': f'Waypoint "{name}" added'
                })
            except Exception as e:
                self.get_logger().warn(f'Could not emit waypoint update: {e}')
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error adding waypoint: {e}')
            return False
    
    def get_waypoints(self):
        """Get all collected waypoints"""
        if not hasattr(self, 'waypoints'):
            self.waypoints = []
        return self.waypoints
    
    def clear_waypoints(self):
        """Clear all waypoints"""
        if not hasattr(self, 'waypoints'):
            self.waypoints = []
        else:
            self.waypoints.clear()
        self.get_logger().info('All waypoints cleared')
        
        # Emit waypoint update to frontend via SocketIO
        try:
            # Use the global socketio instance
            global socketio
            socketio.emit('waypoint_update', {
                'waypoint_count': 0,
                'waypoints': [],
                'message': 'Waypoints cleared'
            })
        except Exception as e:
            self.get_logger().warn(f'Could not emit waypoint update: {e}')
    
    def start_navigation_with_waypoints(self):
        """Start navigation by following all waypoints using Nav2 waypoint follower"""
        try:
            if not hasattr(self, 'waypoints') or not self.waypoints:
                self.get_logger().error('No waypoints to navigate to')
                return False
            
            # Wait for the waypoint follower action server
            if not self.waypoint_follower_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('Waypoint follower action server not available')
                return False
            
            # Create the action goal
            goal_msg = FollowWaypoints.Goal()
            
            # Convert waypoints to PoseStamped messages
            for waypoint in self.waypoints:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = 'map'
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                
                # Set position
                pose_stamped.pose.position.x = float(waypoint['position'][0])
                pose_stamped.pose.position.y = float(waypoint['position'][1])
                pose_stamped.pose.position.z = float(waypoint['position'][2])
                
                # Set orientation
                pose_stamped.pose.orientation.x = float(waypoint['orientation'][0])
                pose_stamped.pose.orientation.y = float(waypoint['orientation'][1])
                pose_stamped.pose.orientation.z = float(waypoint['orientation'][2])
                pose_stamped.pose.orientation.w = float(waypoint['orientation'][3])
                
                goal_msg.poses.append(pose_stamped)
            
            # Send the goal
            self.get_logger().info(f'Sending waypoint navigation goal with {len(self.waypoints)} waypoints')
            
            # Send goal with callbacks
            send_goal_future = self.waypoint_follower_client.send_goal_async(
                goal_msg,
                feedback_callback=self.waypoint_feedback_callback
            )
            
            # Add callback for when goal is accepted/rejected
            send_goal_future.add_done_callback(self.waypoint_goal_response_callback)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error starting waypoint navigation: {e}')
            return False
    
    def waypoint_goal_response_callback(self, future):
        """Callback for waypoint goal response"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Waypoint navigation goal rejected')
                return
            
            self.current_waypoint_goal_handle = goal_handle
            self.get_logger().info('Waypoint navigation goal accepted')
            
            # Get the result
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.waypoint_result_callback)
            
        except Exception as e:
            self.get_logger().error(f'Error in waypoint goal response: {e}')
    
    def waypoint_feedback_callback(self, feedback_msg):
        """Callback for waypoint navigation feedback"""
        try:
            feedback = feedback_msg.feedback
            current_waypoint = feedback.current_waypoint
            total_waypoints = len(self.waypoints) if hasattr(self, 'waypoints') else 0
            
            self.get_logger().info(f'Waypoint navigation progress: {current_waypoint}/{total_waypoints}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing waypoint feedback: {e}')
    
    def waypoint_result_callback(self, future):
        """Callback for waypoint navigation result"""
        global socketio
        
        try:
            result = future.result().result
            status = future.result().status
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                waypoint_count = len(self.waypoints) if hasattr(self, 'waypoints') else 0
                self.get_logger().info(f'Waypoint navigation completed successfully! Traversed {waypoint_count} waypoints.')
                
                # Clear waypoints after successful navigation
                self.clear_waypoints()
                self.get_logger().info('Waypoints cleared after successful navigation')
                
                # Emit navigation completion event
                try:
                    socketio.emit('navigation_complete', {
                        'success': True,
                        'message': f'Navigation completed successfully! Visited {waypoint_count} waypoints.',
                        'waypoints_traversed': waypoint_count,
                        'waypoints_cleared': True,
                        'button_state': 'navigation_complete_success'
                    })
                except Exception as e:
                    self.get_logger().warn(f'Could not emit navigation complete event: {e}')
                    
            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn('Waypoint navigation was canceled')
                try:
                    socketio.emit('navigation_complete', {
                        'success': False,
                        'message': 'Navigation was canceled',
                        'status': 'canceled',
                        'button_state': 'navigation_canceled'
                    })
                except Exception as e:
                    self.get_logger().warn(f'Could not emit navigation complete event: {e}')
                    
            elif status == GoalStatus.STATUS_ABORTED:
                self.get_logger().error('Waypoint navigation was aborted')
                try:
                    socketio.emit('navigation_complete', {
                        'success': False,
                        'message': 'Navigation was aborted due to an error',
                        'status': 'aborted',
                        'button_state': 'navigation_failed'
                    })
                except Exception as e:
                    self.get_logger().warn(f'Could not emit navigation complete event: {e}')
            else:
                self.get_logger().warn(f'Waypoint navigation ended with status: {status}')
                try:
                    socketio.emit('navigation_complete', {
                        'success': False,
                        'message': f'Navigation ended with status: {status}',
                        'status': str(status),
                        'button_state': 'navigation_failed'
                    })
                except Exception as e:
                    self.get_logger().warn(f'Could not emit navigation complete event: {e}')
            
            self.current_waypoint_goal_handle = None
            
        except Exception as e:
            self.get_logger().error(f'Error processing waypoint result: {e}')
    
    def cancel_waypoint_navigation(self):
        """Cancel current waypoint navigation"""
        try:
            if self.current_waypoint_goal_handle is not None:
                self.get_logger().info('Canceling waypoint navigation')
                cancel_future = self.current_waypoint_goal_handle.cancel_goal_async()
                return True
            else:
                self.get_logger().warn('No active waypoint navigation to cancel')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error canceling waypoint navigation: {e}')
            return False
    
    def is_navigation_active(self):
        """Check if waypoint navigation is currently active"""
        return self.current_waypoint_goal_handle is not None

class MapLoader:
    def __init__(self, maps_dir):
        self.maps_dir = maps_dir
    
    def get_available_maps(self):
        """Get list of available map files"""
        maps = []
        yaml_files = [f for f in os.listdir(self.maps_dir) if f.endswith('.yaml')]
        
        for yaml_file in yaml_files:
            map_name = yaml_file.replace('.yaml', '')
            maps.append({
                'name': map_name,
                'yaml_file': yaml_file,
                'pgm_file': map_name + '.pgm'
            })
        return maps
    
    def load_map_info(self, map_name):
        """Load map information from YAML file"""
        yaml_path = os.path.join(self.maps_dir, f"{map_name}.yaml")
        
        if not os.path.exists(yaml_path):
            return None
            
        with open(yaml_path, 'r') as file:
            map_info = yaml.safe_load(file)
        
        return map_info
    
    def get_nav_poses(self, map_name):
        """Get navigation poses from map YAML file"""
        map_info = self.load_map_info(map_name)
        if not map_info or 'nav_poses' not in map_info:
            return {}
        
        nav_poses = map_info['nav_poses']
        # Convert to a more convenient format
        poses = {}
        for pose_id, pose_data in nav_poses.items():
            poses[pose_id] = {
                'id': pose_id,
                'name': pose_data.get('name', pose_id),
                'position': pose_data.get('position', [0.0, 0.0, 0.0]),
                'orientation': pose_data.get('orientation', [0.0, 0.0, 0.0, 1.0])
            }
        
        return poses
    
    def convert_pgm_to_base64(self, map_name):
        """Convert PGM file to base64 for web display"""
        pgm_path = os.path.join(self.maps_dir, f"{map_name}.pgm")
        
        if not os.path.exists(pgm_path):
            return None
        
        # Load PGM image
        img = Image.open(pgm_path)
        
        # Convert to RGB if it's grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array for processing
        img_array = np.array(img)
        
        # For ROS maps: 
        # - White (255) = free space
        # - Black (0) = occupied space
        # - Gray (127) = unknown space
        
        # Convert back to PIL Image
        processed_img = Image.fromarray(img_array)
        
        # Save as PNG in memory and convert to base64
        import io
        import base64
        
        buffer = io.BytesIO()
        processed_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64

map_loader = MapLoader(MAPS_DIR)

@app.route('/')
def index():
    """Main page"""
    available_maps = map_loader.get_available_maps()
    return render_template('index.html', maps=available_maps)

@app.route('/api/maps')
def get_maps():
    """API endpoint to get available maps"""
    return jsonify(map_loader.get_available_maps())

@app.route('/api/map/<map_name>')
def get_map_data(map_name):
    """API endpoint to get specific map data"""
    map_info = map_loader.load_map_info(map_name)
    
    if not map_info:
        return jsonify({'error': 'Map not found'}), 404
    
    img_base64 = map_loader.convert_pgm_to_base64(map_name)
    
    if not img_base64:
        return jsonify({'error': 'Map image not found'}), 404
    
    nav_poses = map_loader.get_nav_poses(map_name)
    
    return jsonify({
        'map_info': map_info,
        'image_data': img_base64,
        'map_name': map_name,
        'nav_poses': nav_poses
    })

@app.route('/maps/<filename>')
def serve_map_file(filename):
    """Serve map files directly"""
    return send_from_directory(MAPS_DIR, filename)

@app.route('/api/nav_poses/<map_name>')
def get_nav_poses(map_name):
    """API endpoint to get navigation poses for a specific map"""
    nav_poses = map_loader.get_nav_poses(map_name)
    
    if not nav_poses:
        return jsonify({'error': 'No navigation poses found for this map'}), 404
    
    return jsonify({
        'map_name': map_name,
        'nav_poses': nav_poses
    })

@app.route('/api/navigation/set_initial_pose', methods=['POST'])
def set_initial_pose():
    """API endpoint to set the initial pose for navigation"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 or TF listener not available'}), 503
    
    try:
        data = request.get_json()
        if not data or 'pose' not in data:
            return jsonify({'error': 'Missing pose data'}), 400
        
        pose_data = data['pose']
        
        # Extract position and orientation from pose data
        position = pose_data.get('position', [0.0, 0.0, 0.0])
        orientation = pose_data.get('orientation', [0.0, 0.0, 0.0, 1.0])
        
        # Call the service (with fallback) to set initial pose
        success = tf_listener.set_initial_pose_service(position, orientation, 'map')
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Initial pose set to {pose_data.get("name", "pose")} at ({position[0]:.2f}, {position[1]:.2f})',
                'pose': pose_data,
                'method': 'service_or_fallback'
            })
        else:
            return jsonify({'error': 'Failed to set initial pose (service & fallback failed)'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/navigation/add_goal_pose', methods=['POST'])
def add_goal_pose():
    """API endpoint to add a waypoint to the navigation plan"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 or TF listener not available'}), 503
    
    try:
        data = request.get_json()
        if not data or 'pose' not in data:
            return jsonify({'error': 'Missing pose data'}), 400
        
        pose_data = data['pose']
        
        # Extract position and orientation from pose data
        position = pose_data.get('position', [0.0, 0.0, 0.0])
        orientation = pose_data.get('orientation', [0.0, 0.0, 0.0, 1.0])
        
        # Add waypoint to the navigation plan
        success = tf_listener.add_waypoint(position, orientation, pose_data.get('name', 'waypoint'))
        
        if success:
            waypoint_count = len(tf_listener.get_waypoints())
            return jsonify({
                'success': True,
                'message': f'Waypoint "{pose_data["name"]}" added to navigation plan ({waypoint_count} waypoints total)',
                'pose': pose_data,
                'waypoint_count': waypoint_count
            })
        else:
            return jsonify({'error': 'Failed to add waypoint'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/navigation/start_navigation', methods=['POST'])
def start_navigation():
    """API endpoint to start navigation with collected waypoints"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 or TF listener not available'}), 503
    
    try:
        waypoints = tf_listener.get_waypoints()
        
        if not waypoints:
            return jsonify({'error': 'No waypoints available. Add waypoints first.'}), 400
        
        # Publish all waypoints and start navigation
        success = tf_listener.start_navigation_with_waypoints()
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Navigation started with {len(waypoints)} waypoints',
                'waypoint_count': len(waypoints),
                'waypoints': waypoints
            })
        else:
            return jsonify({'error': 'Failed to start navigation'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/navigation/clear_waypoints', methods=['POST'])
def clear_waypoints():
    """API endpoint to clear all waypoints"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 or TF listener not available'}), 503
    
    try:
        tf_listener.clear_waypoints()
        return jsonify({
            'success': True,
            'message': 'All waypoints cleared',
            'waypoint_count': 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/navigation/cancel_navigation', methods=['POST'])
def cancel_navigation():
    """API endpoint to cancel current waypoint navigation"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 or TF listener not available'}), 503
    
    try:
        success = tf_listener.cancel_waypoint_navigation()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Navigation canceled successfully'
            })
        else:
            return jsonify({'error': 'No active navigation to cancel or failed to cancel'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/navigation/status', methods=['GET'])
def get_navigation_status():
    """API endpoint to get navigation status"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 or TF listener not available'}), 503
    
    try:
        is_active = tf_listener.is_navigation_active()
        waypoints = tf_listener.get_waypoints()
        
        return jsonify({
            'is_navigation_active': is_active,
            'waypoint_count': len(waypoints),
            'waypoints': waypoints
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/navigation/waypoints', methods=['GET'])
def get_waypoints():
    """API endpoint to get current waypoints"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 or TF listener not available'}), 503
    
    try:
        waypoints = tf_listener.get_waypoints()
        return jsonify({
            'waypoints': waypoints,
            'waypoint_count': len(waypoints)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/navigation/plan', methods=['POST'])
def plan_navigation():
    """API endpoint to plan navigation with waypoints"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 or TF listener not available'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing navigation data'}), 400
        
        init_pose = data.get('init_pose')
        goal_pose = data.get('goal_pose')
        waypoints = data.get('waypoints', [])
        
        if not init_pose or not goal_pose:
            return jsonify({'error': 'Initial pose and goal pose are required'}), 400
        
        # Here you would interface with your ROS2 navigation stack
        # This could involve:
        # 1. Setting the initial pose
        # 2. Planning a path through waypoints to the goal
        # 3. Returning the planned path
        
        navigation_plan = {
            'init_pose': init_pose,
            'goal_pose': goal_pose,
            'waypoints': waypoints,
            'status': 'planned',
            'path_length': len(waypoints) + 2  # init + waypoints + goal
        }
        
        return jsonify({
            'success': True,
            'message': 'Navigation planned successfully',
            'plan': navigation_plan
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tf/frames')
def get_tf_frames():
    """API endpoint to get available TF frames"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 TF not available'}), 503
    
    try:
        frame_tree = tf_listener.get_transform_tree()
        return jsonify({
            'frames': list(frame_tree.keys()) if frame_tree else [],
            'frame_tree': frame_tree
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tf/transforms')
def get_tf_transforms():
    """API endpoint to get all current transforms"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 TF not available'}), 503
    
    try:
        transforms = tf_listener.get_all_current_transforms()
        return jsonify({
            'transforms': transforms,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tf/transform/<target_frame>/<source_frame>')
def get_specific_transform(target_frame, source_frame):
    """API endpoint to get transform between specific frames"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 TF not available'}), 503
    
    try:
        transform = tf_listener.get_transform(target_frame, source_frame)
        if transform:
            return jsonify({'transform': transform})
        else:
            return jsonify({'error': f'Transform from {source_frame} to {target_frame} not available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/robot/pose')
def get_robot_pose():
    """API endpoint to get robot pose"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 TF not available'}), 503
    
    try:
        # Try different common frame combinations
        base_frames = ['base_link', 'base_footprint', 'robot_base']
        map_frames = ['map', 'odom', 'world']
        
        for map_frame in map_frames:
            for base_frame in base_frames:
                pose = tf_listener.get_robot_pose(base_frame, map_frame)
                if pose:
                    return jsonify({'pose': pose})
        
        return jsonify({'error': 'Robot pose not available in any known frame combination'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/goal/pose')
def get_goal_pose():
    """API endpoint to get goal pose"""
    if not ROS2_AVAILABLE or not tf_listener:
        return jsonify({'error': 'ROS2 not available'}), 503
    
    try:
        goal_pose = tf_listener.get_goal_pose()
        if goal_pose:
            return jsonify({'goal_pose': goal_pose})
        else:
            return jsonify({'error': 'No goal pose available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# YOLO Integration Endpoints
@app.route('/api/yolo/video_feed')
def yolo_video_proxy():
    """Proxy endpoint for YOLO video feed to avoid CORS issues"""
    def generate():
        try:
            # Connect to YOLO service
            response = requests.get('http://localhost:5001/video_feed', stream=True, timeout=5)
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
            else:
                # Return error frame if YOLO service is not available
                yield generate_error_frame('YOLO Service Unavailable')
        except Exception as e:
            print(f"Error proxying YOLO video feed: {e}")
            # Return error frame
            yield generate_error_frame('Connection Error')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_error_frame(message):
    """Generate an error frame with a message"""
    if CV2_AVAILABLE:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, message, (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
    else:
        # Fallback: create a simple error image using PIL
        from PIL import Image, ImageDraw, ImageFont
        error_img = Image.new('RGB', (640, 480), color='black')
        draw = ImageDraw.Draw(error_img)
        try:
            font = ImageFont.load_default()
            draw.text((50, 240), message, fill=(255, 0, 0), font=font)
        except:
            draw.text((50, 240), message, fill=(255, 0, 0))
        
        import io
        buffer = io.BytesIO()
        error_img.save(buffer, format='JPEG')
        frame_bytes = buffer.getvalue()
    
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/api/yolo/status')
def yolo_status():
    """API endpoint to check YOLO service status"""
    try:
        response = requests.get('http://localhost:5001/', timeout=3)
        if response.status_code == 200:
            return jsonify({
                'status': 'connected',
                'port': 5001,
                'service': 'available'
            })
        else:
            return jsonify({
                'status': 'error',
                'port': 5001,
                'service': 'unavailable',
                'error': f'HTTP {response.status_code}'
            }), 503
    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 'disconnected',
            'port': 5001,
            'service': 'unavailable',
            'error': str(e)
        }), 503

@app.route('/api/yolo/status/<int:port>')
def yolo_status_custom_port(port):
    """API endpoint to check YOLO service status on custom port"""
    try:
        response = requests.get(f'http://localhost:{port}/', timeout=3)
        if response.status_code == 200:
            return jsonify({
                'status': 'connected',
                'port': port,
                'service': 'available'
            })
        else:
            return jsonify({
                'status': 'error',
                'port': port,
                'service': 'unavailable',
                'error': f'HTTP {response.status_code}'
            }), 503
    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 'disconnected',
            'port': port,
            'service': 'unavailable',
            'error': str(e)
        }), 503

@app.route('/api/yolo/stats')
def yolo_stats():
    """API endpoint to get YOLO detection statistics"""
    try:
        response = requests.get('http://localhost:5001/api/stats', timeout=3)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                'error': f'HTTP {response.status_code}',
                'person_count': 0,
                'fps': 0.0,
                'active': False
            }), 503
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': str(e),
            'person_count': 0,
            'fps': 0.0,
            'active': False
        }), 503

@app.route('/api/yolo/stats/<int:port>')
def yolo_stats_custom_port(port):
    """API endpoint to get YOLO detection statistics from custom port"""
    try:
        response = requests.get(f'http://localhost:{port}/api/stats', timeout=3)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                'error': f'HTTP {response.status_code}',
                'person_count': 0,
                'fps': 0.0,
                'active': False
            }), 503
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': str(e),
            'person_count': 0,
            'fps': 0.0,
            'active': False
        }), 503

# SocketIO events for real-time TF updates
@socketio.on('start_tf_updates')
def handle_start_tf_updates():
    """Start sending real-time TF updates"""
    if not ROS2_AVAILABLE or not tf_listener:
        emit('tf_error', {'error': 'ROS2 TF not available'})
        return
    
    # Start sending updates every 100ms
    def send_tf_updates():
        while True:
            try:
                transforms = tf_listener.get_all_current_transforms()
                
                # Get robot pose
                robot_pose = None
                base_frames = ['base_link', 'base_footprint', 'robot_base']
                map_frames = ['map', 'odom', 'world']
                
                for map_frame in map_frames:
                    for base_frame in base_frames:
                        pose = tf_listener.get_robot_pose(base_frame, map_frame)
                        if pose:
                            robot_pose = pose
                            break
                    if robot_pose:
                        break
                
                socketio.emit('tf_update', {
                    'transforms': transforms,
                    'robot_pose': robot_pose,
                    'goal_pose': tf_listener.get_goal_pose(),
                    'timestamp': time.time()
                })
            except Exception as e:
                socketio.emit('tf_error', {'error': str(e)})
                break
            time.sleep(0.1)  # 10 Hz updates
    
    # Start update thread
    update_thread = threading.Thread(target=send_tf_updates, daemon=True)
    update_thread.start()

@socketio.on('stop_tf_updates')
def handle_stop_tf_updates():
    """Stop sending real-time TF updates"""
    # Updates will stop when the socket disconnects
    pass

# Global variables for plan updates
plan_update_thread = None
plan_updates_active = False

@socketio.on('start_plan_updates')
def handle_start_plan_updates():
    """Start sending real-time plan updates"""
    global plan_update_thread, plan_updates_active
    
    if not ROS2_AVAILABLE or not tf_listener:
        emit('plan_error', {'error': 'ROS2 not available'})
        return
    
    if plan_updates_active:
        return  # Already active
    
    plan_updates_active = True
    
    # Start sending updates every 200ms
    def send_plan_updates():
        global plan_updates_active
        while plan_updates_active:
            try:
                robot_plan = tf_listener.get_robot_plan()
                if robot_plan:
                    socketio.emit('plan_update', {
                        'plan': robot_plan['points'],
                        'frame_id': robot_plan['frame_id'],
                        'timestamp': time.time()
                    })
            except Exception as e:
                socketio.emit('plan_error', {'error': str(e)})
                break
            time.sleep(0.2)  # 5 Hz updates
    
    # Start update thread
    plan_update_thread = threading.Thread(target=send_plan_updates, daemon=True)
    plan_update_thread.start()

@socketio.on('stop_plan_updates')
def handle_stop_plan_updates():
    """Stop sending real-time plan updates"""
    global plan_updates_active
    plan_updates_active = False

if __name__ == '__main__':
    # Start ROS2 in background thread
    start_ros2_thread()
    
    try:
        socketio.run(app, debug=True, host='0.0.0.0', port=5003, allow_unsafe_werkzeug=True)
    finally:
        # Clean up ROS2
        stop_ros2()
