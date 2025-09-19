# app.py
import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import atexit
import time
import threading

app = Flask(__name__)

# Global detection statistics
detection_stats = {
    'person_count': 0,
    'fps': 0.0,
    'last_update': time.time(),
    'total_detections': 0,
    'active': False
}
stats_lock = threading.Lock()

# Initialize YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

# Initialize camera
print("Initializing camera...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    cap = None
else:
    print("Camera initialized successfully")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Reduce camera buffer to minimize delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Cleanup function
def cleanup():
    global cap
    if cap is not None:
        print("Releasing camera...")
        cap.release()

atexit.register(cleanup)

def generate_frames():
    global detection_stats
    
    if cap is None:
        # Return error image
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, 'Camera not available', (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    # Initialize FPS calculation variables with rolling average
    frame_times = []
    fps = 0
    frame_count = 0
    current_detections = []
    fps_window_size = 30  # Calculate FPS over 30 frames for stability
    
    # Update stats to show we're active
    with stats_lock:
        detection_stats['active'] = True
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera")
            break

        frame_count += 1
        current_time = time.time()
        
        # Calculate rolling average FPS
        frame_times.append(current_time)
        if len(frame_times) > fps_window_size:
            frame_times.pop(0)
        
        if len(frame_times) >= 2:
            time_span = frame_times[-1] - frame_times[0]
            if time_span > 0:
                fps = (len(frame_times) - 1) / time_span

        # Run YOLO detection every 3rd frame to reduce processing load
        if frame_count % 3 == 0:
            try:
                # Resize frame for faster inference (reduce resolution by half)
                h, w = frame.shape[:2]
                small_frame = cv2.resize(frame, (w//2, h//2))
                
                # Run inference on smaller frame with optimized settings
                results = model(small_frame, classes=[0], conf=0.5, imgsz=320, verbose=False)[0]
                
                # Store detection results for use in subsequent frames
                current_detections = []
                person_count = 0
                if results.boxes is not None and len(results.boxes) > 0:
                    for result in results.boxes:
                        confidence = float(result.conf[0].item())
                        x1, y1, x2, y2 = map(int, result.xyxy[0] * 2)  # Scale by 2
                        current_detections.append((x1, y1, x2, y2, confidence))
                        person_count += 1
                
                # Update detection statistics
                with stats_lock:
                    detection_stats['person_count'] = person_count
                    detection_stats['fps'] = round(fps, 1)
                    detection_stats['last_update'] = current_time
                    detection_stats['total_detections'] += person_count
                    
            except Exception as e:
                print(f"YOLO detection error: {e}")
                current_detections = []
                with stats_lock:
                    detection_stats['person_count'] = 0
        
        # Draw stored detections on every frame
        if current_detections:
            for x1, y1, x2, y2, confidence in current_detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Person {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display FPS and detection count on frame
        fps_text = f'FPS: {fps:.1f}'
        cv2.putText(frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display person count
        if current_detections:
            count_text = f'Persons: {len(current_detections)}'
            cv2.putText(frame, count_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Mark as inactive when loop ends
    with stats_lock:
        detection_stats['active'] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_detection_stats():
    """API endpoint to get current detection statistics"""
    with stats_lock:
        current_stats = detection_stats.copy()
    
    # Check if stats are recent (within last 2 seconds)
    time_since_update = time.time() - current_stats['last_update']
    if time_since_update > 2.0:
        current_stats['active'] = False
        current_stats['fps'] = 0.0
    
    return jsonify(current_stats)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'camera_available': cap is not None,
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=5004, debug=False)
