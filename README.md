# app_backup

Two Flask services:
- rviz_web (Python 3.12): ROS 2 TF/Navigation web UI with Socket.IO
- yolo_web (Python 3.11): YOLOv8 camera streaming and detection stats

## Prerequisites
- Linux with bash
- Python 3.12 and 3.11 available as `python3.12` and `python3.11`
- ROS 2 Jazzy installed and sourceable: `/opt/ros/jazzy/setup.bash` (for rviz_web)
- Camera available at `/dev/video0` (or adjust in code)
- Internet access to install Python packages

## First-time setup
1) Make scripts executable
```
chmod +x build_envs.sh create_rviz_web.sh create_yolo_web.sh \
  rviz_web/start_rviz_web.sh yolo_web/start_yolo_web.sh
```

2) Create virtual environments and install requirements
- Build both at once:
```
./build_envs.sh
```
- Or build individually:
```
./create_rviz_web.sh
./create_yolo_web.sh
```

3) (Optional) Verify Python versions
```
rviz_web/.venv-py3.12/bin/python -V
yolo_web/.venv-py3.11/bin/python -V
```

## Running the apps

### Start YOLO service (yolo_web)
Default port: 5004
```
./start_yolo_web.sh
```
Open http://localhost:5004

Note: rviz_web proxies YOLO video from http://localhost:5001 by default. To integrate seamlessly you can:
- Change yolo_web to run on port 5001 (edit `app.py` or export a PORT variable if you add support), or
- Keep yolo_web on 5004 and adjust the rviz_web code for the proxy URL (video proxy currently hard-coded to 5001).

### Start RVIZ web UI (rviz_web)
Default port: 5003
```
./start_rviz_web.sh
```
Open http://localhost:5003

## Useful endpoints
- rviz_web
  - `/api/tf/frames`, `/api/tf/transforms`, `/api/robot/pose`
  - YOLO proxy: `/api/yolo/status`, `/api/yolo/stats` (defaults to port 5001)
- yolo_web
  - Video: `/video_feed`
  - Stats: `/api/stats`
  - Health: `/api/health`

## Troubleshooting
- python3.12/python3.11 not found: install via your package manager or pyenv.
- Virtualenv missing: re-run build scripts.
- Camera not available: ensure user is in the `video` group or run with appropriate permissions.
- ROS 2 not sourced: source `/opt/ros/jazzy/setup.bash` before running rviz_web.

## Development
- Virtualenvs are ignored by Git via per-app `.gitignore` files.
- After edits, reinstall dependencies if you changed `requirements.txt`:
```
# in project folder after activating the venv
pip install -r requirements.txt
```
