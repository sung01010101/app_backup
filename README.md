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

## Install Python 3.12 and Python 3.11 (Skip if installed)
1) Install DeadSnakes PPA
```
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```
2) Check which version(s) you haven't install yet
```
python3 --version
```

3) Install python version you don't have
```
sudo apt install python3.12
sudo apt install python3.11
```

4) Verify installation
```
python3 --version
```

## First-time setup
1) Download package and cd into package
```
git clone https://github.com/sung01010101/app_backup.git
cd ./app_backup
```

2) Make scripts executable
```
sudo chmod +x ./scripts/*.sh
```

3) Create virtual environments and install requirements
- Build both at once:
```
./scripts/build_both_envs.sh
```
- Or build individually:
```
./scripts/build_rviz_env.sh
./scripts/build_yolo_env.sh
```

4) (Optional) Verify Python versions
```
rviz_web/.venv-py3.12/bin/python -V
yolo_web/.venv-py3.11/bin/python -V
```

## Running the apps

### Start/Stop YOLO service (yolo_web)
Default port: 5004
```
./scripts/start_yolo_web.sh
# ... later
./scripts/stop_yolo_web.sh
```
Open http://localhost:5004

### Start/Stop RVIZ web UI (rviz_web)
Default port: 5003
```
./scripts/start_rviz_web.sh
# ... later
./scripts/stop_rviz_web.sh
```
Open http://localhost:5003

## Useful endpoints
- rviz_web
  - `/api/tf/frames`, `/api/tf/transforms`, `/api/robot/pose`
  - YOLO proxy: `/api/yolo/status`, `/api/yolo/stats`
- yolo_web
  - Video: `/video_feed`
  - Stats: `/api/stats`
  - Health: `/api/health`

## Troubleshooting
- python3.12/python3.11 not found: install via your package manager or pyenv.
- Virtualenv missing: re-run build scripts under scripts/.
- Camera not available: ensure user is in the `video` group or run with appropriate permissions.
- ROS 2 not sourced: source `/opt/ros/jazzy/setup.bash` before running rviz_web.

## Development
- Virtualenvs are ignored by Git via per-app `.gitignore` files.
- After edits, reinstall dependencies if you changed `requirements.txt`:
```
# in project folder after activating the venv
pip install -r requirements.txt
```
