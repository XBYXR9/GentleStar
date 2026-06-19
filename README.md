# Duckietown Autonomous Navigation — TUM

Autonomous navigation system for Duckietown built by Yahia Taha and Dorde Vidakovic at TUM.

## What it does
- Lane following using OpenCV (yellow + white line detection)
- Duck detection and avoidance (stop or go-around maneuver)
- Red stop line detection (stops 3 seconds with cooldown)
- A* path planning for intersection navigation
- Live camera stream with detection overlays at http://localhost:5000

## Requirements
- Python 3.9+
- duck3 robot running with rosbridge on port 9001
- Windows with WSL2 (Ubuntu) or any Linux machine

## Setup
```bash
python3 -m venv duckietown-env
source duckietown-env/bin/activate
pip install roslibpy opencv-python flask numpy pyyaml
```

## Networking (Windows only)
Run in PowerShell as admin after every reboot:
```powershell
netsh interface portproxy add v4tov4 listenport=22 listenaddress=0.0.0.0 connectport=22 connectaddress=192.168.137.233
netsh interface portproxy add v4tov4 listenport=9001 listenaddress=0.0.0.0 connectport=9001 connectaddress=192.168.137.233
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=192.168.137.233
netsh interface portproxy add v4tov4 listenport=9000 listenaddress=0.0.0.0 connectport=9000 connectaddress=192.168.137.233
```
Note: Replace `192.168.137.233` with duck3's current IP from Windows Mobile Hotspot settings.

## Running the robot

### 1. Start the robot containers
```bash
ssh duckie@duck3.local   # password: quackquack
docker start duckiebot-interface
docker start rosbridge-websocket
```

### 2. Run the main navigation script
```bash
source duckietown-env/bin/activate
python3 packages/navigation/src/navigate.py
```
Open http://localhost:5000 to see the live camera feed with all detections.

Place the robot at tile (0,4) on the TUM track facing East.

### 3. Manual keyboard control (optional)
```bash
python3 packages/navigation/src/keyboard_drive.py
```
- W = forward
- S = backward  
- A = turn left
- D = turn right
- Space = stop
- Q = quit

## Files
| File | Description |
|------|-------------|
| `packages/navigation/src/navigate.py` | Main script: lane following, duck detection, intersection navigation |
| `packages/navigation/src/a_star_planner.py` | A* path planner for the TUM track |
| `packages/navigation/src/tum_map.yaml` | TUM lab track map (7x6 tile grid) |
| `packages/navigation/src/keyboard_drive.py` | Manual keyboard control |

## Route
- Start: tile (0,4) facing East
- Goal: tile (5,2)
- Intersection at (3,4): turn LEFT
- After intersection: GOAL REACHED, robot stops

## Architecture
Everything runs on the laptop via roslibpy connecting to duck3's rosbridge websocket (port 9001).
No custom code is deployed to the robot — only standard Duckietown Docker containers run on duck3.

## System Overview
Laptop (WSL)                    duck3 robot

─────────────────               ──────────────────

navigate.py                     duckiebot-interface

│ OpenCV detection               │ camera

│ A* planning                    │ screen

│ State machine              rosbridge-websocket

└──── roslibpy ─────────────────►│ port 9001

websocket               car-interface

│ wheels

## State Machine
- LANE_FOLLOWING → normal driving using lane detection
- DUCK_STOP → stopped because duck is blocking path
- GO_AROUND → executing maneuver around duck
- RED_STOP → stopped at red stop line for 3 seconds
- INTERSECTION → executing A* planned turn

## PPO Reinforcement Learning (attempted)
We trained a PPO agent for 183M steps in simulation using stable-baselines3.
The agent learned to follow lanes in simulation but exhibited circular motion
on the real robot due to the sim-to-real gap. We documented this as a research
finding and switched to the classical OpenCV approach.
