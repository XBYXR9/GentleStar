import roslibpy
import base64
import numpy as np
import cv2
import time
import atexit
import signal
import sys
import os
import threading
import yaml
from flask import Flask, Response, request

# --- BACKGROUND CONTAINER MANAGEMENT ---
print("Deactivating baseline navigation container paths...")
os.system("ssh duckie@duck3.local 'docker stop demo_indefinite_navigation' > /dev/null 2>&1 &")

app = Flask(__name__)
latest_jpeg = None
lock = threading.Lock()
simulation_mode = False

# ==============================================================================
# 1. 🗺️ DIRECT EMBEDDED A* PLANNER LOGIC & TOPO MATRIX SPECIFICATIONS
# ==============================================================================
MAP_PATH = os.path.expanduser('~/tum_map.yaml')

# Compass primitives matching a_star_planner.py
DELTAS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
LEFT_OF = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}

OPENINGS = {
    "straight/N": {"N", "S"}, "straight/S": {"N", "S"},
    "straight/E": {"E", "W"}, "straight/W": {"E", "W"},
    "curve_left/N": {"S", "W"}, "curve_left/E": {"W", "N"},
    "curve_left/S": {"N", "E"}, "curve_left/W": {"E", "S"},
    "curve_right/N": {"S", "E"}, "curve_right/E": {"W", "S"},
    "curve_right/S": {"N", "W"}, "curve_right/W": {"E", "N"},
    "3way_left/N": {"N", "S", "W"}, "3way_left/E": {"N", "E", "W"},
    "3way_left/S": {"N", "S", "E"}, "3way_left/W": {"S", "E", "W"},
    "4way": {"N", "S", "E", "W"},
}

def load_track_map():
    """Dynamically parses tiles straight from the local yaml track file."""
    try:
        with open(MAP_PATH, 'r') as f:
            data = yaml.safe_load(f)
        raw_tiles = data["tiles"]
        tiles = [[str(cell).strip() for cell in row] for row in raw_tiles]
        h = len(tiles)
        w = max(len(r) for r in tiles) if h > 0 else 0
        return tiles, w, h
    except Exception as e:
        print(f"⚠️ Map file unavailable ({e}). Using default TUM 7x6 matrix.")
        fallback = [
            ["grass","grass","curve_right/N","straight/E","straight/E","curve_left/N"],
            ["grass","curve_right/N","curve_right/S","grass","grass","straight/N"],
            ["curve_right/N","curve_right/S","grass","curve_right/N","straight/E","3way_left/N"],
            ["straight/N","grass","grass","straight/N","grass","straight/N"],
            ["3way_left/S","straight/E","straight/E","4way","straight/E","3way_left/N"],
            ["straight/N","grass","grass","straight/N","grass","straight/N"],
            ["curve_right/W","straight/E","straight/E","3way_left/E","straight/E","curve_right/S"]
        ]
        return fallback, 6, 7

MAP_TILES, MAP_W, MAP_H = load_track_map()

def build_adjacency_graph():
    adj = {}
    for j in range(MAP_H):
        for i in range(MAP_W):
            kind = MAP_TILES[j][i]
            if kind not in OPENINGS: continue
            node = (i, j)
            adj.setdefault(node, [])
            for d in OPENINGS[kind]:
                di, dj = DELTAS[d]
                ni, nj = i + di, j + dj
                if 0 <= ni < MAP_W and 0 <= nj < MAP_H:
                    nk = MAP_TILES[nj][ni]
                    if nk in OPENINGS and OPPOSITE[d] in OPENINGS[nk]:
                        adj[node].append((ni, nj))
    return adj

GRAPH_ADJ = build_adjacency_graph()

def astar_search(start, goal):
    if start not in GRAPH_ADJ or goal not in GRAPH_ADJ: return [start]
    import heapq
    open_heap = [(abs(start[0]-goal[0]) + abs(start[1]-goal[1]), 0.0, start)]
    came_from = {}
    best_g = {start: 0.0}
    
    while open_heap:
        _, g, n = heapq.heappop(open_heap)
        if g > best_g.get(n, float('inf')): continue
        if n == goal:
            path = [n]
            while n in came_from:
                n = came_from[n]
                path.append(n)
            path.reverse()
            return path
        for nb in GRAPH_ADJ.get(n, []):
            tentative_g = g + 1.0
            if tentative_g < best_g.get(nb, float('inf')):
                best_g[nb] = tentative_g
                came_from[nb] = n
                f = tentative_g + (abs(nb[0]-goal[0]) + abs(nb[1]-goal[1]))
                heapq.heappush(open_heap, (f, tentative_g, nb))
    return [start]

def compute_path_turn_decisions(path):
    turn_tiles = {}
    intersection_tiles = []
    if len(path) < 3: return turn_tiles, intersection_tiles
    
    for k in range(1, len(path) - 1):
        prev, cur, nxt = path[k - 1], path[k], path[k + 1]
        kind = MAP_TILES[cur[1]][cur[0]]
        if not (kind.startswith("3way_") or kind == "4way"): continue
        
        in_d = None
        for d, (x, y) in DELTAS.items():
            if (cur[0]-prev[0], cur[1]-prev[1]) == (x, y): in_d = d
        out_d = None
        for d, (x, y) in DELTAS.items():
            if (nxt[0]-cur[0], nxt[1]-cur[1]) == (x, y): out_d = d
            
        if in_d and out_d:
            intersection_tiles.append(cur)
            if in_d == out_d: turn_tiles[cur] = "straight"
            elif LEFT_OF[in_d] == out_d: turn_tiles[cur] = "left"
            elif RIGHT_OF[in_d] == out_d: turn_tiles[cur] = "right"
    return turn_tiles, intersection_tiles

# --- PATH DATA ENGINE INITIAL STATE ---
ROUTE = [(0,4), (1,4), (2,4), (3,4), (3,3), (3,2), (4,2), (5,2)]
ROUTE_START = (0,4)
ROUTE_GOAL = (5,2)
ROUTE_TURN_TILES = {(3,4): "left"}
ROUTE_INTERSECTION_TILES = [(3,4)]

map_marker_tile = ROUTE_START
map_intersections_passed = 0

def update_active_trajectory(start_coord, goal_coord):
    """Calculates A* paths on the fly from the web interface inputs."""
    global ROUTE, ROUTE_START, ROUTE_GOAL, ROUTE_TURN_TILES, ROUTE_INTERSECTION_TILES
    global map_marker_tile, map_intersections_passed, current_state
    
    computed_path = astar_search(start_coord, goal_coord)
    turn_maps, inter_lists = compute_path_turn_decisions(computed_path)
    
    with lock:
        ROUTE_START = start_coord
        ROUTE_GOAL = goal_coord
        ROUTE = computed_path
        ROUTE_TURN_TILES = turn_maps
        ROUTE_INTERSECTION_TILES = inter_lists
        map_marker_tile = ROUTE_START
        map_intersections_passed = 0
        current_state = STATE_LANE_FOLLOWING
    print(f"🎯 Strategized fresh A* vector route mapping: {ROUTE}")

def map_reach_intersection():
    global map_marker_tile, map_intersections_passed
    if map_intersections_passed < len(ROUTE_INTERSECTION_TILES):
        map_marker_tile = ROUTE_INTERSECTION_TILES[map_intersections_passed]
        map_intersections_passed += 1

def map_finish_turn():
    global map_marker_tile
    map_marker_tile = ROUTE_GOAL

# ==============================================================================
# 🏎️ AUTONOMOUS SPEED CONTROLLERS
# ==============================================================================
LANE_SPEED = 0.075             
TRACK_HALF_WIDTH = 155         
STEERING_MAX_YAW = 2.4         
STEERING_TORQUE_FLOOR = 0.52   

# --- STATE + TIMING CONSTANTS ---
STATE_LANE_FOLLOWING = "lane_following"
STATE_RED_STOP = "red_line_stopped"
STATE_GO_AROUND = "go_around_maneuver"
STATE_INTERSECTION_TURN = "intersection_maneuver"
COOLDOWN_DURATION = 6
turn_sequence_active = []
turn_step_index = 0

current_state = STATE_LANE_FOLLOWING
state_start_time = time.time()
last_red_line_time = 0
last_overtake_time = 0

# Maneuver context frames
active_turn_direction = 'none'
maneuver_step = 0
step_start_time = time.time()

# Timed step configurations.
# NOTE: these are now FALLBACK upper bounds only. Vision breakout (see
# STATE_INTERSECTION_TURN) hands control back to lane following as soon as a real
# exit lane is reacquired, so exact durations no longer need to be perfect.
# Retuned so that, absent any vision, each arc still ends roughly aligned with
# the exit lane instead of massively over-rotating:
#   step0 = brief creep into the box, step1 = the ~90 deg arc (v>0 so it is an arc,
#   not an in-place spin), step2 = short settle straight.
# (v, omega, seconds). Old left/right step1 were pure spins ~200 deg+; fixed.
INTERSECTION_LEFT_STEPS = [(0.08, 0.0, 0.8), (0.10, 1.20, 1.8), (0.07, 0.0, 1.0)]
INTERSECTION_RIGHT_STEPS = [(0.08, 0.0, 0.5), (0.09, -1.40, 1.4), (0.07, 0.0, 1.0)]
INTERSECTION_STRAIGHT_STEPS = [(0.09, 0.0, 3.0)]

# --- INTERSECTION EXIT / MANEUVER TUNING (easy-to-tweak in one place) ---
# Minimum seconds to commit to a turn before we trust vision enough to break out,
# so we don't immediately snap back onto the lane we just left (the entry lane can
# still look valid in the first moments of a turn). Vision breakout is PRIMARY; the
# step durations above are the fallback upper bound. Left/right need enough time to
# physically rotate away from the entry lane; straight is collinear so exiting early
# is harmless. Tune down if turns overshoot, up if they cut the turn short.
MANEUVER_MIN_S = {'left': 1.5, 'right': 1.5, 'straight': 1.0}
# Right-turn white-line hug tuning
WHITE_HUG_TARGET_FRAC = 0.78   # keep the white line ~78% across the frame (to our right)
WHITE_HUG_GAIN = 2.0           # P gain on white-line offset -> omega
WHITE_HUG_CLAMP = 2.2          # clamp on hug omega
RIGHT_SEARCH_V = 0.06          # when no white visible yet, creep+curve right to find it
RIGHT_SEARCH_OMEGA = -1.6
RIGHT_HUG_MAX_S = 6.0          # absolute cap: never hug forever, hand back after this

# --- DUCK AVOIDANCE TUNING ---
DUCK_AVOIDANCE_ON = True        # RE-ENABLED (was gated off with False)
DUCK_TRIGGER_FRAMES = 3         # consecutive confirmed-duck frames before we swerve (debounce)
DUCK_TRIGGER_AREA = 3500        # min blob area (closeness) worth avoiding; is_duck() floor is 2500
DUCK_COOLDOWN_S = 8             # min seconds between go-around maneuvers
GO_AROUND_STEP3_MAX_S = 2.5     # max search time in the swerve-back reacquire step
GO_AROUND_MAX_S = 12.0          # overall cap on the whole go-around before we hold/stop

# --- INTERACTIVE KEYBOARD MANUAL LATCH ---
keyboard_engaged = False
manual_v = 0.0
manual_omega = 0.0

# ==============================================================================
# 🤖 ROS NODE HANDSHAKE CHANNELS
# ==============================================================================
try:
    print("Reaching out to local proxy websocket container ports...")
    client_ros = roslibpy.Ros(host='localhost', port=9001)
    client_ros.run(timeout=2)
    
    if client_ros.is_connected:
        print("Connected to duckie3 Central Navigation Hub!")
        cmd_pub = roslibpy.Topic(client_ros, '/duck3/car_cmd_switch_node/cmd', 'duckietown_msgs/Twist2DStamped')
        override_pub = roslibpy.Topic(client_ros, '/duck3/joy_mapper_node/joystick_override', 'duckietown_msgs/BoolStamped')
    else:
        raise Exception("ROS network down.")
except Exception as e:
    print("⚠️ Hardware Link Offline. Waking up Onboard Visual Simulation Engine...")
    simulation_mode = True

def publish_drive(v, omega):
    if simulation_mode: return
    try:
        override_pub.publish(roslibpy.Message({'header': {'stamp': {'secs':0,'nsecs':0}, 'frame_id':''}, 'data': True}))
        cmd_pub.publish(roslibpy.Message({'header': {'stamp': {'secs':0,'nsecs':0}, 'frame_id':''}, 'v': float(v), 'omega': float(omega)}))
    except:
        pass

def release_override():
    if simulation_mode: return
    try:
        cmd_pub.publish(roslibpy.Message({'header': {'stamp': {'secs':0,'nsecs':0}, 'frame_id':''}, 'v': 0.0, 'omega': 0.0}))
        override_pub.publish(roslibpy.Message({'header': {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''}, 'data': False}))
    except:
        pass

atexit.register(release_override)
signal.signal(signal.SIGTERM, lambda s, f: release_override() or os._exit(0))
signal.signal(signal.SIGINT, lambda s, f: release_override() or os._exit(0))

def is_duck(contour):
    area = cv2.contourArea(contour)
    if area < 2500 or area > 50000: return False
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if aspect_ratio > 1.8 or aspect_ratio < 0.4: return False
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False
    solidity = area / hull_area
    rect = cv2.minAreaRect(contour)
    w_rot, h_rot = rect[1]
    rect_area = w_rot * h_rot
    if rect_area == 0: return False
    rotated_rectangularity = area / rect_area
    long_side = max(w_rot, h_rot)
    short_side = min(w_rot, h_rot)
    rotated_aspect_ratio = long_side / short_side if short_side > 0 else 0
    if cv2.arcLength(contour, True) == 0: return False
    approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)
    if len(approx) <= 6 and solidity > 0.85: return False  
    if rotated_aspect_ratio > 1.55 and rotated_rectangularity > 0.72: return False  
    if rotated_aspect_ratio > 1.50: return False
    return True

# ==============================================================================
# 👁️ FRAME SEGREGATION PROCESSING SUITE
# ==============================================================================
def scan_band(yellow_mask, white_mask, w, y_top, y_bot, frame_center_x):
    """Return (yellow_cx, white_cx) for one horizontal band, applying the
    white-must-be-right-of-yellow rule."""
    y_slice = yellow_mask[y_top:y_bot, :int(w * 0.55)]
    w_slice = white_mask[y_top:y_bot, int(w * 0.35):]
    ys = np.where(y_slice > 0)[1]
    ws = np.where(w_slice > 0)[1]
    ycx = int(np.mean(ys)) if len(ys) > 0 else None
    pwhite = int(np.mean(ws)) + int(w * 0.35) if len(ws) > 0 else None
    wcx = None
    if pwhite is not None:
        if ycx is not None:
            if pwhite > ycx + 40:
                wcx = pwhite
        elif pwhite >= frame_center_x:
            wcx = pwhite
    return ycx, wcx

def band_lane_center(ycx, wcx, frame_center_x, half_width):
    if ycx is not None and wcx is not None:
        return (ycx + wcx) // 2
    elif ycx is not None:
        return ycx + half_width
    elif wcx is not None:
        return wcx - half_width
    return None

# How much to weight the FAR (look-ahead) band vs NEAR, in [0,1]. Track is mostly
# curves, so far-weighted. Lower toward 0.4 if it cuts corners / oscillates; raise
# toward 0.7 if it still doesn't turn into curves early enough. This is now WIRED
# INTO the multi-height sampler (previously it was declared but unused): the
# farthest sampled row carries FAR_WEIGHT of the vote, the nearest (1-FAR_WEIGHT).
FAR_WEIGHT = 0.6

# --- LANE-FOLLOWING STEERING TUNING (all in one place; used by the new logic) ---
# STEERING_MAX_YAW / STEERING_TORQUE_FLOOR live in the speed-controller block above.
STEERING_DEADBAND_PX = 12       # ignore centering errors this small -> go straight (kills straight-line hunting)
STEERING_FLOOR_ACTIVATE = 0.12  # only apply the stiction floor once |omega| exceeds this (avoids bang-bang on the gentlest curves)
YELLOW_CENTER_MARGIN = 20       # yellow within this many px of center counts as "at center" (was the hardcoded -20)
OFFTRACK_PERSIST_FRAMES = 4     # consecutive off-track frames required before the recovery turn fires
OFFTRACK_RECOVER_V = 0.06
OFFTRACK_RECOVER_OMEGA = -1.2   # firm corrective turn away from the yellow center line (the old snap-turn, now gated)

# --- STEERING MEMORY (temporal smoothing + lane-loss hold + off-track detect) ---
import collections
_omega_hist = collections.deque(maxlen=5)   # rolling average of recent steering
_last_good_omega = 0.0                       # last steering when lane WAS visible
_lane_lost_frames = 0
_yellow_center_frames = 0                    # consecutive frames yellow sat at/right of center (off-track candidate)
_duck_seen_frames = 0                        # consecutive frames a real (is_duck) blob was confirmed (trigger debounce)

def sample_lane_center(frame, yellow_mask, white_mask, w, h, frame_center_x, draw=False):
    """Weighted multi-height lane sampler — the SINGLE source of truth for
    'where is the lane'. Samples the lane at several look-ahead rows and returns
    one blended lane-center x (or None if nothing lane-like is visible).

    Shared by:
      * normal lane following      (draw=True, keeps the green dashboard trace),
      * intersection exit reacquire (STATE_INTERSECTION_TURN),
      * go-around reacquire         (STATE_GO_AROUND step 3),
    so a lane is detected the SAME robust way everywhere instead of each place
    re-checking near-band pixels differently. FAR_WEIGHT tunes far vs near rows."""
    _rows = [int(h * 0.58), int(h * 0.66), int(h * 0.74), int(h * 0.82)]
    _n = len(_rows)
    _targets = []
    for _i, _sy in enumerate(_rows):
        _yrow = yellow_mask[_sy:_sy + 4, :int(w * 0.60)]
        _yx = np.where(_yrow > 0)[1]
        _ycx = int(np.mean(_yx)) if len(_yx) > 2 else None
        # white: if yellow present, only count white right of it; else accept
        # white on the right half of the frame.
        _wcx = None
        if _ycx is not None:
            _wrow = white_mask[_sy:_sy + 4, _ycx + 40:]
            _wx = np.where(_wrow > 0)[1]
            if len(_wx) > 2:
                _wcx = int(np.mean(_wx)) + _ycx + 40
        else:
            _wrow = white_mask[_sy:_sy + 4, frame_center_x:]
            _wx = np.where(_wrow > 0)[1]
            if len(_wx) > 2:
                _wcx = int(np.mean(_wx)) + frame_center_x
        # YELLOW IS PRIMARY (left boundary); WHITE is the fallback.
        if _ycx is not None and _wcx is not None:
            _t = (_ycx + _wcx) // 2
        elif _ycx is not None:
            _t = _ycx + TRACK_HALF_WIDTH
        elif _wcx is not None:
            _t = _wcx - TRACK_HALF_WIDTH        # follow white, hold offset to its LEFT
        else:
            continue
        # FAR_WEIGHT-driven look-ahead weighting: far row -> FAR_WEIGHT, near row ->
        # (1-FAR_WEIGHT), middle rows interpolate linearly. Now FAR_WEIGHT is a real
        # knob (older code hardcoded 1.0 + (n-1-i)*0.5 and ignored FAR_WEIGHT).
        _frac_far = (_n - 1 - _i) / (_n - 1) if _n > 1 else 0.0
        _wgt = (1.0 - FAR_WEIGHT) + (2.0 * FAR_WEIGHT - 1.0) * _frac_far
        _targets.append((_wgt, _t, _sy))
        if draw:
            cv2.circle(frame, (_t, _sy), 4, (0, 255, 0), -1)
    if not _targets:
        return None
    _sw = sum(wt for wt, _, _ in _targets)
    lane_center = int(sum(wt * t for wt, t, _ in _targets) / _sw)
    if draw and len(_targets) >= 2:
        cv2.polylines(frame, [np.array([(t, sy) for _, t, sy in _targets], np.int32)], False, (0, 255, 0), 2)
    return lane_center

def valid_lane_reacquired(lane_center, yellow_cx, white_cx, frame_center_x):
    """Shared 'is there a REAL lane in front of me' test used when EXITING a
    maneuver (intersection turn or go-around). Trusts the multi-height sampler
    for a coherent center AND requires the near-band structure to look like our
    lane (yellow on the left, or a proper yellow+white pair) — so we don't snap
    onto a stray white edge / the lane we just left mid-turn."""
    if lane_center is None:
        return False
    yellow_on_left = (yellow_cx is not None and yellow_cx < frame_center_x)
    yellow_white_pair = (yellow_cx is not None and white_cx is not None and white_cx > yellow_cx + 40)
    return yellow_on_left or yellow_white_pair

def process_image_frame(frame):
    global latest_jpeg, current_state, state_start_time, last_red_line_time, last_overtake_time
    global turn_sequence_active, turn_step_index, active_turn_direction, TRACK_HALF_WIDTH
    global keyboard_engaged, manual_v, manual_omega, maneuver_step, step_start_time
    # steering / trigger memory (single global declaration up front)
    global _last_good_omega, _lane_lost_frames, _yellow_center_frames, _duck_seen_frames

    h, w = frame.shape[:2]
    frame_center_x = w // 2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([40, 255, 255]))
    white_mask = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 50, 255]))
    red_mask = cv2.inRange(hsv, np.array([0, 110, 60]), np.array([15, 255, 255])) + cv2.inRange(hsv, np.array([160, 110, 60]), np.array([180, 255, 255]))

    # Panoramic duck scanner logic
    duck_roi = np.zeros_like(yellow_mask)
    duck_roi[int(h*0.45):int(h*0.92), 0:w] = yellow_mask[int(h*0.45):int(h*0.92), 0:w]
    duck_contours, _ = cv2.findContours(duck_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    duck_found = False
    duck_x = None
    duck_area = 0                      # area of the closest confirmed duck (0 if none)
    for contour in duck_contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if is_duck(contour):           # is_duck() is the shape filter that keeps non-duck yellow out
            a = cv2.contourArea(contour)
            duck_found = True
            if a > duck_area:          # track the BIGGEST (closest) duck so duck_x/area are meaningful
                duck_area = a
                duck_x = x + cw // 2
            cv2.rectangle(frame, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
            cv2.putText(frame, "DUCK", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif cv2.contourArea(contour) > 800:
            # Rejected yellow -> drawn separately (orange, thin) so false positives are visible.
            cv2.rectangle(frame, (x, y), (x+cw, y+ch), (0, 200, 255), 1)
            cv2.putText(frame, "yellow obj", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

    # Debounce: count consecutive confirmed-duck frames so a single stray blob that
    # slips past is_duck() cannot trigger a full swerve (used by the go-around gate).
    if duck_found:
        _duck_seen_frames += 1
    else:
        _duck_seen_frames = 0

    # FEATURE 7: Scan box locked strictly to right 50% width
    red_roi = np.zeros_like(red_mask)
    red_roi[int(h*0.60):h, :] = red_mask[int(h*0.60):h, :]

    red_line_found = False
    stop_line_contours, _ = cv2.findContours(red_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in stop_line_contours:
        area = cv2.contourArea(contour)
        if area > 60:  # draw even small red so we can see EVERYTHING
            x, y, cw, ch = cv2.boundingRect(contour)
            ar = float(cw) / ch if ch > 0 else 0
            width_pct = int(100 * cw / w)
            ox, oy = x, y + int(h*0.60)
            wide_enough = cw > int(w * 0.20)
            long_enough = ar > 2.2
            if long_enough and wide_enough and area > 150:
                red_line_found = True
                cv2.rectangle(frame, (ox, oy), (ox + cw, oy + ch), (0, 0, 255), 3)
                cv2.putText(frame, f"STOP LINE ar={ar:.1f} w={width_pct}%", (ox, oy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (ox, oy), (ox + cw, oy + ch), (0, 100, 200), 1)
                reason = "thin" if not long_enough else ("narrow" if not wide_enough else "small")
                cv2.putText(frame, f"red? ar={ar:.1f} w={width_pct}% [{reason}]", (ox, oy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 230), 1)

    # NEAR band (in front of bot): centering + feeds all downstream logic
    near_top, near_bot = int(h * 0.72), int(h * 0.78)
    yellow_cx, white_cx = scan_band(yellow_mask, white_mask, w, near_top, near_bot, frame_center_x)

    # FAR band (down the road): anticipates curves
    far_top, far_bot = int(h * 0.55), int(h * 0.61)
    far_yellow_cx, far_white_cx = scan_band(yellow_mask, white_mask, w, far_top, far_bot, frame_center_x)

    # red scan zone (where stop lines are looked for)
    cv2.rectangle(frame, (0, int(h*0.60)), (w-1, h-1), (40, 40, 120), 1)
    cv2.putText(frame, "RED SCAN ZONE", (8, int(h*0.60) + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 160), 1)

    if yellow_cx is not None:
        cv2.rectangle(frame, (yellow_cx - 15, near_top), (yellow_cx + 15, near_bot), (0, 255, 255), 2)
        cv2.putText(frame, "YELLOW(near)", (yellow_cx - 30, near_top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    if white_cx is not None:
        cv2.rectangle(frame, (white_cx - 15, near_top), (white_cx + 15, near_bot), (255, 255, 255), 2)
        cv2.putText(frame, "WHITE(near)", (white_cx - 25, near_top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if far_yellow_cx is not None:
        cv2.rectangle(frame, (far_yellow_cx - 12, far_top), (far_yellow_cx + 12, far_bot), (0, 200, 200), 1)
        cv2.putText(frame, "yellow(far)", (far_yellow_cx - 24, far_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
    if far_white_cx is not None:
        cv2.rectangle(frame, (far_white_cx - 12, far_top), (far_white_cx + 12, far_bot), (200, 200, 200), 1)
        cv2.putText(frame, "white(far)", (far_white_cx - 22, far_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    now = time.time()
    time_in_step = now - step_start_time
    arrow_v, arrow_omega = LANE_SPEED, 0.0

    if keyboard_engaged:
        # HARD CUTOVER: keyboard owns the bot. No lane following, no maneuvers.
        publish_drive(manual_v, manual_omega)
        arrow_v, arrow_omega = manual_v, manual_omega
        current_state = STATE_LANE_FOLLOWING
        cv2.putText(frame, "MANUAL OVERRIDE - AUTONOMY DISABLED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        if current_state == STATE_LANE_FOLLOWING:
            # The multi-height sampler ALWAYS runs first (draw=True keeps the green
            # dashboard trace). The old hardcoded snap-turn used to pre-empt this
            # whole block whenever near-band yellow crossed center — which also
            # happens naturally on sharp curves. It is now folded in below and only
            # fires when it PERSISTS *and* the sampler finds no lane.
            lane_center = sample_lane_center(frame, yellow_mask, white_mask, w, h, frame_center_x, draw=True)

            # Off-track counter: increment ONLY when yellow (the center line) sits at
            # /right of center AND the sampler finds no lane on the same frame. We
            # reset whenever a lane IS found, so a long yellow-near-center CURVE (which
            # steers fine) can't pre-load the counter and then false-fire the recovery
            # snap on a single dropped frame. Both conditions must persist together.
            yellow_at_center = (yellow_cx is not None and yellow_cx >= (frame_center_x - YELLOW_CENTER_MARGIN))
            if yellow_at_center and lane_center is None:
                _yellow_center_frames += 1
            else:
                _yellow_center_frames = 0

            if lane_center is not None:
                # NORMAL WEIGHTED STEERING — bends smoothly with the curve, including
                # the yellow-near-center case. The snap-turn can no longer fight this.
                error = lane_center - frame_center_x
                raw_omega = - (float(error) / frame_center_x) * STEERING_MAX_YAW
                _omega_hist.append(raw_omega)                 # temporal smoothing (kills single-frame jerks)
                omega = sum(_omega_hist) / len(_omega_hist)
                if abs(error) <= STEERING_DEADBAND_PX:
                    omega = 0.0                               # DEADBAND: tiny error -> go straight (no straight-line hunting)
                elif abs(omega) > STEERING_FLOOR_ACTIVATE:
                    # real turn wanted: apply the stiction floor so the wheels actually move
                    omega = max(STEERING_TORQUE_FLOOR, omega) if omega > 0 else min(-STEERING_TORQUE_FLOOR, omega)
                _last_good_omega = omega
                _lane_lost_frames = 0
                publish_drive(LANE_SPEED, omega)
                arrow_v, arrow_omega = LANE_SPEED, omega
            elif _yellow_center_frames >= OFFTRACK_PERSIST_FRAMES:
                # CONFIRMED OFF TRACK: yellow has sat at/right of center for several
                # frames AND the sampler finds no lane -> we've drifted over the center
                # line. This is the OLD snap-turn, now properly gated so it can't fire
                # on ordinary curves. Firm corrective turn away from the yellow.
                publish_drive(OFFTRACK_RECOVER_V, OFFTRACK_RECOVER_OMEGA)
                arrow_v, arrow_omega = OFFTRACK_RECOVER_V, OFFTRACK_RECOVER_OMEGA
                _last_good_omega = OFFTRACK_RECOVER_OMEGA     # keep turning this way if the lane stays lost
                _lane_lost_frames = 0
                cv2.putText(frame, f"OFF TRACK - recovering ({_yellow_center_frames})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # LANE LOST (not confirmed off-track): hold the last good turn briefly
                # so we ride out a curve / bad frame, then stop. (memory unchanged)
                _lane_lost_frames += 1
                if _lane_lost_frames < 25:   # ~2-3s of memory
                    hold = _last_good_omega * 0.92   # decay slowly
                    publish_drive(LANE_SPEED * 0.7, hold)
                    arrow_v, arrow_omega = LANE_SPEED * 0.7, hold
                    cv2.putText(frame, f"LANE LOST - holding turn ({_lane_lost_frames})", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    publish_drive(0.0, 0.0)   # truly lost for a while: stop, don't wander
                    arrow_v, arrow_omega = 0.0, 0.0
                    cv2.putText(frame, "LANE LOST - stopped", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # DUCK AVOIDANCE (re-enabled via DUCK_AVOIDANCE_ON). Trigger only on a
            # confirmed duck that is: shape-valid (is_duck), close enough
            # (duck_area), seen for several frames (_duck_seen_frames debounce), and
            # past the cooldown. state_start_time is stamped so the go-around has an
            # overall timeout and can't loop forever.
            if (DUCK_AVOIDANCE_ON and duck_found and duck_area >= DUCK_TRIGGER_AREA
                    and _duck_seen_frames >= DUCK_TRIGGER_FRAMES
                    and (now - last_overtake_time) > DUCK_COOLDOWN_S):
                current_state = STATE_GO_AROUND; maneuver_step = 0
                state_start_time = now; step_start_time = now
                print("DUCK confirmed -> go-around maneuver")
            elif red_line_found and (now - last_red_line_time) > COOLDOWN_DURATION:
                # Every real stop line triggers an intersection stop (A* gating removed).
                current_state = STATE_RED_STOP; state_start_time = now
                map_reach_intersection()
                print("STOP LINE -> stopping at intersection, awaiting direction (W/A/D)")

        elif current_state == STATE_RED_STOP:
            publish_drive(0.0, 0.0)
            arrow_v, arrow_omega = 0.0, 0.0

        elif current_state == STATE_INTERSECTION_TURN:
            # ROBUST EXIT REACQUISITION: use the SAME multi-height sampler as lane
            # following (draw=True for the dashboard), not just near-band pixels,
            # then confirm the near-band structure looks like our lane. This is what
            # makes turns reliably re-enter lane following instead of falling through
            # to "lost".
            exit_center = sample_lane_center(frame, yellow_mask, white_mask, w, h, frame_center_x, draw=True)
            valid_exit_lane = valid_lane_reacquired(exit_center, yellow_cx, white_cx, frame_center_x)

            time_in_turn = now - state_start_time
            min_turn_s = MANEUVER_MIN_S.get(active_turn_direction, 1.0)
            # Break out the MOMENT a real exit lane is confirmed, but only after a
            # per-direction minimum time so we don't snap back onto the lane we just
            # left. Decoupled from turn_step_index — the old code required
            # turn_step_index >= 1 for left/right, but the right-turn hug below
            # early-returned every frame and never advanced that index, so right
            # turns could miss their reacquisition window. The time guard fixes it.
            can_breakout = valid_exit_lane and time_in_turn >= min_turn_s

            if can_breakout:
                last_red_line_time = time.time()
                map_finish_turn()
                _omega_hist.clear()          # fresh steering memory on the new lane
                current_state = STATE_LANE_FOLLOWING
                arrow_v, arrow_omega = LANE_SPEED, 0.0
                cv2.putText(frame, "EXIT LANE ACQUIRED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            elif active_turn_direction == 'right':
                # RIGHT TURN = actively hug the white line until yellow reappears on
                # the left (handled by can_breakout above). No early return now, so
                # the dashboard video keeps updating through the whole turn.
                WHITE_HUG_TARGET = int(w * WHITE_HUG_TARGET_FRAC)   # want white ~78% across (to our right)
                if white_cx is not None:
                    err = white_cx - WHITE_HUG_TARGET
                    # err<0 (white too central) -> steer left; err>0 (white too far right) -> steer right.
                    hug_omega = - (float(err) / frame_center_x) * WHITE_HUG_GAIN
                    hug_omega = max(-WHITE_HUG_CLAMP, min(WHITE_HUG_CLAMP, hug_omega))
                    publish_drive(0.07, hug_omega)
                    arrow_v, arrow_omega = 0.07, hug_omega
                    cv2.putText(frame, "RIGHT TURN: hugging white line", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                else:
                    publish_drive(RIGHT_SEARCH_V, RIGHT_SEARCH_OMEGA)   # no white yet: curve right to find it
                    arrow_v, arrow_omega = RIGHT_SEARCH_V, RIGHT_SEARCH_OMEGA
                    cv2.putText(frame, "RIGHT TURN: searching for white line", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                # Absolute cap: never hug forever. If vision never confirms the exit
                # lane, hand back to lane following (which has its own lost-lane hold).
                if time_in_turn > RIGHT_HUG_MAX_S:
                    last_red_line_time = time.time()
                    map_finish_turn()
                    _omega_hist.clear()
                    current_state = STATE_LANE_FOLLOWING
                    cv2.putText(frame, "RIGHT TURN: timeout -> lane follow", (10, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                # LEFT / STRAIGHT: run the open-loop arc as a fallback; the vision
                # breakout above cuts it short as soon as the exit lane appears.
                if turn_step_index < len(turn_sequence_active):
                    v, omega, duration = turn_sequence_active[turn_step_index]
                    if time_in_step < duration:
                        publish_drive(v, omega)
                        arrow_v, arrow_omega = v, omega
                    else:
                        turn_step_index += 1; step_start_time = now
                else:
                    last_red_line_time = time.time()
                    map_finish_turn()
                    _omega_hist.clear()
                    current_state = STATE_LANE_FOLLOWING

        elif current_state == STATE_GO_AROUND:
            go_around_elapsed = now - state_start_time   # overall timeout reference
            if maneuver_step == 0:
                # Step 0: swerve OUT (left) to clear the duck (open-loop).
                publish_drive(0.08, 1.2)
                arrow_v, arrow_omega = 0.08, 1.2
                if time_in_step > 1.4: maneuver_step = 1; step_start_time = now
            elif maneuver_step == 1:
                # Step 1: drive straight past the duck until it falls behind on the
                # left (duck_x left of center), or a timeout.
                publish_drive(0.08, 0.0)
                arrow_v, arrow_omega = 0.08, 0.0
                if duck_x is not None and duck_x < frame_center_x - 30:
                    maneuver_step = 2; step_start_time = now
                elif time_in_step > 4.0:
                    maneuver_step = 2; step_start_time = now
            elif maneuver_step == 2:
                # Step 2: swerve BACK (right) toward the lane (open-loop).
                publish_drive(0.07, -1.3)
                arrow_v, arrow_omega = 0.07, -1.3
                if time_in_step > 1.6: maneuver_step = 3; step_start_time = now
            elif maneuver_step == 3:
                # Step 3: VERIFY a real lane is reacquired (same multi-height +
                # structural check as intersection exit) before returning to lane
                # following — replaces the old bare "any yellow/white OR 1.5s".
                rea_center = sample_lane_center(frame, yellow_mask, white_mask, w, h, frame_center_x, draw=True)
                if valid_lane_reacquired(rea_center, yellow_cx, white_cx, frame_center_x):
                    last_overtake_time = now
                    _omega_hist.clear()
                    current_state = STATE_LANE_FOLLOWING
                    cv2.putText(frame, "GO-AROUND: lane reacquired", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                elif time_in_step > GO_AROUND_STEP3_MAX_S or go_around_elapsed > GO_AROUND_MAX_S:
                    # FALLBACK so we never loop forever: if the duck is STILL ahead,
                    # HOLD (stop) rather than charge it; if it's gone, hand back to
                    # lane following (its own lost-lane logic takes over). We stay in
                    # step 3 while holding and keep re-checking for a lane each frame.
                    if duck_found:
                        publish_drive(0.0, 0.0)
                        arrow_v, arrow_omega = 0.0, 0.0
                        cv2.putText(frame, "GO-AROUND FAILED - holding (duck ahead)", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        last_overtake_time = now
                        _omega_hist.clear()
                        current_state = STATE_LANE_FOLLOWING
                else:
                    # keep easing forward-left, searching for the lane
                    publish_drive(0.06, 0.5)
                    arrow_v, arrow_omega = 0.06, 0.5
                    cv2.putText(frame, "GO-AROUND: searching for lane", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # 🏹 FEATURE 8: TESLA DYNAMIC ARROW OVERLAY
    if abs(arrow_omega) < 0.15:
        cv2.arrowedLine(frame, (frame_center_x, int(h * 0.85)), (frame_center_x, int(h * 0.68)), (0, 255, 0), 4, cv2.LINE_AA, 0, 0.25)
    else:
        target_pt_x = frame_center_x - int(arrow_omega * 110)
        target_pt_y = int(h * 0.68)
        arrow_color = (0, 165, 255) if abs(arrow_omega) > 0.8 else (0, 255, 0)
        cv2.arrowedLine(frame, (frame_center_x, int(h * 0.85)), (target_pt_x, target_pt_y), arrow_color, 4, cv2.LINE_AA, 0, 0.25)

    draw_minimap(frame, map_marker_tile)
    _, jpeg = cv2.imencode('.jpg', frame)
    with lock: latest_jpeg = jpeg.tobytes()

# ==============================================================================
# 🎮 SIMULATION FALLBACK LOOP
# ==============================================================================
def simulation_hardware_loop():
    print("🤖 Virtual Camera Loop Streaming Active!")
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "SIMULATION ADAPTER FEED: ROBOT NOT DETECTED", (15, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        cv2.line(frame, (180, 480), (280, 260), (0, 255, 255), 3)   
        cv2.line(frame, (490, 480), (390, 260), (255, 255, 255), 3) 
        if current_state == STATE_RED_STOP:
            cv2.rectangle(frame, (320, 340), (450, 365), (0, 0, 255), -1)
        process_image_frame(frame)
        time.sleep(0.033)

if simulation_mode:
    threading.Thread(target=simulation_hardware_loop, daemon=True).start()
else:
    camera_sub = roslibpy.Topic(client_ros, '/duck3/camera_node/image/compressed', 'sensor_msgs/CompressedImage', throttle_rate=100, queue_length=1)
    camera_sub.subscribe(lambda msg: process_image_frame(cv2.imdecode(np.frombuffer(base64.b64decode(msg['data']), np.uint8), cv2.IMREAD_COLOR)))

# ==============================================================================
# 🌐 DASHBOARD ENGINE SVG GENERATORS
# ==============================================================================
def render_map_svg(current_tile):
    cell = 34
    W, H = MAP_W * cell, MAP_H * cell
    p = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']
    route_set = set(ROUTE)
    for j in range(MAP_H):
        for i in range(MAP_W):
            kind = MAP_TILES[j][i]
            x, y = i*cell, j*cell
            fill = "#244a28" if kind == "grass" else "#3a3a3a"
            p.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#111" stroke-width="1"/>')
            if (i, j) in route_set and kind != "grass":
                p.append(f'<rect x="{x+2}" y="{y+2}" width="{cell-4}" height="{cell-4}" fill="none" stroke="#1f6feb" stroke-width="1.5" opacity="0.5"/>')
    if len(ROUTE) > 1:
        pts = " ".join(f"{i*cell+cell//2},{j*cell+cell//2}" for (i,j) in ROUTE)
        p.append(f'<polyline points="{pts}" fill="none" stroke="#1f9bff" stroke-width="3" opacity="0.85"/>')
    p.append(f'<circle cx="{ROUTE_START[0]*cell+cell//2}" cy="{ROUTE_START[1]*cell+cell//2}" r="10" fill="#22c55e"/>')
    p.append(f'<text x="{ROUTE_START[0]*cell+cell//2}" y="{ROUTE_START[1]*cell+cell//2+4}" font-size="11" fill="black" text-anchor="middle" font-weight="bold">S</text>')
    p.append(f'<circle cx="{ROUTE_GOAL[0]*cell+cell//2}" cy="{ROUTE_GOAL[1]*cell+cell//2}" r="10" fill="#ef4444"/>')
    p.append(f'<text x="{ROUTE_GOAL[0]*cell+cell//2}" y="{ROUTE_GOAL[1]*cell+cell//2+4}" font-size="11" fill="white" text-anchor="middle" font-weight="bold">G</text>')
    for (ti, tj), d in ROUTE_TURN_TILES.items():
        arrow = {"left":"L","right":"R","straight":"^"}.get(d, "?")
        p.append(f'<text x="{ti*cell+cell//2}" y="{tj*cell+cell//2+5}" font-size="13" fill="#ffd000" text-anchor="middle" font-family="sans-serif" font-weight="bold">{arrow}</text>')
    p.append(f'<circle cx="{current_tile[0]*cell+cell//2}" cy="{current_tile[1]*cell+cell//2}" r="9" fill="#ffe600" stroke="#000" stroke-width="2"><animate attributeName="r" values="9;12;9" dur="1s" repeatCount="indefinite"/></circle></svg>')
    return "".join(p)

def draw_minimap(frame, current_tile):
    h, w = frame.shape[:2]
    cell = 11
    mw, mh = MAP_W*cell, MAP_H*cell
    ox, oy = w - mw - 10, 75
    cv2.rectangle(frame, (ox-4, oy-4), (ox+mw+4, oy+mh+4), (20,20,20), -1)
    route_set = set(ROUTE)
    for j in range(MAP_H):
        for i in range(MAP_W):
            color = (36,74,40) if MAP_TILES[j][i] == "grass" else (58,58,58)
            cv2.rectangle(frame, (ox+i*cell, oy+j*cell), (ox+i*cell+cell-1, oy+j*cell+cell-1), color, -1)
            if (i,j) in route_set and MAP_TILES[j][i] != "grass":
                cv2.rectangle(frame, (ox+i*cell, oy+j*cell), (ox+i*cell+cell-1, oy+j*cell+cell-1), (255,140,0), 1)
    cv2.circle(frame, (ox+ROUTE_START[0]*cell+cell//2, oy+ROUTE_START[1]*cell+cell//2), 3, (0,200,0), -1)
    cv2.circle(frame, (ox+ROUTE_GOAL[0]*cell+cell//2, oy+ROUTE_GOAL[1]*cell+cell//2), 3, (0,0,230), -1)
    cv2.circle(frame, (ox+current_tile[0]*cell+cell//2, oy+current_tile[1]*cell+cell//2), 4, (0,230,255), -1)
    cv2.putText(frame, "MAP", (ox, oy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

# ==============================================================================
# 🌐 WEB DASHBOARD INTERFACE
# ==============================================================================
@app.route('/')
def index():
    # Build scannable coordinate select fields straight from the loaded topo map parameters
    start_options = ""
    goal_options = ""
    for j in range(MAP_H):
        for i in range(MAP_W):
            if MAP_TILES[j][i] in OPENINGS:
                s_sel = 'selected' if (i,j) == ROUTE_START else ''
                g_sel = 'selected' if (i,j) == ROUTE_GOAL else ''
                start_options += f'<option value="{i},{j}" {s_sel}>Tile ({i},{j}) - {MAP_TILES[j][i]}</option>'
                goal_options += f'<option value="{i},{j}" {g_sel}>Tile ({i},{j}) - {MAP_TILES[j][i]}</option>'

    # --- Duckietown-themed stylesheet (plain string so the f-string below stays
    # brace-safe; interpolating {page_style} inserts these braces as a VALUE). ---
    page_style = """<style>
      * { box-sizing: border-box; }
      body { background:#141414; color:#f4f4f4; text-align:center;
             font-family:'Segoe UI', system-ui, sans-serif; margin:0; padding:0 20px 30px; }
      /* asphalt header with a dashed yellow road centre-line along the bottom */
      .road-header { margin:0 -20px 20px; padding:20px 20px 24px; border-bottom:5px solid #000;
        background:
          repeating-linear-gradient(90deg, #f6c915 0 46px, transparent 46px 92px) center bottom / 100% 7px no-repeat,
          #0e0e0e; }
      .road-header h1 { margin:0; font-size:1.55em; letter-spacing:.5px; color:#f6c915; text-shadow:0 2px 0 #000; }
      .road-header .sub { color:#b9b9b9; font-size:.82em; margin-top:5px; }
      .card { background:#1e1e1e; border:2px solid #2c2c2c; border-radius:12px; padding:12px 16px; display:inline-block; }
      .card.accent { border-color:#f6c915; }
      #status_box { margin-bottom:16px; background:#f6c915; color:#111; padding:13px; display:inline-block;
        border-radius:12px; font-weight:700; font-size:1.12em; width:78%; border:2px solid #000; box-shadow:0 3px 0 #000; }
      .panel-title { color:#f6c915; font-size:1.05em; font-weight:700; }
      label { color:#d8d8d8; font-size:.9em; }
      select, button { background:#262626; color:#f4f4f4; padding:5px 8px; border-radius:7px;
        border:1px solid #3a3a3a; font-size:.9em; }
      select:focus, button:focus { outline:2px solid #f6c915; }
      button.go { background:#f6c915; color:#111; font-weight:700; border:0; padding:6px 15px; cursor:pointer; }
      button.go:hover { background:#ffd733; }
      .divider { border-left:1px solid #3a3a3a; padding-left:15px; display:flex; gap:10px; align-items:center; }
      #video-frame { width:60%; border:4px solid #f6c915; border-radius:10px; background:#000; }
      #mapbox { background:#141414; border:2px solid #2c2c2c; border-radius:10px; padding:8px; display:inline-block; }
      .maplabel { color:#f6c915; font-size:.75em; margin-bottom:4px; font-weight:700; letter-spacing:.5px; }
      .help { margin-top:12px; background:#1b1b1b; padding:10px 12px; border-radius:10px; border:1px solid #2c2c2c;
        font-size:.75em; text-align:left; color:#cfcfcf; max-width:255px; }
      .help strong { color:#f6c915; }
      kbd { display:inline-block; background:#2c2c2c; border:1px solid #000; border-bottom-width:3px; border-radius:6px;
        padding:1px 6px; font-family:inherit; font-weight:700; color:#f4f4f4; font-size:.95em; }
      kbd.warn { background:#f2a900; color:#111; }
      kbd.stop { background:#2f8f4e; color:#fff; }
      kbd.kill { background:#c0342e; color:#fff; }
    </style>"""

    return f'''
    <html>
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>🦆 duck3 Control Hub</title>
    {page_style}
    </head>
    <body>
        <div class="road-header">
            <h1>🦆 Duckietown · duck3 Control Hub</h1>
            <div class="sub">Live vision telemetry &amp; manual override</div>
        </div>

        <div id="status_box">Autonomous Run Initialized.</div>
        <br>

        <div class="card accent" style="margin-bottom:15px; width:78%; text-align:left;">
            <span class="panel-title">🗺️ Dynamic A* Route Planning Framework</span>
            <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 15px; align-items: center;">
                <div>
                    <label>🏁 Presets: </label>
                    <select id="preset_sel" onchange="applyPreset(this.value)">
                        <option value="none">--- Select Route Preset ---</option>
                        <option value="0,4|5,2">Preset 1: Default Test Track Run (0,4 -> 5,2)</option>
                        <option value="3,6|3,0">Preset 2: North/South Intersection Cross (3,6 -> 3,0)</option>
                        <option value="5,1|0,4">Preset 3: Extended Backtrack Loop (5,1 -> 0,4)</option>
                    </select>
                </div>
                <div class="divider">
                    <div>
                        <label>🟢 Start: </label>
                        <select id="start_coord">{start_options}</select>
                    </div>
                    <div>
                        <label>🔴 Goal: </label>
                        <select id="goal_coord">{goal_options}</select>
                    </div>
                    <button class="go" onclick="calculateCustomPath()">Compute Route</button>
                </div>
            </div>
        </div>
        <br>

        <div style="display:flex; justify-content:center; align-items:flex-start; gap:15px; margin-top:10px;">
            <img id="video-frame" src="/video">
            <div style="text-align:center;">
                <div class="maplabel">🚦 A* LIVE PATH MATRIX</div>
                <div id="mapbox"></div>
                <div class="help">
                    <strong>[FEATURE 6: LATCH INTERLOCK]</strong><br>
                    <kbd class="warn">E</kbd> Toggle Manual Override Mode<br>
                    <kbd>W</kbd>/<kbd>S</kbd> Latch Speed &nbsp; <kbd>A</kbd>/<kbd>D</kbd> Latch Spin<br>
                    <kbd class="stop">Space</kbd> Stop &nbsp; <kbd class="kill">Q</kbd> KILL<br>
                    <hr style="border:0; border-top:1px solid #2c2c2c; margin:6px 0;">
                    <strong>[OFFLINE SIM SHORTCUTS]</strong><br>
                    <kbd>S</kbd> trips a virtual Red Stop Line.
                </div>
            </div>
        </div>
        
        <script>
            function applyPreset(val) {{
                if (val === 'none') return;
                let parts = val.split('|');
                document.getElementById('start_coord').value = parts[0];
                document.getElementById('goal_coord').value = parts[1];
                calculateCustomPath();
            }}

            function calculateCustomPath() {{
                let s = document.getElementById('start_coord').value;
                let g = document.getElementById('goal_coord').value;
                fetch('/set_path?start=' + s + '&goal=' + g)
                    .then(r => r.text())
                    .then(msg => console.log("A* Updated: " + msg));
            }}

            setInterval(function() {{ fetch('/get_map').then(r => r.text()).then(t => {{ document.getElementById("mapbox").innerHTML = t; }}); }}, 300);
            setInterval(function() {{ fetch('/get_status').then(response => response.text()).then(text => {{
                let box = document.getElementById("status_box"); box.innerText = text;
                if (text.includes("AWAITING") || text.includes("DETECTED")) {{ box.style.background = "#c0342e"; box.style.color = "#fff"; }}
                else if (text.includes("LATCH")) {{ box.style.background = "#f2a900"; box.style.color = "#111"; }}
                else {{ box.style.background = "#f6c915"; box.style.color = "#111"; }}
            }}); }}, 300);
            
            document.addEventListener('keydown', function(e) {{
                let k = e.key.toLowerCase(); if (e.key === ' ') k = 'space';
                if (['w', 'a', 's', 'd', 'e', 'q', 'space'].includes(k)) {{ if (k === 'space') e.preventDefault(); fetch('/control?key=' + k); }}
            }});
        </script>
    </body>
    </html>
    '''

@app.route('/set_path')
def set_path():
    try:
        s_parts = request.args.get('start', '0,4').split(',')
        g_parts = request.args.get('goal', '5,2').split(',')
        start_node = (int(s_parts[0]), int(s_parts[1]))
        goal_node = (int(g_parts[0]), int(g_parts[1]))
        update_active_trajectory(start_node, goal_node)
        return f"Calculated path from {start_node} to {goal_node}"
    except Exception as e:
        return f"Error: {e}"

@app.route('/get_map')
def get_map(): return render_map_svg(map_marker_tile)

@app.route('/get_status')
def get_status():
    if keyboard_engaged: return f"🚨 FEATURE 6: KEYBOARD LATCH ACTIVE -> v={manual_v:.2f}, omega={manual_omega:.2f}"
    if current_state == STATE_RED_STOP: return "🛑 INTERSECTION ENCOUNTERED: AWAITING ROUTING DIRECTION (W/A/D)"
    if current_state == STATE_INTERSECTION_TURN: return f"🚦 MANEUVER ACTIVE: Executing closed-loop turn phase ({active_turn_direction})..."
    if current_state == STATE_GO_AROUND: return f"🦆 FEATURE 2: EVADING OBSTACLE (Phase {maneuver_step})"
    return "🟢 AUTONOMOUS RUNNING: Centering between lanes."

@app.route('/control')
def control():
    global keyboard_engaged, manual_v, manual_omega, current_state, turn_sequence_active, turn_step_index, step_start_time, active_turn_direction, state_start_time
    key = request.args.get('key', '').lower()
    if key == 'q': release_override(); os._exit(0)
    elif key == 'e':
        keyboard_engaged = not keyboard_engaged; manual_v = 0.0; manual_omega = 0.0
        if not keyboard_engaged: current_state = STATE_LANE_FOLLOWING
        return "OK"
    if simulation_mode and key == 's' and current_state == STATE_LANE_FOLLOWING:
        current_state = STATE_RED_STOP; map_reach_intersection(); return "OK"
    if current_state == STATE_RED_STOP and key in ['w', 'a', 'd']:
        active_turn_direction = {'w':'straight','a':'left','d':'right'}[key]
        turn_sequence_active = {'w':INTERSECTION_STRAIGHT_STEPS, 'a':INTERSECTION_LEFT_STEPS, 'd':INTERSECTION_RIGHT_STEPS}[key]
        # stamp BOTH step_start_time (per-step timing) and state_start_time (whole-
        # turn timing) so the vision breakout's min-time guard and the right-hug cap
        # have a reference. active_turn_direction stays manual (W/A/D) as required.
        turn_step_index = 0; step_start_time = time.time(); state_start_time = time.time()
        current_state = STATE_INTERSECTION_TURN
        return "OK"
    elif keyboard_engaged:
        if key == 'w': manual_v = 0.10; manual_omega = 0.0
        elif key == 's': manual_v = -0.10; manual_omega = 0.0
        elif key == 'a': manual_v = 0.04; manual_omega = 0.75
        elif key == 'd': manual_v = 0.04; manual_omega = -0.75
        elif key == 'space': manual_v = 0.0; manual_omega = 0.0
    return "OK"

@app.route('/video')
def video():
    def generate():
        while True:
            with lock: frame = latest_jpeg
            if frame: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

print("\n🚀 Fully dynamic A* topology mapping activated! Launching laptop testing server...")
app.run(host='0.0.0.0', port=5000, threaded=True)
