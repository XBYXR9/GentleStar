#!/usr/bin/env python3
"""
navigate.py - pure vision navigation for duck3 with facing-aware A* path planning,
duck stop + DYNAMIC VISION-BASED DUCK OVERTAKE, 4.5s/tile dead-reckoning
(2.25s final tile), 2.0s red stop hold, closed-loop right turns, and live web dashboard.

Duck behaviour (new):
  1. Duck seen in lane        -> full stop.
  2. Hold DUCK_WAIT_DURATION  -> watch whether the duck actually moves.
                                 - duck leaves    -> resume lane following
                                 - duck shifts    -> reset the timer, give it more time
                                 - duck static    -> plan a way around it
  3. Overtake (closed loop, NOT a canned move sequence):
       cross  : steer left until the YELLOW line is tracked over to the right side
       pass   : lane-follow inside the LEFT lane (white on the left, yellow on the right)
       merge  : steer right until the yellow line is back on the left AND white on the
                right, i.e. a real right-lane re-acquisition
     Every phase is driven by where the lines actually are in the image, with time
     limits only as a safety net.
"""
import base64
import collections
import os
import signal
import threading
import time

import cv2
import numpy as np
import roslibpy
import yaml
from flask import Flask, Response, jsonify, request

print("Stopping baseline navigation container...")
os.system("ssh duckie@duck3.local 'docker stop demo_indefinite_navigation' > /dev/null 2>&1 &")

app = Flask(__name__)
latest_jpeg = None
latest_mask_jpeg = None
lock = threading.Lock()
simulation_mode = False

# ==============================================================================
# TUNING
# ==============================================================================
ROBOT = 'duck3'
ROSBRIDGE_HOST = 'localhost'
ROSBRIDGE_PORT = 9001
FRAME_STALE_S = 0.4

# Speeds & Basic Steering
LANE_SPEED = 0.098
CURVE_SLOWDOWN = 0.35
STEERING_MAX_YAW = 2.4
STEERING_STICTION = 0.15
STEERING_DEADBAND_PX = 12
SMOOTH_FRAMES = 5
STEERING_TRIM = -0.04

# Timing Controls
RED_STOP_DURATION = 2.0         # Stop for 2.0 seconds at red intersection line
SECONDS_PER_TILE = 4.5          # Standard time per tile (4.5s)
FINAL_TILE_SECONDS = 2.25       # Half-time for final tile (2.25s) to stop centered

# Lane Geometry Sampling
SAMPLE_ROW_FRACS = [0.58, 0.66, 0.74, 0.82]
LOOKAHEAD_WEIGHT = 2.0
HALF_WIDTH_FAR = 95
HALF_WIDTH_NEAR = 195
HALF_WIDTH_ADAPT = 0.05
MIN_LINE_GAP_PX = 40
RUN_GAP_PX = 8
RUN_MIN_PX = 3

# Off-Track Recovery
LANE_LOST_HOLD_FRAMES = 25
OFFTRACK_YELLOW_MARGIN = 40
OFFTRACK_PERSIST_FRAMES = 4
OFFTRACK_RECOVER_V = 0.078
OFFTRACK_RECOVER_OMEGA = -1.2

# Stop Line Detection
STOPLINE_MIN_AR = 1.5
STOPLINE_MIN_WIDTH_FRAC = 0.20
STOPLINE_MIN_AREA = 150
STOPLINE_MIN_BOTTOM_FRAC = 0.80
STOPLINE_COOLDOWN_S = 5.0

# Open-Loop Turn Fallbacks (Left & Straight)
INTERSECTION_LEFT_STEPS     = [(0.08,  0.00, 0.4), (0.10,  1.20, 1.4), (0.07, 0.0, 0.6)]
INTERSECTION_STRAIGHT_STEPS = [(0.09,  0.00, 2.4)]

# Closed-Loop Right Turn (Immediate Corner Hug)
WHITE_HUG_TARGET_FRAC = 0.72
WHITE_HUG_GAIN        = 2.8
WHITE_HUG_CLAMP       = 2.6
RIGHT_TURN_SPEED      = 0.065
RIGHT_SEARCH_OMEGA    = -2.4
RIGHT_HUG_MAX_S       = 4.5
MANEUVER_MIN_ROT      = {'left': 1.10, 'right': 1.15, 'straight': 0.0}
MANEUVER_MIN_S        = {'left': 1.0,  'right': 0.8,  'straight': 0.6}
EXIT_MAX_ERR_PX       = 130

# Duck Detection (Stop & Wait)
DUCK_DETECTION_ON    = True
DUCK_TRIGGER_FRAMES  = 2
DUCK_TRIGGER_AREA    = 2500
DUCK_MIN_BOTTOM_FRAC = 0.50
DUCK_PATH_TOL_PX     = 140
DUCK_LINE_REJECT_PX  = 40

# ---- Duck patience: is it a statue or is it walking? -------------------------
DUCK_WAIT_DURATION   = 2.0    # must stay still this long before we plan around it
DUCK_MAX_WAIT_S      = 10.0   # hard cap: overtake even if it keeps twitching
DUCK_MOVED_PX        = 22     # centroid shift that counts as "it moved"
DUCK_MOVED_AREA_FRAC = 0.30   # relative area change that counts as "it moved"
DUCK_CLEAR_FRAMES    = 4      # frames without a blocking duck -> path is free
DUCK_RETRIGGER_COOL  = 2.5    # after an overtake, ignore ducks this long (it is behind us)

# ---- Dynamic overtake (closed loop on the yellow line) ----------------------
OVERTAKE_ENABLED       = True
OVERTAKE_SPEED         = 0.072   # forward speed during the whole manoeuvre

# phase 1: cross into the left lane
CROSS_TARGET_FRAC      = 0.62    # want the yellow line parked here (right of centre)
CROSS_GAIN             = 2.4
CROSS_OMEGA_MIN        = 0.55    # always keep some left rotation on
CROSS_OMEGA_MAX        = 1.9
CROSS_BLIND_OMEGA      = 1.15    # yellow not visible yet -> gentle open-loop sweep
CROSS_MIN_S            = 0.45
CROSS_MAX_S            = 4.0     # safety net only

# phase 2: run down the left lane
PASS_MIN_S             = 1.4     # never merge back before this
PASS_MAX_S             = 7.0     # safety net only
PASS_CLEAR_FRAMES      = 4       # duck out of view for this many frames = passed
PASS_EXTRA_S           = 1.3     # keep going this long after the duck disappears
PASS_LOST_LANE_OMEGA   = 0.0     # nothing visible -> just creep straight

# phase 3: merge back into the right lane
MERGE_TARGET_FRAC      = 0.30    # want the yellow line back over here (left of centre)
MERGE_GAIN             = 2.4
MERGE_OMEGA_MIN        = 0.50    # magnitude; applied negative (right turn)
MERGE_OMEGA_MAX        = 1.9
MERGE_BLIND_OMEGA      = -1.10
MERGE_MIN_S            = 0.45
MERGE_MAX_S            = 4.5     # safety net only

# how much of the overtake time counts as real forward progress towards the goal
OVERTAKE_PROGRESS_FRAC = 0.60

# HSV Thresholds
HSV_YELLOW = (np.array([10,  70,  70]), np.array([40, 255, 255]))
HSV_WHITE  = (np.array([ 0,   0, 150]), np.array([180,  45, 255]))
HSV_RED_A  = (np.array([ 0, 110,  60]), np.array([ 15, 255, 255]))
HSV_RED_B  = (np.array([160, 110, 60]), np.array([180, 255, 255]))

# ==============================================================================
# A* PATH PLANNER WITH HEADING CONSTRAINT
# ==============================================================================
MAP_PATH = os.path.expanduser('~/tum_map.yaml')
DELTAS   = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
LEFT_OF  = {"N": "W", "W": "S", "S": "E", "E": "N"}
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
    try:
        with open(MAP_PATH, 'r') as f:
            data = yaml.safe_load(f)
        raw_tiles = data["tiles"]
        tiles = [[str(cell).strip() for cell in row] for row in raw_tiles]
        h = len(tiles)
        w = max(len(r) for r in tiles) if h > 0 else 0
        return tiles, w, h
    except Exception:
        fallback = [
            ["grass", "grass", "curve_right/N", "straight/E", "straight/E", "curve_left/N"],
            ["grass", "curve_right/N", "curve_right/S", "grass", "grass", "straight/N"],
            ["curve_right/N", "curve_right/S", "grass", "curve_right/N", "straight/E", "3way_left/N"],
            ["straight/N", "grass", "grass", "straight/N", "grass", "straight/N"],
            ["3way_left/S", "straight/E", "straight/E", "4way", "straight/E", "3way_left/N"],
            ["straight/N", "grass", "grass", "straight/N", "grass", "straight/N"],
            ["curve_right/W", "straight/E", "straight/E", "3way_left/E", "straight/E", "curve_right/S"],
        ]
        return fallback, 6, 7


MAP_TILES, MAP_W, MAP_H = load_track_map()


def build_adjacency_graph():
    adj = {}
    for j in range(MAP_H):
        for i in range(MAP_W):
            kind = MAP_TILES[j][i]
            if kind not in OPENINGS:
                continue
            node = (i, j)
            adj.setdefault(node, [])
            for d in OPENINGS[kind]:
                di, dj = DELTAS[d]
                ni, nj = i + di, j + dj
                if 0 <= ni < MAP_W and 0 <= nj < MAP_H:
                    nk = MAP_TILES[nj][ni]
                    if nk in OPENINGS and OPPOSITE[d] in OPENINGS[nk]:
                        adj[node].append(((ni, nj), d))
    return adj


GRAPH_ADJ = build_adjacency_graph()


def astar_search(start, goal, required_heading=None):
    if start not in GRAPH_ADJ or goal not in GRAPH_ADJ:
        return [start]
    import heapq
    open_heap = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0.0, start, [start])]
    visited = set()
    while open_heap:
        f, g, n, path = heapq.heappop(open_heap)
        if n == goal:
            return path
        state_key = (n, len(path))
        if state_key in visited:
            continue
        visited.add(state_key)
        for nb, d in GRAPH_ADJ.get(n, []):
            if len(path) == 1 and required_heading and d != required_heading:
                continue
            if nb in path:
                continue
            tentative_g = g + 1.0
            h = abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
            heapq.heappush(open_heap, (tentative_g + h, tentative_g, nb, path + [nb]))

    if required_heading:
        return astar_search(start, goal, required_heading=None)
    return [start]


def compute_path_turn_decisions(path):
    turn_tiles = {}
    intersection_order = []
    if len(path) < 3:
        return turn_tiles, intersection_order
    for k in range(1, len(path) - 1):
        prev, cur, nxt = path[k - 1], path[k], path[k + 1]
        kind = MAP_TILES[cur[1]][cur[0]]
        if not (kind.startswith("3way_") or kind == "4way"):
            continue
        in_d = None
        for d, (x, y) in DELTAS.items():
            if (cur[0] - prev[0], cur[1] - prev[1]) == (x, y):
                in_d = d
        out_d = None
        for d, (x, y) in DELTAS.items():
            if (nxt[0] - cur[0], nxt[1] - cur[1]) == (x, y):
                out_d = d
        if in_d and out_d:
            intersection_order.append(cur)
            if in_d == out_d:
                turn_tiles[cur] = "straight"
            elif LEFT_OF[in_d] == out_d:
                turn_tiles[cur] = "left"
            elif RIGHT_OF[in_d] == out_d:
                turn_tiles[cur] = "right"
    return turn_tiles, intersection_order


ROUTE = []
ROUTE_START = None
ROUTE_GOAL = None
ROUTE_TURN_TILES = {}
ROUTE_INTERSECTION_ORDER = []
bot_start_tile = None
bot_heading = 'E'
AUTO_PATH_MODE = False
path_intersections_passed = 0

# Dead-Reckoning Goal Tracking
post_intersections_tracking = False
post_intersection_start_time = 0.0
tiles_after_final_turn = 0
final_turn_route_index = 0
current_tracked_tile_index = 0
goal_total_duration = 0.0


def compute_route():
    global ROUTE, ROUTE_TURN_TILES, ROUTE_INTERSECTION_ORDER, path_intersections_passed
    global post_intersections_tracking, current_tracked_tile_index, tiles_after_final_turn
    global final_turn_route_index, goal_total_duration
    if bot_start_tile is None or ROUTE_GOAL is None:
        return False, "Set both Start and Goal tiles first."
    path = astar_search(bot_start_tile, ROUTE_GOAL, required_heading=bot_heading)
    if len(path) < 2 or path[-1] != ROUTE_GOAL:
        ROUTE, ROUTE_TURN_TILES, ROUTE_INTERSECTION_ORDER = [], {}, []
        return False, "No drivable path found."
    turn_tiles, order = compute_path_turn_decisions(path)
    ROUTE = path
    ROUTE_TURN_TILES = turn_tiles
    ROUTE_INTERSECTION_ORDER = order
    path_intersections_passed = 0
    post_intersections_tracking = False
    current_tracked_tile_index = 0

    if order:
        last_inter = order[-1]
        final_turn_route_index = path.index(last_inter)
        tiles_after_final_turn = len(path) - 1 - final_turn_route_index
    else:
        final_turn_route_index = 0
        tiles_after_final_turn = len(path) - 1

    # Total duration = full tiles (4.5s) + half-time for final tile (2.25s)
    if tiles_after_final_turn > 1:
        goal_total_duration = (tiles_after_final_turn - 1) * SECONDS_PER_TILE + FINAL_TILE_SECONDS
    elif tiles_after_final_turn == 1:
        goal_total_duration = FINAL_TILE_SECONDS
    else:
        goal_total_duration = 0.0
    return True, f"Route: {len(path)} tiles ({len(order)} turns)"


def render_map_svg():
    cell = 34
    pad = 22
    grid_w = MAP_W * cell
    grid_h = MAP_H * cell
    total_w = grid_w + 2 * pad
    total_h = grid_h + 2 * pad
    p = [f'<svg width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">']
    p.append(f'<text x="{pad + grid_w // 2}" y="{pad - 7}" font-size="12" font-weight="bold" fill="#f6c915" text-anchor="middle" font-family="sans-serif">&#9650; NORTH (N)</text>')
    p.append(f'<text x="{pad + grid_w // 2}" y="{pad + grid_h + 16}" font-size="12" font-weight="bold" fill="#f6c915" text-anchor="middle" font-family="sans-serif">&#9660; SOUTH (S)</text>')
    p.append(f'<text x="{pad - 7}" y="{pad + grid_h // 2 + 4}" font-size="12" font-weight="bold" fill="#f6c915" text-anchor="end" font-family="sans-serif">&#9664; W</text>')
    p.append(f'<text x="{pad + grid_w + 7}" y="{pad + grid_h // 2 + 4}" font-size="12" font-weight="bold" fill="#f6c915" text-anchor="start" font-family="sans-serif">E &#9654;</text>')
    route_set = set(ROUTE)
    for j in range(MAP_H):
        for i in range(MAP_W):
            kind = MAP_TILES[j][i]
            x, y = pad + i * cell, pad + j * cell
            fill = "#244a28" if kind == "grass" else "#3a3a3a"
            p.append(f'<rect class="map-tile" x="{x}" y="{y}" width="{cell}" height="{cell}" '
                     f'fill="{fill}" stroke="#111" stroke-width="1" '
                     f'style="cursor:pointer;" onclick="mapTileClick({i},{j})"/>')
            if (i, j) in route_set and kind != "grass":
                p.append(f'<rect x="{x+2}" y="{y+2}" width="{cell-4}" height="{cell-4}" '
                         f'fill="none" stroke="#1f6feb" stroke-width="1.5" opacity="0.6" pointer-events="none"/>')
    if len(ROUTE) > 1:
        pts = " ".join(f"{pad + i*cell+cell//2},{pad + j*cell+cell//2}" for (i, j) in ROUTE)
        p.append(f'<polyline points="{pts}" fill="none" stroke="#1f9bff" stroke-width="3" opacity="0.85" pointer-events="none"/>')
    if bot_start_tile:
        sx, sy = bot_start_tile
        p.append(f'<circle cx="{pad + sx*cell+cell//2}" cy="{pad + sy*cell+cell//2}" r="10" fill="#22c55e" pointer-events="none"/>')
        p.append(f'<text x="{pad + sx*cell+cell//2}" y="{pad + sy*cell+cell//2+4}" font-size="11" fill="black" text-anchor="middle" font-weight="bold" pointer-events="none">S</text>')
    if ROUTE_GOAL:
        gx, gy = ROUTE_GOAL
        p.append(f'<circle cx="{pad + gx*cell+cell//2}" cy="{pad + gy*cell+cell//2}" r="10" fill="#ef4444" pointer-events="none"/>')
        p.append(f'<text x="{pad + gx*cell+cell//2}" y="{pad + gy*cell+cell//2+4}" font-size="11" fill="white" text-anchor="middle" font-weight="bold" pointer-events="none">G</text>')
    for (ti, tj), d in ROUTE_TURN_TILES.items():
        arrow = {"left": "L", "right": "R", "straight": "^"}.get(d, "?")
        p.append(f'<text x="{pad + ti*cell+cell//2}" y="{pad + tj*cell+cell//2+5}" font-size="13" fill="#ffd000" text-anchor="middle" font-family="sans-serif" font-weight="bold" pointer-events="none">{arrow}</text>')
    # Live position marker (Intersections or Dead Reckoning Tile Tracking)
    if AUTO_PATH_MODE and ROUTE:
        if path_intersections_passed < len(ROUTE_INTERSECTION_ORDER):
            cx_, cy_ = ROUTE_INTERSECTION_ORDER[path_intersections_passed]
        elif post_intersections_tracking and current_tracked_tile_index < len(ROUTE):
            cx_, cy_ = ROUTE[current_tracked_tile_index]
        else:
            cx_, cy_ = ROUTE[-1]

        p.append(f'<circle cx="{pad + cx_*cell+cell//2}" cy="{pad + cy_*cell+cell//2}" r="9" fill="#ffe600" stroke="#000" stroke-width="2" pointer-events="none"><animate attributeName="r" values="9;12;9" dur="1s" repeatCount="indefinite"/></circle>')
    p.append('</svg>')
    return "".join(p)


# ==============================================================================
# STATE & DETECTIONS
# ==============================================================================
STATE_LANE_FOLLOWING = "lane_following"
STATE_RED_STOP = "red_line_stopped"
STATE_DUCK_STOP = "duck_stopped"
STATE_DUCK_CROSS = "duck_overtake_cross"
STATE_DUCK_PASS = "duck_overtake_pass"
STATE_DUCK_MERGE = "duck_overtake_merge"
STATE_INTERSECTION_TURN = "intersection_maneuver"
STATE_GOAL_REACHED = "goal_reached"

OVERTAKE_STATES = (STATE_DUCK_CROSS, STATE_DUCK_PASS, STATE_DUCK_MERGE)

current_state = STATE_LANE_FOLLOWING
state_start_time = time.time()
step_start_time = time.time()
last_red_line_time = 0.0
SETUP_COMPLETE = False

turn_sequence_active = []
turn_step_index = 0
active_turn_direction = 'none'

keyboard_engaged = False
manual_v = 0.0
manual_omega = 0.0

_omega_hist = collections.deque(maxlen=SMOOTH_FRAMES)
_last_good_omega = 0.0
_lane_lost_frames = 0
_yellow_right_frames = 0
_duck_seen_frames = 0
_duck_clear_frames = 0
_turn_rot = 0.0
_prev_frame_t = time.time()
_last_frame_time = time.time()
_fps = 0.0
link_stale = False
_link_error = ""
_last_publish_warn = 0.0

# --- duck wait / overtake bookkeeping ---
_duck_ref_x = None
_duck_ref_area = 0.0
_duck_stop_clock = 0.0      # when the whole duck episode started (robot stationary)
_overtake_clock = 0.0       # when the overtake manoeuvre itself started
_cross_seen_right = False   # yellow line has been observed right of centre
_pass_clear_frames = 0
_pass_clear_time = 0.0
_overtake_count = 0
_overtake_note = "idle"
_duck_cooldown_until = 0.0


def _init_half_widths():
    f0, f1 = SAMPLE_ROW_FRACS[0], SAMPLE_ROW_FRACS[-1]
    out = []
    for f in SAMPLE_ROW_FRACS:
        t = (f - f0) / (f1 - f0) if f1 != f0 else 0.0
        out.append(HALF_WIDTH_FAR + t * (HALF_WIDTH_NEAR - HALF_WIDTH_FAR))
    return out


_half_width = _init_half_widths()
_half_width_init = list(_half_width)

TEL = {"state": current_state, "v": 0.0, "omega": 0.0, "rows": 0,
       "yellow": False, "white": False, "duck": False, "duck_area": 0,
       "fps": 0.0, "link": "ok", "note": "", "overtake": "idle",
       "overtake_count": 0, "overtake_on": OVERTAKE_ENABLED}

# ==============================================================================
# ROS LINK
# ==============================================================================
try:
    print(f"Connecting to rosbridge at {ROSBRIDGE_HOST}:{ROSBRIDGE_PORT} ...")
    client_ros = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
    client_ros.run(timeout=3)
    if not client_ros.is_connected:
        raise RuntimeError("rosbridge did not answer")
    print(f"Connected to {ROBOT}.")
    cmd_pub = roslibpy.Topic(client_ros, f'/{ROBOT}/car_cmd_switch_node/cmd',
                             'duckietown_msgs/Twist2DStamped')
    override_pub = roslibpy.Topic(client_ros, f'/{ROBOT}/joy_mapper_node/joystick_override',
                                  'duckietown_msgs/BoolStamped')
except Exception as e:
    print(f"Hardware link offline ({e}). Starting simulation feed.")
    simulation_mode = True


def _stamp():
    return {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''}


def publish_drive(v, omega):
    global _link_error, _last_publish_warn
    if simulation_mode:
        return
    try:
        override_pub.publish(roslibpy.Message({'header': _stamp(), 'data': True}))
        cmd_pub.publish(roslibpy.Message({'header': _stamp(),
                                          'v': float(v), 'omega': float(omega)}))
        _link_error = ""
    except Exception as e:
        _link_error = str(e)
        if time.time() - _last_publish_warn > 2.0:
            print(f"publish_drive failed: {e}")
            _last_publish_warn = time.time()


def release_override():
    if simulation_mode:
        return
    try:
        cmd_pub.publish(roslibpy.Message({'header': _stamp(), 'v': 0.0, 'omega': 0.0}))
        override_pub.publish(roslibpy.Message({'header': _stamp(), 'data': False}))
    except Exception:
        pass


import atexit
atexit.register(release_override)
signal.signal(signal.SIGTERM, lambda s, f: release_override() or os._exit(0))
signal.signal(signal.SIGINT, lambda s, f: release_override() or os._exit(0))


def watchdog_loop():
    global link_stale
    while True:
        time.sleep(0.1)
        if simulation_mode:
            continue
        stale = (time.time() - _last_frame_time) > FRAME_STALE_S
        if stale and not link_stale:
            print("No camera frames - stopping wheels.")
        if stale:
            publish_drive(0.0, 0.0)
        link_stale = stale


threading.Thread(target=watchdog_loop, daemon=True).start()


# ==============================================================================
# VISION HELPERS
# ==============================================================================
def _column_runs(band, gap=RUN_GAP_PX, min_cols=RUN_MIN_PX):
    if band.size == 0:
        return []
    cols = np.flatnonzero(band.any(axis=0))
    if len(cols) == 0:
        return []
    runs, s, p = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - p > gap:
            runs.append((s, p))
            s = c
        p = c
    runs.append((s, p))
    return [(int(a), int(b)) for a, b in runs if (b - a + 1) >= min_cols]


def find_yellow_col(yellow, y0, y1, x_limit):
    """Right-lane view: yellow centre line lives on the LEFT of the image."""
    x_limit = max(1, min(int(x_limit), yellow.shape[1]))
    runs = _column_runs(yellow[y0:y1, :x_limit])
    if not runs:
        return None
    a, b = runs[-1]
    return (a + b) // 2


def find_white_col(white, y0, y1, x_from):
    """Right-lane view: white outer edge lives on the RIGHT of the image."""
    x_from = max(0, int(x_from))
    if x_from >= white.shape[1]:
        return None
    runs = _column_runs(white[y0:y1, x_from:])
    if not runs:
        return None
    a, b = runs[0]
    return (a + b) // 2 + x_from


# ---- mirrored helpers: used while driving in the LEFT (oncoming) lane --------
def find_yellow_col_right(yellow, y0, y1, x_from):
    """Left-lane view: yellow centre line is now on the RIGHT of the image.
    Take the first (left-most) yellow run at or right of x_from."""
    x_from = max(0, int(x_from))
    if x_from >= yellow.shape[1]:
        return None
    runs = _column_runs(yellow[y0:y1, x_from:])
    if not runs:
        return None
    a, b = runs[0]
    return (a + b) // 2 + x_from


def find_white_col_left(white, y0, y1, x_limit):
    """Left-lane view: the lane's white outer edge is on the LEFT of the image.
    Take the last (right-most) white run left of x_limit."""
    x_limit = max(1, min(int(x_limit), white.shape[1]))
    runs = _column_runs(white[y0:y1, :x_limit])
    if not runs:
        return None
    a, b = runs[-1]
    return (a + b) // 2


def scan_band(yellow, white, w, y0, y1, cx):
    ycx = find_yellow_col(yellow, y0, y1, int(w * 0.60))
    if ycx is not None:
        wcx = find_white_col(white, y0, y1, ycx + MIN_LINE_GAP_PX)
    else:
        wcx = find_white_col(white, y0, y1, cx)
    return ycx, wcx


def scan_band_left_lane(yellow, white, w, y0, y1, cx):
    """Mirror of scan_band for the oncoming lane: white | robot | yellow."""
    ycx = find_yellow_col_right(yellow, y0, y1, int(w * 0.40))
    if ycx is not None:
        wcx = find_white_col_left(white, y0, y1, ycx - MIN_LINE_GAP_PX)
    else:
        wcx = find_white_col_left(white, y0, y1, cx)
    return ycx, wcx


def sample_lane_center(frame, yellow, white, w, h, cx, draw=False):
    targets = []
    n = len(SAMPLE_ROW_FRACS)
    for i, frac in enumerate(SAMPLE_ROW_FRACS):
        sy = int(h * frac)
        y0, y1 = sy, min(h, sy + 4)
        ycx, wcx = scan_band(yellow, white, w, y0, y1, cx)
        hw = _half_width[i]
        if ycx is not None and wcx is not None:
            t = (ycx + wcx) // 2
            measured = (wcx - ycx) / 2.0
            lo, hi = 0.45 * _half_width_init[i], 2.0 * _half_width_init[i]
            if lo <= measured <= hi:
                _half_width[i] = (1 - HALF_WIDTH_ADAPT) * hw + HALF_WIDTH_ADAPT * measured
        elif ycx is not None:
            t = int(ycx + hw)
        elif wcx is not None:
            t = int(wcx - hw)
        else:
            continue
        frac_far = (n - 1 - i) / (n - 1) if n > 1 else 0.0
        wgt = 1.0 + LOOKAHEAD_WEIGHT * frac_far
        targets.append((wgt, t, sy))
        if draw:
            cv2.circle(frame, (max(0, min(w - 1, t)), sy), 4, (0, 255, 0), -1)
    if not targets:
        return None, 0
    sw = sum(g for g, _, _ in targets)
    lane_center = int(sum(g * t for g, t, _ in targets) / sw)
    if draw and len(targets) >= 2:
        cv2.polylines(frame, [np.array([(max(0, min(w - 1, t)), sy) for _, t, sy in targets],
                                       np.int32)], False, (0, 255, 0), 2)
    return lane_center, len(targets)


def sample_left_lane_center(frame, yellow, white, w, h, cx, draw=False):
    """Same weighted multi-row estimator, but for the oncoming lane.
    Half widths are NOT adapted here so the right-lane calibration stays clean."""
    targets = []
    n = len(SAMPLE_ROW_FRACS)
    for i, frac in enumerate(SAMPLE_ROW_FRACS):
        sy = int(h * frac)
        y0, y1 = sy, min(h, sy + 4)
        ycx, wcx = scan_band_left_lane(yellow, white, w, y0, y1, cx)
        hw = _half_width_init[i]
        if ycx is not None and wcx is not None:
            measured = (ycx - wcx) / 2.0
            lo, hi = 0.45 * hw, 2.0 * hw
            t = (ycx + wcx) // 2 if lo <= measured <= hi else int(ycx - hw)
        elif ycx is not None:
            t = int(ycx - hw)
        elif wcx is not None:
            t = int(wcx + hw)
        else:
            continue
        frac_far = (n - 1 - i) / (n - 1) if n > 1 else 0.0
        wgt = 1.0 + LOOKAHEAD_WEIGHT * frac_far
        targets.append((wgt, t, sy))
        if draw:
            cv2.circle(frame, (max(0, min(w - 1, t)), sy), 4, (255, 0, 255), -1)
    if not targets:
        return None, 0
    sw = sum(g for g, _, _ in targets)
    lane_center = int(sum(g * t for g, t, _ in targets) / sw)
    if draw and len(targets) >= 2:
        cv2.polylines(frame, [np.array([(max(0, min(w - 1, t)), sy) for _, t, sy in targets],
                                       np.int32)], False, (255, 0, 255), 2)
    return lane_center, len(targets)


def valid_lane_reacquired(lane_center, yellow_cx, white_cx, cx):
    if lane_center is None:
        return False
    if abs(lane_center - cx) > EXIT_MAX_ERR_PX:
        return False
    yellow_on_left = (yellow_cx is not None and yellow_cx < cx - 15)
    yellow_white_pair = (yellow_cx is not None and white_cx is not None
                         and white_cx > yellow_cx + MIN_LINE_GAP_PX)
    return yellow_on_left and (white_cx is not None or yellow_white_pair)


def is_duck(contour):
    area = cv2.contourArea(contour)
    if area < 2000 or area > 50000:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    if aspect_ratio > 1.9 or aspect_ratio < 0.4:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0:
        return False
    solidity = area / hull_area
    rect = cv2.minAreaRect(contour)
    w_rot, h_rot = rect[1]
    rect_area = w_rot * h_rot
    if rect_area == 0:
        return False
    rectangularity = area / rect_area
    long_side, short_side = max(w_rot, h_rot), min(w_rot, h_rot)
    rot_aspect = long_side / short_side if short_side > 0 else 0
    per = cv2.arcLength(contour, True)
    if per == 0:
        return False
    approx = cv2.approxPolyDP(contour, 0.018 * per, True)
    if len(approx) <= 6 and solidity > 0.85:
        return False
    if rot_aspect > 1.55 and rectangularity > 0.72:
        return False
    return True


def steer_from_error(error, cx):
    global _omega_hist
    if abs(error) <= STEERING_DEADBAND_PX:
        raw = 0.0
    else:
        raw = -(float(error) / cx) * STEERING_MAX_YAW
        raw = max(-STEERING_MAX_YAW, min(STEERING_MAX_YAW, raw))
    _omega_hist.append(raw)
    omega = sum(_omega_hist) / len(_omega_hist)
    if abs(omega) < 1e-3:
        omega = 0.0
    else:
        sign = 1.0 if omega > 0 else -1.0
        mag = STEERING_STICTION + (1.0 - STEERING_STICTION / STEERING_MAX_YAW) * abs(omega)
        omega = sign * min(STEERING_MAX_YAW, mag)
    omega = max(-STEERING_MAX_YAW, min(STEERING_MAX_YAW, omega + STEERING_TRIM))
    v = LANE_SPEED * (1.0 - CURVE_SLOWDOWN * min(1.0, abs(omega) / STEERING_MAX_YAW))
    return v, omega


def start_intersection_turn(direction):
    global active_turn_direction, turn_sequence_active, turn_step_index
    global _turn_rot, step_start_time, state_start_time, current_state
    active_turn_direction = direction
    turn_sequence_active = {'straight': INTERSECTION_STRAIGHT_STEPS,
                            'left': INTERSECTION_LEFT_STEPS,
                            'right': []}[direction]
    turn_step_index = 0
    _turn_rot = 0.0
    step_start_time = state_start_time = time.time()
    current_state = STATE_INTERSECTION_TURN
    print(f"Intersection Turn Commenced: {direction}")


def _credit_goal_timer(seconds):
    """Push the dead-reckoning clock forward so a duck episode does not eat
    into the distance estimate towards the goal tile."""
    global post_intersection_start_time
    if post_intersections_tracking and seconds > 0:
        post_intersection_start_time += seconds


def _finish_duck_episode(now, overtook):
    """Return to lane following and repay the goal timer for time lost."""
    global current_state, _lane_lost_frames, _duck_seen_frames, _duck_clear_frames
    global _overtake_note, _duck_cooldown_until
    if overtook:
        waited = max(0.0, _overtake_clock - _duck_stop_clock)
        manoeuvre = max(0.0, now - _overtake_clock)
        _credit_goal_timer(waited + manoeuvre * (1.0 - OVERTAKE_PROGRESS_FRAC))
        # the duck we just passed is beside/behind us - do not stop for it again
        _duck_cooldown_until = now + DUCK_RETRIGGER_COOL
    else:
        _credit_goal_timer(max(0.0, now - _duck_stop_clock))
    _omega_hist.clear()
    _lane_lost_frames = 0
    _duck_seen_frames = 0
    _duck_clear_frames = 0
    _overtake_note = "idle"
    current_state = STATE_LANE_FOLLOWING


def start_overtake(now):
    global current_state, state_start_time, _overtake_clock, _cross_seen_right
    global _pass_clear_frames, _pass_clear_time, _overtake_count, _overtake_note
    current_state = STATE_DUCK_CROSS
    state_start_time = now
    _overtake_clock = now
    _cross_seen_right = False
    _pass_clear_frames = 0
    _pass_clear_time = 0.0
    _overtake_count += 1
    _overtake_note = "crossing into left lane"
    _omega_hist.clear()
    print("Duck is not moving -> planning around it (crossing into the left lane).")


# ==============================================================================
# MAIN FRAME PROCESSING
# ==============================================================================
def process_image_frame(frame):
    global latest_jpeg, latest_mask_jpeg, current_state, state_start_time, last_red_line_time
    global turn_sequence_active, turn_step_index, active_turn_direction
    global keyboard_engaged, manual_v, manual_omega, step_start_time
    global _last_good_omega, _lane_lost_frames, _yellow_right_frames
    global _duck_seen_frames, _duck_clear_frames, _turn_rot
    global _prev_frame_t, _last_frame_time, _fps
    global path_intersections_passed, post_intersections_tracking, post_intersection_start_time
    global current_tracked_tile_index
    global _duck_ref_x, _duck_ref_area, _duck_stop_clock, _overtake_clock
    global _cross_seen_right, _pass_clear_frames, _pass_clear_time, _overtake_note
    global _duck_cooldown_until

    if frame is None or frame.size == 0:
        return

    now = time.time()
    dt = min(0.5, max(0.0, now - _prev_frame_t))
    _prev_frame_t = now
    _last_frame_time = now
    if dt > 0:
        _fps = 0.85 * _fps + 0.15 * (1.0 / dt)

    h, w = frame.shape[:2]
    cx = w // 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, *HSV_YELLOW)
    white_mask = cv2.inRange(hsv, *HSV_WHITE)
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, *HSV_RED_A),
                              cv2.inRange(hsv, *HSV_RED_B))

    # --- Ducks ---
    duck_roi = np.zeros_like(yellow_mask)
    duck_roi[int(h * 0.45):int(h * 0.98), :] = yellow_mask[int(h * 0.45):int(h * 0.98), :]
    duck_contours, _ = cv2.findContours(duck_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    duck_found = False
    duck_x = None
    duck_area = 0
    duck_bottom = 0
    duck_boxes = []
    duck_contours_confirmed = []

    for contour in duck_contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if is_duck(contour):
            a = cv2.contourArea(contour)
            duck_found = True
            duck_boxes.append((x, y, cw, ch))
            duck_contours_confirmed.append(contour)
            if a > duck_area:
                duck_area, duck_x, duck_bottom = a, x + cw // 2, y + ch
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
            cv2.putText(frame, "DUCK", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif cv2.contourArea(contour) > 800:
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 200, 255), 1)

    # ducks are yellow: cut them out before any lane-line reasoning
    yellow_lane = yellow_mask.copy()
    for (x, y, cw, ch) in duck_boxes:
        yellow_lane[y:y + ch, x:x + cw] = 0

    duck_blocking = duck_found and duck_bottom >= int(h * DUCK_MIN_BOTTOM_FRAC) \
        and duck_area >= DUCK_TRIGGER_AREA * 0.6

    # --- Stop Line ---
    red_roi = np.zeros_like(red_mask)
    red_roi[int(h * 0.60):h, :] = red_mask[int(h * 0.60):h, :]
    red_line_found = False
    red_line_ahead = False
    stop_contours, _ = cv2.findContours(red_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stop_contours_confirmed = []
    for contour in stop_contours:
        area = cv2.contourArea(contour)
        if area <= 60:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        ar = float(cw) / ch if ch > 0 else 0
        ox, oy = x, y
        bottom_frac = (oy + ch) / float(h)
        shape_ok = (ar > STOPLINE_MIN_AR and cw > int(w * STOPLINE_MIN_WIDTH_FRAC)
                    and area > STOPLINE_MIN_AREA)
        close_ok = bottom_frac >= STOPLINE_MIN_BOTTOM_FRAC
        if shape_ok and close_ok:
            red_line_found = True
            cv2.rectangle(frame, (ox, oy), (ox + cw, oy + ch), (0, 0, 255), 3)
            cv2.putText(frame, f"STOP LINE ar={ar:.1f}", (ox, oy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            stop_contours_confirmed.append(contour)
        elif shape_ok:
            red_line_ahead = True
            cv2.rectangle(frame, (ox, oy), (ox + cw, oy + ch), (0, 140, 230), 2)

    # --- Lane Bands (right lane reference) ---
    near_top, near_bot = int(h * 0.72), int(h * 0.78)
    yellow_cx, white_cx = scan_band(yellow_lane, white_mask, w, near_top, near_bot, cx)
    far_top, far_bot = int(h * 0.55), int(h * 0.61)
    far_yellow_cx, far_white_cx = scan_band(yellow_lane, white_mask, w, far_top, far_bot, cx)

    # --- Where is the yellow line, anywhere in the frame? (overtake reference) ---
    # During the overtake the yellow line sweeps across the whole image, so we
    # search the full width instead of assuming a side.
    ot_runs = _column_runs(yellow_lane[near_top:near_bot, :])
    if not ot_runs:
        ot_runs = _column_runs(yellow_lane[far_top:far_bot, :])
    yellow_any_cx = None
    if ot_runs:
        if current_state == STATE_DUCK_CROSS:
            # while swinging left, the relevant edge is the right-most yellow run
            a, b = ot_runs[-1]
        else:
            a, b = ot_runs[0] if current_state == STATE_DUCK_MERGE else ot_runs[-1]
        yellow_any_cx = (a + b) // 2

    time_in_step = now - step_start_time
    arrow_v, arrow_omega = 0.0, 0.0
    note = ""
    rows_hit = 0

    # ==========================================================================
    # STATE MACHINE
    # ==========================================================================
    if current_state == STATE_GOAL_REACHED:
        publish_drive(0.0, 0.0)
        arrow_v, arrow_omega = 0.0, 0.0
        note = "GOAL REACHED - STOPPED"
        cv2.putText(frame, "DESTINATION REACHED - STOPPED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    elif not SETUP_COMPLETE:
        publish_drive(0.0, 0.0)
        arrow_v, arrow_omega = 0.0, 0.0
        note = "Waiting for setup completion"
        cv2.putText(frame, "SETUP - Select mode on dashboard", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    elif keyboard_engaged:
        publish_drive(manual_v, manual_omega)
        arrow_v, arrow_omega = manual_v, manual_omega
        current_state = STATE_LANE_FOLLOWING
        note = "Manual override"
        cv2.putText(frame, "MANUAL OVERRIDE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    elif current_state == STATE_LANE_FOLLOWING:
        # --- Timed Tile Count to Goal ---
        if AUTO_PATH_MODE and post_intersections_tracking:
            elapsed_post = now - post_intersection_start_time
            blocks_passed = int(elapsed_post // SECONDS_PER_TILE)
            current_tracked_tile_index = min(len(ROUTE) - 1, final_turn_route_index + blocks_passed)

            # Goal reached threshold check (4.5s/tile + 2.25s final center tile)
            if elapsed_post >= goal_total_duration:
                current_state = STATE_GOAL_REACHED
                publish_drive(0.0, 0.0)
                print(f"Goal Reached after {elapsed_post:.1f}s!")

        lane_center, rows_hit = sample_lane_center(frame, yellow_lane, white_mask,
                                                   w, h, cx, draw=True)

        crossed_center = (yellow_cx is not None
                          and yellow_cx > cx + OFFTRACK_YELLOW_MARGIN
                          and white_cx is None)
        _yellow_right_frames = _yellow_right_frames + 1 if crossed_center else 0

        if _yellow_right_frames >= OFFTRACK_PERSIST_FRAMES:
            publish_drive(OFFTRACK_RECOVER_V, OFFTRACK_RECOVER_OMEGA)
            arrow_v, arrow_omega = OFFTRACK_RECOVER_V, OFFTRACK_RECOVER_OMEGA
            _last_good_omega = OFFTRACK_RECOVER_OMEGA
            _lane_lost_frames = 0
            note = "Off track - recovering"
        elif lane_center is not None and current_state != STATE_GOAL_REACHED:
            error = lane_center - cx
            v, omega = steer_from_error(error, cx)
            _last_good_omega = omega
            _lane_lost_frames = 0
            publish_drive(v, omega)
            arrow_v, arrow_omega = v, omega

            if AUTO_PATH_MODE and post_intersections_tracking:
                rem_time = max(0.0, goal_total_duration - (now - post_intersection_start_time))
                note = f"Driving to Goal ({rem_time:.1f}s left, {rows_hit}/4 rows)"
            else:
                note = f"Tracking ({rows_hit}/4 rows, err {error:+d}px)"
        elif current_state != STATE_GOAL_REACHED:
            _lane_lost_frames += 1
            if _lane_lost_frames < LANE_LOST_HOLD_FRAMES:
                _last_good_omega *= 0.92
                publish_drive(LANE_SPEED * 0.7, _last_good_omega)
                arrow_v, arrow_omega = LANE_SPEED * 0.7, _last_good_omega
                note = f"Lane lost - holding ({_lane_lost_frames})"
            else:
                publish_drive(0.0, 0.0)
                note = "Lane lost - stopped"

        duck_in_path = False
        if duck_found and duck_x is not None:
            close_enough = (duck_area >= DUCK_TRIGGER_AREA
                            and duck_bottom >= int(h * DUCK_MIN_BOTTOM_FRAC))
            in_path = (lane_center is None or abs(duck_x - lane_center) <= DUCK_PATH_TOL_PX)
            y0 = max(0, duck_bottom - 8)
            y1 = min(h, duck_bottom + 2)
            line_cols = [((a + b) // 2) for a, b in _column_runs(yellow_lane[y0:y1, :])]
            on_line = any(abs(duck_x - c) < DUCK_LINE_REJECT_PX for c in line_cols)
            duck_in_path = close_enough and in_path and not on_line

        _duck_seen_frames = _duck_seen_frames + 1 if duck_in_path else 0

        if now < _duck_cooldown_until:
            _duck_seen_frames = 0        # just overtook one; ignore it while it slides past

        if DUCK_DETECTION_ON and duck_in_path and _duck_seen_frames >= DUCK_TRIGGER_FRAMES:
            current_state = STATE_DUCK_STOP
            state_start_time = now
            _duck_stop_clock = now
            _duck_clear_frames = 0
            _duck_ref_x = duck_x
            _duck_ref_area = float(duck_area)
            _overtake_note = "waiting to see if it moves"
            publish_drive(0.0, 0.0)
            print("Duck detected in lane -> Stopped, watching whether it moves...")
        elif red_line_found and (now - last_red_line_time) > STOPLINE_COOLDOWN_S \
                and current_state != STATE_GOAL_REACHED:
            current_state = STATE_RED_STOP
            state_start_time = now
            publish_drive(0.0, 0.0)
            print("Stop line -> Stopped, holding 2.0s before proceeding...")

    # ------------------------------------------------------------------
    # DUCK: stop and find out whether it is alive or a lawn ornament
    # ------------------------------------------------------------------
    elif current_state == STATE_DUCK_STOP:
        publish_drive(0.0, 0.0)
        arrow_v, arrow_omega = 0.0, 0.0

        still_for = now - state_start_time
        waited_total = now - _duck_stop_clock

        if not duck_blocking:
            _duck_clear_frames += 1
        else:
            _duck_clear_frames = 0
            # did it actually shuffle along?
            moved = False
            if duck_x is not None and _duck_ref_x is not None:
                if abs(duck_x - _duck_ref_x) > DUCK_MOVED_PX:
                    moved = True
            if _duck_ref_area > 0 and duck_area > 0:
                if abs(duck_area - _duck_ref_area) / _duck_ref_area > DUCK_MOVED_AREA_FRAC:
                    moved = True
            if moved:
                _duck_ref_x = duck_x
                _duck_ref_area = float(duck_area)
                state_start_time = now      # it is moving: restart the patience timer
                still_for = 0.0
                _overtake_note = "duck is moving - giving it room"

        if _duck_clear_frames >= DUCK_CLEAR_FRAMES:
            print("Duck cleared on its own -> Resuming lane following")
            _finish_duck_episode(now, overtook=False)
        elif OVERTAKE_ENABLED and DUCK_DETECTION_ON and \
                (still_for >= DUCK_WAIT_DURATION or waited_total >= DUCK_MAX_WAIT_S):
            start_overtake(now)
        else:
            rem = max(0.0, DUCK_WAIT_DURATION - still_for)
            note = f"Duck blocking - waiting {rem:.1f}s to see if it moves"
            _overtake_note = f"patience {still_for:.1f}/{DUCK_WAIT_DURATION:.1f}s"
            cv2.putText(frame, f"DUCK - WAITING {rem:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # ------------------------------------------------------------------
    # OVERTAKE PHASE 1 - cross the yellow line into the left lane
    # ------------------------------------------------------------------
    elif current_state == STATE_DUCK_CROSS:
        t_in = now - state_start_time
        target_px = int(w * CROSS_TARGET_FRAC)
        cv2.line(frame, (target_px, int(h * 0.5)), (target_px, h), (255, 0, 255), 2)

        if yellow_any_cx is not None:
            if yellow_any_cx > cx + 10:
                _cross_seen_right = True
            err = yellow_any_cx - target_px          # negative -> yellow still too far left
            omega = -(float(err) / cx) * CROSS_GAIN  # negative err -> positive omega (left)
            omega = max(CROSS_OMEGA_MIN, min(CROSS_OMEGA_MAX, omega))
            note = f"Overtake: crossing left (yellow @ {yellow_any_cx}px)"
        else:
            omega = CROSS_BLIND_OMEGA
            note = "Overtake: crossing left (yellow not visible)"

        publish_drive(OVERTAKE_SPEED, omega)
        arrow_v, arrow_omega = OVERTAKE_SPEED, omega
        _overtake_note = "phase 1/3 cross"

        crossed = (yellow_any_cx is not None and yellow_any_cx >= target_px)
        # if we saw it swing right and then lose it, we are already across
        lost_after_right = (_cross_seen_right and yellow_any_cx is None)
        if (crossed or lost_after_right) and t_in >= CROSS_MIN_S:
            current_state = STATE_DUCK_PASS
            state_start_time = now
            _pass_clear_frames = 0
            _pass_clear_time = 0.0
            _omega_hist.clear()
            print("Overtake: in the left lane -> passing the duck.")
        elif t_in > CROSS_MAX_S:
            current_state = STATE_DUCK_PASS
            state_start_time = now
            _pass_clear_frames = 0
            _pass_clear_time = 0.0
            _omega_hist.clear()
            print("Overtake: cross timed out -> passing anyway.")

    # ------------------------------------------------------------------
    # OVERTAKE PHASE 2 - lane-follow inside the left lane until past the duck
    # ------------------------------------------------------------------
    elif current_state == STATE_DUCK_PASS:
        t_in = now - state_start_time
        left_center, rows_hit = sample_left_lane_center(frame, yellow_lane, white_mask,
                                                        w, h, cx, draw=True)
        if left_center is not None:
            error = left_center - cx
            _, omega = steer_from_error(error, cx)
            publish_drive(OVERTAKE_SPEED, omega)
            arrow_v, arrow_omega = OVERTAKE_SPEED, omega
            note = f"Overtake: passing in left lane ({rows_hit}/4 rows, err {error:+d}px)"
        else:
            publish_drive(OVERTAKE_SPEED, PASS_LOST_LANE_OMEGA)
            arrow_v, arrow_omega = OVERTAKE_SPEED, PASS_LOST_LANE_OMEGA
            note = "Overtake: passing (left lane lines not visible)"
        _overtake_note = "phase 2/3 pass"

        # duck out of view (it has slid past the camera) -> start the clearance timer
        if duck_blocking:
            _pass_clear_frames = 0
            _pass_clear_time = 0.0
        else:
            _pass_clear_frames += 1
            if _pass_clear_frames == PASS_CLEAR_FRAMES:
                _pass_clear_time = now

        past_duck = (_pass_clear_time > 0.0 and (now - _pass_clear_time) >= PASS_EXTRA_S)
        if (past_duck and t_in >= PASS_MIN_S) or t_in > PASS_MAX_S:
            current_state = STATE_DUCK_MERGE
            state_start_time = now
            _omega_hist.clear()
            print("Overtake: duck is behind us -> merging back into the right lane.")

    # ------------------------------------------------------------------
    # OVERTAKE PHASE 3 - merge back right, confirmed by real line geometry
    # ------------------------------------------------------------------
    elif current_state == STATE_DUCK_MERGE:
        t_in = now - state_start_time
        target_px = int(w * MERGE_TARGET_FRAC)
        cv2.line(frame, (target_px, int(h * 0.5)), (target_px, h), (255, 0, 255), 2)

        lane_center, rows_hit = sample_lane_center(frame, yellow_lane, white_mask,
                                                   w, h, cx, draw=True)

        if yellow_any_cx is not None:
            err = yellow_any_cx - target_px           # positive -> yellow still too far right
            omega = -(float(err) / cx) * MERGE_GAIN   # positive err -> negative omega (right)
            omega = max(-MERGE_OMEGA_MAX, min(-MERGE_OMEGA_MIN, omega))
            note = f"Overtake: merging right (yellow @ {yellow_any_cx}px)"
        else:
            omega = MERGE_BLIND_OMEGA
            note = "Overtake: merging right (yellow not visible)"

        publish_drive(OVERTAKE_SPEED, omega)
        arrow_v, arrow_omega = OVERTAKE_SPEED, omega
        _overtake_note = "phase 3/3 merge"

        back_home = valid_lane_reacquired(lane_center, yellow_cx, white_cx, cx)
        if back_home and t_in >= MERGE_MIN_S:
            print("Overtake complete - right lane re-acquired.")
            _finish_duck_episode(now, overtook=True)
        elif t_in > MERGE_MAX_S:
            print("Overtake: merge timed out - handing back to lane follower.")
            _finish_duck_episode(now, overtook=True)

    elif current_state == STATE_RED_STOP:
        publish_drive(0.0, 0.0)
        elapsed_in_stop = now - state_start_time
        rem_stop = max(0.0, RED_STOP_DURATION - elapsed_in_stop)
        note = f"Stopped at red line ({rem_stop:.1f}s left)"
        cv2.putText(frame, f"STOP LINE HOLD: {rem_stop:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2.0s hold duration check
        if elapsed_in_stop >= RED_STOP_DURATION:
            if AUTO_PATH_MODE and path_intersections_passed < len(ROUTE_INTERSECTION_ORDER):
                tile = ROUTE_INTERSECTION_ORDER[path_intersections_passed]
                direction = ROUTE_TURN_TILES.get(tile, 'straight')
                path_intersections_passed += 1
                last_red_line_time = now
                start_intersection_turn(direction)
            elif AUTO_PATH_MODE:
                # Arrived / Route complete
                current_state = STATE_GOAL_REACHED
                publish_drive(0.0, 0.0)
            else:
                note = "Waiting for W / A / D key"

    elif current_state == STATE_INTERSECTION_TURN:
        exit_center, rows_hit = sample_lane_center(frame, yellow_lane, white_mask,
                                                   w, h, cx, draw=True)
        valid_exit = valid_lane_reacquired(exit_center, yellow_cx, white_cx, cx)
        time_in_turn = now - state_start_time
        min_s = MANEUVER_MIN_S.get(active_turn_direction, 0.8)
        min_rot = MANEUVER_MIN_ROT.get(active_turn_direction, 0.0)
        can_breakout = valid_exit and (time_in_turn >= min_s) and (abs(_turn_rot) >= min_rot)

        if can_breakout:
            last_red_line_time = now
            _omega_hist.clear()
            _lane_lost_frames = 0
            current_state = STATE_LANE_FOLLOWING
            note = "Exit lane acquired"

            # Start Dead-Reckoning after the final planned turn
            if AUTO_PATH_MODE and path_intersections_passed >= len(ROUTE_INTERSECTION_ORDER):
                post_intersections_tracking = True
                post_intersection_start_time = now
                print(f"Final turn complete. Driving {goal_total_duration:.2f}s to center of goal tile...")

        elif active_turn_direction == 'right':
            target_px = int(w * WHITE_HUG_TARGET_FRAC)
            if white_cx is not None:
                err = white_cx - target_px
                hug = -(float(err) / cx) * WHITE_HUG_GAIN
                hug = max(-WHITE_HUG_CLAMP, min(-0.8, hug))
                publish_drive(RIGHT_TURN_SPEED, hug)
                arrow_v, arrow_omega = RIGHT_TURN_SPEED, hug
                note = f"Right turn - hugging white line (w={hug:.2f})"
            else:
                publish_drive(RIGHT_TURN_SPEED, RIGHT_SEARCH_OMEGA)
                arrow_v, arrow_omega = RIGHT_TURN_SPEED, RIGHT_SEARCH_OMEGA
                note = "Right turn - corner hook"

            if time_in_turn > RIGHT_HUG_MAX_S:
                last_red_line_time = now
                current_state = STATE_LANE_FOLLOWING
                if AUTO_PATH_MODE and path_intersections_passed >= len(ROUTE_INTERSECTION_ORDER):
                    post_intersections_tracking = True
                    post_intersection_start_time = now
        else:
            if turn_step_index < len(turn_sequence_active):
                v, omega, duration = turn_sequence_active[turn_step_index]
                if time_in_step < duration:
                    publish_drive(v, omega)
                    arrow_v, arrow_omega = v, omega
                    note = f"{active_turn_direction} turn step {turn_step_index}"
                else:
                    turn_step_index += 1
                    step_start_time = now
            else:
                last_red_line_time = now
                current_state = STATE_LANE_FOLLOWING
                if AUTO_PATH_MODE and path_intersections_passed >= len(ROUTE_INTERSECTION_ORDER):
                    post_intersections_tracking = True
                    post_intersection_start_time = now

    if current_state == STATE_INTERSECTION_TURN:
        _turn_rot += arrow_omega * dt

    # HUD banner for the overtake
    if current_state in OVERTAKE_STATES:
        cv2.putText(frame, f"OVERTAKING DUCK - {_overtake_note}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # HUD Arrow Overlay
    if abs(arrow_omega) < 0.15:
        cv2.arrowedLine(frame, (cx, int(h * 0.85)), (cx, int(h * 0.68)),
                        (0, 255, 0), 4, cv2.LINE_AA, 0, 0.25)
    else:
        tx = cx - int(arrow_omega * 110)
        color = (0, 165, 255) if abs(arrow_omega) > 0.8 else (0, 255, 0)
        cv2.arrowedLine(frame, (cx, int(h * 0.85)), (tx, int(h * 0.68)),
                        color, 4, cv2.LINE_AA, 0, 0.25)

    TEL.update({"state": current_state, "v": round(arrow_v, 3), "omega": round(arrow_omega, 2),
                "rows": rows_hit, "yellow": yellow_cx is not None,
                "white": white_cx is not None, "duck": duck_found,
                "duck_area": int(duck_area), "fps": round(_fps, 1),
                "link": ("simulation" if simulation_mode else
                         ("stale" if link_stale else (_link_error or "ok"))),
                "note": note, "stopline_ahead": red_line_ahead,
                "auto_path_mode": AUTO_PATH_MODE,
                "setup_complete": SETUP_COMPLETE,
                "overtake": _overtake_note,
                "overtake_count": _overtake_count,
                "overtake_on": OVERTAKE_ENABLED,
                "path_progress": f"{path_intersections_passed}/{len(ROUTE_INTERSECTION_ORDER)}"
                                  if ROUTE_INTERSECTION_ORDER else "no route"})

    ok, jpeg = cv2.imencode('.jpg', frame)
    if ok:
        with lock:
            latest_jpeg = jpeg.tobytes()

    mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
    mask_frame[yellow_lane > 0] = (0, 255, 255)
    mask_frame[white_mask > 0] = (255, 255, 255)
    if stop_contours_confirmed:
        cv2.drawContours(mask_frame, stop_contours_confirmed, -1, (255, 120, 0), -1)
    if duck_contours_confirmed:
        cv2.drawContours(mask_frame, duck_contours_confirmed, -1, (0, 0, 255), -1)
    ok2, mjpeg = cv2.imencode('.jpg', mask_frame)
    if ok2:
        with lock:
            latest_mask_jpeg = mjpeg.tobytes()


# ==============================================================================
# SIMULATION FEED / ROS CAMERA SUBSCRIPTION
# ==============================================================================
def simulation_hardware_loop():
    print("Virtual camera loop active.")
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "SIMULATION FEED - ROBOT NOT CONNECTED", (15, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        cv2.line(frame, (180, 480), (280, 260), (0, 255, 255), 3)
        cv2.line(frame, (490, 480), (390, 260), (255, 255, 255), 3)
        if current_state == STATE_RED_STOP:
            cv2.rectangle(frame, (200, 400), (470, 430), (0, 0, 255), -1)
        process_image_frame(frame)
        time.sleep(0.033)


if simulation_mode:
    threading.Thread(target=simulation_hardware_loop, daemon=True).start()
else:
    camera_sub = roslibpy.Topic(client_ros, f'/{ROBOT}/camera_node/image/compressed',
                                'sensor_msgs/CompressedImage')

    def _on_frame(msg):
        try:
            buf = np.frombuffer(base64.b64decode(msg['data']), np.uint8)
            process_image_frame(cv2.imdecode(buf, cv2.IMREAD_COLOR))
        except Exception as e:
            print(f"frame decode error: {e}")

    camera_sub.subscribe(_on_frame)

# ==============================================================================
# DASHBOARD
# ==============================================================================
PAGE = """
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>duck3 Control Hub</title>
<style>
  * { box-sizing: border-box; }
  body { background:#141414; color:#f4f4f4; text-align:center; font-family:'Segoe UI', system-ui, sans-serif; margin:0; padding:0 20px 30px; }
  .road-header { margin:0 -20px 20px; padding:16px 20px 20px; border-bottom:5px solid #000; background:#0e0e0e; }
  .road-header h1 { margin:0; font-size:1.55em; color:#f6c915; }
  #status_box { margin-bottom:16px; background:#f6c915; color:#111; padding:12px; display:inline-block; border-radius:10px; font-weight:700; font-size:1.05em; width:80%; border:2px solid #000; }
  .side { text-align:left; width:290px; }
  .card { background:#1b1b1b; border:1px solid #2c2c2c; border-radius:10px; padding:10px 12px; margin-bottom:12px; font-size:.82em; color:#cfcfcf; }
  .card strong { color:#f6c915; display:block; margin-bottom:6px; }
  table { width:100%; border-collapse:collapse; }
  td { padding:2px 0; }
  td.k { color:#9a9a9a; }
  td.v { text-align:right; }
  .ok { color:#5ad07a; } .bad { color:#ff6b6b; }
  button { background:#262626; color:#f4f4f4; border:1px solid #3a3a3a; border-radius:6px; padding:6px 10px; font-size:.85em; cursor:pointer; }
  button:hover { background:#333; }
  button.active { background:#f6c915 !important; color:#111 !important; font-weight:bold; }
  button.go { background:#f6c915; color:#111; font-weight:700; border:0; }
  button.go:hover { background:#ffd733; }
  kbd { display:inline-block; background:#2c2c2c; border:1px solid #000; border-radius:4px; padding:1px 5px; font-weight:700; }
</style>
</head>
<body>
  <!-- MODAL POPUP -->
  <div id="setup_overlay" style="position:fixed; inset:0; background:rgba(10,10,10,0.92); z-index:1000; display:flex; align-items:center; justify-content:center;">
    <div style="background:#1b1b1b; border:2px solid #f6c915; border-radius:14px; padding:20px 24px; max-width:470px; width:94%; text-align:center; box-shadow:0 10px 30px rgba(0,0,0,0.8);">
      <h2 style="color:#f6c915; margin:0 0 4px;">Choose Navigation Mode</h2>
      <p style="color:#aaa; font-size:.82em; margin:0 0 14px;">duck3 is stopped until you pick a mode.</p>
      <div id="setup_step_choose">
        <button onclick="showAStarSetup()" style="width:100%; padding:14px; margin-bottom:12px; font-size:1.05em; font-weight:bold; background:#262626; border:1px solid #f6c915; color:#f6c915;">
          A* Autonomous Route Planner
        </button>
        <button onclick="confirmManual()" style="width:100%; padding:12px; font-size:0.95em;">
          Manual Mode (WASD)
        </button>
      </div>
      <div id="setup_step_astar" style="display:none; text-align:left;">
        <div style="text-align:center; margin-bottom:10px;">
          <div id="modal_mapbox" style="display:inline-block; background:#111; border:2px solid #333; border-radius:8px; padding:4px;"></div>
        </div>
        <div style="display:flex; gap:8px; margin-bottom:8px;">
          <button onclick="setClickMode('start')" id="modal_btn_start" style="flex:1;">1. Set Start</button>
          <button onclick="setClickMode('goal')" id="modal_btn_goal" style="flex:1;">2. Set Goal</button>
        </div>
        <div style="margin-bottom:10px; font-size:.85em; display:flex; align-items:center; justify-content:space-between;">
          <span>Initial Facing:</span>
          <div>
            <button onclick="setHeading('N')" id="head_N">N</button>
            <button onclick="setHeading('E')" id="head_E">E</button>
            <button onclick="setHeading('S')" id="head_S">S</button>
            <button onclick="setHeading('W')" id="head_W">W</button>
            <span id="heading_val" style="color:#f6c915; font-weight:700; margin-left:6px;">E</span>
          </div>
        </div>
        <button class="go" onclick="computePath()" style="width:100%; padding:10px; margin-bottom:8px; font-size:0.95em;">3. Compute Route</button>
        <div id="astar_setup_status" style="color:#f2a900; font-size:.82em; margin-bottom:12px; text-align:center;">Click Set Start, then click a tile on the grid above.</div>
        <button onclick="confirmAutoPath()" style="width:100%; padding:12px; font-size:1em; font-weight:bold; background:#22c55e; color:#000; border:0; border-radius:6px; margin-bottom:8px;">
          Start Driving (A*)
        </button>
        <button onclick="backToChoose()" style="width:100%; padding:6px; font-size:.8em;">Back</button>
      </div>
    </div>
  </div>

  <div class="road-header">
    <h1>Duckietown &middot; duck3 Control Hub</h1>
  </div>

  <div id="status_box">Starting up...</div>

  <div style="display:flex; justify-content:center; align-items:flex-start; gap:15px; flex-wrap:wrap;">
    <div style="width:100%; max-width:640px;">
      <img id="video-frame" src="/video" style="width:100%; border:3px solid #f6c915; border-radius:10px; background:#000;">
      <div style="color:#9a9a9a; font-size:.78em; margin:6px 0;">Recognition Mask</div>
      <img id="mask-frame" src="/mask_video" style="width:100%; border:3px solid #333; border-radius:10px; background:#000;">
    </div>

    <div class="side">
      <div class="card">
        <strong>Telemetry</strong>
        <table>
          <tr><td class="k">State</td><td class="v" id="t_state">-</td></tr>
          <tr><td class="k">Speed v</td><td class="v" id="t_v">-</td></tr>
          <tr><td class="k">Yaw w</td><td class="v" id="t_omega">-</td></tr>
          <tr><td class="k">Lines</td><td class="v" id="t_lines">-</td></tr>
          <tr><td class="k">Duck</td><td class="v" id="t_duck">-</td></tr>
          <tr><td class="k">FPS</td><td class="v" id="t_fps">-</td></tr>
          <tr><td class="k">Link</td><td class="v" id="t_link">-</td></tr>
        </table>
      </div>

      <div class="card">
        <strong>Duck Avoidance</strong>
        <table>
          <tr><td class="k">Overtake</td><td class="v" id="t_ot_on">-</td></tr>
          <tr><td class="k">Phase</td><td class="v" id="t_ot_phase">-</td></tr>
          <tr><td class="k">Completed</td><td class="v" id="t_ot_count">-</td></tr>
        </table>
        <div style="margin-top:6px; color:#8a8a8a;">Stop &rarr; wait 2s &rarr; cross yellow &rarr; pass in left lane &rarr; merge back.</div>
      </div>

      <div class="card">
        <strong>Controls</strong>
        <kbd>E</kbd> Manual Override &nbsp; <kbd>Space</kbd> Stop &nbsp; <kbd>Q</kbd> Quit<br>
        <kbd>W/A/S/D</kbd> Drive manually or turn at red lines.<br>
        <kbd>Y</kbd> Toggle AUTO_PATH_MODE &nbsp; <kbd>O</kbd> Toggle overtaking<br>
        <kbd>R</kbd> Reset run state
      </div>

      <div class="card">
        <strong>Live Route Map (with Compass)</strong>
        <div id="mapbox" style="margin-top:6px; text-align:center;"></div>
      </div>
    </div>
  </div>

<script>
  function cls(good) { return good ? 'ok' : 'bad'; }

  setInterval(function() {
    fetch('/telemetry').then(r => r.json()).then(t => {
      document.getElementById('t_state').innerText = t.state;
      document.getElementById('t_v').innerText = t.v.toFixed(3);
      document.getElementById('t_omega').innerText = (t.omega >= 0 ? '+' : '') + t.omega.toFixed(2);
      var lines = document.getElementById('t_lines');
      lines.innerHTML = '<span class="' + cls(t.yellow) + '">Y</span> / <span class="' + cls(t.white) + '">W</span> (' + t.rows + '/4)';
      document.getElementById('t_duck').innerText = t.duck ? ('YES (' + t.duck_area + 'px)') : 'None';
      document.getElementById('t_fps').innerText = t.fps.toFixed(1);
      var link = document.getElementById('t_link');
      link.innerText = t.link;
      link.className = 'v ' + (t.link === 'ok' ? 'ok' : 'bad');

      var otOn = document.getElementById('t_ot_on');
      otOn.innerText = t.overtake_on ? 'ENABLED' : 'OFF';
      otOn.className = 'v ' + (t.overtake_on ? 'ok' : 'bad');
      document.getElementById('t_ot_phase').innerText = t.overtake;
      document.getElementById('t_ot_count').innerText = t.overtake_count;

      var box = document.getElementById('status_box');
      box.innerText = t.note ? (t.state + ' - ' + t.note) : t.state;
      if (t.state === 'goal_reached') { box.style.background = '#22c55e'; box.style.color = '#000'; }
      else if (t.state === 'red_line_stopped') { box.style.background = '#c0342e'; box.style.color = '#fff'; }
      else if (t.state === 'duck_stopped') { box.style.background = '#e67e22'; box.style.color = '#fff'; }
      else if (t.state.indexOf('duck_overtake') === 0) { box.style.background = '#a855f7'; box.style.color = '#fff'; }
      else { box.style.background = '#f6c915'; box.style.color = '#111'; }
    });
  }, 250);

  function reloadMap() {
    fetch('/map_svg').then(r => r.text()).then(svg => {
      var sideBox = document.getElementById('mapbox');
      if (sideBox) sideBox.innerHTML = svg;
      var modalBox = document.getElementById('modal_mapbox');
      if (modalBox) modalBox.innerHTML = svg;
    });
  }
  setInterval(reloadMap, 600);

  var clickMode = null;
  function setClickMode(mode) {
    clickMode = (clickMode === mode) ? null : mode;
    var bStart = document.getElementById('modal_btn_start');
    var bGoal = document.getElementById('modal_btn_goal');
    if (bStart) bStart.className = (clickMode === 'start') ? 'active' : '';
    if (bGoal) bGoal.className = (clickMode === 'goal') ? 'active' : '';

    var status = document.getElementById('astar_setup_status');
    if (clickMode === 'start') status.innerText = 'Click a tile to place START marker (S).';
    else if (clickMode === 'goal') status.innerText = 'Click a tile to place GOAL marker (G).';
  }

  function mapTileClick(x, y) {
    if (clickMode === 'start') {
      fetch('/set_start?x=' + x + '&y=' + y).then(() => {
        setClickMode(null);
        document.getElementById('astar_setup_status').innerText = 'Start set! Now click 2. Set Goal.';
        reloadMap();
      });
    } else if (clickMode === 'goal') {
      fetch('/set_goal?x=' + x + '&y=' + y).then(() => {
        setClickMode(null);
        document.getElementById('astar_setup_status').innerText = 'Goal set! Click 3. Compute Route.';
        reloadMap();
      });
    }
  }

  function setHeading(dir) {
    fetch('/set_heading?dir=' + dir).then(r => r.json()).then(d => {
      if (d.ok) {
        document.getElementById('heading_val').innerText = d.heading;
        ['N','E','S','W'].forEach(k => {
          document.getElementById('head_' + k).className = (k === d.heading) ? 'active' : '';
        });
      }
    });
  }

  function computePath() {
    fetch('/compute_path').then(r => r.json()).then(d => {
      var status = document.getElementById('astar_setup_status');
      status.innerText = d.message;
      status.style.color = d.ok ? '#5ad07a' : '#ff6b6b';
      reloadMap();
    });
  }

  function confirmManual() {
    fetch('/confirm_manual').then(() => {
      document.getElementById('setup_overlay').style.display = 'none';
    });
  }

  function showAStarSetup() {
    document.getElementById('setup_step_choose').style.display = 'none';
    document.getElementById('setup_step_astar').style.display = 'block';
    setHeading('E');
    reloadMap();
  }

  function backToChoose() {
    document.getElementById('setup_step_choose').style.display = 'block';
    document.getElementById('setup_step_astar').style.display = 'none';
  }

  function confirmAutoPath() {
    fetch('/confirm_auto_path').then(r => r.json()).then(d => {
      if (d.ok) {
        document.getElementById('setup_overlay').style.display = 'none';
      } else {
        var status = document.getElementById('astar_setup_status');
        status.innerText = 'Error: ' + d.error;
        status.style.color = '#ff6b6b';
      }
    });
  }

  document.addEventListener('keydown', function(e) {
    var k = e.key.toLowerCase();
    if (e.key === ' ') k = 'space';
    if (['w','a','s','d','e','q','r','y','o','space'].indexOf(k) >= 0) {
      if (k === 'space') e.preventDefault();
      fetch('/control?key=' + k);
    }
  });
</script>
</body>
</html>
"""


def _mjpeg_stream(getter):
    boundary = b'--frame\r\n'
    while True:
        with lock:
            buf = getter()
        if buf is not None:
            yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + buf + b'\r\n'
        time.sleep(0.04)


@app.route('/')
def index():
    return PAGE


@app.route('/video')
def video():
    return Response(_mjpeg_stream(lambda: latest_jpeg),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/mask_video')
def mask_video():
    return Response(_mjpeg_stream(lambda: latest_mask_jpeg),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/telemetry')
def telemetry():
    return jsonify(TEL)


@app.route('/map_svg')
def map_svg():
    return Response(render_map_svg(), mimetype='image/svg+xml')


@app.route('/set_start')
def set_start():
    global bot_start_tile, ROUTE_START
    try:
        x = int(request.args.get('x'))
        y = int(request.args.get('y'))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad coordinates"})
    if not (0 <= x < MAP_W and 0 <= y < MAP_H):
        return jsonify({"ok": False, "error": "off map"})
    if MAP_TILES[y][x] not in OPENINGS:
        return jsonify({"ok": False, "error": "not a road tile"})
    bot_start_tile = (x, y)
    ROUTE_START = (x, y)
    return jsonify({"ok": True, "start": [x, y]})


@app.route('/set_goal')
def set_goal():
    global ROUTE_GOAL
    try:
        x = int(request.args.get('x'))
        y = int(request.args.get('y'))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad coordinates"})
    if not (0 <= x < MAP_W and 0 <= y < MAP_H):
        return jsonify({"ok": False, "error": "off map"})
    if MAP_TILES[y][x] not in OPENINGS:
        return jsonify({"ok": False, "error": "not a road tile"})
    ROUTE_GOAL = (x, y)
    return jsonify({"ok": True, "goal": [x, y]})


@app.route('/set_heading')
def set_heading():
    global bot_heading
    d = (request.args.get('dir') or '').upper()
    if d not in DELTAS:
        return jsonify({"ok": False, "error": "bad heading"})
    bot_heading = d
    return jsonify({"ok": True, "heading": bot_heading})


@app.route('/compute_path')
def compute_path_route():
    ok, message = compute_route()
    return jsonify({"ok": ok, "message": message})


@app.route('/confirm_manual')
def confirm_manual():
    global AUTO_PATH_MODE, SETUP_COMPLETE
    AUTO_PATH_MODE = False
    SETUP_COMPLETE = True
    print("Manual mode confirmed.")
    return jsonify({"ok": True})


@app.route('/confirm_auto_path')
def confirm_auto_path():
    global AUTO_PATH_MODE, SETUP_COMPLETE
    if len(ROUTE) < 2:
        return jsonify({"ok": False, "error": "compute a route first"})
    AUTO_PATH_MODE = True
    SETUP_COMPLETE = True
    print(f"A* mode confirmed. Route: {ROUTE}")
    return jsonify({"ok": True})


@app.route('/control')
def control():
    global keyboard_engaged, manual_v, manual_omega, AUTO_PATH_MODE, OVERTAKE_ENABLED
    global current_state, state_start_time, last_red_line_time
    global path_intersections_passed, post_intersections_tracking, current_tracked_tile_index
    global _omega_hist, _lane_lost_frames, _duck_seen_frames, _duck_clear_frames, _overtake_note

    key = (request.args.get('key') or '').lower()
    now = time.time()

    if key == 'q':
        release_override()
        threading.Timer(0.2, lambda: os._exit(0)).start()
        return jsonify({"ok": True, "action": "quit"})

    if key == 'e':
        keyboard_engaged = not keyboard_engaged
        manual_v = manual_omega = 0.0
        publish_drive(0.0, 0.0)
        return jsonify({"ok": True, "manual": keyboard_engaged})

    if key == 'space':
        manual_v = manual_omega = 0.0
        publish_drive(0.0, 0.0)
        return jsonify({"ok": True, "action": "stop"})

    if key == 'y':
        AUTO_PATH_MODE = not AUTO_PATH_MODE
        return jsonify({"ok": True, "auto_path_mode": AUTO_PATH_MODE})

    if key == 'o':
        OVERTAKE_ENABLED = not OVERTAKE_ENABLED
        return jsonify({"ok": True, "overtake_enabled": OVERTAKE_ENABLED})

    if key == 'r':
        current_state = STATE_LANE_FOLLOWING
        state_start_time = now
        last_red_line_time = 0.0
        path_intersections_passed = 0
        post_intersections_tracking = False
        current_tracked_tile_index = 0
        _omega_hist.clear()
        _lane_lost_frames = 0
        _duck_seen_frames = 0
        _duck_clear_frames = 0
        _overtake_note = "idle"
        return jsonify({"ok": True, "action": "reset"})

    if key in ('w', 'a', 's', 'd'):
        # At a red line in manual routing, W/A/D pick the intersection manoeuvre.
        if current_state == STATE_RED_STOP and not keyboard_engaged and key != 's':
            direction = {'w': 'straight', 'a': 'left', 'd': 'right'}[key]
            last_red_line_time = now
            start_intersection_turn(direction)
            return jsonify({"ok": True, "turn": direction})

        keyboard_engaged = True
        if key == 'w':
            manual_v, manual_omega = LANE_SPEED, 0.0
        elif key == 's':
            manual_v, manual_omega = -LANE_SPEED, 0.0
        elif key == 'a':
            manual_v, manual_omega = LANE_SPEED * 0.6, 1.5
        elif key == 'd':
            manual_v, manual_omega = LANE_SPEED * 0.6, -1.5
        publish_drive(manual_v, manual_omega)
        return jsonify({"ok": True, "v": manual_v, "omega": manual_omega})

    return jsonify({"ok": False, "error": "unknown key"})


if __name__ == '__main__':
    print("Dashboard: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
