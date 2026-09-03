#!/usr/bin/env python3
"""
====================================================================
 DUCKIETOWN AUTONOMOUS NAVIGATION STACK - EXPLAINED VERSION
====================================================================
This script drives a Duckiebot (a small self-driving toy car) around
a track made of tiles. In plain terms, it does four jobs at once:

1. VISION: reads the camera, finds the yellow center line, the white
   edge line, red stop lines, and rubber ducks sitting on the road.
2. DRIVING: uses that vision to steer down the middle of the lane
   (a simple "PD controller" - explained below), stop at red lines,
   turn at intersections, and swerve around ducks.
3. PLANNING: knows the shape of the whole track (a grid of tiles)
   and uses the A* pathfinding algorithm (the same idea used in many
   games) to work out the shortest route with the fewest turns from
   a start tile to a goal tile.
4. DASHBOARD: runs a small website (using Flask) so a human can watch
   the camera feed, see robot stats, click a start/goal on a map, and
   drive the robot manually with the keyboard if they want.

Most behavior is kept camera-only and intentionally simple. Comments explain
the controls and state-machine decisions so each part is easy to follow.
====================================================================
"""
import base64
import csv
import datetime
import os
import signal
import threading
import time
import atexit
import cv2                 # OpenCV: image processing (finding colors, shapes, etc.)
import numpy as np         # Fast math on grids of numbers (used for images)
import roslibpy             # Lets Python talk to the robot's ROS software over the network
import yaml                 # Reads the map file (a simple text format)
from flask import Flask, Response, jsonify, request  # Runs the small web dashboard

# When this script starts, it first makes sure any older/default navigation
# program running on the robot is stopped, so this script is the only thing
# sending drive commands (otherwise the two would fight over the wheels).
print("Stopping baseline navigation container...")
os.system("ssh duckie@duck3.local 'docker stop demo_indefinite_navigation' > /dev/null 2>&1 &")

app = Flask(__name__)  # The Flask app that serves the dashboard web page

# ---- shared "latest camera frame" storage ----
# The camera thread constantly overwrites these with the newest picture,
# and the web dashboard thread reads them to stream video to the browser.
# `lock` prevents the two threads from reading/writing at the exact same
# instant and corrupting the image.
latest_jpeg = None       # Most recent camera frame (with drawings on it), as JPEG bytes
latest_mask_jpeg = None  # Most recent "what the robot sees" debug view (colored blobs)
lock = threading.Lock()
simulation_mode = False  # True if we couldn't connect to the real robot (fallback demo mode)

VEHICLE = os.environ.get("VEHICLE_NAME", "duck3")  # The robot's name (used in ROS topic names)
ROSBRIDGE_HOST = 'localhost'
ROSBRIDGE_PORT = 9001
FRAME_STALE_S = 0.4  # If no new camera frame arrives for this many seconds, treat the link as broken and stop the wheels

# ==============================================================================
# LANE FOLLOWING
# ==============================================================================
# The robot steers using a classic control technique called a "PD controller"
# (Proportional-Derivative). In simple terms:
#   - "Proportional" (KP): the further the lane center is from the middle of
#     the camera image, the harder the robot steers to correct it.
#   - "Derivative" (KD): if that error is changing quickly (the robot is
#     swinging), it also reacts to the *rate* of change, which smooths out
#     the steering instead of overshooting side to side.
# Think of it like a driver correcting the wheel: turn more if you're far
# off-center, and ease off if you're already turning quickly toward center.
KP = 4.0            # How strongly to react to being off-center (steering "stiffness")
KD = 2.0            # How strongly to react to the error changing quickly (damping/smoothing)
V_BAR = 0.10         # Normal forward driving speed
OMEGA_MAX = 6.0      # Fastest the robot is allowed to spin/turn
V_MIN = 0.05         # Slowest forward speed (never drop below this while still moving)
SLOWDOWN_STRENGTH = 0.8  # How much to slow down when steering hard (sharper turn = slower speed, like a real car)
ROI_FRACTION = 0.40  # Only look at the bottom 40% of the camera image for lane lines (that's the road right in front of the robot; the top of the image is mostly distant background)

# Road layout as seen by the camera, left to right:
# [white edge] [oncoming lane] [YELLOW centre line] [our lane] [white edge]
# Without limits, the line-detector could accidentally grab the *far* white
# edge (across the road) instead of the near one, especially on a curve,
# and steer the robot the wrong way. These "gates" restrict where in the
# image each color is allowed to be searched for, so each line search stays
# on its own correct side.
YELLOW_SEARCH_MAX = 0.70   # Only look for yellow in the left 70% of the image
WHITE_SEARCH_MIN = 0.30    # Only look for white starting from the right 30% of the image
MIN_LANE_WIDTH_PX = 20     # If the detected white line is too close to the yellow line, it's probably noise, not a real lane edge
HALF_LANE_FRAC = 0.25      # If only one line (yellow or white) is visible, estimate the lane center by offsetting from it by this fraction of the image width
LANE_MEMORY_DECAY = 0.85   # If the lane is briefly lost, keep steering like before but fade the turn amount toward straight each frame
LANE_LOST_MAX_FRAMES = 15  # After this many frames with no lane found at all, give up and stop instead of guessing forever

def centroid_x(mask, x_lo=None, x_hi=None):
    """
    Given a black-and-white "mask" image (white pixels = the color we're
    looking for, e.g. yellow paint), find the horizontal (x) position of
    the middle of that blob of color - like finding the center of gravity
    of all the white pixels. Returns None if there isn't enough of the
    color to be confident it's really a line (avoids reacting to a couple
    of stray pixels).

    x_lo / x_hi optionally restrict the search to a horizontal band of the
    image (see the "gates" explained above) - implemented by blanking out
    everything outside that band rather than cropping the image, so the
    returned x position is still measured in the original full-image
    coordinates.
    """
    if x_lo is not None or x_hi is not None:
        gated = np.zeros_like(mask)
        lo = 0 if x_lo is None else max(0, int(x_lo))
        hi = mask.shape[1] if x_hi is None else min(mask.shape[1], int(x_hi))
        if hi <= lo:
            return None
        gated[:, lo:hi] = mask[:, lo:hi]
        mask = gated
    m = cv2.moments(mask)          # OpenCV's built-in "center of mass" calculation for a shape
    if m["m00"] < 500:             # m00 is roughly "how many white pixels" - too few means no real line
        return None
    return m["m10"] / m["m00"]     # This division gives the x-coordinate of the center of mass

# ==============================================================================
# VISION THRESHOLDS + DUCK / STOP-LINE RECOGNITION
# ==============================================================================
# HSV color ranges used to pick out each color of interest from the camera
# image. HSV (Hue, Saturation, Value) is used instead of plain RGB because
# it separates "what color" from "how bright/washed out", which makes color
# detection far more reliable under different lighting.
HSV_YELLOW = (np.array([10, 70, 70]), np.array([40, 255, 255]))
HSV_WHITE = (np.array([0, 0, 150]), np.array([180, 45, 255]))
# Red wraps around both ends of the hue wheel in HSV, so two ranges are
# needed and then combined (RED_A covers reds near 0, RED_B covers reds
# near 180 - both ends of the color wheel are "red").
HSV_RED_A = (np.array([0, 110, 60]), np.array([15, 255, 255]))
HSV_RED_B = (np.array([160, 110, 60]), np.array([180, 255, 255]))

DUCK_MIN_AREA = 2000     # A blob of yellow smaller than this (in pixels) is too small to be a duck
DUCK_MAX_AREA = 160000   # A blob bigger than this is too big to be a duck (probably the yellow lane line itself)
DUCK_LARGE_AREA = 40000  # Above this size, the duck is very close to the camera and may be cut off by the frame edge

STOPLINE_MIN_AR = 1.5           # Stop lines are wide rectangles - minimum width-to-height ratio to count as one
STOPLINE_MIN_WIDTH_FRAC = 0.20  # Must span at least this fraction of the image width
STOPLINE_MIN_AREA = 150
STOPLINE_MIN_BOTTOM_FRAC = 0.80 # Must be near the bottom of the image (i.e., close to the robot, not far away)
STOPLINE_COOLDOWN_S = 5.0       # After stopping at a line, ignore stop-line detections for this long (so the same line isn't re-triggered while driving away from it)
RED_STOP_HOLD_S = 2.0            # How many seconds to sit still at a stop line before continuing

def is_duck(contour):
    """
    Decide whether a detected yellow blob's outline ("contour") actually
    looks like a duck, as opposed to a lane marking or other yellow object.
    This uses several shape checks stacked together - a duck is roughly
    oval/blobby, not a thin straight rectangle like a road line.
    """
    area = cv2.contourArea(contour)
    if area < DUCK_MIN_AREA or area > DUCK_MAX_AREA:
        return False  # Wrong size entirely
    x, y, w, h = cv2.boundingRect(contour)  # Smallest upright rectangle that contains the shape
    aspect_ratio = float(w) / h if h > 0 else 0
    # A duck that's very close to the robot often gets clipped by the edge
    # of the camera frame, which distorts its normal proportions - so the
    # width/height ratio check is loosened for big (close) blobs. The other
    # shape checks below (solidity, rectangularity) still catch and reject
    # anything that's actually a lane marking rather than a duck.
    if area >= DUCK_LARGE_AREA:
        if aspect_ratio > 2.4 or aspect_ratio < 0.3:
            return False
    elif aspect_ratio > 1.9 or aspect_ratio < 0.4:
        return False
    # "Solidity" = how much of the shape's convex hull (the tightest rubber
    # band wrapped around it) is actually filled in. A rectangle-like lane
    # marking is very solid (close to 1.0); a duck's irregular outline is less so.
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0:
        return False
    solidity = area / hull_area
    # "Rectangularity" = how much of a tightly-fit rotated rectangle the
    # shape fills. A real rectangle (lane marking) fills nearly all of it.
    rect = cv2.minAreaRect(contour)
    w_rot, h_rot = rect[1]
    rect_area = w_rot * h_rot
    if rect_area == 0:
        return False
    rectangularity = area / rect_area
    long_side, short_side = max(w_rot, h_rot), min(w_rot, h_rot)
    rot_aspect = long_side / short_side if short_side > 0 else 0
    # approxPolyDP simplifies the outline down to its corner points - a
    # rectangle simplifies to about 4-6 points; a duck's rounded shape needs
    # more points to approximate, so few points + high solidity = "too
    # rectangle-like to be a duck" -> reject.
    per = cv2.arcLength(contour, True)
    if per == 0:
        return False
    approx = cv2.approxPolyDP(contour, 0.018 * per, True)
    if len(approx) <= 6 and solidity > 0.85:
        return False
    if rot_aspect > 1.55 and rectangularity > 0.72:
        return False
    return True  # Passed every shape test - treat it as a duck

# ==============================================================================
# DUCK AVOIDANCE (movement)
#
# Behaviour, in plain terms: when a duck is seen close enough in the lane,
# the robot stops completely. It then watches the duck for a couple of
# seconds to figure out if it's a real (possibly moving) duck or just a
# stationary obstacle:
#   - If the duck shifts position/size noticeably, that resets the "has it
#     been still long enough" timer, because it might walk away on its own.
#   - If it disappears / moves out of the way on its own, the robot just
#     resumes driving - no need to go around it.
#   - If it sits still for the full wait time (or a maximum timeout is hit
#     regardless), the robot commits to steering around it.
#
# The go-around itself is simple: for a few seconds, instead of aiming for
# the usual "half a lane to the right of the yellow line" target, the robot
# aims to put the YELLOW LINE ITSELF in the center of the camera image.
# That nudges the robot left, around the duck, using the exact same PD
# steering controller as normal lane following - just pointed at a
# different target. Because it's still closed-loop (reacting to what the
# camera sees every frame) rather than a blind fixed turn, it can't run
# away or overshoot the way an open-loop timed swerve could.
# ==============================================================================
DUCK_AVOIDANCE_ON = True
DUCK_TRIGGER_FRAMES = 3       # Duck must be seen for this many consecutive frames before reacting (avoids reacting to a one-frame glitch)
DUCK_TRIGGER_AREA = 3500      # Duck's blob must be at least this big (i.e. close enough) to be worth stopping for
DUCK_COOLDOWN_S = 8           # Minimum seconds between separate duck-avoidance episodes (stops the robot re-triggering on the same duck as it drives away)

# ---- step 0: stop and watch --------------------------------------------
DUCK_WAIT_DURATION = 2.0      # Duck must sit still for this long before the robot plans to go around it
DUCK_MAX_WAIT_S = 10.0        # Hard safety cap: go around it anyway after this long, even if it keeps shifting slightly
DUCK_MOVED_PX = 22            # If the duck's center shifts more than this many pixels, count it as "it moved"
DUCK_MOVED_AREA_FRAC = 0.30   # Or if its blob size changes by more than this fraction, also count as "it moved"
DUCK_CLEAR_FRAMES = 4         # This many consecutive frames with no blocking duck means the path is free again
DUCK_RETRIGGER_COOL = 2.5     # After finishing a go-around, ignore ducks for a bit (the one we just passed is now behind us)

# ---- go-around: centre on the yellow line for a fixed duration -----------
OVERTAKE_SPEED = 0.072
DUCK_FOLLOW_YELLOW_S = 3.0     # How long to ride the yellow line before switching back to normal lane following
AVOID_OMEGA_MAX = 2.4          # Tighter (gentler) steering limit while going around the duck, for a smoother swerve
YELLOW_SEARCH_SWERVE = 0.95    # Widen how far right the yellow-line search is allowed to look while re-centring on it

# ==============================================================================
# INTERSECTION MANEUVERS (movement)
# ==============================================================================
# Pre-planned short sequences of (speed, turn-rate, duration) steps used to
# physically carry the robot through a left turn or a straight crossing at
# an intersection, before normal lane-following can see the new lane and
# take back over. Each tuple is (v, omega, seconds).
INTERSECTION_LEFT_STEPS = [(0.08, 0.0, 0.8), (0.10, 1.20, 1.8), (0.07, 0.0, 1.0)]
INTERSECTION_STRAIGHT_STEPS = [(0.09, 0.0, 3.0)]

# A right turn instead works by "hugging" the white line on the outside of
# the curve at a target position in the image, steering to keep it there.
WHITE_HUG_TARGET_FRAC = 0.78
WHITE_HUG_GAIN = 2.0
WHITE_HUG_CLAMP = 2.2
RIGHT_SEARCH_V = 0.06     # If the white line isn't found yet during a right turn, creep forward slowly...
RIGHT_SEARCH_OMEGA = -1.6 # ...while turning right to go looking for it
RIGHT_HUG_MAX_S = 6.0     # Safety timeout for the whole right-turn maneuver

# During an explicit start U-turn the robot rotates in place at this low yaw
# rate, so it can keep checking camera geometry every frame instead of doing
# a blind timed spin.
UTURN_OMEGA = 1.0

# This is only a safety fallback: if the camera never confirms the opposite
# lane within this many seconds, stop and return to lane following anyway.
UTURN_TIMEOUT_S = 5.0

# ==============================================================================
# A* PATH PLANNER WITH HEADING CONSTRAINT
# ==============================================================================
# The track is represented as a grid of tiles (like a simple board game
# map). Each tile has a "kind" (straight road, curve, 3-way or 4-way
# intersection, or grass/off-road) and the compass directions it's allowed
# to connect to. A* is a well-known shortest-path algorithm (used in maps
# apps and games) that finds the best route between a start tile and a goal
# tile using this connectivity information.
MAP_PATH = os.path.expanduser('~/tum_map.yaml')
DELTAS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}   # How each compass direction moves on the grid (x, y)
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
LEFT_OF = {"N": "W", "W": "S", "S": "E", "E": "N"}    # "Turning left from heading X ends up heading..."
RIGHT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}   # "Turning right from heading X ends up heading..."

# For each tile "kind", which compass directions can a road actually enter
# or exit from? E.g. a tile that curves left when approached from the
# north connects North and West.
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

# A "3way_*" or "4way" tile is the only place a red stop line actually
# exists on the real track - driving through one always costs a real
# stop-and-wait plus a turn maneuver, which takes much longer than simply
# driving across one more ordinary tile. This default penalty is used when
# the dashboard's "avoid intersections" setting is enabled; switching it to
# 0.0 makes A* behave like a plain shortest-path planner.
DEFAULT_INTERSECTION_PENALTY = 1000.0

# A U-turn is slower than a tile but far cheaper than an extra intersection,
# so the planner should accept one to avoid a stop line.
UTURN_PENALTY = 3.0

# These time values are estimates until real runs are logged. They are used
# only by the route-comparison endpoint to give a rough side-by-side total.
EST_TILE_TIME_S = 4.5
EST_TURN_TIME_S = 3.0
EST_UTURN_TIME_S = 4.0

intersection_penalty_value = DEFAULT_INTERSECTION_PENALTY

def is_intersection_kind(kind):
    return kind.startswith("3way_") or kind == "4way"

def current_intersection_penalty():
    return intersection_penalty_value

def load_track_map():
    """Load the grid of tiles that make up the track from a YAML file on
    disk. If that file isn't there or can't be read, fall back to a
    hard-coded default 7-row by 6-column map so the robot can still run."""
    try:
        with open(MAP_PATH, 'r') as f:
            data = yaml.safe_load(f)
        raw_tiles = data["tiles"]
        tiles = [[str(cell).strip() for cell in row] for row in raw_tiles]
        h = len(tiles)
        w = max(len(r) for r in tiles) if h > 0 else 0
        return tiles, w, h
    except Exception as e:
        print(f"Map file unavailable ({e}). Using default TUM 7x6 matrix.")
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
    """Turn the grid of tiles into a graph: for every road tile, work out
    which neighboring tiles it's actually connected to (both tiles' road
    openings must line up), and in which compass direction. This graph is
    what A* searches over."""
    adj = {}
    for j in range(MAP_H):
        for i in range(MAP_W):
            kind = MAP_TILES[j][i]
            if kind not in OPENINGS:
                continue  # Not a road tile (e.g. grass) - skip it
            node = (i, j)
            adj.setdefault(node, [])
            for d in OPENINGS[kind]:
                di, dj = DELTAS[d]
                ni, nj = i + di, j + dj
                if 0 <= ni < MAP_W and 0 <= nj < MAP_H:
                    nk = MAP_TILES[nj][ni]
                    # Only a real connection if the neighboring tile also
                    # opens back toward this one (roads must match up on
                    # both sides)
                    if nk in OPENINGS and OPPOSITE[d] in OPENINGS[nk]:
                        adj[node].append(((ni, nj), d))
    return adj

GRAPH_ADJ = build_adjacency_graph()

def _step_dir(a, b):
    """Compass direction of the one-tile move from a to b."""
    di, dj = b[0] - a[0], b[1] - a[1]
    for d, (x, y) in DELTAS.items():
        if (di, dj) == (x, y):
            return d
    return None

def _path_first_heading(path):
    if len(path) < 2:
        return None
    return _step_dir(path[0], path[1])

def _count_path_intersections(path):
    return sum(1 for tile in path[:-1] if is_intersection_kind(MAP_TILES[tile[1]][tile[0]]))

def _count_route_intersections_for_summary(path):
    count = _count_path_intersections(path)
    if path and count < 2 and is_intersection_kind(MAP_TILES[path[-1][1]][path[-1][0]]):
        count += 1
    return count

def _route_tile_count(path):
    return max(0, len(path) - 1)

def _route_uturn_count(turn_tiles):
    return sum(1 for d in turn_tiles.values() if d == "uturn")

def _estimate_route_time(path, turn_tiles):
    tiles = _route_tile_count(path)
    intersections = _count_route_intersections_for_summary(path)
    uturns = _route_uturn_count(turn_tiles)
    return (EST_TILE_TIME_S * tiles
            + intersections * (RED_STOP_HOLD_S + EST_TURN_TIME_S)
            + uturns * EST_UTURN_TIME_S)

def _astar_search_once(start, goal, required_heading=None, penalty=None):
    """
    Weighted A* pathfinding between two tiles for a single heading policy.

    Every tile normally costs 1.0 to drive onto, plus the active intersection
    penalty if it's a stop-line intersection - except the very last (goal) tile,
    since arriving there ends the trip rather than triggering a turn (this
    matches compute_path_turn_decisions below, which likewise never treats
    the first or last tile of the route as a turn/stop).

    The Manhattan distance (straight grid distance, ignoring intersection
    penalties) is used as A*'s "heuristic" - an estimate of remaining
    distance that never overestimates the true cost, since every real move
    costs at least 1.0. That guarantee is what keeps A* finding the truly
    best route rather than just a decent one.
    """
    if penalty is None:
        penalty = current_intersection_penalty()
    if start not in GRAPH_ADJ or goal not in GRAPH_ADJ:
        return [start], 0.0
    import heapq
    # Each entry in the priority queue: (estimated total cost, negative
    # intersection count for deterministic plain-route ties, cost so far,
    # current tile, path taken so far)
    open_heap = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0, 0.0, start, [start])]
    best_seen = {}
    while open_heap:
        f, neg_intersections, g, n, path = heapq.heappop(open_heap)  # Always expand the most promising option next
        if n == goal:
            return path, g  # Found it - this is the best route
        seen_g, seen_neg_intersections = best_seen.get(n, (float("inf"), 0))
        if g > seen_g or (g == seen_g and neg_intersections >= seen_neg_intersections):
            continue
        best_seen[n] = (g, neg_intersections)
        for nb, d in GRAPH_ADJ.get(n, []):
            if len(path) == 1 and required_heading and d != required_heading:
                continue  # Skip moves that don't match the robot's starting facing direction
            if nb in path:
                continue  # Don't allow the route to loop back on itself
            edge_cost = 1.0
            if nb != goal and is_intersection_kind(MAP_TILES[nb[1]][nb[0]]):
                edge_cost += penalty
            tentative_g = g + edge_cost
            h = abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
            nb_intersections = -neg_intersections + (1 if nb != goal and is_intersection_kind(MAP_TILES[nb[1]][nb[0]]) else 0)
            heapq.heappush(open_heap, (tentative_g + h, -nb_intersections, tentative_g, nb, path + [nb]))
    return [start], float("inf")  # No route found at all - just stay put

def astar_search(start, goal, required_heading=None, penalty=None):
    """
    Weighted A* pathfinding between two tiles, with an explicit start U-turn
    when the robot's physical heading makes the best route unreachable.
    """
    unrestricted, unrestricted_cost = _astar_search_once(start, goal, None, penalty)
    if not required_heading:
        return unrestricted

    heading_path, heading_cost = _astar_search_once(start, goal, required_heading, penalty)
    if heading_path[-1] == goal and heading_cost <= unrestricted_cost + UTURN_PENALTY:
        return heading_path
    if unrestricted[-1] == goal:
        return unrestricted
    return heading_path

def compute_path_turn_decisions(path, start_uturn=False):
    """
    Walk through the computed route tile by tile and, for every
    intersection tile in the middle of the path, work out whether the
    robot needs to go straight, turn left, or turn right there, based on
    which direction it was heading coming in versus which direction it
    needs to head going out.

    Returns:
      turn_tiles: a dict of {tile: "left"/"right"/"straight"/"uturn"} for each
                  intersection on the route
      intersection_order: the same intersections, in the order the robot
                  will reach them while driving the route
    """
    turn_tiles = {}
    intersection_order = []
    if start_uturn and path:
        turn_tiles[path[0]] = "uturn"
        intersection_order.append(path[0])
    if len(path) < 3:
        return turn_tiles, intersection_order
    for k in range(1, len(path) - 1):
        prev, cur, nxt = path[k - 1], path[k], path[k + 1]
        kind = MAP_TILES[cur[1]][cur[0]]
        if not is_intersection_kind(kind):
            continue
        in_d = None   # Direction the robot was travelling to arrive at this tile
        for d, (x, y) in DELTAS.items():
            if (cur[0] - prev[0], cur[1] - prev[1]) == (x, y):
                in_d = d
        out_d = None  # Direction the robot needs to travel to leave this tile
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
            elif OPPOSITE[in_d] == out_d:
                turn_tiles[cur] = "uturn"
    return turn_tiles, intersection_order

# ---- route / map state -------------------------------------------------
# These global variables hold the currently planned route and where the
# robot is along it. They get filled in once the user picks a start tile,
# a goal tile, and clicks "Compute Route" on the dashboard.
ROUTE = []                       # The list of tiles the robot will drive through, in order
ROUTE_GOAL = None                # The chosen destination tile
ROUTE_TURN_TILES = {}            # Which intersections need left/right/straight (see compute_path_turn_decisions)
ROUTE_INTERSECTION_ORDER = []    # The intersections in the order they'll be reached
COMPARE_ROUTES = None            # Optional route-comparison result drawn on the dashboard map
bot_start_tile = None            # Chosen starting tile
bot_heading = 'E'                # Which way the robot is facing at the start (N/E/S/W)
AUTO_PATH_MODE = False           # True = robot drives the planned route automatically; False = manual/keyboard control
SETUP_COMPLETE = False           # True once the user has picked a mode on the dashboard (robot won't move before this)
path_intersections_passed = 0    # How many of the planned intersections have been driven through so far

# ---- dead-reckoning goal timer -------------------------------------------
# Once the robot has made its LAST planned turn, there are no more
# intersections left to count on the way to the goal - so instead of
# detecting anything further, the robot just times itself: it assumes each
# remaining tile takes about SECONDS_PER_TILE to cross, and stops once
# enough time has passed. This is called "dead reckoning" - estimating
# position purely from elapsed time/speed rather than by sensing anything.
SECONDS_PER_TILE = 4.5
FINAL_TILE_SECONDS = 2.25
post_intersections_tracking = False  # True once we're in this "just count down the clock" phase
post_intersection_start_time = 0.0
final_turn_route_index = 0
tiles_after_final_turn = 0
current_tracked_tile_index = 0
goal_total_duration = 0.0

run_metrics = None
run_started_at = None
run_start_wall = None
run_frame_count = 0
run_fps_sum = 0.0
run_logged = False

def compute_route():
    """
    Called when the dashboard's "Compute Route" button is pressed. Runs
    A* from the chosen start tile to the chosen goal tile, works out where
    the turns are, and also pre-calculates how long the final "dead
    reckoning" stretch (after the last turn) should take, so the robot
    knows when it's reached the goal even with no more intersections to
    detect.
    """
    global ROUTE, ROUTE_TURN_TILES, ROUTE_INTERSECTION_ORDER, path_intersections_passed
    global post_intersections_tracking, current_tracked_tile_index
    global final_turn_route_index, tiles_after_final_turn, goal_total_duration
    if bot_start_tile is None or ROUTE_GOAL is None:
        return False, "Set both Start and Goal tiles first."
    path = astar_search(bot_start_tile, ROUTE_GOAL, required_heading=bot_heading)
    if len(path) < 2 or path[-1] != ROUTE_GOAL:
        ROUTE, ROUTE_TURN_TILES, ROUTE_INTERSECTION_ORDER = [], {}, []
        return False, "No drivable path found."
    start_uturn = _path_first_heading(path) != bot_heading
    turn_tiles, order = compute_path_turn_decisions(path, start_uturn=start_uturn)
    ROUTE = path
    ROUTE_TURN_TILES = turn_tiles
    ROUTE_INTERSECTION_ORDER = order
    path_intersections_passed = 0
    post_intersections_tracking = False
    current_tracked_tile_index = 0
    if order:
        # There's at least one turn - work out how many tiles remain after it
        last_inter = order[-1]
        final_turn_route_index = path.index(last_inter)
        tiles_after_final_turn = len(path) - 1 - final_turn_route_index
    else:
        # No turns at all on this route - the whole thing is "dead reckoning"
        final_turn_route_index = 0
        tiles_after_final_turn = len(path) - 1
    if tiles_after_final_turn > 1:
        goal_total_duration = (tiles_after_final_turn - 1) * SECONDS_PER_TILE + FINAL_TILE_SECONDS
    elif tiles_after_final_turn == 1:
        goal_total_duration = FINAL_TILE_SECONDS
    else:
        goal_total_duration = 0.0
    return True, f"Route: {_route_tile_count(path)} tiles ({len(order)} actions)"

def _route_summary(start, goal, heading, penalty):
    path = astar_search(start, goal, required_heading=heading, penalty=penalty)
    start_uturn = bool(heading and _path_first_heading(path) != heading and path[-1] == goal)
    turn_tiles, order = compute_path_turn_decisions(path, start_uturn=start_uturn)
    return {
        "path": [list(t) for t in path],
        "tile_count": _route_tile_count(path),
        "intersection_count": _count_route_intersections_for_summary(path),
        "uturn_count": _route_uturn_count(turn_tiles),
        "estimated_time_s": round(_estimate_route_time(path, turn_tiles), 2),
        "actions": [{"tile": list(tile), "turn": turn_tiles[tile]} for tile in order],
    }

def compare_routes(start, goal, heading):
    return {
        "penalty_on": _route_summary(start, goal, heading, DEFAULT_INTERSECTION_PENALTY),
        "penalty_off": _route_summary(start, goal, heading, 0.0),
        "estimate_note": "Times are estimates until real runs are logged.",
    }

def _credit_goal_timer(seconds):
    """
    While the robot is on the final "dead reckoning" stretch, if it has to
    sit still for a while (waiting at a red line, or waiting out a duck),
    that stopped time shouldn't count as progress toward the goal.
    This function pushes the dead-reckoning start time forward by however
    long the robot was stopped, so the timer effectively "pauses" during
    stops instead of ticking down while the robot isn't actually moving.
    """
    global post_intersection_start_time
    if post_intersections_tracking and seconds > 0:
        post_intersection_start_time += seconds

def render_map_svg():
    """
    Draws the whole track map as an SVG picture (a small vector graphic)
    for the dashboard: each tile as a colored square, the planned route as
    a blue line, turn arrows, the Start (green) and Goal (red) markers, a
    compass, and - while auto-driving - a pulsing yellow dot showing where
    the robot currently is along the route.
    """
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
            # Each tile is clickable - clicking calls back into JavaScript
            # (mapTileClick) so the user can set Start/Goal by clicking the map.
            p.append(f'<rect class="map-tile" x="{x}" y="{y}" width="{cell}" height="{cell}" '
                     f'fill="{fill}" stroke="#111" stroke-width="1" '
                     f'style="cursor:pointer;" onclick="mapTileClick({i},{j})"/>')
            if (i, j) in route_set and kind != "grass":
                p.append(f'<rect x="{x+2}" y="{y+2}" width="{cell-4}" height="{cell-4}" '
                         f'fill="none" stroke="#1f6feb" stroke-width="1.5" opacity="0.6" pointer-events="none"/>')
    if len(ROUTE) > 1:
        pts = " ".join(f"{pad + i*cell+cell//2},{pad + j*cell+cell//2}" for (i, j) in ROUTE)
        p.append(f'<polyline points="{pts}" fill="none" stroke="#1f9bff" stroke-width="3" opacity="0.85" pointer-events="none"/>')
    if COMPARE_ROUTES:
        for key, color, dash in (("penalty_off", "#ff8c42", "5 3"), ("penalty_on", "#22c55e", "0")):
            route = COMPARE_ROUTES.get(key, {}).get("path", [])
            if len(route) > 1:
                pts = " ".join(f"{pad + i*cell+cell//2},{pad + j*cell+cell//2}" for (i, j) in route)
                p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="4" '
                         f'stroke-dasharray="{dash}" opacity="0.75" pointer-events="none"/>')
    if bot_start_tile:
        sx, sy = bot_start_tile
        p.append(f'<circle cx="{pad + sx*cell+cell//2}" cy="{pad + sy*cell+cell//2}" r="10" fill="#22c55e" pointer-events="none"/>')
        p.append(f'<text x="{pad + sx*cell+cell//2}" y="{pad + sy*cell+cell//2+4}" font-size="11" fill="black" text-anchor="middle" font-weight="bold" pointer-events="none">S</text>')
    if ROUTE_GOAL:
        gx, gy = ROUTE_GOAL
        p.append(f'<circle cx="{pad + gx*cell+cell//2}" cy="{pad + gy*cell+cell//2}" r="10" fill="#ef4444" pointer-events="none"/>')
        p.append(f'<text x="{pad + gx*cell+cell//2}" y="{pad + gy*cell+cell//2+4}" font-size="11" fill="white" text-anchor="middle" font-weight="bold" pointer-events="none">G</text>')
    for (ti, tj), d in ROUTE_TURN_TILES.items():
        arrow = {"left": "L", "right": "R", "straight": "^", "uturn": "U"}.get(d, "?")
        p.append(f'<text x="{pad + ti*cell+cell//2}" y="{pad + tj*cell+cell//2+5}" font-size="13" fill="#ffd000" text-anchor="middle" font-family="sans-serif" font-weight="bold" pointer-events="none">{arrow}</text>')
    if COMPARE_ROUTES:
        for key, color in (("penalty_off", "#ff8c42"), ("penalty_on", "#22c55e")):
            for action in COMPARE_ROUTES.get(key, {}).get("actions", []):
                if action.get("turn") == "uturn":
                    ti, tj = action["tile"]
                    p.append(f'<text x="{pad + ti*cell+cell//2}" y="{pad + tj*cell+cell//2+5}" font-size="13" fill="{color}" text-anchor="middle" font-family="sans-serif" font-weight="bold" pointer-events="none">U</text>')
    if AUTO_PATH_MODE and ROUTE:
        # Figure out which tile to show the pulsing "you are here" dot on:
        # still driving toward the next planned turn, or already in the
        # final dead-reckoning stretch, or finished.
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
# STATE MACHINE
# ==============================================================================
# The robot's overall behavior is organized as a "state machine": at any
# moment it is in exactly one named state, and each state has its own
# rules for what to do and when to switch to a different state. This is a
# very common and simple way to structure robot/game logic. The states
# here are:
STATE_LANE_FOLLOWING = "lane_following"        # Normal driving, following the lane
STATE_RED_STOP = "red_line_stopped"            # Stopped at a red stop line, waiting
STATE_DUCK_STOP = "duck_stopped"               # Stopped for a duck, watching to see if it moves
STATE_DUCK_OVERTAKE = "duck_overtake"          # Actively steering around a duck
STATE_INTERSECTION_TURN = "intersection_maneuver"  # Mid-turn at an intersection
STATE_UTURN = "u_turn"                         # Rotating in place until the opposite lane is visible
STATE_GOAL_REACHED = "goal_reached"            # Arrived at the destination, fully stopped
DUCK_OVERTAKE_STATES = (STATE_DUCK_STOP, STATE_DUCK_OVERTAKE)

current_state = STATE_LANE_FOLLOWING
state_start_time = time.time()   # When the current state was entered (used for timeouts)
step_start_time = time.time()    # When the current step within a state began (e.g. one leg of a turn)
last_red_line_time = 0.0
last_duck_avoid_time = 0.0
turn_sequence_active = []        # The list of (v, omega, duration) steps for the turn currently in progress
turn_step_index = 0              # Which step of that sequence we're on
active_turn_direction = 'none'
uturn_pending = False
keyboard_engaged = False         # True while a human is driving manually via keyboard
manual_v = 0.0
manual_omega = 0.0

# lane-following memory (PD controller state) - carried between frames
prev_error = 0.0   # The lane-center error from the previous frame (needed to compute the "derivative" part of PD)
last_omega = 0.0    # The last steering command sent, used to keep turning gently if the lane briefly disappears
lost_frames = 0      # How many frames in a row the lane has been completely undetectable

# duck avoidance bookkeeping
_duck_seen_frames = 0
_duck_ref_x = None
_duck_ref_area = 0.0
_duck_stop_clock = 0.0     # When the whole duck episode began (robot came to a stop)
_duck_clear_frames = 0     # Counts frames without a blocking duck, used to decide the path is clear again
_overtake_note = "idle"     # Human-readable status text shown on the dashboard

# link / fps telemetry
_prev_frame_t = time.time()
_last_frame_time = time.time()
_fps = 0.0
link_stale = False           # True if the camera feed has stopped updating (connection problem)
_link_error = ""
_last_publish_warn = 0.0

# TEL ("telemetry") is the snapshot of the robot's current status that gets
# sent to the dashboard's browser every quarter second so the human can see
# what's happening live.
TEL = {"state": current_state, "v": 0.0, "omega": 0.0,
       "yellow": False, "white": False, "duck": False, "duck_area": 0,
       "fps": 0.0, "link": "ok", "note": "", "auto_path_mode": False,
       "setup_complete": False, "duck_phase": "idle", "path_progress": "no route",
       "intersection_penalty": DEFAULT_INTERSECTION_PENALTY}

# ==============================================================================
# ROS LINK
# ==============================================================================
# ROS (Robot Operating System) is the standard software framework
# Duckiebots run on. "rosbridge" lets this ordinary Python script talk to
# the robot's ROS system over a normal network connection (via
# "roslibpy"), instead of needing to be written in ROS's native style.
# If the robot can't be reached at all (e.g. testing on a laptop with no
# robot around), the script falls back to "simulation_mode": a fake camera
# feed is generated instead, so the same code can still be exercised.
try:
    print(f"Connecting to rosbridge at {ROSBRIDGE_HOST}:{ROSBRIDGE_PORT} ...")
    client_ros = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
    client_ros.run(timeout=3)
    if not client_ros.is_connected:
        raise RuntimeError("rosbridge did not answer")
    print(f"Connected to {VEHICLE}.")
    # "Topics" are ROS's named channels for sending/receiving messages.
    # cmd_pub sends drive commands (speed + turn rate); override_pub tells
    # the robot's joystick-safety system that this script (not a human
    # joystick) is currently allowed to drive.
    cmd_pub = roslibpy.Topic(client_ros, f'/{VEHICLE}/car_cmd_switch_node/cmd',
                              'duckietown_msgs/Twist2DStamped')
    override_pub = roslibpy.Topic(client_ros, f'/{VEHICLE}/joy_mapper_node/joystick_override',
                                   'duckietown_msgs/BoolStamped')
except Exception as e:
    print(f"Hardware link offline ({e}). Starting simulation feed.")
    simulation_mode = True

def _stamp():
    """A minimal ROS message 'header' (timestamp placeholder). Required by
    the message format but not actually used for timing logic here."""
    return {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''}

def publish_drive(v, omega):
    """
    Send a drive command to the real robot: v = forward speed,
    omega = turn rate (positive/negative = left/right). Every call also
    re-asserts the "override" flag so the robot keeps listening to this
    script instead of a joystick. Does nothing in simulation mode (there's
    no real robot to command). Errors are caught and remembered (not
    crashed on) so a temporary network hiccup doesn't kill the whole
    program - the watchdog thread below is what actually keeps the robot
    safe if publishing keeps failing.
    """
    global _link_error, _last_publish_warn
    if simulation_mode:
        return
    try:
        override_pub.publish(roslibpy.Message({'header': _stamp(), 'data': True}))
        cmd_pub.publish(roslibpy.Message({'header': _stamp(), 'v': float(v), 'omega': float(omega)}))
        _link_error = ""
    except Exception as e:
        _link_error = str(e)
        if time.time() - _last_publish_warn > 2.0:   # Don't spam the console - only log this every 2 seconds
            print(f"publish_drive failed: {e}")
            _last_publish_warn = time.time()

def valid_lane_exit_geometry(yc, wc, cx):
    """
    Return True when the camera sees a plausible lane centered in front of
    the robot: yellow on the left, white on the right, enough lane width, and
    the lane midpoint close to the image center.
    """
    if yc is None or wc is None:
        return False
    lane_width_ok = wc > yc + MIN_LANE_WIDTH_PX
    line_sides_ok = yc < cx and wc > cx
    lane_center = (yc + wc) / 2.0
    center_ok = abs(lane_center - cx) < 15
    return lane_width_ok and line_sides_ok and center_ok

def release_override():
    """Called when the script is shutting down: stop the wheels and hand
    control back (turn off the 'override' flag) so the robot doesn't stay
    locked out of manual/joystick control after this program exits."""
    if simulation_mode:
        return
    try:
        cmd_pub.publish(roslibpy.Message({'header': _stamp(), 'v': 0.0, 'omega': 0.0}))
        override_pub.publish(roslibpy.Message({'header': _stamp(), 'data': False}))
    except Exception:
        pass

# Make sure the wheels are stopped and control is released no matter how
# the script ends - normal exit, Ctrl+C, or a "terminate" signal from the
# operating system. This is an important safety measure so the robot never
# keeps driving after the controlling program has died.
atexit.register(release_override)
signal.signal(signal.SIGTERM, lambda s, f: release_override() or os._exit(0))
signal.signal(signal.SIGINT, lambda s, f: release_override() or os._exit(0))

def watchdog_loop():
    """
    Safety watchdog, running in its own background thread forever.

    Its only job: if no new camera frame has arrived recently (the camera
    connection dropped, or frame processing crashed/hung), immediately
    command the wheels to stop. This means that even if something else in
    the program goes wrong, the robot fails safe (stops) rather than
    continuing to drive blind on stale commands. It deliberately never
    lets an exception kill this thread - it just logs and keeps checking.
    """
    global link_stale
    while True:
        try:
            time.sleep(0.1)
            if simulation_mode:
                continue
            stale = (time.time() - _last_frame_time) > FRAME_STALE_S
            if stale and not link_stale:
                print("No camera frames (or frame processing failed) - stopping wheels.")
            if stale:
                publish_drive(0.0, 0.0)
            link_stale = stale
        except Exception as e:
            print(f"watchdog error (continuing): {e}")
            time.sleep(0.1)

threading.Thread(target=watchdog_loop, daemon=True).start()

def start_intersection_turn(direction):
    """Begin executing a turn maneuver ('straight', 'left', or 'right') at
    an intersection: switches the state machine into
    STATE_INTERSECTION_TURN and loads up the right pre-planned sequence of
    steps for that direction (a right turn instead uses the white-line-hug
    logic directly in the main loop, so it has an empty step list here)."""
    global active_turn_direction, turn_sequence_active, turn_step_index
    global step_start_time, state_start_time, current_state
    active_turn_direction = direction
    turn_sequence_active = {'straight': INTERSECTION_STRAIGHT_STEPS,
                             'left': INTERSECTION_LEFT_STEPS,
                             'right': []}[direction]
    turn_step_index = 0
    step_start_time = state_start_time = time.time()
    current_state = STATE_INTERSECTION_TURN
    print(f"Intersection turn commenced: {direction}")

def start_u_turn():
    """Begin a planned in-place U-turn at the start tile."""
    global current_state, state_start_time, step_start_time, uturn_pending
    current_state = STATE_UTURN
    state_start_time = step_start_time = time.time()
    uturn_pending = False
    if run_metrics is not None:
        run_metrics["uturns"] += 1
    print("Planned U-turn commenced at start tile.")

def _advance_route_after_turn():
    """
    Called every time a turn maneuver finishes. If that was the LAST
    planned turn on the route, there's nothing left to detect (no more
    intersections coming up) - so this switches the robot from "count
    intersections as they're passed" mode into "just time the remaining
    drive" (dead reckoning) mode, and starts that countdown clock.
    """
    global post_intersections_tracking, post_intersection_start_time
    if AUTO_PATH_MODE and path_intersections_passed >= len(ROUTE_INTERSECTION_ORDER):
        post_intersections_tracking = True
        post_intersection_start_time = time.time()
        print(f"Final turn complete. Driving {goal_total_duration:.2f}s to the goal tile...")

def start_run_logging():
    """Start collecting one run's timing, route, event, and camera-rate data."""
    global run_metrics, run_started_at, run_start_wall, run_frame_count, run_fps_sum, run_logged
    now = time.time()
    run_started_at = now
    run_start_wall = datetime.datetime.now()
    run_frame_count = 0
    run_fps_sum = 0.0
    run_logged = False
    run_metrics = {
        "start_tile": bot_start_tile,
        "goal_tile": ROUTE_GOAL,
        "heading": bot_heading,
        "penalty_setting": current_intersection_penalty(),
        "tiles_driven": _route_tile_count(ROUTE),
        "intersections_crossed": _count_path_intersections(ROUTE),
        "red_stops": 0,
        "duck_stops": 0,
        "duck_overtakes": 0,
        "uturns": 0,
        "lane_lost_frames": 0,
        "goal_reached": False,
    }

def log_run_result(goal_reached):
    """Write the current run metrics to a timestamped CSV in runs/."""
    global run_logged
    if run_metrics is None or run_started_at is None or run_logged:
        return
    os.makedirs("runs", exist_ok=True)
    finished_at = datetime.datetime.now()
    stamp = finished_at.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs", f"run_{stamp}.csv")
    row = dict(run_metrics)
    row["wall_clock_time_s"] = round(time.time() - run_started_at, 3)
    row["goal_reached"] = bool(goal_reached)
    row["mean_camera_fps"] = round(run_fps_sum / run_frame_count, 2) if run_frame_count else 0.0
    row["started_at"] = run_start_wall.isoformat(timespec="seconds") if run_start_wall else ""
    row["finished_at"] = finished_at.isoformat(timespec="seconds")
    fieldnames = ["started_at", "finished_at", "start_tile", "goal_tile", "heading",
                  "penalty_setting", "wall_clock_time_s", "tiles_driven",
                  "intersections_crossed", "red_stops", "duck_stops",
                  "duck_overtakes", "uturns", "lane_lost_frames",
                  "mean_camera_fps", "goal_reached"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    run_logged = True
    print(f"Run log written: {path}")

# ==============================================================================
# MAIN FRAME PROCESSING
# ==============================================================================
def process_image_frame(frame):
    """
    This is the heart of the whole program - it runs once for every new
    camera frame (many times per second) and does everything in one pass:

      1. Detect colors of interest in the image (yellow line, white line,
         red stop line, yellow duck blobs).
      2. Work out where the center of the lane is.
      3. Look at the current state (see the STATE_* constants above) and
         decide what to do: keep lane-following, stop for a duck or red
         line, execute a turn, etc. - and actually send that drive command
         to the robot.
      4. Possibly switch to a different state for the next frame.
      5. Draw debug info onto the image and update the telemetry dict so
         the dashboard can show what's going on.

    Because this function runs on every frame, everything it changes uses
    Python's `global` keyword to update the shared state variables defined
    above, so those values persist and continue evolving frame after frame.
    """
    global latest_jpeg, latest_mask_jpeg, current_state, state_start_time
    global last_red_line_time, last_duck_avoid_time
    global turn_sequence_active, turn_step_index, active_turn_direction, step_start_time
    global keyboard_engaged, manual_v, manual_omega
    global prev_error, last_omega, lost_frames, _duck_seen_frames
    global _duck_ref_x, _duck_ref_area, _duck_stop_clock
    global _duck_clear_frames, _overtake_note
    global _prev_frame_t, _last_frame_time, _fps
    global path_intersections_passed, post_intersections_tracking, post_intersection_start_time
    global current_tracked_tile_index
    global run_frame_count, run_fps_sum

    if frame is None or frame.size == 0:
        return  # Got a bad/empty frame - skip it rather than crashing

    # ---- measure how fast frames are arriving (for the FPS readout) ----
    now = time.time()
    dt = min(0.5, max(0.0, now - _prev_frame_t))
    _prev_frame_t = now
    if dt > 0:
        _fps = 0.85 * _fps + 0.15 * (1.0 / dt)  # Smoothed ("exponential moving average") FPS estimate
    if run_metrics is not None:
        run_frame_count += 1
        run_fps_sum += _fps

    h, w = frame.shape[:2]
    cx = w // 2.0  # The horizontal center of the camera image - i.e. "straight ahead"

    # ---- color detection: convert to HSV, then build a black/white mask
    #      for each color we care about (white pixels = "this color is here") ----
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, *HSV_YELLOW)
    white_mask = cv2.inRange(hsv, *HSV_WHITE)
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, *HSV_RED_A), cv2.inRange(hsv, *HSV_RED_B))

    # ==========================================================================
    # DUCK RECOGNITION
    # Only look for ducks in a horizontal strip roughly in the middle-lower
    # part of the image (45%-98% down) - that's where the road surface is;
    # ignoring the sky/background above it avoids false positives.
    # ==========================================================================
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
            if a > duck_area:  # Track the single biggest (closest) confirmed duck
                duck_area = a
                duck_x = x + cw // 2
                duck_bottom = y + ch
            # Draw a red box + label on the frame for the dashboard operator to see
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
            cv2.putText(frame, "DUCK", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif cv2.contourArea(contour) > 800:
            # A sizeable yellow blob that failed the duck shape-test - draw
            # it in a different color so an operator watching the video
            # can tell the difference (probably part of the lane line)
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 200, 255), 1)
            cv2.putText(frame, "yellow obj", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

    # Ducks are yellow, just like the lane's centre line - so before using
    # the yellow mask for LANE detection, "erase" the duck's own bounding
    # boxes from a copy of it. Otherwise the robot could mistake a duck for
    # part of the yellow lane line and steer toward it.
    yellow_lane = yellow_mask.copy()
    for (bx, by, bw, bh) in duck_boxes:
        yellow_lane[by:by + bh, bx:bx + bw] = 0

    if duck_found:
        _duck_seen_frames += 1
    else:
        _duck_seen_frames = 0

    # "blocking" = a duck that's both detected AND big/close enough to
    # actually be worth reacting to. This flag is used later by the
    # wait/pass duck-avoidance states to judge whether the duck has really
    # moved out of the way. IMPORTANT: this uses the same underlying
    # conditions (duck_found + an area threshold) as the original trigger
    # below, just at a slightly lower area bar (0.6x) so a duck that's
    # started to move away isn't instantly declared "gone".
    duck_blocking = duck_found and duck_area >= DUCK_TRIGGER_AREA * 0.6

    # ==========================================================================
    # STOP-LINE RECOGNITION
    # Only look for red stop lines in the bottom 40% of the image - a stop
    # line only matters once it's close to the robot.
    # ==========================================================================
    red_roi = np.zeros_like(red_mask)
    red_roi[int(h * 0.60):h, :] = red_mask[int(h * 0.60):h, :]
    red_line_found = False
    stop_contours, _ = cv2.findContours(red_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stop_contours_confirmed = []
    for contour in stop_contours:
        area = cv2.contourArea(contour)
        if area <= 60:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        ar = float(cw) / ch if ch > 0 else 0
        bottom_frac = (y + ch) / float(h)
        # A real stop line is a wide, flat rectangle sitting near the
        # bottom of the frame (close to the robot) - these checks filter
        # out other red things (like a red duck, or red image noise).
        shape_ok = (ar > STOPLINE_MIN_AR and cw > int(w * STOPLINE_MIN_WIDTH_FRAC)
                    and area > STOPLINE_MIN_AREA)
        close_ok = bottom_frac >= STOPLINE_MIN_BOTTOM_FRAC
        if shape_ok and close_ok:
            red_line_found = True
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 0, 255), 3)
            cv2.putText(frame, f"STOP LINE ar={ar:.1f}", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            stop_contours_confirmed.append(contour)
        elif shape_ok:
            # Shaped like a stop line but too far away yet - draw it in a
            # dimmer color as a preview, don't trigger the stop yet
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 140, 230), 2)

    # ==========================================================================
    # LANE FOLLOWING - work out where the middle of the lane is
    # ==========================================================================
    # Crop to just the bottom slice of the image (the road right in front
    # of the robot) before searching for the yellow/white lines.
    roi_y0 = int(h * (1.0 - ROI_FRACTION))
    yellow_roi = yellow_lane[roi_y0:, :]
    white_roi = white_mask[roi_y0:, :]
    yc = centroid_x(yellow_roi, x_hi=w * YELLOW_SEARCH_MAX)   # x-position of the yellow centre line, or None
    wc = centroid_x(white_roi, x_lo=w * WHITE_SEARCH_MIN)     # x-position of the white edge line, or None
    if yc is not None and wc is not None and wc <= yc + MIN_LANE_WIDTH_PX:
        # If the "white line" found is suspiciously close to (or left of)
        # the yellow line, it's not really a separate lane edge - discard it
        wc = None
    if yc is not None:
        cv2.circle(frame, (int(yc), int(h * 0.9)), 6, (0, 255, 255), -1)  # Yellow dot marker for the dashboard video
    if wc is not None:
        cv2.circle(frame, (int(wc), int(h * 0.9)), 6, (255, 255, 255), -1)  # White dot marker
    # Decide the lane center to steer toward:
    if yc is not None and wc is not None:
        lane_center = (yc + wc) / 2.0            # Both lines visible - aim for the midpoint between them
    elif yc is not None:
        lane_center = yc + w * HALF_LANE_FRAC    # Only yellow visible - estimate center by offsetting right
    elif wc is not None:
        lane_center = wc - w * HALF_LANE_FRAC    # Only white visible - estimate center by offsetting left
    else:
        lane_center = None                        # Neither line visible - lane is "lost" this frame

    note = ""              # Human-readable status text for this frame, shown on the dashboard
    arrow_v, arrow_omega = 0.0, 0.0   # What speed/turn to show as the HUD direction arrow
    time_in_step = now - step_start_time

    # ==========================================================================
    # STATE MACHINE - decide what to actually do this frame
    # ==========================================================================
    if current_state == STATE_GOAL_REACHED:
        # Trip is over - stay stopped and show a banner
        publish_drive(0.0, 0.0)
        note = "GOAL REACHED - stopped"
        cv2.putText(frame, "DESTINATION REACHED - STOPPED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    elif not SETUP_COMPLETE:
        # Waiting for a human to choose Manual or A* mode on the dashboard
        # before the robot is allowed to move at all
        publish_drive(0.0, 0.0)
        note = "Waiting for setup on the dashboard"
        cv2.putText(frame, "SETUP - Select a mode on the dashboard", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    elif keyboard_engaged:
        # A human has pressed 'E' to take manual control - just drive
        # whatever speed/turn the last keypress set, ignore all autonomy
        publish_drive(manual_v, manual_omega)
        arrow_v, arrow_omega = manual_v, manual_omega
        note = "Manual override"
        cv2.putText(frame, "MANUAL OVERRIDE - AUTONOMY DISABLED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    elif current_state == STATE_LANE_FOLLOWING:
        # --- if we're in the final "dead reckoning" leg of an A* route,
        #     check whether enough time has passed to call the goal reached ---
        if AUTO_PATH_MODE and post_intersections_tracking:
            elapsed_post = now - post_intersection_start_time
            blocks_passed = int(elapsed_post // SECONDS_PER_TILE)
            current_tracked_tile_index = min(len(ROUTE) - 1, final_turn_route_index + blocks_passed) if ROUTE else 0
            if elapsed_post >= goal_total_duration:
                current_state = STATE_GOAL_REACHED
                publish_drive(0.0, 0.0)
                if run_metrics is not None:
                    run_metrics["goal_reached"] = True
                log_run_result(True)
                print(f"Goal reached after {elapsed_post:.1f}s!")

        if current_state == STATE_LANE_FOLLOWING:
            # --- the actual PD steering control ---
            if lane_center is not None:
                error = (lane_center - cx) / cx   # How far off-center we are, as a fraction (-1 to +1)
                d_error = error - prev_error       # How much that error changed since last frame
                prev_error = error
                # PD formula: steer harder the further off-center we are
                # (KP * error), and also react to how fast that's changing
                # (KD * d_error) to avoid oscillating back and forth.
                omega = float(np.clip(-(KP * error + KD * d_error), -OMEGA_MAX, OMEGA_MAX))
                # Slow down when steering hard (turning sharply), just like
                # a real car easing off the gas mid-turn
                v = max(V_MIN, V_BAR * (1.0 - SLOWDOWN_STRENGTH * error * error))
                last_omega = omega
                lost_frames = 0
                publish_drive(v, omega)
                arrow_v, arrow_omega = v, omega
                if AUTO_PATH_MODE and post_intersections_tracking:
                    rem_time = max(0.0, goal_total_duration - (now - post_intersection_start_time))
                    note = f"Driving to goal ({rem_time:.1f}s left)"
                else:
                    note = f"Tracking (err {error:+.2f})"
            else:
                # Lane briefly not visible - don't panic-stop immediately.
                # Keep creeping forward slowly while gradually straightening
                # out the last known turn amount (LANE_MEMORY_DECAY fades it
                # toward zero each frame). Only fully stop if this goes on
                # too long (LANE_LOST_MAX_FRAMES).
                lost_frames += 1
                if run_metrics is not None:
                    run_metrics["lane_lost_frames"] += 1
                if lost_frames <= LANE_LOST_MAX_FRAMES:
                    last_omega *= LANE_MEMORY_DECAY
                    publish_drive(V_MIN, last_omega)
                    arrow_v, arrow_omega = V_MIN, last_omega
                    note = f"Lane lost - holding ({lost_frames})"
                else:
                    publish_drive(0.0, 0.0)
                    note = "Lane lost - stopped"

            # --- duck avoidance trigger: stop first, decide what to do later ---
            if (DUCK_AVOIDANCE_ON and duck_found and duck_area >= DUCK_TRIGGER_AREA
                    and _duck_seen_frames >= DUCK_TRIGGER_FRAMES
                    and (now - last_duck_avoid_time) > DUCK_COOLDOWN_S):
                current_state = STATE_DUCK_STOP
                state_start_time = now
                _duck_stop_clock = now
                _duck_ref_x = duck_x
                _duck_ref_area = float(duck_area)
                _overtake_note = "waiting to see if it moves"
                if run_metrics is not None:
                    run_metrics["duck_stops"] += 1
                publish_drive(0.0, 0.0)
                print("Duck detected in lane -> stopped, watching whether it moves...")

            # --- stop-line trigger (only checked if we didn't just trigger
            #     a duck-stop above, since 'elif' only runs one or the other) ---
            elif red_line_found and (now - last_red_line_time) > STOPLINE_COOLDOWN_S:
                current_state = STATE_RED_STOP
                state_start_time = now
                if run_metrics is not None:
                    run_metrics["red_stops"] += 1
                publish_drive(0.0, 0.0)
                print("Stop line -> stopped, holding before proceeding...")

    elif current_state == STATE_RED_STOP:
        # Sit completely still at the stop line for RED_STOP_HOLD_S seconds
        publish_drive(0.0, 0.0)
        elapsed_in_stop = now - state_start_time
        rem_stop = max(0.0, RED_STOP_HOLD_S - elapsed_in_stop)
        note = f"Stopped at red line ({rem_stop:.1f}s left)" if AUTO_PATH_MODE else \
               "Stopped at red line - press W/A/D to choose a turn"
        cv2.putText(frame, f"STOP LINE: {rem_stop:.1f}s" if AUTO_PATH_MODE else "STOP LINE - awaiting W/A/D",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if AUTO_PATH_MODE and elapsed_in_stop >= RED_STOP_HOLD_S:
            # Waited long enough - time to go. Look up the pre-planned
            # direction for this intersection (or default to straight if
            # we've already used up the planned route, e.g. driving past
            # the goal by mistake) and start the turn.
            _credit_goal_timer(RED_STOP_HOLD_S)
            if path_intersections_passed < len(ROUTE_INTERSECTION_ORDER):
                tile = ROUTE_INTERSECTION_ORDER[path_intersections_passed]
                direction = ROUTE_TURN_TILES.get(tile, 'straight')
                path_intersections_passed += 1
            else:
                direction = 'straight'
            last_red_line_time = now
            if direction == 'uturn':
                start_u_turn()
            else:
                start_intersection_turn(direction)

    elif current_state == STATE_UTURN:
        valid_exit = valid_lane_exit_geometry(yc, wc, cx)
        elapsed = now - state_start_time
        if valid_exit:
            publish_drive(0.0, 0.0)
            last_red_line_time = now
            prev_error = 0.0
            lost_frames = 0
            current_state = STATE_LANE_FOLLOWING
            note = "U-turn complete - opposite lane acquired"
            _advance_route_after_turn()
            print("U-turn complete on lane geometry.")
        elif elapsed >= UTURN_TIMEOUT_S:
            publish_drive(0.0, 0.0)
            last_red_line_time = now
            prev_error = 0.0
            lost_frames = 0
            current_state = STATE_LANE_FOLLOWING
            note = "U-turn timeout - lane follow"
            _advance_route_after_turn()
            print("U-turn timeout - stopped wheels and returned to lane following.")
        else:
            publish_drive(0.0, UTURN_OMEGA)
            arrow_v, arrow_omega = 0.0, UTURN_OMEGA
            note = "U-turn - searching for opposite lane"

    elif current_state == STATE_INTERSECTION_TURN:
        # Executing a turn. "valid_exit" checks whether the camera can now
        # see a plausible NEW lane ahead (yellow line clearly to the left
        # of center, or both lines visible with a real gap between them) -
        # this is the vision-based signal that the turn is basically done,
        # on top of the minimum-time safety checks below.
        yc_t, wc_t = yc, wc  # Same centroids computed earlier in this frame, now re-used mid-turn
        valid_exit = valid_lane_exit_geometry(yc_t, wc_t, cx)

        if active_turn_direction == 'right':
            # Right turns steer by keeping the white outer line at a fixed
            # target position in the image ("hugging" it), rather than
            # following a pre-set timed sequence like left/straight do.
            target_px = int(w * WHITE_HUG_TARGET_FRAC)
            if wc is not None:
                err = wc - target_px
                hug = -(float(err) / cx) * WHITE_HUG_GAIN
                hug = max(-WHITE_HUG_CLAMP, min(WHITE_HUG_CLAMP, hug))
                publish_drive(0.07, hug)
                arrow_v, arrow_omega = 0.07, hug
                note = "Right turn - hugging white line"
            else:
                # Can't see the white line yet - creep forward while
                # turning right to go looking for it
                publish_drive(RIGHT_SEARCH_V, RIGHT_SEARCH_OMEGA)
                arrow_v, arrow_omega = RIGHT_SEARCH_V, RIGHT_SEARCH_OMEGA
                note = "Right turn - searching for white line"
            if valid_exit and time_in_step > 1.5:
                # Turn looks complete and enough time has passed - hand
                # back to normal lane following
                last_red_line_time = now
                prev_error = 0.0
                current_state = STATE_LANE_FOLLOWING
                note = "Exit lane acquired"
                _advance_route_after_turn()
            elif time_in_step > RIGHT_HUG_MAX_S:
                # Safety timeout - give up trying to detect the exit and
                # just resume lane following anyway rather than getting stuck
                last_red_line_time = now
                prev_error = 0.0
                current_state = STATE_LANE_FOLLOWING
                note = "Right turn timeout - lane follow"
                _advance_route_after_turn()
        else:
            # Left turn or straight-through: primarily driven by the
            # pre-planned step sequence (turn_sequence_active), but can
            # also finish early if the camera confirms a valid exit lane.
            if valid_exit and time_in_step > 1.0:
                last_red_line_time = now
                prev_error = 0.0
                current_state = STATE_LANE_FOLLOWING
                note = "Exit lane acquired"
                _advance_route_after_turn()
            elif turn_step_index < len(turn_sequence_active):
                v, omega, duration = turn_sequence_active[turn_step_index]
                if time_in_step < duration:
                    publish_drive(v, omega)
                    arrow_v, arrow_omega = v, omega
                    note = f"{active_turn_direction} turn step {turn_step_index}"
                else:
                    turn_step_index += 1
                    step_start_time = now
            else:
                # Ran out of pre-planned steps without a confirmed exit -
                # resume lane following anyway (safety fallback)
                last_red_line_time = now
                prev_error = 0.0
                current_state = STATE_LANE_FOLLOWING
                _advance_route_after_turn()

    # --------------------------------------------------------------------
    # DUCK STEP 0: stopped, watching to see if it's a statue or it's alive
    # --------------------------------------------------------------------
    elif current_state == STATE_DUCK_STOP:
        publish_drive(0.0, 0.0)
        arrow_v, arrow_omega = 0.0, 0.0
        still_for = now - state_start_time     # How long since the "it's holding still" timer last restarted
        waited_total = now - _duck_stop_clock   # How long since the whole duck episode began
        if not duck_blocking:
            # Duck's no longer detected as blocking - count consecutive
            # "clear" frames; once there are enough, treat the lane as free
            _duck_clear_frames += 1
        else:
            _duck_clear_frames = 0
            moved = False
            if duck_x is not None and _duck_ref_x is not None and abs(duck_x - _duck_ref_x) > DUCK_MOVED_PX:
                moved = True
            if _duck_ref_area > 0 and duck_area > 0 and \
                    abs(duck_area - _duck_ref_area) / _duck_ref_area > DUCK_MOVED_AREA_FRAC:
                moved = True
            if moved:
                # The duck shifted noticeably - it might be walking around
                # on its own, so restart the "has it been still" timer and
                # remember its new position/size as the new reference point
                _duck_ref_x = duck_x
                _duck_ref_area = float(duck_area)
                state_start_time = now
                still_for = 0.0
                _overtake_note = "duck is moving - giving it room"
        if _duck_clear_frames >= DUCK_CLEAR_FRAMES:
            # The duck cleared out of the lane on its own - no need to go
            # around it, just resume normal driving
            last_duck_avoid_time = now
            _credit_goal_timer(waited_total)
            prev_error = 0.0
            current_state = STATE_LANE_FOLLOWING
            note = "Duck cleared on its own - resuming"
            print("Duck cleared on its own -> resuming lane following")
        elif still_for >= DUCK_WAIT_DURATION or waited_total >= DUCK_MAX_WAIT_S:
            # Either it's been still long enough to be confident it's a
            # stationary obstacle, or we've hit the hard maximum wait time
            # regardless - either way, commit to steering around it now
            current_state = STATE_DUCK_OVERTAKE
            state_start_time = now
            _overtake_note = f"centring on yellow line ({DUCK_FOLLOW_YELLOW_S:.0f}s)"
            note = "Duck is not moving - centring on the yellow line to go around it"
            if run_metrics is not None:
                run_metrics["duck_overtakes"] += 1
            print(f"Duck is not moving -> centring on the yellow line for "
                  f"{DUCK_FOLLOW_YELLOW_S:.1f}s.")
        else:
            rem = max(0.0, DUCK_WAIT_DURATION - still_for)
            note = f"Duck blocking - waiting {rem:.1f}s to see if it moves"
            cv2.putText(frame, f"DUCK - WAITING {rem:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # --------------------------------------------------------------------
    # DUCK GO-AROUND: for a fixed few seconds, aim to put the yellow line
    # itself in the middle of the camera image (instead of the usual
    # "half a lane to the right of yellow" target), which steers the robot
    # left and around the duck - then switch back to normal lane
    # following. It's still the exact same PD steering loop the whole
    # time, just aimed at a different target, so it reacts every single
    # frame instead of blindly committing to a fixed turn - much safer
    # than an open-loop timed swerve.
    # --------------------------------------------------------------------
    elif current_state == STATE_DUCK_OVERTAKE:
        elapsed = now - state_start_time
        yc_sw = centroid_x(yellow_roi, x_hi=w * YELLOW_SEARCH_SWERVE)
        if yc_sw is not None:
            error = (yc_sw - cx) / cx   # Target IS the yellow line - steer to put it dead-center
            d_error = error - prev_error
            prev_error = error
            omega = float(np.clip(-(KP * error + KD * d_error), -AVOID_OMEGA_MAX, AVOID_OMEGA_MAX))
            v = OVERTAKE_SPEED
        else:
            # No yellow line visible to steer against right now - coast
            # straight rather than holding onto whatever turn amount we
            # had a moment ago. Holding a hard turn blind (with nothing to
            # measure against) is exactly what would spin the robot around.
            omega = 0.0
            v = V_MIN
        publish_drive(v, omega)
        arrow_v, arrow_omega = v, omega
        rem = max(0.0, DUCK_FOLLOW_YELLOW_S - elapsed)
        note = (f"Overtake: centring on yellow line ({rem:.1f}s left)" if yc_sw is not None
                else f"Overtake: yellow line lost - coasting ({rem:.1f}s left)")
        _overtake_note = f"centring on yellow line ({rem:.1f}s left)"
        if elapsed >= DUCK_FOLLOW_YELLOW_S:
            # Overtake window is over - return to normal lane following.
            # Also shift the cooldown clock so the NEXT duck-avoidance
            # trigger is allowed sooner than the usual full cooldown,
            # since the duck we just passed is now behind the robot and
            # shouldn't block a genuinely new duck up ahead for as long.
            last_duck_avoid_time = now - DUCK_COOLDOWN_S + DUCK_RETRIGGER_COOL
            _credit_goal_timer(now - _duck_stop_clock)
            prev_error = 0.0
            current_state = STATE_LANE_FOLLOWING
            _overtake_note = "idle"
            note = "Overtake complete - back to normal lane following"
            print("Overtake complete -> resuming normal lane following.")

    if current_state in DUCK_OVERTAKE_STATES:
        cv2.putText(frame, f"OVERTAKING DUCK - {_overtake_note}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # ---- HUD arrow: draws a green/orange arrow on the video showing the
    #      current speed/turn direction, purely for the human watching the
    #      dashboard - has no effect on the robot's actual driving ----
    cxi = int(cx)
    if abs(arrow_omega) < 0.15:
        cv2.arrowedLine(frame, (cxi, int(h * 0.85)), (cxi, int(h * 0.68)), (0, 255, 0), 4, cv2.LINE_AA, 0, 0.25)
    else:
        tx = cxi - int(arrow_omega * 60)
        color = (0, 165, 255) if abs(arrow_omega) > 3.0 else (0, 255, 0)
        cv2.arrowedLine(frame, (cxi, int(h * 0.85)), (tx, int(h * 0.68)), color, 4, cv2.LINE_AA, 0, 0.25)

    # ---- update the shared telemetry dict the dashboard polls ----
    TEL.update({"state": current_state, "v": round(arrow_v, 3), "omega": round(arrow_omega, 2),
                "yellow": yc is not None, "white": wc is not None, "duck": duck_found,
                "duck_area": int(duck_area), "fps": round(_fps, 1),
                "link": ("simulation" if simulation_mode else
                         ("stale" if link_stale else (_link_error or "ok"))),
                "note": note, "auto_path_mode": AUTO_PATH_MODE,
                "setup_complete": SETUP_COMPLETE,
                "duck_phase": _overtake_note,
                "intersection_penalty": current_intersection_penalty(),
                "path_progress": f"{path_intersections_passed}/{len(ROUTE_INTERSECTION_ORDER)}"
                                  if ROUTE_INTERSECTION_ORDER else "no route"})

    # ---- encode the annotated frame as JPEG for the dashboard's live video ----
    ok, jpeg = cv2.imencode('.jpg', frame)
    if ok:
        with lock:
            latest_jpeg = jpeg.tobytes()

    # ---- build the "recognition mask" debug view: a plain black image with
    #      each detected feature painted in a solid color, so an operator can
    #      see exactly what the vision pipeline is picking up, separate from
    #      the normal camera view ----
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

    _last_frame_time = now   # Tells the watchdog thread "a frame was just processed successfully"

# ==============================================================================
# SIMULATION FEED / ROS CAMERA SUBSCRIPTION
# ==============================================================================
def simulation_hardware_loop():
    """
    Used only when there's no real robot connected. Generates a very
    simple fake camera image (a black background with a yellow line and a
    white line drawn on it, roughly ~30 times per second) and feeds it
    through the exact same process_image_frame() function the real robot
    uses. This lets the whole vision/state-machine/dashboard code be
    tested and demoed without any hardware.
    """
    print("Virtual camera loop active.")
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "SIMULATION FEED - ROBOT NOT CONNECTED", (15, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        cv2.line(frame, (180, 480), (280, 260), (0, 255, 255), 3)
        cv2.line(frame, (490, 480), (390, 260), (255, 255, 255), 3)
        if current_state == STATE_RED_STOP:
            cv2.rectangle(frame, (200, 400), (470, 430), (0, 0, 255), -1)
        try:
            process_image_frame(frame)
        except Exception as e:
            # Never let a processing bug leave the robot driving blind -
            # stop the wheels first, then just log the error and continue
            publish_drive(0.0, 0.0)
            print(f"frame processing error (wheels stopped): {e}")
        time.sleep(0.033)  # ~30 frames per second

if simulation_mode:
    threading.Thread(target=simulation_hardware_loop, daemon=True).start()
else:
    # Subscribe to the robot's real camera feed over ROS. Every time a new
    # compressed JPEG frame arrives, decode it and run it through the same
    # process_image_frame() pipeline.
    camera_sub = roslibpy.Topic(client_ros, f'/{VEHICLE}/camera_node/image/compressed',
                                 'sensor_msgs/CompressedImage')

    def _on_frame(msg):
        try:
            buf = np.frombuffer(base64.b64decode(msg['data']), np.uint8)
            process_image_frame(cv2.imdecode(buf, cv2.IMREAD_COLOR))
        except Exception as e:
            publish_drive(0.0, 0.0)  # Same safety principle as above: stop first, then log
            print(f"frame error (wheels stopped): {e}")

    camera_sub.subscribe(_on_frame)

# ==============================================================================
# DASHBOARD
# ==============================================================================
# PAGE is the entire HTML/CSS/JavaScript for the web dashboard, served as
# one string by Flask. It shows:
#  - A first-time setup popup to choose Manual or A* Auto-Path mode
#  - The live camera feed and the debug "recognition mask" feed
#  - A telemetry panel (state, speed, turn rate, which lines are seen, etc.)
#  - The clickable map for picking Start/Goal tiles and computing a route
#  - Keyboard control instructions
# The small <script> block at the bottom polls the Flask server every
# 250ms for telemetry and every 600ms for the map, and turns keypresses
# into calls to the /control endpoint below.
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
  .toggle-row { display:flex; gap:8px; align-items:center; justify-content:space-between; margin-top:8px; }
  .compare-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }
  .compare-cell { background:#111; border:1px solid #333; border-radius:6px; padding:6px; }
  kbd { display:inline-block; background:#2c2c2c; border:1px solid #000; border-radius:4px; padding:1px 5px; font-weight:700; }
</style>
</head>
<body>
  <!-- Setup popup shown until the user picks Manual or A* mode -->
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
      <div style="color:#9a9a9a; font-size:.78em; margin:6px 0;">Recognition Mask (yellow / white / stop line / duck)</div>
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
        <strong>A* Route</strong>
        <table>
          <tr><td class="k">Auto-Path</td><td class="v" id="t_auto">-</td></tr>
          <tr><td class="k">Penalty</td><td class="v" id="t_penalty">-</td></tr>
          <tr><td class="k">Turns done</td><td class="v" id="t_progress">-</td></tr>
        </table>
        <div class="toggle-row">
          <span>Avoid intersections</span>
          <button id="penalty_btn" onclick="togglePenalty()">ON</button>
        </div>
        <button onclick="compareCurrentRoute()" style="width:100%; margin-top:8px;">Compare Routes</button>
        <div id="compare_box" class="compare-grid" style="display:none;"></div>
        <div style="margin-top:6px; color:#8a8a8a;">Each stop line auto-picks the next turn until the plan runs out, then the bot times itself to the goal tile.</div>
      </div>
      <div class="card">
        <strong>Duck Avoidance</strong>
        <table>
          <tr><td class="k">Phase</td><td class="v" id="t_duck_phase">idle</td></tr>
        </table>
        <div style="margin-top:6px; color:#8a8a8a;">Stop &rarr; watch 2s &rarr; if still there, centre on the yellow line for 5s, then return to normal lane following.</div>
      </div>
      <div class="card">
        <strong>Controls</strong>
        <kbd>E</kbd> Toggle Manual Override<br>
        <kbd>W</kbd>/<kbd>S</kbd> Latch Speed &nbsp; <kbd>A</kbd>/<kbd>D</kbd> Latch Spin<br>
        <kbd>Space</kbd> Stop (manual) &nbsp; <kbd>Q</kbd> Kill<br>
        <kbd>Y</kbd> Toggle Auto-Path Mode &nbsp; <kbd>R</kbd> Reset run state<br>
        <div style="margin-top:6px; color:#8a8a8a;">At a red line in manual routing: <kbd>W</kbd> straight, <kbd>A</kbd> left, <kbd>D</kbd> right.</div>
      </div>
      <div class="card">
        <strong>Live Route Map (with Compass)</strong>
        <div id="mapbox" style="margin-top:6px; text-align:center;"></div>
      </div>
    </div>
  </div>
<script>
  function cls(good) { return good ? 'ok' : 'bad'; }
  // Poll the robot's status 4 times a second and refresh the numbers on screen
  setInterval(function() {
    fetch('/telemetry').then(r => r.json()).then(t => {
      document.getElementById('t_state').innerText = t.state;
      document.getElementById('t_v').innerText = t.v.toFixed(3);
      document.getElementById('t_omega').innerText = (t.omega >= 0 ? '+' : '') + t.omega.toFixed(2);
      var lines = document.getElementById('t_lines');
      lines.innerHTML = '<span class="' + cls(t.yellow) + '">Y</span> / <span class="' + cls(t.white) + '">W</span>';
      document.getElementById('t_duck').innerText = t.duck ? ('YES (' + t.duck_area + 'px)') : 'None';
      document.getElementById('t_fps').innerText = t.fps.toFixed(1);
      var link = document.getElementById('t_link');
      link.innerText = t.link;
      link.className = 'v ' + (t.link === 'ok' ? 'ok' : 'bad');
      var auto = document.getElementById('t_auto');
      auto.innerText = t.auto_path_mode ? 'ON' : 'OFF';
      auto.className = 'v ' + (t.auto_path_mode ? 'ok' : 'bad');
      document.getElementById('t_progress').innerText = t.path_progress;
      document.getElementById('t_penalty').innerText = t.intersection_penalty.toFixed(0);
      var pbtn = document.getElementById('penalty_btn');
      if (pbtn) {
        pbtn.innerText = t.intersection_penalty > 0 ? 'ON' : 'OFF';
        pbtn.className = t.intersection_penalty > 0 ? 'active' : '';
      }
      var dphase = document.getElementById('t_duck_phase');
      dphase.innerText = t.duck_phase;
      dphase.className = 'v ' + (t.duck_phase === 'idle' ? 'ok' : 'bad');
      var box = document.getElementById('status_box');
      box.innerText = t.note ? (t.state + ' - ' + t.note) : t.state;
      // Color the big status banner according to what the robot is doing
      if (t.state === 'goal_reached') { box.style.background = '#22c55e'; box.style.color = '#000'; }
      else if (t.state === 'red_line_stopped') { box.style.background = '#c0342e'; box.style.color = '#fff'; }
      else if (t.state === 'duck_stopped') { box.style.background = '#e67e22'; box.style.color = '#fff'; }
      else if (t.state.indexOf('duck_overtake') === 0) { box.style.background = '#a855f7'; box.style.color = '#fff'; }
      else if (t.state === 'u_turn') { box.innerText = 'U-TURN' + (t.note ? ' - ' + t.note : ''); box.style.background = '#06b6d4'; box.style.color = '#001018'; }
      else { box.style.background = '#f6c915'; box.style.color = '#111'; }
    });
  }, 250);
  // Refresh the map picture (route, position dot, etc.) a bit over once a second
  function reloadMap() {
    fetch('/map_svg').then(r => r.text()).then(svg => {
      var sideBox = document.getElementById('mapbox');
      if (sideBox) sideBox.innerHTML = svg;
      var modalBox = document.getElementById('modal_mapbox');
      if (modalBox) modalBox.innerHTML = svg;
    });
  }
  setInterval(reloadMap, 600);
  // ---- A* route setup wizard logic ----
  var clickMode = null;  // null, 'start', or 'goal' - which marker the next map click will place
  var currentStart = null;
  var currentGoal = null;
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
        currentStart = x + ',' + y;
        setClickMode(null);
        document.getElementById('astar_setup_status').innerText = 'Start set! Now click 2. Set Goal.';
        reloadMap();
      });
    } else if (clickMode === 'goal') {
      fetch('/set_goal?x=' + x + '&y=' + y).then(() => {
        currentGoal = x + ',' + y;
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
  function togglePenalty() {
    var enabled = document.getElementById('penalty_btn').innerText !== 'ON';
    fetch('/set_intersection_penalty?enabled=' + (enabled ? '1' : '0')).then(() => computePath());
  }
  function compareCurrentRoute() {
    var box = document.getElementById('compare_box');
    if (!currentStart || !currentGoal) {
      box.style.display = 'block';
      box.innerHTML = '<div class="compare-cell" style="grid-column:1/3; color:#ff6b6b;">Set start and goal first.</div>';
      return;
    }
    fetch('/compare_route?start=' + encodeURIComponent(currentStart) +
          '&goal=' + encodeURIComponent(currentGoal) +
          '&heading=' + encodeURIComponent(document.getElementById('heading_val').innerText))
      .then(r => r.json()).then(d => {
        if (!d.ok) {
          box.style.display = 'block';
          box.innerHTML = '<div class="compare-cell" style="grid-column:1/3; color:#ff6b6b;">' + d.error + '</div>';
          return;
        }
        box.style.display = 'grid';
        box.innerHTML =
          '<div class="compare-cell"><strong style="color:#22c55e;">Penalty on</strong><br>' +
          d.penalty_on.tile_count + ' tiles<br>' + d.penalty_on.intersection_count + ' intersections<br>' + d.penalty_on.estimated_time_s + 's est.</div>' +
          '<div class="compare-cell"><strong style="color:#ff8c42;">Penalty off</strong><br>' +
          d.penalty_off.tile_count + ' tiles<br>' + d.penalty_off.intersection_count + ' intersections<br>' + d.penalty_off.estimated_time_s + 's est.</div>';
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
  // ---- keyboard shortcuts for manual driving / mode toggles ----
  document.addEventListener('keydown', function(e) {
    var k = e.key.toLowerCase(); if (e.key === ' ') k = 'space';
    if (['w', 'a', 's', 'd', 'e', 'q', 'r', 'y', 'space'].includes(k)) {
      if (k === 'space') e.preventDefault();
      fetch('/control?key=' + k);
    }
  });
</script>
</body>
</html>
"""

def _mjpeg_stream(getter):
    """
    A generator function that turns the latest stored JPEG frame into an
    MJPEG video stream (a very simple 'motion JPEG' format that browsers
    can play directly as a <img> tag) - it just keeps sending
    "here's a new frame" packets forever, ~25 times per second.
    """
    boundary = b'--frame\r\n'
    while True:
        with lock:
            buf = getter()
        if buf is not None:
            yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + buf + b'\r\n'
        time.sleep(0.04)

# ==============================================================================
# FLASK WEB ROUTES (the dashboard's backend "API")
# Each function below handles one URL that the dashboard's browser page
# calls, either to load the page/video, or to respond to a button click /
# keypress by changing the robot's state.
# ==============================================================================
@app.route('/')
def index():
    """The dashboard's home page - just returns the big HTML string above."""
    return PAGE

@app.route('/video')
def video():
    """The main camera video stream (with debug drawings overlaid)."""
    return Response(_mjpeg_stream(lambda: latest_jpeg), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mask_video')
def mask_video():
    """The 'what the robot sees' debug video (solid colored blobs)."""
    return Response(_mjpeg_stream(lambda: latest_mask_jpeg), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    """Returns the current TEL status dict as JSON, polled by the dashboard."""
    return jsonify(TEL)

@app.route('/map_svg')
def map_svg():
    """Returns the live track-map picture as SVG, polled by the dashboard."""
    return Response(render_map_svg(), mimetype='image/svg+xml')

@app.route('/set_start')
def set_start():
    """Called when the user clicks a tile on the map while in 'Set Start' mode."""
    global bot_start_tile
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
    return jsonify({"ok": True, "start": [x, y]})

@app.route('/set_goal')
def set_goal():
    """Called when the user clicks a tile on the map while in 'Set Goal' mode."""
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
    """Called when the user clicks one of the N/E/S/W compass buttons to
    tell the planner which way the robot is currently facing at the start."""
    global bot_heading
    d = (request.args.get('dir') or '').upper()
    if d not in DELTAS:
        return jsonify({"ok": False, "error": "bad heading"})
    bot_heading = d
    return jsonify({"ok": True, "heading": bot_heading})

@app.route('/compute_path')
def compute_path_route():
    """Called by the 'Compute Route' button - runs A* and returns the result."""
    ok, message = compute_route()
    return jsonify({"ok": ok, "message": message})

@app.route('/set_intersection_penalty')
def set_intersection_penalty():
    """Toggle the runtime A* intersection penalty between default and zero."""
    global intersection_penalty_value
    enabled = (request.args.get('enabled', '1') == '1')
    intersection_penalty_value = DEFAULT_INTERSECTION_PENALTY if enabled else 0.0
    return jsonify({"ok": True, "enabled": enabled, "penalty": intersection_penalty_value})

def _parse_route_tile_arg(name):
    raw = request.args.get(name, "")
    if "," in raw:
        a, b = raw.split(",", 1)
        return int(a), int(b)
    return int(request.args.get(name + "_x")), int(request.args.get(name + "_y"))

@app.route('/compare_route')
def compare_route_endpoint():
    """Return penalty-on and penalty-off route summaries for start/goal/heading."""
    global COMPARE_ROUTES
    try:
        start = _parse_route_tile_arg("start")
        goal = _parse_route_tile_arg("goal")
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad start or goal; use start=x,y&goal=x,y"}), 400
    heading = (request.args.get("heading") or bot_heading or "E").upper()
    if heading not in DELTAS:
        return jsonify({"ok": False, "error": "bad heading"}), 400
    if start not in GRAPH_ADJ or goal not in GRAPH_ADJ:
        return jsonify({"ok": False, "error": "start or goal is not drivable"}), 400
    COMPARE_ROUTES = compare_routes(start, goal, heading)
    return jsonify({"ok": True, **COMPARE_ROUTES})

@app.route('/confirm_manual')
def confirm_manual():
    """Called when the user picks 'Manual Mode' on the setup popup - this
    is what actually allows the robot to start moving (SETUP_COMPLETE)."""
    global AUTO_PATH_MODE, SETUP_COMPLETE
    AUTO_PATH_MODE = False
    SETUP_COMPLETE = True
    print("Manual mode confirmed.")
    return jsonify({"ok": True})

@app.route('/confirm_auto_path')
def confirm_auto_path():
    """Called when the user clicks 'Start Driving (A*)' after computing a
    route - locks in autonomous driving mode."""
    global AUTO_PATH_MODE, SETUP_COMPLETE, post_intersections_tracking, post_intersection_start_time
    global path_intersections_passed
    if len(ROUTE) < 2:
        return jsonify({"ok": False, "error": "compute a route first"})
    AUTO_PATH_MODE = True
    SETUP_COMPLETE = True
    start_run_logging()
    if ROUTE_INTERSECTION_ORDER and ROUTE_TURN_TILES.get(ROUTE_INTERSECTION_ORDER[0]) == "uturn":
        path_intersections_passed = 1
        start_u_turn()
        return jsonify({"ok": True})
    if not ROUTE_INTERSECTION_ORDER:
        # The route has no intersections at all (e.g. start and goal are on
        # the same straight stretch) - there's nothing to detect along the
        # way, so start the dead-reckoning countdown to the goal immediately
        post_intersections_tracking = True
        post_intersection_start_time = time.time()
        print(f"A* mode confirmed. No intersections on route - "
              f"dead-reckoning {goal_total_duration:.2f}s to the goal tile.")
    else:
        print(f"A* mode confirmed. Route: {ROUTE}")
    return jsonify({"ok": True})

@app.route('/control')
def control():
    """
    Handles every keyboard shortcut sent from the dashboard's browser page.
    One endpoint, dispatched by the 'key' query parameter:
      q      - kill switch: release control and shut the whole program down
      e      - toggle manual/autonomous driving
      y      - toggle A* auto-path mode on/off
      r      - reset the run's progress (start over on the current route)
      w/a/d  - at a red stop line in manual routing mode, choose straight/left/right
      w/s/a/d/space - while in manual driving mode, set the drive speed/turn
    """
    global keyboard_engaged, manual_v, manual_omega, current_state
    global turn_sequence_active, turn_step_index, step_start_time, active_turn_direction, state_start_time
    global last_red_line_time, AUTO_PATH_MODE
    global path_intersections_passed, post_intersections_tracking, current_tracked_tile_index
    global prev_error, lost_frames
    key = request.args.get('key', '').lower()
    now = time.time()
    if key == 'q':
        # Emergency stop + full shutdown: release the ROS override so the
        # robot returns to normal joystick control, then exit the process
        # shortly after (small delay lets the HTTP response actually send first)
        release_override()
        threading.Timer(0.2, lambda: os._exit(0)).start()
        return jsonify({"ok": True, "action": "quit"})
    if key == 'e':
        # Flip manual override on/off. Turning it OFF resets the driving
        # state back to normal lane-following from a clean slate.
        keyboard_engaged = not keyboard_engaged
        manual_v = 0.0
        manual_omega = 0.0
        if not keyboard_engaged:
            current_state = STATE_LANE_FOLLOWING
            state_start_time = now
            last_red_line_time = now
            prev_error = 0.0
            lost_frames = 0
        return jsonify({"ok": True, "manual": keyboard_engaged})
    if key == 'y':
        AUTO_PATH_MODE = not AUTO_PATH_MODE
        return jsonify({"ok": True, "auto_path_mode": AUTO_PATH_MODE})
    if key == 'r':
        # Reset the robot's run progress (route position, stop-line
        # cooldown, PD controller memory) back to a fresh start, without
        # forgetting the planned route itself - useful for re-running the
        # same route from the beginning.
        keyboard_engaged = False
        manual_v = 0.0
        manual_omega = 0.0
        current_state = STATE_LANE_FOLLOWING
        state_start_time = now
        last_red_line_time = 0.0
        path_intersections_passed = 0
        post_intersections_tracking = False
        current_tracked_tile_index = 0
        prev_error = 0.0
        lost_frames = 0
        return jsonify({"ok": True, "action": "reset"})
    if current_state == STATE_RED_STOP and not keyboard_engaged and not AUTO_PATH_MODE and key in ('w', 'a', 'd'):
        # Manual routing: while stopped at a red line (and not in A* mode),
        # let the human pick which way to turn
        direction = {'w': 'straight', 'a': 'left', 'd': 'right'}[key]
        last_red_line_time = now
        start_intersection_turn(direction)
        return jsonify({"ok": True, "turn": direction})
    if keyboard_engaged:
        # Basic tank-style manual driving controls, latched (they set a
        # speed/turn that stays active until the next keypress changes it,
        # rather than needing to be held down)
        if key == 'w':
            manual_v, manual_omega = 0.10, 0.0
        elif key == 's':
            manual_v, manual_omega = -0.10, 0.0
        elif key == 'a':
            manual_v, manual_omega = 0.04, 0.75
        elif key == 'd':
            manual_v, manual_omega = 0.04, -0.75
        elif key == 'space':
            manual_v, manual_omega = 0.0, 0.0
        return jsonify({"ok": True, "v": manual_v, "omega": manual_omega})
    return jsonify({"ok": False, "error": "unknown key or not applicable in this state"})

if __name__ == '__main__':
    print("\nMerged navigation stack ready. Dashboard: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
