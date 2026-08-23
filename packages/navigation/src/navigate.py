#!/usr/bin/env python3
"""
duck3 Navigation Hub
=====================

Self-contained autonomous-driving stack for a single Duckiebot ("duck3").
This process:

  1. Talks to the robot over rosbridge (roslibpy) - subscribes to the camera
     feed and publishes wheel commands. If rosbridge can't be reached (e.g.
     while developing on a laptop with no robot nearby) it transparently
     falls back to a synthetic simulation feed so the whole state machine
     and dashboard can still be exercised.
  2. Runs a classical computer-vision pipeline (OpenCV, HSV colour
     thresholding) to find the yellow centre line, the white lane edges,
     red stop lines, and rubber-duck obstacles in every camera frame.
  3. Drives a finite-state machine that turns that vision into behaviour:
     lane following (PD control), stopping at red lines, executing turn
     manoeuvres at intersections, and going around ducks blocking the lane.
  4. Optionally plans a route across a known tile map with A* and drives it
     autonomously, choosing the correct turn at every intersection and
     dead-reckoning the final stretch to the goal tile.
  5. Serves a small Flask dashboard (video/telemetry/map) so a human can
     watch, take manual control, or set up an A* route from a browser.

Sections in this file, top to bottom:
    - Lane following (PD controller tuning)
    - Vision thresholds + duck / stop-line recognition
    - Duck avoidance behaviour
    - Intersection manoeuvres
    - A* path planner with heading constraint
    - Finite-state machine + main per-frame vision/control pipeline
    - ROS link (publishing / subscribing) and safety watchdog
    - Simulation feed (used when no robot is reachable)
    - Flask dashboard (HTML page + HTTP endpoints)

Nothing here changes the robot's actual driving logic - this is a
documentation/readability pass over the original script.
"""
import base64
import os
import signal
import threading
import time
import atexit
import cv2
import numpy as np
import roslibpy
import yaml
from flask import Flask, Response, jsonify, request

# Any previous demo container left running on the robot would fight this
# script for control of the wheels, so make sure it's stopped before we
# start publishing our own commands. Fire-and-forget over SSH; we don't
# want a slow/failed SSH to block this process from starting.
print("Stopping baseline navigation container...")
os.system("ssh duckie@duck3.local 'docker stop demo_indefinite_navigation' > /dev/null 2>&1 &")

app = Flask(__name__)

# Most recently encoded JPEG frames served by the /video and /mask_video
# MJPEG streams. Written by process_image_frame(), read by the streaming
# generator below - guarded by `lock` since they're touched from different
# threads (ROS callback thread / simulation thread vs. Flask's request
# threads).
latest_jpeg = None
latest_mask_jpeg = None
lock = threading.Lock()

simulation_mode = False
VEHICLE = os.environ.get("VEHICLE_NAME", "duck3")
ROSBRIDGE_HOST = 'localhost'
ROSBRIDGE_PORT = 9001
FRAME_STALE_S = 0.4  # no fresh camera frame for this long -> watchdog kills the wheels

# ==============================================================================
# LANE FOLLOWING
# ==============================================================================
# Classic PD (proportional-derivative) controller on the horizontal offset
# between the lane centre and the image centre. `error` is normalised to
# [-1, 1] (fraction of half the image width), so these gains are tuned in
# "radians of omega per unit of normalised error", not pixels.
KP = 4.0                # proportional gain: how hard we steer for a given offset
KD = 2.0                 # derivative gain: damps oscillation / reacts to fast drift
V_BAR = 0.10              # nominal forward speed (m/s) when driving straight
OMEGA_MAX = 6.0           # hard clamp on yaw rate (rad/s) for normal lane following
V_MIN = 0.05              # never crawl slower than this while still trying to drive
SLOWDOWN_STRENGTH = 0.8   # how much speed bleeds off as the steering error grows
ROI_FRACTION = 0.40       # only look at the bottom 40% of the frame for line centroids

# Road layout as seen by the camera: [white edge][oncoming lane][YELLOW
# centre][our lane][white edge]. Ungated searches let the far white edge
# hijack steering on curves -- these gates keep each line search on its own
# side of the frame.
YELLOW_SEARCH_MAX = 0.70   # yellow centreline search stays left of 70% of frame width
WHITE_SEARCH_MIN = 0.30    # white edge search stays right of 30% of frame width
MIN_LANE_WIDTH_PX = 20     # if the detected white edge is closer to yellow than this, discard it
HALF_LANE_FRAC = 0.25      # assumed half lane-width (as a frame fraction) when only one line is visible
LANE_MEMORY_DECAY = 0.85   # how fast we forget the last steering command while the lane is lost
LANE_LOST_MAX_FRAMES = 15  # give up and stop after this many consecutive lost-lane frames


def centroid_x(mask, x_lo=None, x_hi=None):
    """Return the x-coordinate of a binary mask's centre of mass, or None.

    `x_lo`/`x_hi` restrict the search to a horizontal band. This is done by
    *blanking* the mask outside the band (not cropping it), so the returned
    centroid stays in full-image x-coordinates and can be compared directly
    against other centroids or the image centre.

    Returns None if the mask is empty (or too small to trust) inside the
    given band - `m["m00"]` is the zeroth image moment, i.e. total mask
    area in pixels, and doubles as a noise-rejection threshold here.
    """
    if x_lo is not None or x_hi is not None:
        gated = np.zeros_like(mask)
        lo = 0 if x_lo is None else max(0, int(x_lo))
        hi = mask.shape[1] if x_hi is None else min(mask.shape[1], int(x_hi))
        if hi <= lo:
            return None
        gated[:, lo:hi] = mask[:, lo:hi]
        mask = gated
    m = cv2.moments(mask)
    if m["m00"] < 500:  # not enough pixels to trust the centroid - treat as "no line"
        return None
    return m["m10"] / m["m00"]


# ==============================================================================
# VISION THRESHOLDS + DUCK / STOP-LINE RECOGNITION
# ==============================================================================
# HSV colour ranges used to segment the camera frame. Red wraps around the
# hue circle (0 and 180 are both "red" in OpenCV's 0-180 hue range), so it
# needs two ranges OR'd together.
HSV_YELLOW = (np.array([10, 70, 70]), np.array([40, 255, 255]))
HSV_WHITE = (np.array([0, 0, 150]), np.array([180, 45, 255]))
HSV_RED_A = (np.array([0, 110, 60]), np.array([15, 255, 255]))
HSV_RED_B = (np.array([160, 110, 60]), np.array([180, 255, 255]))

# Duck-blob area gates, in pixels. Anything smaller than DUCK_MIN_AREA is
# noise; anything larger than DUCK_MAX_AREA is almost certainly a mislabeled
# patch of yellow lane paint rather than an actual duck.
DUCK_MIN_AREA = 2000
DUCK_MAX_AREA = 160000
DUCK_LARGE_AREA = 40000   # above this the blob is close and may be frame-clipped

STOPLINE_MIN_AR = 1.5              # stop lines are wide and short (width/height ratio)
STOPLINE_MIN_WIDTH_FRAC = 0.20     # must span at least this fraction of the frame width
STOPLINE_MIN_AREA = 150
STOPLINE_MIN_BOTTOM_FRAC = 0.80    # must sit in the bottom 20% of the frame (i.e. close)
STOPLINE_COOLDOWN_S = 5.0          # ignore further stop-line triggers for this long after one fires
RED_STOP_HOLD_S = 2.0              # how long to sit at a stop line before continuing


def is_duck(contour):
    """Heuristic shape filter that decides whether a yellow contour is a
    rubber duck rather than a patch of yellow lane paint.

    Combines several classical shape descriptors so no single noisy metric
    can misclassify a contour on its own:
      - area (already pre-filtered by the caller's ROI + area gates)
      - aspect ratio of the axis-aligned bounding box
      - solidity (contour area / convex-hull area) - ducks have a fairly
        solid silhouette, thin lane-paint slivers don't
      - rectangularity (contour area / minimum-area rotated rectangle) -
        lane markings are close to perfect rectangles, ducks aren't
      - polygon approximation vertex count - a near-rectangle with very
        few vertices and high solidity reads as a lane marking, not a duck
    """
    area = cv2.contourArea(contour)
    if area < DUCK_MIN_AREA or area > DUCK_MAX_AREA:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    # A duck close enough to matter is often clipped by the frame edge, so its
    # bounding box stops looking duck-shaped. Loosen the aspect gate for big
    # blobs; the rectangularity checks below still reject lane markings.
    if area >= DUCK_LARGE_AREA:
        if aspect_ratio > 2.4 or aspect_ratio < 0.3:
            return False
    elif aspect_ratio > 1.9 or aspect_ratio < 0.4:
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

    # A near-convex shape with few vertices is almost certainly a simple
    # rectangle/blob of paint, not the more irregular silhouette of a duck.
    if len(approx) <= 6 and solidity > 0.85:
        return False
    # A long, thin, highly rectangular shape reads as a painted lane line
    # segment rather than a duck.
    if rot_aspect > 1.55 and rectangularity > 0.72:
        return False
    return True


# ==============================================================================
# DUCK AVOIDANCE (movement)
#
# Behaviour: duck seen and close enough -> full stop. Wait DUCK_WAIT_DURATION
# (2s) watching whether it's actually a duck walking around or a stationary
# obstacle -- if it shifts, the patience timer resets and we keep watching;
# if it clears out of the way on its own, we just resume. Only once it has
# sat still for the full wait (or DUCK_MAX_WAIT_S has elapsed regardless) do
# we run the go-around.
#
# The go-around is a literal 3-step SEQUENCE (left, forward, right), built
# the same way the intersection turns and navigator.py's turn maneuvers are
# built: a small ordered list of steps, each with a minimum time before it's
# even allowed to end and a maximum time as a safety net, advanced by a
# single step index (mirrors navigator.py's phases_for()/do_turn() pattern).
# Unlike a truly blind timed swerve, each step still ends on a real vision
# check (not just a clock) -- LEFT ends once the mirrored left-lane geometry
# reads as acquired, FORWARD lane-follows down the left lane using that same
# mirrored geometry so a curve doesn't carry the bot off the road, and RIGHT
# ends once the right lane (yellow back on our left) is confirmed again.
# ==============================================================================
DUCK_AVOIDANCE_ON = True
DUCK_TRIGGER_FRAMES = 3       # consecutive confirmed-duck frames before stopping
DUCK_TRIGGER_AREA = 3500      # min blob area (closeness) worth stopping for
DUCK_COOLDOWN_S = 8           # min seconds between duck-avoidance episodes

# ---- step 0: stop and watch --------------------------------------------
DUCK_WAIT_DURATION = 2.0      # must sit still this long before we plan around it
DUCK_MAX_WAIT_S = 10.0        # hard cap: go around it even if it keeps twitching
DUCK_MOVED_PX = 22            # centroid shift that counts as "it moved"
DUCK_MOVED_AREA_FRAC = 0.30   # relative area change that counts as "it moved"
DUCK_CLEAR_FRAMES = 4         # frames without a blocking duck -> path is free
DUCK_RETRIGGER_COOL = 2.5     # after finishing, ignore ducks this long (it's behind us)

# ---- go-around: centre on the yellow line for a fixed duration -----------
# Once the duck is confirmed not moving, steer to put the YELLOW LINE ITSELF
# at the centre of the frame (instead of the usual half-lane-right-of-yellow
# target) for DUCK_FOLLOW_YELLOW_S seconds, then hand back to normal lane
# following. Still the same closed-loop PD controller the whole time, just
# aimed at a different target -- so it can't run away or overshoot the way
# a fixed-omega open turn can.
OVERTAKE_SPEED = 0.072
DUCK_FOLLOW_YELLOW_S = 3.0     # how long to ride the yellow line before returning
AVOID_OMEGA_MAX = 2.4          # tighter steering clamp while doing this
YELLOW_SEARCH_SWERVE = 0.95    # widen the yellow search while re-centring on it


# ==============================================================================
# INTERSECTION MANEUVERS (movement)
# ==============================================================================
# Each tuple is (v, omega, duration_seconds): an open-loop step driven for at
# least `duration_seconds` before the state machine is allowed to check for
# a valid exit lane. Left turns are a 3-step arc (ease left, sweep through
# the turn, straighten out); driving straight through an intersection is a
# single step that just powers through until the exit lane is (re)acquired.
INTERSECTION_LEFT_STEPS = [(0.08, 0.0, 0.8), (0.10, 1.20, 1.8), (0.07, 0.0, 1.0)]
INTERSECTION_STRAIGHT_STEPS = [(0.09, 0.0, 3.0)]

# Right turns aren't a timed sequence at all - instead the bot closes a
# control loop on the white outer edge of the turn ("hugs" it), which
# handles the tighter, more variable radius of a right turn better than an
# open-loop timed arc would.
WHITE_HUG_TARGET_FRAC = 0.78   # keep the white edge at this fraction of frame width
WHITE_HUG_GAIN = 2.0
WHITE_HUG_CLAMP = 2.2
RIGHT_SEARCH_V = 0.06          # crawl speed while no white edge is visible yet
RIGHT_SEARCH_OMEGA = -1.6      # and spin right in place (ish) looking for it
RIGHT_HUG_MAX_S = 6.0          # safety timeout - give up hugging and just resume lane following


# ==============================================================================
# A* PATH PLANNER WITH HEADING CONSTRAINT
# ==============================================================================
# The track is modelled as a grid of tiles, each either "grass" (undrivable)
# or a road tile with a `kind` (straight/curve/3-way/4-way) that determines
# which of its four edges (N/E/S/W) a car can enter or exit from. Path
# planning is graph search over tile-to-tile edges gated by that openings
# table, not free-form geometry.
MAP_PATH = os.path.expanduser('~/tum_map.yaml')

DELTAS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
LEFT_OF = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}

# For each tile kind + the heading it's drawn facing in the map file, the
# set of compass directions a road actually connects to. E.g. a tile drawn
# as "curve_left/N" connects South (where you enter from) to West (where
# you exit to) - a left-hand curve into a northbound road.
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

# A "3way_*"/"4way" tile is the only place a red stop line lives on the map --
# every one the route passes through costs a real stop-and-wait plus the turn
# maneuver itself, which is a lot more time than just driving across one more
# tile. INTERSECTION_PENALTY is added to the A* edge cost for entering such a
# tile, so the planner finds the route through the FEWEST intersections
# first, breaking ties by fewest tiles driven. It's large enough to dominate
# any plausible tile-count difference on this map (worst case a few dozen
# tiles) -- lower it toward something like 2-4 if you'd rather the planner
# trade "one more intersection" for "many extra tiles" instead of avoiding
# intersections almost regardless of distance.
INTERSECTION_PENALTY = 1000.0


def is_intersection_kind(kind):
    """True for any tile kind that carries a stop line (3-way / 4-way)."""
    return kind.startswith("3way_") or kind == "4way"


def load_track_map():
    """Load the tile grid from MAP_PATH (a YAML file with a `tiles` matrix
    of tile-kind strings), falling back to a small hardcoded 6x7 TUM demo
    map if the file is missing or malformed. Returns (tiles, width, height).
    """
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
    """Turn the tile grid + OPENINGS table into a directed graph: for every
    drivable tile, the list of (neighbour_tile, direction_driven) edges that
    are geometrically legal - i.e. this tile opens onto the neighbour AND
    the neighbour opens back onto this tile from the opposite side.
    """
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
    """Weighted A*: every tile costs 1.0 to enter, plus INTERSECTION_PENALTY
    if that tile is a stop-line intersection -- except the GOAL tile itself,
    since arriving there ends the trip rather than triggering a turn (this
    matches compute_path_turn_decisions below, which also never counts the
    first or last tile of the path as an intersection stop). The Manhattan
    heuristic is still a valid lower bound (every edge costs >= 1.0), so the
    search remains optimal for this weighted cost, not just for tile count.

    `required_heading`, if given, constrains the very first edge out of
    `start` to that compass direction (the bot can't instantly change which
    way it's already facing) - and if no path satisfies that, the search is
    retried once without the constraint rather than failing outright.
    """
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
            if nb in path:  # no revisits - keeps the search a simple path
                continue
            edge_cost = 1.0
            if nb != goal and is_intersection_kind(MAP_TILES[nb[1]][nb[0]]):
                edge_cost += INTERSECTION_PENALTY
            tentative_g = g + edge_cost
            h = abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
            heapq.heappush(open_heap, (tentative_g + h, tentative_g, nb, path + [nb]))
    if required_heading:
        return astar_search(start, goal, required_heading=None)
    return [start]


def compute_path_turn_decisions(path):
    """Walk a tile path and, for every intersection tile strictly between
    the start and goal, work out which way the bot needs to turn there
    (based on the heading it enters on vs. the heading it needs to leave
    on) and record it. Returns:
        turn_tiles:         {tile: "straight" | "left" | "right"}
        intersection_order: [tile, ...] in the order they're driven through
    """
    turn_tiles = {}
    intersection_order = []
    if len(path) < 3:
        return turn_tiles, intersection_order
    for k in range(1, len(path) - 1):
        prev, cur, nxt = path[k - 1], path[k], path[k + 1]
        kind = MAP_TILES[cur[1]][cur[0]]
        if not is_intersection_kind(kind):
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


# ---- route / map state -------------------------------------------------
# All of the below is mutated by the /set_start, /set_goal, /set_heading,
# /compute_path and /confirm_auto_path dashboard endpoints, and consumed by
# the state machine while AUTO_PATH_MODE is on.
ROUTE = []
ROUTE_GOAL = None
ROUTE_TURN_TILES = {}
ROUTE_INTERSECTION_ORDER = []
bot_start_tile = None
bot_heading = 'E'
AUTO_PATH_MODE = False
SETUP_COMPLETE = False
path_intersections_passed = 0

# ---- dead-reckoning goal timer (timestamp-based tile counting after the
# last planned turn -- there is nothing left to count via stop lines once
# the last turn is done, so the last leg is timed instead) ----
SECONDS_PER_TILE = 4.5     # rough time to cross one straight tile at V_BAR
FINAL_TILE_SECONDS = 2.25  # only drive halfway into the goal tile, not through it
post_intersections_tracking = False
post_intersection_start_time = 0.0
final_turn_route_index = 0
tiles_after_final_turn = 0
current_tracked_tile_index = 0
goal_total_duration = 0.0


def compute_route():
    """Recompute ROUTE (and its derived turn/timing state) from the current
    bot_start_tile / ROUTE_GOAL / bot_heading. Returns (ok, message) for the
    /compute_path dashboard endpoint to relay to the user.
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

    turn_tiles, order = compute_path_turn_decisions(path)
    ROUTE = path
    ROUTE_TURN_TILES = turn_tiles
    ROUTE_INTERSECTION_ORDER = order
    path_intersections_passed = 0
    post_intersections_tracking = False
    current_tracked_tile_index = 0

    # Work out how long the final, un-supervised leg (after the last
    # planned turn, where there are no more stop lines to key off of) will
    # take to drive, so the state machine can time it out instead.
    if order:
        last_inter = order[-1]
        final_turn_route_index = path.index(last_inter)
        tiles_after_final_turn = len(path) - 1 - final_turn_route_index
    else:
        final_turn_route_index = 0
        tiles_after_final_turn = len(path) - 1

    if tiles_after_final_turn > 1:
        goal_total_duration = (tiles_after_final_turn - 1) * SECONDS_PER_TILE + FINAL_TILE_SECONDS
    elif tiles_after_final_turn == 1:
        goal_total_duration = FINAL_TILE_SECONDS
    else:
        goal_total_duration = 0.0

    return True, f"Route: {len(path)} tiles ({len(order)} turns)"


def _credit_goal_timer(seconds):
    """Push the dead-reckoning clock forward so time spent stopped (a red
    line, a duck go-around) on the post-final-turn leg doesn't get counted
    as driving progress towards the goal tile."""
    global post_intersection_start_time
    if post_intersections_tracking and seconds > 0:
        post_intersection_start_time += seconds


def render_map_svg():
    """Render the tile map, planned route, start/goal markers, per-turn
    arrows and a pulsing "you are here" marker as an inline SVG string for
    the dashboard's live map panel."""
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

    # Pulsing marker showing current progress along the route: still working
    # through planned turns -> next intersection; past the last turn and
    # dead-reckoning -> current timed tile estimate; otherwise -> the goal.
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
# STATE MACHINE
# ==============================================================================
# Top-level behavioural states the robot can be in. Exactly one is active at
# a time; process_image_frame() below is both the vision pipeline and the
# state machine's per-frame `update()`.
STATE_LANE_FOLLOWING = "lane_following"
STATE_RED_STOP = "red_line_stopped"
STATE_DUCK_STOP = "duck_stopped"
STATE_DUCK_OVERTAKE = "duck_overtake"
STATE_INTERSECTION_TURN = "intersection_maneuver"
STATE_GOAL_REACHED = "goal_reached"
DUCK_OVERTAKE_STATES = (STATE_DUCK_STOP, STATE_DUCK_OVERTAKE)

current_state = STATE_LANE_FOLLOWING
state_start_time = time.time()   # when we entered current_state
step_start_time = time.time()    # when we entered the current sub-step (e.g. a turn step)
last_red_line_time = 0.0         # last time a stop-line episode ended (cooldown anchor)
last_duck_avoid_time = 0.0       # last time a duck-avoidance episode ended (cooldown anchor)

# STATE_INTERSECTION_TURN bookkeeping (see start_intersection_turn()).
turn_sequence_active = []
turn_step_index = 0
active_turn_direction = 'none'

# Manual ("keyboard") drive override, toggled by the dashboard's E key.
keyboard_engaged = False
manual_v = 0.0
manual_omega = 0.0

# Lane-following PD-controller memory, carried across frames.
prev_error = 0.0
last_omega = 0.0
lost_frames = 0

# Duck-avoidance episode bookkeeping (see STATE_DUCK_STOP / STATE_DUCK_OVERTAKE).
_duck_seen_frames = 0
_duck_ref_x = None
_duck_ref_area = 0.0
_duck_stop_clock = 0.0     # when the whole duck episode started (robot stationary)
_duck_clear_frames = 0     # used by the STOP/wait phase (duck cleared on its own)
_overtake_note = "idle"

# Link / fps telemetry.
_prev_frame_t = time.time()
_last_frame_time = time.time()
_fps = 0.0
link_stale = False
_link_error = ""
_last_publish_warn = 0.0

# Snapshot of current status, refreshed every frame and polled by the
# dashboard's /telemetry endpoint.
TEL = {"state": current_state, "v": 0.0, "omega": 0.0,
       "yellow": False, "white": False, "duck": False, "duck_area": 0,
       "fps": 0.0, "link": "ok", "note": "", "auto_path_mode": False,
       "setup_complete": False, "duck_phase": "idle", "path_progress": "no route"}


# ==============================================================================
# ROS LINK
# ==============================================================================
# Connect to rosbridge and set up the publishers this script drives the
# robot with. If the connection fails (no rosbridge reachable - typically
# because no physical robot is on the network), fall back to
# `simulation_mode` so the rest of the stack (state machine + dashboard)
# still runs against a synthetic camera feed instead of crashing.
try:
    print(f"Connecting to rosbridge at {ROSBRIDGE_HOST}:{ROSBRIDGE_PORT} ...")
    client_ros = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
    client_ros.run(timeout=3)
    if not client_ros.is_connected:
        raise RuntimeError("rosbridge did not answer")
    print(f"Connected to {VEHICLE}.")
    cmd_pub = roslibpy.Topic(client_ros, f'/{VEHICLE}/car_cmd_switch_node/cmd',
                              'duckietown_msgs/Twist2DStamped')
    override_pub = roslibpy.Topic(client_ros, f'/{VEHICLE}/joy_mapper_node/joystick_override',
                                   'duckietown_msgs/BoolStamped')
except Exception as e:
    print(f"Hardware link offline ({e}). Starting simulation feed.")
    simulation_mode = True


def _stamp():
    """Minimal ROS Header-compatible stamp (we don't need real timestamps
    for this to work, just a well-formed message)."""
    return {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''}


def publish_drive(v, omega):
    """Publish a velocity command to the robot, asserting joystick override
    first (Duckietown's stock stack otherwise ignores /cmd unless the
    joystick override flag is set). No-ops in simulation mode. Failures are
    swallowed (logged at most every 2s) so a flaky ROS link doesn't crash
    the vision/control loop - the watchdog thread is what actually protects
    against a stuck robot."""
    global _link_error, _last_publish_warn
    if simulation_mode:
        return
    try:
        override_pub.publish(roslibpy.Message({'header': _stamp(), 'data': True}))
        cmd_pub.publish(roslibpy.Message({'header': _stamp(), 'v': float(v), 'omega': float(omega)}))
        _link_error = ""
    except Exception as e:
        _link_error = str(e)
        if time.time() - _last_publish_warn > 2.0:
            print(f"publish_drive failed: {e}")
            _last_publish_warn = time.time()


def release_override():
    """Stop the wheels and hand control back to the joystick/default stack.
    Registered for atexit + SIGTERM/SIGINT so however this process ends,
    the robot doesn't drive off unattended."""
    if simulation_mode:
        return
    try:
        cmd_pub.publish(roslibpy.Message({'header': _stamp(), 'v': 0.0, 'omega': 0.0}))
        override_pub.publish(roslibpy.Message({'header': _stamp(), 'data': False}))
    except Exception:
        pass


atexit.register(release_override)
signal.signal(signal.SIGTERM, lambda s, f: release_override() or os._exit(0))
signal.signal(signal.SIGINT, lambda s, f: release_override() or os._exit(0))


def watchdog_loop():
    """Background safety net: if no camera frame has been processed in the
    last FRAME_STALE_S seconds (frozen callback, dropped topic, an
    exception in the vision pipeline, ...), force the wheels to stop.

    Never let this thread die: it is what stops the wheels if frame
    processing stalls or throws."""
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
    """Enter STATE_INTERSECTION_TURN and arm the open-loop step sequence
    for the given direction ('straight' | 'left' | 'right'). Right turns
    get an empty sequence because they're driven by the closed-loop
    white-line-hugging controller instead (see the STATE_INTERSECTION_TURN
    handling in process_image_frame)."""
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


def _advance_route_after_turn():
    """Called once a turn manoeuvre finishes. If that was the LAST planned
    turn on the route, switch the route-progress tracker from counting
    intersections to timing tiles (dead reckoning) for the final leg."""
    global post_intersections_tracking, post_intersection_start_time
    if AUTO_PATH_MODE and path_intersections_passed >= len(ROUTE_INTERSECTION_ORDER):
        post_intersections_tracking = True
        post_intersection_start_time = time.time()
        print(f"Final turn complete. Driving {goal_total_duration:.2f}s to the goal tile...")


# ==============================================================================
# MAIN FRAME PROCESSING
# ==============================================================================
def process_image_frame(frame):
    """Run one full cycle of the pipeline on a single BGR camera frame:

        1. Vision: threshold for yellow/white/red, find & classify duck and
           stop-line contours, compute lane-centreline centroids.
        2. Control: advance the finite-state machine (lane following / red
           stop / intersection turn / duck stop / duck overtake / goal
           reached) and publish the resulting (v, omega) to the robot.
        3. Telemetry: annotate the frame, encode both the annotated camera
           view and a debug colour mask to JPEG for the dashboard streams,
           and refresh the TEL telemetry dict.

    This function is the callback for both the real camera subscription and
    the simulation loop, so it's the single source of truth for "what does
    the robot do this frame" regardless of whether hardware is attached.
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

    if frame is None or frame.size == 0:
        return

    # ---- fps bookkeeping ----
    now = time.time()
    dt = min(0.5, max(0.0, now - _prev_frame_t))
    _prev_frame_t = now
    if dt > 0:
        _fps = 0.85 * _fps + 0.15 * (1.0 / dt)  # simple exponential moving average

    h, w = frame.shape[:2]
    cx = w // 2.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, *HSV_YELLOW)
    white_mask = cv2.inRange(hsv, *HSV_WHITE)
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, *HSV_RED_A), cv2.inRange(hsv, *HSV_RED_B))

    # ---- duck recognition ----
    # Only search the lower-middle band of the frame: ducks close enough to
    # matter sit low in frame, and this avoids false positives from yellow
    # objects near the horizon.
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
            if a > duck_area:  # track the largest (closest) confirmed duck
                duck_area = a
                duck_x = x + cw // 2
                duck_bottom = y + ch
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
            cv2.putText(frame, "DUCK", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif cv2.contourArea(contour) > 800:
            # Debug overlay only: a sizeable yellow blob that failed the
            # duck shape test (most likely lane paint).
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 200, 255), 1)
            cv2.putText(frame, "yellow obj", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

    # Ducks are yellow, same as the lane's centre line - cut confirmed duck
    # bounding boxes out of the yellow mask before any lane-line reasoning
    # uses it, or a duck sitting near the yellow line would distort the
    # lane centroid.
    yellow_lane = yellow_mask.copy()
    for (bx, by, bw, bh) in duck_boxes:
        yellow_lane[by:by + bh, bx:bx + bw] = 0

    if duck_found:
        _duck_seen_frames += 1
    else:
        _duck_seen_frames = 0

    # "blocking" = still a real, close-enough duck, used by the wait/pass
    # phases below to decide whether it has actually moved out of the way.
    # IMPORTANT: this must stay consistent with the trigger condition below
    # (duck_found + area threshold, nothing else).
    duck_blocking = duck_found and duck_area >= DUCK_TRIGGER_AREA * 0.6

    # ---- stop-line recognition ----
    # Only search the bottom of the frame: a stop line only matters once
    # it's close, and this avoids picking up unrelated red objects further
    # down the road.
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
            # Right shape, but not close enough yet - shown for debugging,
            # doesn't trigger a stop.
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 140, 230), 2)

    # ---- lane following (PD controller) ----
    # Restrict the centreline search to a strip near the bottom of the
    # frame (ROI_FRACTION) so the controller reacts to where the lane is
    # *right in front of the robot*, not far down the road where a curve
    # could point the wrong way.
    roi_y0 = int(h * (1.0 - ROI_FRACTION))
    yellow_roi = yellow_lane[roi_y0:, :]
    white_roi = white_mask[roi_y0:, :]
    yc = centroid_x(yellow_roi, x_hi=w * YELLOW_SEARCH_MAX)
    wc = centroid_x(white_roi, x_lo=w * WHITE_SEARCH_MIN)
    if yc is not None and wc is not None and wc <= yc + MIN_LANE_WIDTH_PX:
        # Sanity check: the white edge should be clearly to the right of
        # the yellow centreline. If it isn't, something got misdetected -
        # drop the white reading and fall back to the yellow-only estimate.
        wc = None

    if yc is not None:
        cv2.circle(frame, (int(yc), int(h * 0.9)), 6, (0, 255, 255), -1)
    if wc is not None:
        cv2.circle(frame, (int(wc), int(h * 0.9)), 6, (255, 255, 255), -1)

    if yc is not None and wc is not None:
        lane_center = (yc + wc) / 2.0             # both lines seen: split the difference
    elif yc is not None:
        lane_center = yc + w * HALF_LANE_FRAC      # only yellow: assume our lane is to its right
    elif wc is not None:
        lane_center = wc - w * HALF_LANE_FRAC      # only white: assume our lane is to its left
    else:
        lane_center = None                          # neither line visible

    note = ""
    arrow_v, arrow_omega = 0.0, 0.0
    time_in_step = now - step_start_time

    # ==========================================================================
    # STATE DISPATCH
    # ==========================================================================
    if current_state == STATE_GOAL_REACHED:
        publish_drive(0.0, 0.0)
        note = "GOAL REACHED - stopped"
        cv2.putText(frame, "DESTINATION REACHED - STOPPED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    elif not SETUP_COMPLETE:
        # Refuse to drive until a human has picked a mode on the dashboard.
        publish_drive(0.0, 0.0)
        note = "Waiting for setup on the dashboard"
        cv2.putText(frame, "SETUP - Select a mode on the dashboard", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    elif keyboard_engaged:
        # Manual override takes priority over every autonomous state.
        publish_drive(manual_v, manual_omega)
        arrow_v, arrow_omega = manual_v, manual_omega
        note = "Manual override"
        cv2.putText(frame, "MANUAL OVERRIDE - AUTONOMY DISABLED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    elif current_state == STATE_LANE_FOLLOWING:
        # --- dead-reckoning check: have we timed our way to the goal tile? ---
        if AUTO_PATH_MODE and post_intersections_tracking:
            elapsed_post = now - post_intersection_start_time
            blocks_passed = int(elapsed_post // SECONDS_PER_TILE)
            current_tracked_tile_index = min(len(ROUTE) - 1, final_turn_route_index + blocks_passed) if ROUTE else 0
            if elapsed_post >= goal_total_duration:
                current_state = STATE_GOAL_REACHED
                publish_drive(0.0, 0.0)
                print(f"Goal reached after {elapsed_post:.1f}s!")

        if current_state == STATE_LANE_FOLLOWING:
            # --- PD steering ---
            if lane_center is not None:
                error = (lane_center - cx) / cx
                d_error = error - prev_error
                prev_error = error
                omega = float(np.clip(-(KP * error + KD * d_error), -OMEGA_MAX, OMEGA_MAX))
                v = max(V_MIN, V_BAR * (1.0 - SLOWDOWN_STRENGTH * error * error))  # slow down on sharp error
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
                # Lane momentarily lost: coast on the decayed last-known
                # steering command for a grace period rather than stopping
                # immediately (handles brief dropouts, e.g. glare or a gap
                # in the paint), then give up and stop.
                lost_frames += 1
                if lost_frames <= LANE_LOST_MAX_FRAMES:
                    last_omega *= LANE_MEMORY_DECAY
                    publish_drive(V_MIN, last_omega)
                    arrow_v, arrow_omega = V_MIN, last_omega
                    note = f"Lane lost - holding ({lost_frames})"
                else:
                    publish_drive(0.0, 0.0)
                    note = "Lane lost - stopped"

            # --- duck avoidance trigger: stop first, decide later ----
            if (DUCK_AVOIDANCE_ON and duck_found and duck_area >= DUCK_TRIGGER_AREA
                    and _duck_seen_frames >= DUCK_TRIGGER_FRAMES
                    and (now - last_duck_avoid_time) > DUCK_COOLDOWN_S):
                current_state = STATE_DUCK_STOP
                state_start_time = now
                _duck_stop_clock = now
                _duck_ref_x = duck_x
                _duck_ref_area = float(duck_area)
                _overtake_note = "waiting to see if it moves"
                publish_drive(0.0, 0.0)
                print("Duck detected in lane -> stopped, watching whether it moves...")
            elif red_line_found and (now - last_red_line_time) > STOPLINE_COOLDOWN_S:
                current_state = STATE_RED_STOP
                state_start_time = now
                publish_drive(0.0, 0.0)
                print("Stop line -> stopped, holding before proceeding...")

    elif current_state == STATE_RED_STOP:
        publish_drive(0.0, 0.0)
        elapsed_in_stop = now - state_start_time
        rem_stop = max(0.0, RED_STOP_HOLD_S - elapsed_in_stop)
        note = f"Stopped at red line ({rem_stop:.1f}s left)" if AUTO_PATH_MODE else \
               "Stopped at red line - press W/A/D to choose a turn"
        cv2.putText(frame, f"STOP LINE: {rem_stop:.1f}s" if AUTO_PATH_MODE else "STOP LINE - awaiting W/A/D",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if AUTO_PATH_MODE and elapsed_in_stop >= RED_STOP_HOLD_S:
            _credit_goal_timer(RED_STOP_HOLD_S)
            if path_intersections_passed < len(ROUTE_INTERSECTION_ORDER):
                # Next intersection on the planned route decides the turn.
                tile = ROUTE_INTERSECTION_ORDER[path_intersections_passed]
                direction = ROUTE_TURN_TILES.get(tile, 'straight')
                path_intersections_passed += 1
            else:
                # Past the planned route (shouldn't normally happen) - just
                # go straight rather than getting stuck.
                direction = 'straight'
            last_red_line_time = now
            start_intersection_turn(direction)
        # In manual routing mode (not AUTO_PATH_MODE), the /control endpoint's
        # W/A/D handling is what actually calls start_intersection_turn().

    elif current_state == STATE_INTERSECTION_TURN:
        yc_t, wc_t = yc, wc  # same centroids computed above, mid-turn
        # "Have we swung far enough to see the new lane's own centreline?"
        valid_exit = (yc_t is not None and yc_t < cx - 15) or \
                     (yc_t is not None and wc_t is not None and wc_t > yc_t + MIN_LANE_WIDTH_PX)

        if active_turn_direction == 'right':
            # Closed-loop: keep the white outer edge at a fixed target
            # position in frame (steer proportionally to the error) rather
            # than following a fixed timed arc.
            target_px = int(w * WHITE_HUG_TARGET_FRAC)
            if wc is not None:
                err = wc - target_px
                hug = -(float(err) / cx) * WHITE_HUG_GAIN
                hug = max(-WHITE_HUG_CLAMP, min(WHITE_HUG_CLAMP, hug))
                publish_drive(0.07, hug)
                arrow_v, arrow_omega = 0.07, hug
                note = "Right turn - hugging white line"
            else:
                # No white edge visible yet - crawl and spin right until one appears.
                publish_drive(RIGHT_SEARCH_V, RIGHT_SEARCH_OMEGA)
                arrow_v, arrow_omega = RIGHT_SEARCH_V, RIGHT_SEARCH_OMEGA
                note = "Right turn - searching for white line"

            if valid_exit and time_in_step > 1.5:
                last_red_line_time = now
                prev_error = 0.0
                current_state = STATE_LANE_FOLLOWING
                note = "Exit lane acquired"
                _advance_route_after_turn()
            elif time_in_step > RIGHT_HUG_MAX_S:
                # Safety timeout - don't hug forever if vision never confirms the exit.
                last_red_line_time = now
                prev_error = 0.0
                current_state = STATE_LANE_FOLLOWING
                note = "Right turn timeout - lane follow"
                _advance_route_after_turn()
        else:
            # 'straight' / 'left': open-loop timed step sequence, but still
            # cut short as soon as vision confirms the exit lane.
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
                # Ran out of steps without a confirmed exit - hand back to
                # lane following anyway rather than sitting idle.
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
        still_for = now - state_start_time
        waited_total = now - _duck_stop_clock

        if not duck_blocking:
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
                _duck_ref_x = duck_x
                _duck_ref_area = float(duck_area)
                state_start_time = now      # it's moving: restart the patience timer
                still_for = 0.0
                _overtake_note = "duck is moving - giving it room"

        if _duck_clear_frames >= DUCK_CLEAR_FRAMES:
            # It wandered off on its own - no need to plan a go-around.
            last_duck_avoid_time = now
            _credit_goal_timer(waited_total)
            prev_error = 0.0
            current_state = STATE_LANE_FOLLOWING
            note = "Duck cleared on its own - resuming"
            print("Duck cleared on its own -> resuming lane following")
        elif still_for >= DUCK_WAIT_DURATION or waited_total >= DUCK_MAX_WAIT_S:
            # Sat still long enough (or we've waited as long as we're
            # willing to) - commit to the go-around.
            current_state = STATE_DUCK_OVERTAKE
            state_start_time = now
            _overtake_note = f"centring on yellow line ({DUCK_FOLLOW_YELLOW_S:.0f}s)"
            note = "Duck is not moving - centring on the yellow line to go around it"
            print(f"Duck is not moving -> centring on the yellow line for "
                  f"{DUCK_FOLLOW_YELLOW_S:.1f}s.")
        else:
            rem = max(0.0, DUCK_WAIT_DURATION - still_for)
            note = f"Duck blocking - waiting {rem:.1f}s to see if it moves"
            cv2.putText(frame, f"DUCK - WAITING {rem:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # --------------------------------------------------------------------
    # DUCK GO-AROUND: put the yellow line itself at the centre of the frame
    # (instead of the usual half-lane-right-of-yellow target) for
    # DUCK_FOLLOW_YELLOW_S seconds, then hand back to normal lane following.
    # Still the same closed-loop PD controller the whole time, just aimed at
    # a different target -- it can't run away or overshoot the way a
    # fixed-omega open turn can.
    # --------------------------------------------------------------------
    elif current_state == STATE_DUCK_OVERTAKE:
        elapsed = now - state_start_time
        yc_sw = centroid_x(yellow_roi, x_hi=w * YELLOW_SEARCH_SWERVE)
        if yc_sw is not None:
            error = (yc_sw - cx) / cx   # target IS the yellow line -- put it at centre
            d_error = error - prev_error
            prev_error = error
            omega = float(np.clip(-(KP * error + KD * d_error), -AVOID_OMEGA_MAX, AVOID_OMEGA_MAX))
            v = OVERTAKE_SPEED
        else:
            # No yellow line to steer against -- coast rather than holding
            # whatever turn we last had. Holding a hard turn with nothing to
            # measure against is what spins the bot around.
            omega = 0.0
            v = V_MIN
        publish_drive(v, omega)
        arrow_v, arrow_omega = v, omega
        rem = max(0.0, DUCK_FOLLOW_YELLOW_S - elapsed)
        note = (f"Overtake: centring on yellow line ({rem:.1f}s left)" if yc_sw is not None
                else f"Overtake: yellow line lost - coasting ({rem:.1f}s left)")
        _overtake_note = f"centring on yellow line ({rem:.1f}s left)"
        if elapsed >= DUCK_FOLLOW_YELLOW_S:
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

    # ---- HUD steering arrow ----
    cxi = int(cx)
    if abs(arrow_omega) < 0.15:
        cv2.arrowedLine(frame, (cxi, int(h * 0.85)), (cxi, int(h * 0.68)), (0, 255, 0), 4, cv2.LINE_AA, 0, 0.25)
    else:
        tx = cxi - int(arrow_omega * 60)
        color = (0, 165, 255) if abs(arrow_omega) > 3.0 else (0, 255, 0)
        cv2.arrowedLine(frame, (cxi, int(h * 0.85)), (tx, int(h * 0.68)), color, 4, cv2.LINE_AA, 0, 0.25)

    # ---- telemetry snapshot for the dashboard ----
    TEL.update({"state": current_state, "v": round(arrow_v, 3), "omega": round(arrow_omega, 2),
                "yellow": yc is not None, "white": wc is not None, "duck": duck_found,
                "duck_area": int(duck_area), "fps": round(_fps, 1),
                "link": ("simulation" if simulation_mode else
                         ("stale" if link_stale else (_link_error or "ok"))),
                "note": note, "auto_path_mode": AUTO_PATH_MODE,
                "setup_complete": SETUP_COMPLETE,
                "duck_phase": _overtake_note,
                "path_progress": f"{path_intersections_passed}/{len(ROUTE_INTERSECTION_ORDER)}"
                                  if ROUTE_INTERSECTION_ORDER else "no route"})

    # ---- encode annotated camera view for /video ----
    ok, jpeg = cv2.imencode('.jpg', frame)
    if ok:
        with lock:
            latest_jpeg = jpeg.tobytes()

    # ---- encode debug colour mask for /mask_video ----
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

    _last_frame_time = now


# ==============================================================================
# SIMULATION FEED / ROS CAMERA SUBSCRIPTION
# ==============================================================================
def simulation_hardware_loop():
    """Runs in place of the real camera subscription when no robot is
    reachable: synthesises a plain frame with two lane lines drawn on it
    (plus a red block while STATE_RED_STOP is active) and feeds it through
    the exact same process_image_frame() pipeline at roughly 30fps, so the
    dashboard and state machine can be developed/demoed without hardware."""
    print("Virtual camera loop active.")
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "SIMULATION FEED - ROBOT NOT CONNECTED", (15, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        cv2.line(frame, (180, 480), (280, 260), (0, 255, 255), 3)   # fake yellow centreline
        cv2.line(frame, (490, 480), (390, 260), (255, 255, 255), 3)  # fake white edge
        if current_state == STATE_RED_STOP:
            cv2.rectangle(frame, (200, 400), (470, 430), (0, 0, 255), -1)  # fake stop line
        try:
            process_image_frame(frame)
        except Exception as e:
            publish_drive(0.0, 0.0)
            print(f"frame processing error (wheels stopped): {e}")
        time.sleep(0.033)  # ~30 fps


if simulation_mode:
    threading.Thread(target=simulation_hardware_loop, daemon=True).start()
else:
    camera_sub = roslibpy.Topic(client_ros, f'/{VEHICLE}/camera_node/image/compressed',
                                 'sensor_msgs/CompressedImage')

    def _on_frame(msg):
        """rosbridge callback: decode the base64 JPEG payload and run it
        through the same pipeline used by the simulation loop. Any failure
        stops the wheels rather than leaving the robot on its last command."""
        try:
            buf = np.frombuffer(base64.b64decode(msg['data']), np.uint8)
            process_image_frame(cv2.imdecode(buf, cv2.IMREAD_COLOR))
        except Exception as e:
            publish_drive(0.0, 0.0)
            print(f"frame error (wheels stopped): {e}")

    camera_sub.subscribe(_on_frame)


# ==============================================================================
# DASHBOARD
# ==============================================================================
# Single-page HTML/CSS/JS dashboard: shows the live annotated camera feed
# and debug mask, telemetry, an interactive A* route planner over the tile
# map, and manual driving controls. Polls /telemetry and /map_svg on a
# timer rather than using websockets, to keep the server side simple.
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
  <!-- Blocking setup modal: no autonomy runs until a mode is picked here (see SETUP_COMPLETE). -->
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
      <!-- A* route setup: pick start tile, goal tile, and initial heading, then compute + confirm. -->
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
    <!-- Live annotated camera feed + debug colour mask -->
    <div style="width:100%; max-width:640px;">
      <img id="video-frame" src="/video" style="width:100%; border:3px solid #f6c915; border-radius:10px; background:#000;">
      <div style="color:#9a9a9a; font-size:.78em; margin:6px 0;">Recognition Mask (yellow / white / stop line / duck)</div>
      <img id="mask-frame" src="/mask_video" style="width:100%; border:3px solid #333; border-radius:10px; background:#000;">
    </div>

    <!-- Telemetry / route / controls / map sidebar -->
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
          <tr><td class="k">Turns done</td><td class="v" id="t_progress">-</td></tr>
        </table>
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

  // Poll telemetry 4x/second and refresh the sidebar + status banner.
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
      var dphase = document.getElementById('t_duck_phase');
      dphase.innerText = t.duck_phase;
      dphase.className = 'v ' + (t.duck_phase === 'idle' ? 'ok' : 'bad');
      var box = document.getElementById('status_box');
      box.innerText = t.note ? (t.state + ' - ' + t.note) : t.state;
      if (t.state === 'goal_reached') { box.style.background = '#22c55e'; box.style.color = '#000'; }
      else if (t.state === 'red_line_stopped') { box.style.background = '#c0342e'; box.style.color = '#fff'; }
      else if (t.state === 'duck_stopped') { box.style.background = '#e67e22'; box.style.color = '#fff'; }
      else if (t.state.indexOf('duck_overtake') === 0) { box.style.background = '#a855f7'; box.style.color = '#fff'; }
      else { box.style.background = '#f6c915'; box.style.color = '#111'; }
    });
  }, 250);

  // Poll the route map SVG separately, a bit slower since it changes less often.
  function reloadMap() {
    fetch('/map_svg').then(r => r.text()).then(svg => {
      var sideBox = document.getElementById('mapbox');
      if (sideBox) sideBox.innerHTML = svg;
      var modalBox = document.getElementById('modal_mapbox');
      if (modalBox) modalBox.innerHTML = svg;
    });
  }
  setInterval(reloadMap, 600);

  // ---- A* setup modal state ----
  var clickMode = null;   // null | 'start' | 'goal' -- which marker the next tile click sets
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

  // ---- keyboard controls -> /control?key=... ----
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
    """Generic multipart/x-mixed-replace MJPEG generator: repeatedly reads
    whatever JPEG `getter()` currently points to (guarded by `lock`) and
    yields it as one multipart frame, ~25 times a second. Shared by both
    the /video and /mask_video endpoints."""
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
    """Annotated camera feed (or simulation feed) as MJPEG."""
    return Response(_mjpeg_stream(lambda: latest_jpeg), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/mask_video')
def mask_video():
    """Debug colour mask (yellow lane / white edge / stop line / duck) as MJPEG."""
    return Response(_mjpeg_stream(lambda: latest_mask_jpeg), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/telemetry')
def telemetry():
    """Current TEL snapshot, polled by the dashboard every 250ms."""
    return jsonify(TEL)


@app.route('/map_svg')
def map_svg():
    """Rendered SVG of the tile map + planned route, polled by the dashboard."""
    return Response(render_map_svg(), mimetype='image/svg+xml')


@app.route('/set_start')
def set_start():
    """Dashboard: place the A* route's start tile at ?x=&y=."""
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
    """Dashboard: place the A* route's goal tile at ?x=&y=."""
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
    """Dashboard: set the bot's initial compass heading (?dir=N/E/S/W) used
    to constrain the first leg of the A* search."""
    global bot_heading
    d = (request.args.get('dir') or '').upper()
    if d not in DELTAS:
        return jsonify({"ok": False, "error": "bad heading"})
    bot_heading = d
    return jsonify({"ok": True, "heading": bot_heading})


@app.route('/compute_path')
def compute_path_route():
    """Dashboard: run A* between the currently set start/goal/heading."""
    ok, message = compute_route()
    return jsonify({"ok": ok, "message": message})


@app.route('/confirm_manual')
def confirm_manual():
    """Dashboard: leave the setup modal in plain manual (WASD) mode."""
    global AUTO_PATH_MODE, SETUP_COMPLETE
    AUTO_PATH_MODE = False
    SETUP_COMPLETE = True
    print("Manual mode confirmed.")
    return jsonify({"ok": True})


@app.route('/confirm_auto_path')
def confirm_auto_path():
    """Dashboard: leave the setup modal and start driving the computed A*
    route autonomously. Requires a valid ROUTE (see /compute_path first)."""
    global AUTO_PATH_MODE, SETUP_COMPLETE, post_intersections_tracking, post_intersection_start_time
    if len(ROUTE) < 2:
        return jsonify({"ok": False, "error": "compute a route first"})
    AUTO_PATH_MODE = True
    SETUP_COMPLETE = True
    if not ROUTE_INTERSECTION_ORDER:
        # No intersections on this route at all - nothing to count via
        # stop lines, so start the dead-reckoning goal clock immediately.
        post_intersections_tracking = True
        post_intersection_start_time = time.time()
        print(f"A* mode confirmed. No intersections on route - "
              f"dead-reckoning {goal_total_duration:.2f}s to the goal tile.")
    else:
        print(f"A* mode confirmed. Route: {ROUTE}")
    return jsonify({"ok": True})


@app.route('/control')
def control():
    """Dashboard keyboard endpoint (?key=...). Handles: manual-mode driving
    keys (w/a/s/d/space), the E manual-override toggle, Y auto-path toggle,
    R full run-state reset, Q kill switch, and W/A/D turn selection while
    stopped at a red line in manual (non-auto-path) routing."""
    global keyboard_engaged, manual_v, manual_omega, current_state
    global turn_sequence_active, turn_step_index, step_start_time, active_turn_direction, state_start_time
    global last_red_line_time, AUTO_PATH_MODE
    global path_intersections_passed, post_intersections_tracking, current_tracked_tile_index
    global prev_error, lost_frames
    key = request.args.get('key', '').lower()
    now = time.time()

    if key == 'q':
        # Kill switch: release the robot and exit the process shortly after
        # responding (so the HTTP response actually makes it back).
        release_override()
        threading.Timer(0.2, lambda: os._exit(0)).start()
        return jsonify({"ok": True, "action": "quit"})

    if key == 'e':
        keyboard_engaged = not keyboard_engaged
        manual_v = 0.0
        manual_omega = 0.0
        if not keyboard_engaged:
            # Handing back to autonomy - reset state cleanly rather than
            # resuming whatever transient state we were in before.
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
        # Full reset of run-time state (not the route/map setup itself).
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
        # Manual routing: human picks the turn at this stop line.
        direction = {'w': 'straight', 'a': 'left', 'd': 'right'}[key]
        last_red_line_time = now
        start_intersection_turn(direction)
        return jsonify({"ok": True, "turn": direction})

    if keyboard_engaged:
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
