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
from flask import Flask, Response, request

# --- BACKGROUND CONTAINER MANAGEMENT ---
print("Stopping background lane dependencies to ensure absolute system control...")
os.system("ssh duckie@duck3.local 'docker stop demo_indefinite_navigation' > /dev/null 2>&1 &")

app = Flask(__name__)
latest_jpeg = None
lock = threading.Lock()

# ==============================================================================
# 🏎️ AUTONOMOUS NAVIGATION TUNING CONFIGURATION
# ==============================================================================
LANE_SPEED = 0.075             # Base forward cruise velocity
TRACK_HALF_WIDTH = 155         # Nominal baseline center track pixel displacement

# ⚡ DYNAMIC FRICTION COMPENSATION CONFIGURATIONS
STEERING_MAX_YAW = 2.4         # Maximum angular rotation limit for tight corners
STEERING_TORQUE_FLOOR = 0.52   # Minimum voltage threshold to break wheel motor friction

# 🦆 Timed Overtaking Sequence (Go-Around Routine)
GO_AROUND_SEQUENCE = [
    (0.075,  1.0, 2.0),        # Step 1: Swerve left into passing lane
    (0.075,  0.0, 2.0),        # Step 2: Drive straight past the obstacle
    (0.075, -1.0, 4.0),        # Step 3: Swerve right back into original lane
    (0.075,  0.75, 2.0),       # Step 4: Straighten out and match track heading
]

# 🚦 CLOSED-LOOP INTERSECTION PHASE CONFIGURATIONS (Durations act as maximum safety timeouts)
INTERSECTION_LEFT_STEPS = [
    (0.07,  0.45, 2.8),        # Step 0: Deep sweeping arc forward-left to clear old lanes
    (0.00,  0.90, 4.0),        # Step 1: Stationary spin adjustment to square up with new heading
    (0.06,  0.0,  3.5)         # Step 2: Insertion roll phase (Vision breakouts unlock exclusively here!)
]

INTERSECTION_RIGHT_STEPS = [
    (0.06,  0.0,  1.1),        # Step 0: Pocket nudge forward past red line to clear tire pivot radius
    (0.00, -1.05, 4.0),        # Step 1: On-axis sharp right spin inside the intersection zone
    (0.06,  0.0,  3.5)         # Step 2: Insertion roll phase (Actively seeks lane lines to exit early!)
]

INTERSECTION_STRAIGHT_STEPS = [
    (0.06,  0.0, 5.0)          # Step 0: Direct crossover (Vision breakouts engage after 1.2 seconds)
]
# ==============================================================================

# --- STATE CONFIGURATIONS ---
STATE_LANE_FOLLOWING = "lane_following"
STATE_GO_AROUND = "go_around_maneuver"
STATE_RED_STOP = "red_line_stopped"
STATE_INTERSECTION_TURN = "intersection_maneuver"

current_state = STATE_LANE_FOLLOWING
state_start_time = 0
last_red_line_time = 0
last_overtake_time = 0
maneuver_step = 0
step_start_time = 0

turn_sequence_active = []
turn_step_index = 0
active_turn_direction = 'none'

COOLDOWN_DURATION = 6          # Intersection stop cooldown
OVERTAKE_COOLDOWN = 8          # Stabilization cooldown between overtakes

# --- INTERACTIVE MANUAL CONTROL MODULE VARIABLES ---
keyboard_engaged = False
manual_v = 0.0
manual_omega = 0.0

# Connect over port 9001 via localhost proxy tunnel
client_ros = roslibpy.Ros(host='localhost', port=9001)
client_ros.run()
print("Connected to duckie3 Central Navigation Hub!")

cmd_pub = roslibpy.Topic(client_ros, '/duck3/car_cmd_switch_node/cmd', 'duckietown_msgs/Twist2DStamped')
override_pub = roslibpy.Topic(client_ros, '/duck3/joy_mapper_node/joystick_override', 'duckietown_msgs/BoolStamped')

def publish_drive(v, omega):
    """Pushes direct hardware velocity frames down the command switch line."""
    try:
        override_pub.publish(roslibpy.Message({'header': {'stamp': {'secs':0,'nsecs':0}, 'frame_id':''}, 'data': True}))
        cmd_pub.publish(roslibpy.Message({
            'header': {'stamp': {'secs':0,'nsecs':0}, 'frame_id':''},
            'v': float(v),
            'omega': float(omega)
        }))
    except Exception as e:
        print(f"Drive transmission error: {e}")

def release_override():
    """Brakes wheels safely on a standard script exit."""
    try:
        cmd_pub.publish(roslibpy.Message({'header': {'stamp': {'secs':0,'nsecs':0}, 'frame_id':''}, 'v': 0.0, 'omega': 0.0}))
        override_pub.publish(roslibpy.Message({
            'header': {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
            'data': True
        }))
        time.sleep(0.1)
    except:
        pass

def handle_exit(sig, frame):
    release_override()
    os._exit(0)

atexit.register(release_override)
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

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

    perimeter = cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
    vertex_count = len(approx_polygon)

    if vertex_count <= 6 and solidity > 0.85: return False  
    if rotated_aspect_ratio > 1.55 and rotated_rectangularity > 0.72: return False  
    if rotated_aspect_ratio > 1.50: return False
    return True

def process_image(message):
    global latest_jpeg, current_state, state_start_time, last_red_line_time, last_overtake_time, maneuver_step, step_start_time
    global turn_sequence_active, turn_step_index, active_turn_direction, TRACK_HALF_WIDTH
    global keyboard_engaged, manual_v, manual_omega
    try:
        img_data = base64.b64decode(message['data'])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None: return

        h, w = frame.shape[:2]
        frame_center_x = w // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Base Wavelength Color Masks
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Expanded Red Mask for enhanced intersection detection
        lower_red1 = np.array([0, 110, 60])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 110, 60])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        # 1. 🦆 PANORAMIC DUCK SCANNING LAYER
        duck_roi = np.zeros_like(yellow_mask)
        duck_roi[int(h*0.45):int(h*0.92), 0:w] = yellow_mask[int(h*0.45):int(h*0.92), 0:w]
        duck_contours, _ = cv2.findContours(duck_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        duck_found = False
        for contour in duck_contours:
            if is_duck(contour):
                duck_found = True
                x, y, cw, ch = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+cw, y+ch), (0, 0, 255), 3)
                cv2.putText(frame, "DUCK OBJECT", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 2. 🚦 HARDENED RED INTERSECTION STOP LINE LAYER
        red_roi = np.zeros_like(red_mask)
        red_roi[int(h*0.72):h, 0:w] = red_mask[int(h*0.72):h, 0:w]

        red_line_found = False
        stop_line_contours, _ = cv2.findContours(red_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in stop_line_contours:
            if cv2.contourArea(contour) > 150:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = float(cw) / ch
                if aspect_ratio > 2.2 and cw > int(w * 0.20):
                    red_line_found = True
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)
                    cv2.putText(frame, "STOP LINE", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 🟡 HORIZON SCAN WINDOW (LOOK-AHEAD ROW SLICE AT 70%-76%)
        scan_y_top = int(h * 0.70)
        scan_y_bot = int(h * 0.76)

        yellow_slice = yellow_mask[scan_y_top:scan_y_bot, :int(w * 0.55)]
        white_slice = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 50, 255]))[scan_y_top:scan_y_bot, int(w * 0.35):]

        yellow_cx = None
        y_indices = np.where(yellow_slice > 0)
        if len(y_indices[1]) > 0:
            yellow_cx = int(np.mean(y_indices[1]))

        white_cx = None
        w_indices = np.where(white_slice > 0)
        if len(w_indices[1]) > 0:
            potential_cx = int(np.mean(w_indices[1])) + int(w * 0.35)
            
            # 🔥 CRITICAL HARDENED SAFETY FILTER: 
            # White line segments detected on the left 50% of the screen are strictly ignored
            if potential_cx >= frame_center_x:
                white_cx = potential_cx

        if yellow_cx is not None:
            cv2.rectangle(frame, (yellow_cx - 15, scan_y_top), (yellow_cx + 15, scan_y_bot), (0, 255, 255), 2)
        if white_cx is not None:
            cv2.rectangle(frame, (white_cx - 15, scan_y_top), (white_cx + 15, scan_y_bot), (255, 255, 255), 2)

        # 🤖 STEERING MASTER CONTROLLER
        now = time.time()
        elapsed = now - state_start_time

        if keyboard_engaged:
            publish_drive(manual_v, manual_omega)
        else:
            if current_state == STATE_LANE_FOLLOWING:
                # Centerline Crossing Safety Rule
                if yellow_cx is not None and yellow_cx >= (frame_center_x - 20):
                    publish_drive(0.07, -1.2)  
                    cv2.putText(frame, "SAFETY ACTION: CROSSING CENTER LINE (SWERVING RIGHT)", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
                else:
                    lane_center = None
                    if yellow_cx is not None and white_cx is not None:
                        lane_center = (yellow_cx + white_cx) // 2
                        TRACK_HALF_WIDTH = abs(white_cx - yellow_cx) // 2
                    elif yellow_cx is not None:
                        lane_center = yellow_cx + TRACK_HALF_WIDTH
                    elif white_cx is not None:
                        lane_center = white_cx - TRACK_HALF_WIDTH

                    if lane_center is not None:
                        error = lane_center - frame_center_x
                        omega = - (float(error) / frame_center_x) * STEERING_MAX_YAW
                        if abs(error) > 4:
                            if omega > 0: omega = max(STEERING_TORQUE_FLOOR, omega)
                            else: omega = min(-STEERING_TORQUE_FLOOR, omega)
                        publish_drive(LANE_SPEED, omega)
                        cv2.circle(frame, (lane_center, int(h*0.73)), 6, (0, 255, 255), -1)
                        cv2.arrowedLine(frame, (frame_center_x, int(h*0.82)), (lane_center, int(h*0.73)), (0, 255, 0), 3)
                    else:
                        publish_drive(0.035, 0.0)

                # --- DUCK MANEUVERS SLASHED OUT ---
                # if duck_found and (now - last_overtake_time) > OVERTAKE_COOLDOWN:
                # ...
                
                if red_line_found and (now - last_red_line_time) > COOLDOWN_DURATION:
                    current_state = STATE_RED_STOP
                    state_start_time = now
                    print("🛑 Intersection detected. Waiting for direction command.")

            elif current_state == STATE_RED_STOP:
                publish_drive(0.0, 0.0)

            elif current_state == STATE_INTERSECTION_TURN:
                time_in_step = now - step_start_time
                
                # 🧠 CLOSED-LOOP VISION BREAKOUT ENGINE
                can_breakout = False
                if active_turn_direction == 'straight' and time_in_step > 1.2:
                    can_breakout = True
                elif active_turn_direction == 'right' and turn_step_index == 2:
                    can_breakout = True
                elif active_turn_direction == 'left' and turn_step_index == 2:
                    can_breakout = True

                if can_breakout and (yellow_cx is not None or white_cx is not None):
                    print(f"Target lines defined! Breaking out of {active_turn_direction} mode early.")
                    last_red_line_time = time.time()
                    current_state = STATE_LANE_FOLLOWING
                    return

                if turn_step_index < len(turn_sequence_active):
                    v, omega, duration = turn_sequence_active[turn_step_index]
                    if time_in_step < duration:
                        publish_drive(v, omega)
                    else:
                        turn_step_index += 1
                        step_start_time = now
                        print(f"Moving to turn phase step {turn_step_index}")
                else:
                    print("Maneuver timeout reached. Re-engaging lane tracker.")
                    last_red_line_time = time.time()
                    current_state = STATE_LANE_FOLLOWING

            elif current_state == STATE_GO_AROUND:
                if maneuver_step < len(GO_AROUND_SEQUENCE):
                    v, omega, duration = GO_AROUND_SEQUENCE[maneuver_step]
                    if (now - step_start_time) < duration:
                        publish_drive(v, omega)
                    else:
                        maneuver_step += 1
                        step_start_time = now
                else:
                    print("Overtake complete! Returning control to lane seeker.")
                    last_overtake_time = time.time()
                    current_state = STATE_LANE_FOLLOWING

        # 📺 HUD OVERLAYS
        if keyboard_engaged:
            cv2.putText(frame, "MANUAL OVERRIDE ENGAGED: KEYBOARD ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        else:
            if current_state == STATE_LANE_FOLLOWING:
                cv2.putText(frame, "STATE: AUTONOMOUS HARDENED LANE KEEPING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            elif current_state == STATE_GO_AROUND:
                cv2.putText(frame, f"STATE: EVADING BLOCK (PHASE {maneuver_step})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            elif current_state == STATE_RED_STOP:
                cv2.putText(frame, "DECIDE NEXT TURN: W (Ahead) | A (Left) | D (Right)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 150, 255), 2)
            elif current_state == STATE_INTERSECTION_TURN:
                cv2.putText(frame, f"STATE: CLOSED-LOOP {active_turn_direction.upper()} MANEUVER (STEP {turn_step_index})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 255), 2)

        # Draw the visual center boundary splitting line down the screen to highlight the 50% constraint layer
        cv2.line(frame, (frame_center_x, 0), (frame_center_x, h), (255, 0, 100), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, scan_y_top), (w, scan_y_bot), (0, 140, 255), 1)
        cv2.rectangle(frame, (0, int(h*0.45)), (w, int(h*0.92)), (0, 165, 255), 1) 
        cv2.rectangle(frame, (0, int(h*0.75)), (w, h), (0, 0, 150), 2) 

        _, jpeg = cv2.imencode('.jpg', frame)
        with lock:
            latest_jpeg = jpeg.tobytes()
    except Exception as e:
        print(f"Telemetry Execution Frame Jam: {e}")

camera_sub = roslibpy.Topic(client_ros, '/duck3/camera_node/image/compressed', 'sensor_msgs/CompressedImage', throttle_rate=100, queue_length=1)
camera_sub.subscribe(process_image)

# ==============================================================================
# 🌐 WEB INTERFACE
# ==============================================================================
@app.route('/')
def index():
    return '''
    <html>
    <body style="background:#111; color:white; text-align:center; font-family:sans-serif; margin:0; padding:20px;">
        <h2>duckie3 Integrated Telemetry & Web Control Hub</h2>
        
        <div id="status_box" style="margin-bottom: 15px; background: #d97706; color: black; padding: 12px; display: inline-block; border-radius: 5px; font-weight: bold; border: 1px solid #ff9900; font-size: 1.1em; width: 75%;">
            Autonomous Tracking Operational.
        </div>
        <br>

        <div style="margin-bottom: 15px; background: #222; padding: 10px; display: inline-block; border-radius: 5px; border: 1px solid #444;">
            <p style="margin: 5px 0; color: #aaa;"><strong>Keyboard Interlock Framework:</strong></p>
            <span style="background:#d97706; padding:2px 6px; border-radius:3px; font-weight:bold; color:black;">E</span> Toggle Manual Override Mode | 
            <span style="background:#333; padding:2px 6px; border-radius:3px;">W</span> Forward / Straight Turn | 
            <span style="background:#333; padding:2px 6px; border-radius:3px;">S</span> Reverse | 
            <span style="background:#333; padding:2px 6px; border-radius:3px;">A</span> Left Turn | 
            <span style="background:#333; padding:2px 6px; border-radius:3px;">D</span> Right Turn | 
            <span style="background:#22c55e; padding:2px 6px; border-radius:3px; color:black; font-weight:bold;">Spacebar</span> Mechanical Brake | 
            <span style="background:#800; padding:2px 6px; border-radius:3px; color:white; font-weight:bold;">Q</span> Emergency KILL
        </div>
        <br>
        <img src="/video" style="width:75%; border:4px solid #333; border-radius:4px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
        
        <script>
            setInterval(function() {
                fetch('/get_status').then(response => response.text()).then(text => {
                    let box = document.getElementById("status_box");
                    box.innerText = text;
                    if (text.includes("DECIDE") || text.includes("DETECTED")) {
                        box.style.background = "#ef4444"; 
                        box.style.color = "white";
                    } else if (text.includes("ACTIVE") || text.includes("MANUAL")) {
                        box.style.background = "#ff9900"; 
                        box.style.color = "black";
                    } else {
                        box.style.background = "#22c55e"; 
                        box.style.color = "black";
                    }
                });
            }, 300);

            document.addEventListener('keydown', function(event) {
                let key = event.key;
                let keyLower = key.toLowerCase();
                
                if (key === ' ' || key === 'Spacebar') {
                    event.preventDefault(); 
                    fetch('/control?key=space');
                } else if (['w', 'a', 's', 'd', 'e', 'q'].includes(keyLower)) {
                    fetch('/control?key=' + keyLower);
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/get_status')
def get_status():
    if keyboard_engaged:
        return "MANUAL MODE ACTIVE: Driving wheels directly via web interface input switches."
    if current_state == STATE_RED_STOP:
        return "INTERSECTION DETECTED: Press W (Straight), A (Left), or D (Right) to select cross street direction!"
    if current_state == STATE_INTERSECTION_TURN:
        return f"EXECUTING INTERSECTION TURN: Turning {active_turn_direction.upper()} into right lane..."
    if current_state == STATE_GO_AROUND:
        return "EVADING TRACK OBSTACLE: Executing go-around maneuver..."
    return "AUTONOMOUS NAVIGATION ACTIVE: Centering smoothly between track tape lines."

@app.route('/control')
def control():
    global keyboard_engaged, manual_v, manual_omega, current_state
    global turn_sequence_active, turn_step_index, step_start_time, active_turn_direction
    key = request.args.get('key', '').lower()
    
    if key == 'q':
        print("\n🛑 EMERGENCY SHUTDOWN IMMINENT — HALTING ACTUATORS")
        for _ in range(5):
            override_pub.publish(roslibpy.Message({'header': {'stamp': {'secs':0,'nsecs':0}, 'frame_id':''}, 'data': True}))
            cmd_pub.publish(roslibpy.Message({'header': {'stamp': {'secs':0,'nsecs':0}, 'frame_id':''}, 'v': 0.0, 'omega': 0.0}))
            time.sleep(0.02)
        os._exit(0)
        
    elif key == 'e':
        keyboard_engaged = not keyboard_engaged
        manual_v = 0.0; manual_omega = 0.0
        if not keyboard_engaged:
            current_state = STATE_LANE_FOLLOWING
        print(f"\n🔄 MODE SWITCH -> Manual Keyboard Mode Active: {keyboard_engaged}")
        return "Mode Toggled"

    if current_state == STATE_RED_STOP:
        if key == 'w':
            print("Route Decision Captured: Processing STRAIGHT crossover route steps.")
            active_turn_direction = 'straight'
            turn_sequence_active = INTERSECTION_STRAIGHT_STEPS
            turn_step_index = 0; step_start_time = time.time()
            current_state = STATE_INTERSECTION_TURN
        elif key == 'a':
            print("Route Decision Captured: Processing LEFT closed-loop intersection steps.")
            active_turn_direction = 'left'
            turn_sequence_active = INTERSECTION_LEFT_STEPS
            turn_step_index = 0; step_start_time = time.time()
            current_state = STATE_INTERSECTION_TURN
        elif key == 'd':
            print("Route Decision Captured: Processing RIGHT closed-loop intersection steps.")
            active_turn_direction = 'right'
            turn_sequence_active = INTERSECTION_RIGHT_STEPS
            turn_step_index = 0; step_start_time = time.time()
            current_state = STATE_INTERSECTION_TURN
        return "Route Selected"

    elif keyboard_engaged:
        if key == 'w':
            manual_v = 0.15; manual_omega = 0.0
        elif key == 's':
            manual_v = -0.15; manual_omega = 0.0
        elif key == 'a':
            manual_v = 0.0; manual_omega = 1.0
        elif key == 'd':
            manual_v = 0.0; manual_omega = -1.0
        elif key == 'space' or key == ' ':
            manual_v = 0.0; manual_omega = 0.0
            
    return "Command Processed"

@app.route('/video')
def video():
    def generate():
        while True:
            with lock: frame = latest_jpeg
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

print("Autonomous Navigation Hub Active! Open control panel at: http://localhost:5000")
app.run(host='0.0.0.0', port=5000, threaded=True)
