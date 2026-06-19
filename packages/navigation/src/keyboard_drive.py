import roslibpy
import sys
import tty
import termios
import time
import threading

client = roslibpy.Ros(host='localhost', port=9001)
client.run()
print("Connected! Hold W/A/S/D to drive, Space to stop, Q to quit")

publisher = roslibpy.Topic(client, '/duck3/wheels_driver_node/wheels_cmd', 'duckietown_msgs/WheelsCmdStamped')
current_left = 0.0
current_right = 0.0
running = True

def send_cmd(left, right):
    publisher.publish(roslibpy.Message({
        'header': {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
        'vel_left': left,
        'vel_right': right
    }))

def publish_loop():
    while running:
        send_cmd(current_left, current_right)
        time.sleep(0.1)

thread = threading.Thread(target=publish_loop)
thread.start()

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

while True:
    key = get_key()
    if key == 'w':
        current_left = 0.3
        current_right = 0.3
        print("Forward")
    elif key == 's':
        current_left = -0.3
        current_right = -0.3
        print("Backward")
    elif key == 'a':
        current_left = -0.2
        current_right = 0.2
        print("Left")
    elif key == 'd':
        current_left = 0.2
        current_right = -0.2
        print("Right")
    elif key == ' ':
        current_left = 0.0
        current_right = 0.0
        print("Stop")
    elif key == 'q':
        running = False
        send_cmd(0.0, 0.0)
        client.terminate()
        break
