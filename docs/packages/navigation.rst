Python Package: navigation
===========================

.. contents::

Autonomous navigation package for the Duckietown ``duck3`` robot. Runs on the
laptop and drives the robot over roslibpy/rosbridge - no custom code is
deployed to the robot itself.

Modules
-------

``navigate.py``
    Main entry point. Runs the state machine: lane following (OpenCV yellow
    and white line detection), duck detection and avoidance, red stop line
    detection, and A* planned intersection turns. Includes a built-in
    weighted A* search (``astar_search``) over the TUM track's tile graph.
    Serves a live camera stream with detection overlays at
    ``http://localhost:5000``.

``tum_map.yaml``
    Map of the TUM lab track as a 7x6 tile grid, used by ``astar_search``.

``keyboard_drive.py``
    Manual keyboard teleop for the robot (forward/back/turn/stop), used for
    testing outside of autonomous runs.

State machine
-------------

- ``LANE_FOLLOWING`` - normal driving using lane detection
- ``DUCK_STOP`` - stopped because a duck is blocking the path
- ``GO_AROUND`` - executing a maneuver around a duck
- ``RED_STOP`` - stopped at a red stop line for 3 seconds
- ``INTERSECTION`` - executing the A*-planned turn

Dependencies
------------

- roslibpy
- opencv-python
- flask
- numpy
- pyyaml
