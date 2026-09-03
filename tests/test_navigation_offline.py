import importlib.util
import os
import sys
import types
import unittest


class FakeMask:
    shape = (480, 640)

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        pass

    def copy(self):
        return self

    def __gt__(self, other):
        return self


class FakeFrame(FakeMask):
    shape = (480, 640, 3)
    size = 480 * 640 * 3


class FakeApp:
    def __init__(self, *args, **kwargs):
        self.routes = {}

    def route(self, path):
        def wrap(fn):
            self.routes[path] = fn
            return fn
        return wrap

    def run(self, *args, **kwargs):
        pass


class FakeRos:
    is_connected = True

    def __init__(self, *args, **kwargs):
        pass

    def run(self, timeout=None):
        pass


class FakeTopic:
    def __init__(self, *args, **kwargs):
        pass

    def publish(self, *args, **kwargs):
        pass

    def subscribe(self, *args, **kwargs):
        pass


class FakeCV2(types.SimpleNamespace):
    COLOR_BGR2HSV = 40
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 0
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def cvtColor(self, frame, code):
        return frame

    def inRange(self, hsv, low, high):
        return FakeMask()

    def bitwise_or(self, a, b):
        return FakeMask()

    def findContours(self, image, mode, method):
        return [], None

    def moments(self, mask):
        return {"m00": 0}

    def contourArea(self, contour):
        return 0

    def boundingRect(self, contour):
        return (0, 0, 0, 0)

    def rectangle(self, *args, **kwargs):
        pass

    def putText(self, *args, **kwargs):
        pass

    def circle(self, *args, **kwargs):
        pass

    def arrowedLine(self, *args, **kwargs):
        pass

    def drawContours(self, *args, **kwargs):
        pass

    def imencode(self, ext, image):
        return False, b""


def load_navigation():
    os.system = lambda *args, **kwargs: 0
    sys.modules["flask"] = types.SimpleNamespace(
        Flask=FakeApp,
        Response=lambda body, **kwargs: body,
        jsonify=lambda *args, **kwargs: dict(args[0] if args else {}, **kwargs),
        request=types.SimpleNamespace(args={}),
    )
    sys.modules["roslibpy"] = types.SimpleNamespace(
        Ros=FakeRos,
        Topic=FakeTopic,
        Message=lambda msg: msg,
    )
    sys.modules["cv2"] = FakeCV2()
    sys.modules["numpy"] = types.SimpleNamespace(
        array=lambda value: value,
        zeros_like=lambda value: FakeMask(),
        zeros=lambda shape, dtype=None: FakeFrame(),
        clip=lambda value, lo, hi: max(lo, min(hi, value)),
        uint8="uint8",
    )
    sys.modules["yaml"] = types.SimpleNamespace(
        safe_load=lambda f: (_ for _ in ()).throw(Exception("force fallback map"))
    )
    spec = importlib.util.spec_from_file_location(
        "navigate", "packages/navigation/src/navigate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NavigationOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nav = load_navigation()

    def setUp(self):
        self.commands = []
        self.nav.publish_drive = lambda v, omega: self.commands.append((v, omega))
        self.nav.SETUP_COMPLETE = True
        self.nav.keyboard_engaged = False
        self.nav.AUTO_PATH_MODE = True
        self.nav.post_intersections_tracking = False
        self.nav.run_metrics = None

    def test_wrong_start_heading_emits_explicit_uturn_action(self):
        nav = self.nav
        nav.bot_start_tile = (0, 5)
        nav.ROUTE_GOAL = (5, 5)
        nav.bot_heading = "N"
        ok, _ = nav.compute_route()
        self.assertTrue(ok)
        self.assertEqual(nav.ROUTE_TURN_TILES[(0, 5)], "uturn")
        self.assertEqual(nav.ROUTE_INTERSECTION_ORDER[0], (0, 5))
        self.assertEqual(nav._path_first_heading(nav.ROUTE), "S")

    def test_uturn_exits_on_lane_geometry_before_timeout(self):
        nav = self.nav
        nav.current_state = nav.STATE_UTURN
        nav.state_start_time = nav.time.time()
        nav.centroid_x = lambda mask, x_lo=None, x_hi=None: 250 if x_hi is not None else 390
        nav.process_image_frame(FakeFrame())
        self.assertEqual(nav.current_state, nav.STATE_LANE_FOLLOWING)
        self.assertIn((0.0, 0.0), self.commands)

    def test_uturn_timeout_stops_and_returns_to_lane_following(self):
        nav = self.nav
        nav.current_state = nav.STATE_UTURN
        nav.state_start_time = nav.time.time() - nav.UTURN_TIMEOUT_S - 0.1
        nav.centroid_x = lambda mask, x_lo=None, x_hi=None: None
        nav.process_image_frame(FakeFrame())
        self.assertEqual(nav.current_state, nav.STATE_LANE_FOLLOWING)
        self.assertIn((0.0, 0.0), self.commands)

    def test_penalty_routes_differ_on_fixture(self):
        nav = self.nav
        result = nav.compare_routes((0, 5), (5, 5), "N")
        self.assertNotEqual(result["penalty_on"]["path"], result["penalty_off"]["path"])
        self.assertEqual(result["penalty_off"]["tile_count"], 7)
        self.assertEqual(result["penalty_on"]["tile_count"], 7)
        self.assertEqual(result["penalty_off"]["intersection_count"], 3)
        self.assertEqual(result["penalty_on"]["intersection_count"], 1)

    def test_compare_route_endpoint_returns_both_routes(self):
        nav = self.nav
        nav.request.args = {"start": "0,5", "goal": "5,5", "heading": "N"}
        result = nav.compare_route_endpoint()
        self.assertTrue(result["ok"])
        self.assertIn("penalty_on", result)
        self.assertIn("penalty_off", result)
        self.assertEqual(result["penalty_on"]["intersection_count"], 1)
        self.assertEqual(result["penalty_off"]["intersection_count"], 3)

    def test_documented_route_fixtures(self):
        nav = self.nav
        cases = [
            ((0, 5), (5, 5), "N", (7, 3), (7, 1)),
            ((0, 2), (5, 2), "S", (9, 3), (9, 1)),
            ((0, 4), (5, 2), "E", (7, 3), (11, 2)),
        ]
        for start, goal, heading, off_expected, on_expected in cases:
            with self.subTest(start=start, goal=goal):
                result = nav.compare_routes(start, goal, heading)
                self.assertEqual(
                    (result["penalty_off"]["tile_count"], result["penalty_off"]["intersection_count"]),
                    off_expected,
                )
                self.assertEqual(
                    (result["penalty_on"]["tile_count"], result["penalty_on"]["intersection_count"]),
                    on_expected,
                )


if __name__ == "__main__":
    unittest.main()
