"""A* pathfinding for Duckietown maps.

Parses a Duckietown map YAML (as shipped in duckietown_world/data/gd1/maps/)
into a tile graph, runs A* between two tile coordinates, and converts the
tile sequence into a list of turn decisions (straight/left/right) at each
intersection tile.

Standalone module. No dependency on gym_duckietown or a running simulator.

CLI usage:
    python a_star_planner.py udem1 --start 1,5 --goal 6,1

Library usage:
    from a_star_planner import plan, is_intersection
    res = plan("udem1", start=(1, 5), goal=(6, 1))
    for t in res.turns:
        print(t.tile, t.kind, t.in_dir, "->", t.out_dir, t.turn)

Tile coordinates are (i, j):
    i increases East  (grid column, maps to world +x)
    j increases South (grid row,    maps to world +z)
That matches gym_duckietown.Simulator.get_grid_coords and user_tile_start.
"""

from __future__ import annotations

import argparse
import heapq
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import yaml

# ---------------------------------------------------------------------------
# Compass primitives
# ---------------------------------------------------------------------------

# Grid deltas per compass direction (i, j).
DELTAS: Dict[str, Tuple[int, int]] = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

# 90 degree rotations in the compass.
LEFT_OF = {"N": "W", "W": "S", "S": "E", "E": "N"}   # CCW
RIGHT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}  # CW

# ---------------------------------------------------------------------------
# Tile openings
# ---------------------------------------------------------------------------
# Which compass sides each (kind/orient) tile is open on. Derived from the
# Duckietown tile convention and cross-checked tile-by-tile against udem1.
#
#   straight/X   : pass-through (two opposite sides open)
#   curve_left/X : enter facing X, exit 90 degrees LEFT
#   curve_right/X: enter facing X, exit 90 degrees RIGHT
#   3way_left/X  : T-intersection; closed side is RIGHT_OF X
#   4way         : all four sides open

OPENINGS: Dict[str, Set[str]] = {
    "straight/N": {"N", "S"}, "straight/S": {"N", "S"},
    "straight/E": {"E", "W"}, "straight/W": {"E", "W"},

    "curve_left/N": {"S", "W"},
    "curve_left/E": {"W", "N"},
    "curve_left/S": {"N", "E"},
    "curve_left/W": {"E", "S"},

    "curve_right/N": {"S", "E"},
    "curve_right/E": {"W", "S"},
    "curve_right/S": {"N", "W"},
    "curve_right/W": {"E", "N"},

    "3way_left/N": {"N", "S", "W"},
    "3way_left/E": {"N", "E", "W"},
    "3way_left/S": {"N", "S", "E"},
    "3way_left/W": {"S", "E", "W"},

    "4way": {"N", "S", "E", "W"},
}

NON_DRIVABLE = {"floor", "grass", "asphalt", "empty", ""}


def openings_of(kind: str) -> Set[str]:
    return OPENINGS.get(kind, set())


def is_drivable(kind: str) -> bool:
    return kind in OPENINGS


def is_intersection(kind: str) -> bool:
    return kind.startswith("3way_") or kind == "4way"


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

@dataclass
class DuckieMap:
    tiles: List[List[str]]   # tiles[j][i] -> tile kind string
    tile_size: float         # meters per tile
    width: int
    height: int

    def tile(self, i: int, j: int) -> str:
        if 0 <= i < self.width and 0 <= j < self.height:
            row = self.tiles[j]
            if 0 <= i < len(row):
                return row[i]
        return "floor"


def _resolve_map_path(name_or_path: str) -> str:
    """Accept either an absolute .yaml path or a bare map name."""
    if os.path.isfile(name_or_path):
        return name_or_path
    try:
        import duckietown_world as dw  # lazy import; optional at library time
        base = os.path.dirname(dw.__file__)
        fname = name_or_path if name_or_path.endswith(".yaml") else f"{name_or_path}.yaml"
        candidate = os.path.join(base, "data", "gd1", "maps", fname)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    raise FileNotFoundError(
        f"Could not locate map '{name_or_path}' (not a file, and not found "
        f"in duckietown_world/data/gd1/maps/)"
    )


def load_map(name_or_path: str) -> DuckieMap:
    path = _resolve_map_path(name_or_path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    raw = data["tiles"]
    tiles = [[str(cell).strip() for cell in row] for row in raw]
    height = len(tiles)
    width = max((len(r) for r in tiles), default=0)
    tile_size = float(data.get("tile_size", 0.585))
    return DuckieMap(tiles=tiles, tile_size=tile_size, width=width, height=height)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

Tile = Tuple[int, int]  # (i, j)


def build_graph(m: DuckieMap) -> Dict[Tile, List[Tile]]:
    """Adjacency list of drivable tile connections.

    (i, j) and (i', j') are connected iff each tile has an opening toward
    the other. This rejects adjacency across asphalt/grass/floor and across
    curve backs / T-stems automatically.
    """
    adj: Dict[Tile, List[Tile]] = {}
    for j in range(m.height):
        for i in range(m.width):
            k = m.tile(i, j)
            if not is_drivable(k):
                continue
            node = (i, j)
            adj.setdefault(node, [])
            for d in openings_of(k):
                di, dj = DELTAS[d]
                ni, nj = i + di, j + dj
                nk = m.tile(ni, nj)
                if not is_drivable(nk):
                    continue
                if OPPOSITE[d] in openings_of(nk):
                    adj[node].append((ni, nj))
    return adj


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _PQItem:
    f: float
    g: float = field(compare=False)
    node: Tile = field(compare=False)


def _heuristic(a: Tile, b: Tile) -> float:
    # Manhattan distance in tile units (admissible for 4-connected grids).
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(adj: Dict[Tile, List[Tile]], start: Tile, goal: Tile) -> List[Tile]:
    """Return the shortest path of tile coordinates from start to goal.

    Raises ValueError if start/goal are not in the graph or no path exists.
    """
    if start not in adj:
        raise ValueError(f"start {start} is not a drivable tile")
    if goal not in adj:
        raise ValueError(f"goal {goal} is not a drivable tile")
    if start == goal:
        return [start]

    open_heap: List[_PQItem] = []
    heapq.heappush(open_heap, _PQItem(_heuristic(start, goal), 0.0, start))
    came_from: Dict[Tile, Tile] = {}
    best_g: Dict[Tile, float] = {start: 0.0}

    while open_heap:
        cur = heapq.heappop(open_heap)
        n = cur.node
        if cur.g > best_g[n]:
            continue  # stale heap entry
        if n == goal:
            path = [n]
            while n in came_from:
                n = came_from[n]
                path.append(n)
            path.reverse()
            return path
        for nb in adj[n]:
            tentative_g = cur.g + 1.0
            if tentative_g < best_g.get(nb, float("inf")):
                best_g[nb] = tentative_g
                came_from[nb] = n
                f = tentative_g + _heuristic(nb, goal)
                heapq.heappush(open_heap, _PQItem(f, tentative_g, nb))

    raise ValueError(f"No path from {start} to {goal}")


# ---------------------------------------------------------------------------
# Turn inference
# ---------------------------------------------------------------------------

def step_dir(a: Tile, b: Tile) -> str:
    """Compass direction of the step from a to b (must be 4-connected)."""
    di, dj = b[0] - a[0], b[1] - a[1]
    for d, (x, y) in DELTAS.items():
        if (di, dj) == (x, y):
            return d
    raise ValueError(f"Tiles {a} and {b} are not 4-connected neighbors")


def relative_turn(incoming: str, outgoing: str) -> str:
    if incoming == outgoing:
        return "straight"
    if LEFT_OF[incoming] == outgoing:
        return "left"
    if RIGHT_OF[incoming] == outgoing:
        return "right"
    return "uturn"


@dataclass
class TurnDecision:
    tile: Tile
    kind: str
    in_dir: str
    out_dir: str
    turn: str  # 'straight' | 'left' | 'right' | 'uturn'


def path_to_turns(path: List[Tile], m: DuckieMap) -> List[TurnDecision]:
    """For each interior tile of the path, record the in/out directions and
    the relative turn classification. The first and last tiles are skipped
    (they have no incoming / outgoing edge respectively)."""
    out: List[TurnDecision] = []
    for k in range(1, len(path) - 1):
        prev, cur, nxt = path[k - 1], path[k], path[k + 1]
        in_d = step_dir(prev, cur)
        out_d = step_dir(cur, nxt)
        out.append(TurnDecision(
            tile=cur,
            kind=m.tile(*cur),
            in_dir=in_d,
            out_dir=out_d,
            turn=relative_turn(in_d, out_d),
        ))
    return out


# ---------------------------------------------------------------------------
# Top-level plan()
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    map: DuckieMap
    start: Tile
    goal: Tile
    path: List[Tile]
    turns: List[TurnDecision]
    turn_by_tile: Dict[Tile, TurnDecision]  # intersection tile -> decision

    def first_out_dir(self) -> Optional[str]:
        """Compass direction the robot should face at the start tile."""
        if len(self.path) < 2:
            return None
        return step_dir(self.path[0], self.path[1])


def plan(map_name_or_path: str, start: Tile, goal: Tile) -> PlanResult:
    m = load_map(map_name_or_path)
    adj = build_graph(m)
    path = astar(adj, start, goal)
    turns = path_to_turns(path, m)
    by_tile = {t.tile: t for t in turns if is_intersection(t.kind)}
    return PlanResult(
        map=m, start=start, goal=goal,
        path=path, turns=turns, turn_by_tile=by_tile,
    )


# ---------------------------------------------------------------------------
# ASCII visualization
# ---------------------------------------------------------------------------

_TURN_ARROW = {"straight": "|", "left": "L", "right": "R", "uturn": "U"}


def render_plan_ascii(res: PlanResult) -> str:
    m = res.map
    path_set = set(res.path)
    turn_map = {t.tile: t for t in res.turns}
    lines: List[str] = []
    for j in range(m.height):
        row_cells: List[str] = []
        for i in range(m.width):
            k = m.tile(i, j)
            if (i, j) == res.start:
                cell = " S "
            elif (i, j) == res.goal:
                cell = " G "
            elif (i, j) in turn_map and is_intersection(turn_map[(i, j)].kind):
                cell = f" {_TURN_ARROW[turn_map[(i, j)].turn]} "
            elif (i, j) in path_set:
                cell = " * "
            elif is_drivable(k):
                cell = " . "
            else:
                cell = "   "
            row_cells.append(cell)
        lines.append("".join(row_cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tile(s: str) -> Tile:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"tile must be 'i,j', got {s!r}")
    return int(parts[0]), int(parts[1])


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Duckietown A* planner")
    ap.add_argument("map", help="map name (e.g. udem1) or path to .yaml")
    ap.add_argument("--start", required=True, type=_parse_tile, help="start tile 'i,j'")
    ap.add_argument("--goal", required=True, type=_parse_tile, help="goal tile 'i,j'")
    args = ap.parse_args(argv)

    res = plan(args.map, args.start, args.goal)
    m = res.map
    dist_m = (len(res.path) - 1) * m.tile_size

    print(f"Map:        {args.map}  ({m.width}x{m.height}, tile_size={m.tile_size} m)")
    print(f"Start:      {res.start}  kind={m.tile(*res.start)}")
    print(f"Goal:       {res.goal}  kind={m.tile(*res.goal)}")
    print(f"Path:       {len(res.path)} tiles  (~{dist_m:.2f} m)")
    print(f"First move: face {res.first_out_dir()}")
    print()
    print("Tile-by-tile:")
    print(f"  --  {res.path[0]}  START")
    for k, t in enumerate(res.turns, start=1):
        flag = "[I]" if is_intersection(t.kind) else "   "
        print(f"  {k:2d}. {flag} {t.tile}  {t.kind:<16s}  "
              f"{t.in_dir}->{t.out_dir}  {t.turn}")
    print(f"  --  {res.path[-1]}  GOAL")
    print()
    print("Map overview (S=start, G=goal, *=path, L/R/|=turns at intersections):")
    print(render_plan_ascii(res))


if __name__ == "__main__":
    main()
