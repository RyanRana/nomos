"""Generate a 3D San Francisco MJCF scene for the Antim/HUD worldsim (Newton).

Reads our existing OSMnx road network (the same graph the kinematic env trains on)
and emits `scene.xml` + `metadata.json`: a flat ground, every road segment as a
thin box ribbon, optional extruded building boxes, and N Ackermann cars (the
car-v1 body, replicated with per-car name prefixes). Drop the output folder under
the worldsim-template `scenes/` dir and it becomes a live env: reset(scene_id),
then drive every car through the generic step(action) tool.

ACTION CONTRACT (concatenated over cars, in spawn order):
    action = [car0_drive_rl, car0_drive_rr, car0_steer_l, car0_steer_r,
              car1_drive_rl, ...]                       # 4 per car
This is exactly what control_bridge.MultiCarController emits.

Usage:
  python -m smoothride.worldsim.build_sf_scene --cars 12 --out scenes/sf-city-v1
  python -m smoothride.worldsim.build_sf_scene --cars 30 --buildings \
      --out /path/to/worldsim-template/scenes/sf-city-v1
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from ..data.map_loader import load_road_network
from ..env.routing import build_route_pool

ROUTE_N = 1024
ROUTE_MAX_M = 1500.0

# contype=2/conaffinity=1: car geoms collide with the FLOOR (contype/conaffinity 1)
# but NOT each other -> cars keep real driving dynamics yet flow through one another
# instead of jamming, so the street can be packed dense (the controller has no
# car-to-car avoidance). (a&b: floor-car 1&1=1 collide; car-car 2&1=0 skip.)
CAR_CT = 'contype="2" conaffinity="1"'
WHEEL = ('<geom type="cylinder" size="0.33 0.12" euler="1.5708 0 0" rgba="0.1 0.1 0.12 1" '
         f'friction="1.6 0.02 0.001" mass="15" {CAR_CT}/>')


def _car_body(i: int, x: float, y: float, yaw: float, rgba: str) -> str:
    """One car body, names prefixed cN_ so many can coexist in one model."""
    p = f"c{i}_"
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)  # quat about +z
    # a street-level chase camera rides car 0 so the render actually shows cars
    chase = ('\n      <camera name="chase" pos="-13 0 5.5" xyaxes="0 -1 0 0.32 0 1" '
             'mode="trackcom"/>' if i == 0 else "")
    return f"""
    <body name="{p}chassis" pos="{x:.2f} {y:.2f} 0.43" quat="{qw:.4f} 0 0 {qz:.4f}">
      <freejoint name="{p}chassis_joint"/>
      <geom name="{p}chassis_geom" type="box" size="2.1 0.85 0.22" mass="1100" rgba="{rgba}" {CAR_CT}/>
      <geom name="{p}cabin_geom" type="box" size="1.1 0.78 0.25" pos="-0.2 0 0.42" mass="120" rgba="0.12 0.14 0.18 1" {CAR_CT}/>{chase}
      <body name="{p}steer_fl" pos="1.35 0.85 -0.1">
        <inertial pos="0 0 0" mass="2" diaginertia="0.05 0.05 0.05"/>
        <joint name="{p}steer_l" type="hinge" axis="0 0 1" range="-0.7 0.7"/>
        <body name="{p}wheel_fl"><joint name="{p}roll_fl" type="hinge" axis="0 1 0"/>{WHEEL}</body>
      </body>
      <body name="{p}steer_fr" pos="1.35 -0.85 -0.1">
        <inertial pos="0 0 0" mass="2" diaginertia="0.05 0.05 0.05"/>
        <joint name="{p}steer_r" type="hinge" axis="0 0 1" range="-0.7 0.7"/>
        <body name="{p}wheel_fr"><joint name="{p}roll_fr" type="hinge" axis="0 1 0"/>{WHEEL}</body>
      </body>
      <body name="{p}wheel_rl" pos="-1.35 0.85 -0.1"><joint name="{p}roll_rl" type="hinge" axis="0 1 0"/>{WHEEL}</body>
      <body name="{p}wheel_rr" pos="-1.35 -0.85 -0.1"><joint name="{p}roll_rr" type="hinge" axis="0 1 0"/>{WHEEL}</body>
    </body>"""


def _car_actuators(i: int) -> str:
    p = f"c{i}_"
    return (f'    <velocity name="{p}drive_rl" joint="{p}roll_rl" kv="120" ctrlrange="-40 80"/>\n'
            f'    <velocity name="{p}drive_rr" joint="{p}roll_rr" kv="120" ctrlrange="-40 80"/>\n'
            f'    <position name="{p}steer_l" joint="{p}steer_l" kp="2000" ctrlrange="-0.7 0.7"/>\n'
            f'    <position name="{p}steer_r" joint="{p}steer_r" kp="2000" ctrlrange="-0.7 0.7"/>\n')


def _car_sensors(i: int) -> str:
    p = f"c{i}_"
    return (f'    <framepos name="{p}pos" objtype="body" objname="{p}chassis"/>\n'
            f'    <framequat name="{p}quat" objtype="body" objname="{p}chassis"/>\n'
            f'    <framelinvel name="{p}linvel" objtype="body" objname="{p}chassis"/>\n')


def _road_geoms(net, cx, cy) -> str:
    """Each directed edge -> a thin asphalt box, recentred to scene origin."""
    xy = net.node_xy
    out = []
    seen = set()
    for u, v in net.edges:
        key = (min(u, v), max(u, v))           # one ribbon per undirected segment
        if key in seen:
            continue
        seen.add(key)
        x0, y0 = xy[u] - (cx, cy)
        x1, y1 = xy[v] - (cx, cy)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1.0:
            continue
        ang = math.atan2(dy, dx)
        # visual-only (contype/conaffinity=0): a road decal cars roll OVER on the
        # flat ground plane — raised collidable ribbons eject the wheels.
        out.append(f'    <geom type="box" size="{length/2:.2f} 4.0 0.02" '
                   f'pos="{mx:.2f} {my:.2f} 0.04" euler="0 0 {ang:.4f}" '
                   f'material="road" contype="0" conaffinity="0"/>')
    return "\n".join(out)


def _building_geoms(net, cx, cy, bbox) -> str:
    """Optional: OSM building footprints as extruded boxes. Network fetch."""
    try:
        import osmnx as ox
    except ImportError:
        return ""
    try:
        gdf = ox.features_from_bbox(bbox, tags={"building": True})
        # drop null/invalid/non-polygon rows BEFORE projecting (avoids NaN-area crash)
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        gdf = ox.projection.project_gdf(gdf, to_crs=net.G.graph["crs"])
    except Exception as e:  # offline / no buildings / API hiccup
        print(f"  [buildings] skipped: {e}")
        return ""
    out, n = [], 0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
            continue                            # skip NaN/inf geometries
        w, h = (maxx - minx) / 2, (maxy - miny) / 2
        if not (2 < w < 80 and 2 < h < 80):     # skip slivers / mega-polys
            continue
        bx = (minx + maxx) / 2 - net.origin[0] - cx
        by = (miny + maxy) / 2 - net.origin[1] - cy
        ht = 8 + 22 * (n % 5) / 4.0             # vary heights for a skyline look
        out.append(f'    <geom type="box" size="{w:.1f} {h:.1f} {ht/2:.1f}" '
                   f'pos="{bx:.1f} {by:.1f} {ht/2:.1f}" material="bldg"/>')
        n += 1
    print(f"  [buildings] {n} footprints")
    return "\n".join(out)


def build(cars: int, out: str, seed: int, buildings: bool):
    net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2       # recenter scene on the map middle
    half_x, half_y = (x1 - x0) / 2 + 30, (y1 - y0) / 2 + 30

    rng = np.random.default_rng(seed)
    palette = ["0.20 0.55 0.85 1", "0.85 0.30 0.30 1", "0.30 0.75 0.45 1",
               "0.90 0.70 0.20 1", "0.65 0.40 0.85 1"]

    # spawn each car at the START of a real route, heading along it — so the
    # RoutePlanner (which rebuilds the SAME pool from the seed below) finds the
    # car already on its route and advances waypoints from step 1.
    pool = build_route_pool(net, n_routes=ROUTE_N, max_length_m=ROUTE_MAX_M, seed=seed)
    car_routes = [int(r) for r in rng.integers(0, pool.n_routes, size=cars)]
    car_bodies, car_acts, car_sens = [], [], []
    for i, r in enumerate(car_routes):
        p0 = pool.xy[r, 0] - (cx, cy)
        p1 = pool.xy[r, 1] - (cx, cy)
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        car_bodies.append(_car_body(i, float(p0[0]), float(p0[1]), yaw,
                                    palette[i % len(palette)]))
        car_acts.append(_car_actuators(i))
        car_sens.append(_car_sensors(i))

    bldg = _building_geoms(net, cx, cy, (x0, y0, x1, y1)) if buildings else ""

    xml = f"""<mujoco model="sf-city-v1">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.004" integrator="implicitfast"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <default>
    <joint damping="0.2"/>
    <geom condim="4" friction="1.3 0.02 0.001"/>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.65 0.80" rgb2="0.25 0.30 0.40" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.42 0.30" rgb2="0.26 0.38 0.26" width="256" height="256"/>
    <material name="ground" texture="grid" texrepeat="80 80"/>
    <material name="road" rgba="0.18 0.18 0.20 1" reflectance="0.02"/>
    <material name="bldg" rgba="0.55 0.57 0.62 1" specular="0.3" shininess="0.3"/>
  </asset>
  <worldbody>
    <light name="sun" pos="{half_x:.0f} {-half_y:.0f} 400" dir="-0.3 0.3 -1" diffuse="0.85 0.85 0.85"/>
    <geom name="floor" type="plane" size="{half_x:.0f} {half_y:.0f} 0.1" material="ground"/>
    <camera name="drone" pos="0 0 {max(half_x, half_y)*1.4:.0f}" xyaxes="1 0 0 0 1 0"/>
    <camera name="oblique" pos="{-half_x:.0f} {-half_y:.0f} {max(half_x,half_y):.0f}" xyaxes="1 -1 0 0.5 0.5 1"/>

    <!-- ROADS -->
{_road_geoms(net, cx, cy)}

    <!-- BUILDINGS -->
{bldg}

    <!-- CARS ({len(car_bodies)}) -->
{''.join(car_bodies)}
  </worldbody>

  <actuator>
{''.join(car_acts)}  </actuator>

  <sensor>
{''.join(car_sens)}  </sensor>
</mujoco>
"""

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "scene.xml"), "w") as f:
        f.write(xml)

    meta = {
        "scene_id": os.path.basename(out.rstrip("/")),
        "description": f"3D San Francisco (downtown OSM graph) with {len(car_bodies)} "
                       f"Ackermann cars. Generated from the SmoothRide road network. "
                       f"Drive via step(action), 4 actuators per car.",
        "source": "generated:smoothride.worldsim.build_sf_scene",
        "format": "mjcf", "engine": "newton",
        "n_cars": len(car_bodies),
        "control": {
            "tool": "step", "action_dim": 4 * len(car_bodies),
            "per_car": 4, "per_car_layout": "[drive_rl, drive_rr, steer_l, steer_r]",
            "car_order": "spawn order; car i actuators at offset 4*i",
            "wheel_radius_m": 0.33, "wheelbase_m": 2.7,
        },
        "cars": [{"body": f"c{i}_chassis", "pos_sensor": f"c{i}_pos"}
                 for i in range(len(car_bodies))],
        "routing": {"route_seed": seed, "n_routes": ROUTE_N,
                    "max_length_m": ROUTE_MAX_M, "car_routes": car_routes},
        "cameras": ["drone", "oblique"],
        "map": {"center_utm_minus_origin": [cx, cy], "origin_utm": list(net.origin),
                "crs": str(net.G.graph["crs"]),
                "note": "scene meters are UTM-origin-shifted, recentred on map middle"},
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {out}/scene.xml + metadata.json  "
          f"(cars={len(car_bodies)}, action_dim={4*len(car_bodies)}, "
          f"buildings={'on' if buildings else 'off'})")
    return os.path.join(out, "scene.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buildings", action="store_true", help="fetch + extrude OSM building footprints")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "scenes", "sf-city-v1"))
    args = ap.parse_args()
    build(args.cars, args.out, args.seed, args.buildings)


if __name__ == "__main__":
    main()
