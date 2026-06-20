"""Procedurally build a small **fleet** of car meshes with trimesh and export them
for both ends of the pipeline (Cesium glTF + MuJoCo/Newton OBJ).

Three body types (sedan / suv / coupe), each a two-tone mesh — painted body +
dark **glass** greenhouse + black wheels — plus a shared color PALETTE so the
fleet looks varied in the 3D view.

Per body type we export:
  * `<body>.glb`            -> Cesium glTF model (body+glass+wheels, face colors)
  * `<body>_body.obj`       -> MuJoCo visual geom (painted shell, recolor per car)
  * `<body>_glass.obj`      -> MuJoCo visual geom (glass greenhouse)
  * `<body>_hull.obj`       -> convex hull collider proxy
plus `palette.json` (names→rgb + body list) for the renderers.

FRAME CONTRACT (shared with the physics scene + 2D state):
  local +X = forward, +Y = left, +Z = up.  heading = atan2(dy,dx); 0 -> +X (east).
  Centered on the chassis origin, same frame as the car-v* box collider.

Run:  python -m smoothride.worldsim.assets.car_mesh
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh

# --------------------------------------------------------------------------
# Body types: each is a longitudinal profile (x, roof_top_z, half_width) tail->nose
# + cabin/glass params + wheels. Tuned so the three read as distinct silhouettes.
# --------------------------------------------------------------------------
_SEDAN = np.array([
    [-2.10, 0.12, 0.55], [-1.85, 0.17, 0.74], [-1.55, 0.18, 0.83],
    [-1.10, 0.40, 0.85], [-0.70, 0.66, 0.84], [ 0.20, 0.68, 0.84],
    [ 0.70, 0.64, 0.83], [ 1.15, 0.34, 0.82], [ 1.55, 0.20, 0.78],
    [ 1.85, 0.17, 0.70], [ 2.10, 0.12, 0.55],
])
_SUV = np.array([   # taller, boxier, long flat roof, bigger
    [-2.20, 0.16, 0.60], [-1.95, 0.30, 0.82], [-1.60, 0.82, 0.92],
    [-1.20, 0.92, 0.94], [-0.40, 0.94, 0.94], [ 0.60, 0.92, 0.93],
    [ 1.05, 0.78, 0.92], [ 1.45, 0.40, 0.88], [ 1.80, 0.26, 0.82],
    [ 2.05, 0.22, 0.72], [ 2.20, 0.16, 0.58],
])
_COUPE = np.array([  # low, sleek, fastback, short cabin
    [-2.05, 0.10, 0.52], [-1.80, 0.16, 0.74], [-1.40, 0.30, 0.82],
    [-0.90, 0.50, 0.83], [-0.30, 0.55, 0.82], [ 0.30, 0.54, 0.82],
    [ 0.70, 0.46, 0.81], [ 1.20, 0.26, 0.80], [ 1.60, 0.16, 0.74],
    [ 1.90, 0.13, 0.64], [ 2.05, 0.09, 0.50],
])

BODY_TYPES = {
    "sedan": dict(profile=_SEDAN, z_floor=-0.22, cabin=(-1.10, 1.15), glass_z=0.40,
                  wheel_r=0.33, wheel_w=0.22, axle_z=-0.10,
                  wheels=[(1.35, 0.85), (1.35, -0.85), (-1.35, 0.85), (-1.35, -0.85)]),
    "suv":   dict(profile=_SUV,  z_floor=-0.26, cabin=(-1.55, 1.05), glass_z=0.52,
                  wheel_r=0.40, wheel_w=0.26, axle_z=-0.04,
                  wheels=[(1.45, 0.92), (1.45, -0.92), (-1.45, 0.92), (-1.45, -0.92)]),
    "coupe": dict(profile=_COUPE, z_floor=-0.20, cabin=(-1.30, 0.90), glass_z=0.30,
                  wheel_r=0.34, wheel_w=0.24, axle_z=-0.10,
                  wheels=[(1.30, 0.83), (1.30, -0.83), (-1.30, 0.83), (-1.30, -0.83)]),
}

# Shared materials. Body color is per-car (PALETTE); glass + tyres are fixed.
PALETTE = {
    "red":    (0.78, 0.12, 0.12), "blue":   (0.13, 0.40, 0.78),
    "white":  (0.90, 0.90, 0.92), "black":  (0.09, 0.09, 0.11),
    "silver": (0.66, 0.69, 0.72), "green":  (0.10, 0.50, 0.30),
    "yellow": (0.93, 0.78, 0.14), "orange": (0.94, 0.45, 0.10),
    "teal":   (0.10, 0.60, 0.62), "maroon": (0.45, 0.10, 0.18),
}
GLASS = (0.11, 0.14, 0.19)
TYRE = (0.06, 0.06, 0.08)
_WHEEL_SECTIONS = 40           # rounder tread than before


def _c255(rgb, a=255):
    return np.array([*(np.array(rgb) * 255), a], dtype=np.uint8)


def _ring(x, cfg):
    p = cfg["profile"]
    hw = float(np.interp(x, p[:, 0], p[:, 2]))
    z1 = float(np.interp(x, p[:, 0], p[:, 1]))
    z0 = cfg["z_floor"]
    zmid = z0 + 0.62 * (z1 - z0)
    roof_hw = hw * 0.72
    return np.array([
        [x, -hw, z0], [x, hw, z0], [x, hw, zmid],
        [x, roof_hw, z1], [x, -roof_hw, z1], [x, -hw, zmid],
    ], dtype=np.float64)


def _hull(cfg, n=48):
    p = cfg["profile"]
    xs = np.linspace(p[0, 0], p[-1, 0], n)
    rings = [_ring(x, cfg) for x in xs]
    R = 6
    verts = np.concatenate(rings, axis=0)
    faces = []
    for i in range(n - 1):
        a, b = i * R, (i + 1) * R
        for k in range(R):
            k2 = (k + 1) % R
            faces.append([a + k, b + k, b + k2])
            faces.append([a + k, b + k2, a + k2])
    tail, nose = list(range(R)), list(range((n - 1) * R, n * R))
    for k in range(1, R - 1):
        faces.append([tail[0], tail[k + 1], tail[k]])
        faces.append([nose[0], nose[k], nose[k + 1]])
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m.merge_vertices(); m.fix_normals()
    return m


def _glass_mask(cfg, mesh) -> np.ndarray:
    """Faces forming the greenhouse glass: high, within the cabin, and not flat-up
    (side windows / windshield / backlight) — the flat roof stays body color."""
    cen = mesh.triangles_center
    nz = np.abs(mesh.face_normals[:, 2])
    cmin, cmax = cfg["cabin"]
    return ((cen[:, 2] > cfg["glass_z"]) & (cen[:, 0] > cmin) &
            (cen[:, 0] < cmax) & (nz < 0.78))


def _submesh(mesh, face_idx) -> trimesh.Trimesh:
    """submesh that always returns a Trimesh (empty if no faces, not an ndarray)."""
    if len(face_idx) == 0:
        return trimesh.Trimesh()
    return mesh.submesh([face_idx], append=True)


def _wheel(cx, cy, cfg):
    w = trimesh.creation.cylinder(radius=cfg["wheel_r"], height=cfg["wheel_w"],
                                  sections=_WHEEL_SECTIONS)
    w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    w.apply_translation([cx, cy, cfg["axle_z"]])
    return w


# --------------------------------------------------------------------------
# Public builders
# --------------------------------------------------------------------------
def split_meshes(body: str):
    """-> (body_shell, glass, wheels_list) as separate Trimeshes (for MuJoCo geoms)."""
    cfg = BODY_TYPES[body]
    hull = _hull(cfg)
    g = _glass_mask(cfg, hull)
    shell = _submesh(hull, np.where(~g)[0])
    glass = _submesh(hull, np.where(g)[0])
    wheels = [_wheel(cx, cy, cfg) for cx, cy in cfg["wheels"]]
    return shell, glass, wheels


def build_car(body: str, color="silver") -> trimesh.Trimesh:
    """One painted, two-tone car (body+glass+wheels) as a single colored mesh -> GLB."""
    rgb = PALETTE[color] if isinstance(color, str) else color
    shell, glass, wheels = split_meshes(body)
    shell.visual.face_colors = _c255(rgb)
    glass.visual.face_colors = _c255(GLASS)
    for w in wheels:
        w.visual.face_colors = _c255(TYRE)
    return trimesh.util.concatenate([shell, glass, *wheels])


def write(outdir: str | None = None) -> dict:
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), "meshes")
    os.makedirs(outdir, exist_ok=True)
    written = {}
    for body in BODY_TYPES:
        shell, glass, wheels = split_meshes(body)
        hull = trimesh.util.concatenate([shell, glass])
        # Cesium GLB needs the glTF **Y-up** convention. trimesh writes geometry
        # as-is (our Z-up) with no conversion node, but Cesium reads glTF as Y-up —
        # so without this the car lies on its side (left flank read as "up"). Rotate
        # -90° about X so forward->+X, up->+Y, then drop the wheels to Y=0 so
        # clamp-to-ground rests them on the street (Y is up in glTF). After Cesium's
        # Y-up->Z-up conversion this round-trips to an upright, forward-facing car.
        # MuJoCo OBJs stay chassis-centered Z-up (physics frame) — untouched.
        car = build_car(body, "silver")
        car.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        # Cesium's VelocityOrientationProperty aligns the model so it drives along a
        # different local axis than our nose (+X); without this the car travels
        # sideways. Yaw the body 90° about glTF-up (+Y) so the nose leads. Verified
        # with the headless orient_test probe (car must point along the red/east line).
        car.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
        car.apply_translation([0, -car.bounds[0, 1], 0])   # min Y (down) -> 0
        car.export(os.path.join(outdir, f"{body}.glb"))
        shell.export(os.path.join(outdir, f"{body}_body.obj"))
        glass.export(os.path.join(outdir, f"{body}_glass.obj"))
        hull.convex_hull.export(os.path.join(outdir, f"{body}_hull.obj"))
        written[body] = f"{body}.glb"
    with open(os.path.join(outdir, "palette.json"), "w") as f:
        json.dump({"bodies": list(BODY_TYPES), "palette": PALETTE,
                   "glass": GLASS, "tyre": TYRE}, f, indent=2)
    return written


if __name__ == "__main__":
    written = write()
    for body in BODY_TYPES:
        car = build_car(body)
        lo, hi = car.bounds
        print(f"{body:6s}: {len(car.vertices):4d} v / {len(car.faces):4d} f  "
              f"bbox x[{lo[0]:.2f},{hi[0]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]")
    print(f"palette: {', '.join(PALETTE)}")
    print(f"wrote {len(written)} GLBs + body/glass/hull OBJs + palette.json")
