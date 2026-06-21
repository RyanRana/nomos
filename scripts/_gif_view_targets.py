"""Compute 4 Cesium camera targets from the champion scene trajectory.

Emits /tmp/gif_views.json consumed by the Playwright capture script:
  whole-scene, dense cluster, sparse pocket, busiest intersection.
"""
import json
import math
import sys

SCENE = "smoothride/demo/cesium/public/scene_champion_mission.json"
d = json.load(open(SCENE))
meta = d["meta"]
cx, cy = meta["center"]
n_steps = meta["n_steps"]
mlat = math.cos(math.radians(cy))


def m_per_deg_lon():
    return 111320.0 * mlat


def dist_m(a, b):
    dx = (a[0] - b[0]) * m_per_deg_lon()
    dy = (a[1] - b[1]) * 111320.0
    return math.hypot(dx, dy)


cars = d["worlds"]["trained"]["cars"]


def active_pos(car, t):
    """(lng,lat) if the car is live (not crashed/arrived) at step t, else None."""
    if car["crash"][t] or car["arr"][t]:
        return None
    return (car["lng"][t], car["lat"][t])


# --- DENSE: step+point with the most live cars within 55 m ---
best_dense = (-1, None, 0)
for t in range(0, n_steps, 3):
    pts = [active_pos(c, t) for c in cars]
    pts = [p for p in pts if p]
    for p in pts:
        k = sum(1 for q in pts if dist_m(p, q) <= 55.0)
        if k > best_dense[2]:
            best_dense = (t, p, k)
dense_t, dense_p, dense_k = best_dense

# --- SPARSE: a moving car that is the most isolated (fewest within 150 m) ---
best_sparse = (-1, None, 10**9)
for t in range(n_steps // 4, n_steps, 5):
    pts = [active_pos(c, t) for c in cars]
    live = [p for p in pts if p]
    for c in cars:
        p = active_pos(c, t)
        if not p or c["spd"][t] < 1.5:          # must be moving, not parked
            continue
        k = sum(1 for q in live if 0 < dist_m(p, q) <= 150.0)
        if k < best_sparse[2]:
            best_sparse = (t, p, k)
sparse_t, sparse_p, sparse_k = best_sparse

# --- INTERSECTION: road vertex shared by >=3 segments, weighted by car traffic ---
vcount = {}
for seg in d["roads"]:
    for lon, lat, *_ in seg:
        key = (round(lon, 5), round(lat, 5))
        vcount[key] = vcount.get(key, 0) + 1
junctions = [k for k, n in vcount.items() if n >= 3]

# all live car positions across time, to score how busy each junction is
allpts = [active_pos(c, t) for c in cars for t in range(0, n_steps, 4)]
allpts = [p for p in allpts if p]
best_int = (None, -1)
for j in junctions:
    busy = sum(1 for p in allpts if dist_m(p, j) <= 35.0)
    if busy > best_int[1]:
        best_int = (j, busy)
int_p = best_int[0] or (cx, cy)
# pick the step when the most cars are near that junction
int_t, int_best = 0, -1
for t in range(n_steps):
    near = sum(1 for c in cars if (lambda p: p and dist_m(p, int_p) <= 40.0)(active_pos(c, t)))
    if near > int_best:
        int_t, int_best = t, near

views = [
    {"name": "1_zoomed_out", "lon": cx, "lat": cy, "height": 3200,
     "pitch": -42, "heading": 0, "t0": 40, "t1": 230,
     "caption": f"Champion v4loo — Mission ({len(cars)} cars), whole scene"},
    {"name": "2_zoom_dense", "lon": dense_p[0], "lat": dense_p[1], "height": 360,
     "pitch": -38, "heading": 25, "t0": max(0, dense_t - 25), "t1": min(n_steps - 1, dense_t + 60),
     "caption": f"Dense traffic — {dense_k} cars clustered"},
    {"name": "3_zoom_sparse", "lon": sparse_p[0], "lat": sparse_p[1], "height": 380,
     "pitch": -38, "heading": -20, "t0": max(0, sparse_t - 30), "t1": min(n_steps - 1, sparse_t + 55),
     "caption": "Sparse — a car running its route alone"},
    {"name": "4_intersection", "lon": int_p[0], "lat": int_p[1], "height": 240,
     "pitch": -52, "heading": 15, "t0": max(0, int_t - 35), "t1": min(n_steps - 1, int_t + 55),
     "caption": f"Intersection — {int_best} cars converging"},
]
json.dump({"meta": meta, "views": views}, open("/tmp/gif_views.json", "w"), indent=2)
print("dense:", dense_t, dense_p, "k=", dense_k)
print("sparse:", sparse_t, sparse_p, "k=", sparse_k)
print("intersection:", int_p, "busy=", best_int[1], "peak_t=", int_t, "near=", int_best)
print("wrote /tmp/gif_views.json")
