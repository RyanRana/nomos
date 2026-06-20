"""Traffic-law compliance check for a kinematic rollout — "is each car breaking
the law this step?".

Two violations are flagged, both derived from the route geometry the env already
uses to drive the cars:

  * OFF-LANE  — the car is more than `offlane_thresh` metres from the *nearest*
                valid lane centerline of the road segment it is currently driving
                (prev waypoint -> current target waypoint). Measuring distance to
                the segment the car is ON — and to the nearest lane, not just its
                assigned one — means normal lane-keeping, legal lane changes, and
                ordinary corner-cutting read as legal; only a car that has left the
                drivable roadway (crossed multiple lanes / into oncoming space)
                trips it.
  * WRONG-WAY — the car's heading points against its route direction while moving.

Post-respawn merge-in (`spawn_grace`) is exempt so a fresh spawn never counts.

Pure JAX, jit/scan-safe: one (env, state) -> per-agent booleans. No I/O.
"""
from __future__ import annotations

import jax.numpy as jnp

from . import kinematic as K

# distance from the nearest lane centerline beyond which the car has clearly left
# the roadway (m). lane_width is 3.5; 5.0 is ~1.5 lanes — past it the car body is
# off its lane AND the neighbouring lane, i.e. into oncoming space or off-road.
# Below this, corner-cutting and lane changes (which momentarily read ~half a lane
# off) stay legal, which matters here: ~3/4 of route waypoints are intersections.
OFFLANE_THRESH = 5.0
# heading-vs-route cosine below this == pointing the wrong way (~>105 deg off).
WRONGWAY_COS = -0.25
# generous upper bound on lanes-per-road; extra lane slots are masked out per car.
MAX_LANES = 8


def evaluate(env: K.Env, st: K.State,
             offlane_thresh: float = OFFLANE_THRESH,
             wrongway_cos: float = WRONGWAY_COS,
             max_lanes: int = MAX_LANES) -> dict:
    """Per-agent legality of `st`. Returns dict of (N,) arrays:
    lateral (m from nearest lane center of the current segment), off_lane,
    wrong_way, illegal."""
    ri, wp = st.route_idx, st.wp_ptr
    prev = env.routes_xy[ri, jnp.maximum(wp - 1, 0)]            # segment start
    cur = env.routes_xy[ri, wp]                                # segment end (target)
    seg = cur - prev
    seglen = jnp.linalg.norm(seg, axis=-1, keepdims=True)
    u = seg / (seglen + 1e-6)                                  # unit along-segment
    right = jnp.stack([u[:, 1], -u[:, 0]], axis=-1)           # unit right-normal

    # point-to-segment distance to each lane (segment shifted right by lane offset),
    # taking the nearest lane: a legal car sits on one of these lines.
    lanes = env.routes_lanes[ri, wp]                          # (N,)
    ls = jnp.arange(max_lanes)
    offs = env.lane_width * (ls.astype(jnp.float32) + 0.5)    # (Lmax,) lane offsets
    valid = ls[None, :] < jnp.maximum(lanes, 1)[:, None]      # (N,Lmax)

    # broadcast segment endpoints over lanes: a,b shape (N,Lmax,2)
    a = prev[:, None, :] + right[:, None, :] * offs[None, :, None]
    b = cur[:, None, :] + right[:, None, :] * offs[None, :, None]
    ab = b - a
    t = jnp.clip(jnp.sum((st.pos[:, None, :] - a) * ab, -1)
                 / (jnp.sum(ab * ab, -1) + 1e-6), 0.0, 1.0)    # (N,Lmax)
    proj = a + t[..., None] * ab
    d = jnp.linalg.norm(st.pos[:, None, :] - proj, axis=-1)    # (N,Lmax)
    d = jnp.where(valid, d, 1e9)
    lateral = d.min(axis=1)                                    # (N,)

    grace = st.spawn_grace > 0
    route_head = jnp.arctan2(u[:, 1], u[:, 0])
    herr = K._wrap(st.heading - route_head)
    wrong_way = (jnp.cos(herr) < wrongway_cos) & (st.speed > env.idle_speed) & ~grace
    off_lane = (lateral > offlane_thresh) & ~grace
    return {"lateral": lateral, "off_lane": off_lane, "wrong_way": wrong_way,
            "illegal": off_lane | wrong_way}
