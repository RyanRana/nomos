"""Scene schema v1 — the render/IO contract.

ONE file format produced by every sim backend (kinematic now, Isaac/PhysX later)
and consumed by the Cesium viewer. Trajectories are reprojected to lon/lat with a
baked ground `z`, rounded, and packed per-car to stay small.

A scene is:
  schema_version: int
  meta:   {dt, n_steps, vmax, center[lon,lat], bounds[[lon,lat],[lon,lat]], zoom}
  roads:  [ [[lon,lat,z],[lon,lat,z]], ... ]        # 3D segments
  buildings: GeoJSON FeatureCollection (properties.height)
  worlds: { name: {summary, trips_series, cars[], peds[]} }
    car:  {lng[], lat[], z[], hdg[], spd[], crash[]}   # hdg = rad CCW from east
    ped:  {lng[], lat[], z[]}
"""
from __future__ import annotations

import json
import os

import numpy as np

SCHEMA_VERSION = 1
_CAR_KEYS = {"lng", "lat", "z", "hdg", "spd", "crash"}


def pack_world(*, car_lon, car_lat, car_z, heading, speed, crashed, goals,
               ped_lon, ped_lat, ped_z, stride: int) -> dict:
    """Reproject-agnostic packer: takes already-lon/lat arrays (T, N) -> world dict."""
    T, N = car_lon.shape
    frames = range(0, T, stride)
    persist_crash = np.cumsum(crashed.astype(np.int32), axis=0) > 0

    cars = []
    for i in range(N):
        cars.append({
            "lng": [round(float(car_lon[t, i]), 6) for t in frames],
            "lat": [round(float(car_lat[t, i]), 6) for t in frames],
            "z":   [round(float(car_z[t, i]), 2) for t in frames],
            "hdg": [round(float(heading[t, i]), 4) for t in frames],
            "spd": [round(float(speed[t, i]), 2) for t in frames],
            "crash": [int(persist_crash[t, i]) for t in frames],
        })

    peds = []
    for j in range(ped_lon.shape[1]):
        peds.append({
            "lng": [round(float(ped_lon[t, j]), 6) for t in frames],
            "lat": [round(float(ped_lat[t, j]), 6) for t in frames],
            "z":   [round(float(ped_z[t, j]), 2) for t in frames],
        })

    moving_end = int(((speed[-1] > 1.0) & ~persist_crash[-1]).sum())
    summary = {
        "cars": int(N), "peds": int(ped_lon.shape[1]),
        "trips_end": int(goals[-1].sum()),
        "crashed_end": int(persist_crash[-1].sum()),
        "moving_end": moving_end,
    }
    trips_series = [int(goals[t].sum()) for t in frames]
    return {"summary": summary, "trips_series": trips_series, "cars": cars, "peds": peds}


def build_scene(*, meta: dict, roads: list, buildings: dict, worlds: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "roads": roads,
        "buildings": buildings,
        "worlds": worlds,
    }


def validate_scene(scene: dict) -> None:
    """Raise ValueError if `scene` does not conform to schema v1."""
    if scene.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, "
                         f"got {scene.get('schema_version')!r}")
    for key in ("meta", "worlds"):
        if key not in scene:
            raise ValueError(f"scene missing required key: {key!r}")
    for wname, world in scene["worlds"].items():
        for car in world.get("cars", []):
            missing = _CAR_KEYS - set(car)
            if missing:
                raise ValueError(f"world {wname!r} car missing keys: {sorted(missing)}")


def write_scene(path: str, scene: dict) -> int:
    """Validate then write compact JSON. Returns bytes written."""
    validate_scene(scene)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(scene, f, separators=(",", ":"))
    return os.path.getsize(path)
