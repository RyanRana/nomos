import numpy as np
from smoothride.demo import scene as S


def test_pack_world_shapes_and_keys():
    T, N, M = 5, 2, 1
    world = S.pack_world(
        car_lon=np.zeros((T, N)), car_lat=np.zeros((T, N)),
        car_z=np.zeros((T, N)), heading=np.zeros((T, N)),
        speed=np.ones((T, N)), crashed=np.zeros((T, N), bool),
        goals=np.zeros((T, N), int),
        ped_lon=np.zeros((T, M)), ped_lat=np.zeros((T, M)), ped_z=np.zeros((T, M)),
        stride=1,
    )
    assert set(world) == {"summary", "trips_series", "cars", "peds"}
    assert len(world["cars"]) == N
    assert set(world["cars"][0]) == {"lng", "lat", "z", "hdg", "spd", "crash"}
    assert len(world["cars"][0]["lng"]) == T


def test_validate_scene_accepts_minimal_valid_scene():
    scene = {
        "schema_version": S.SCHEMA_VERSION,
        "meta": {"dt": 0.2, "n_steps": 1, "vmax": 16.0,
                 "center": [-122.41, 37.79], "bounds": [[0, 0], [1, 1]], "zoom": 15},
        "roads": [[[-122.41, 37.79, 5.0], [-122.40, 37.79, 6.0]]],
        "buildings": {"type": "FeatureCollection", "features": []},
        "worlds": {"trained": S.pack_world(
            car_lon=np.zeros((1, 1)), car_lat=np.zeros((1, 1)), car_z=np.zeros((1, 1)),
            heading=np.zeros((1, 1)), speed=np.zeros((1, 1)),
            crashed=np.zeros((1, 1), bool), goals=np.zeros((1, 1), int),
            ped_lon=np.zeros((1, 0)), ped_lat=np.zeros((1, 0)), ped_z=np.zeros((1, 0)),
            stride=1)},
    }
    S.validate_scene(scene)      # must not raise


def test_validate_scene_rejects_wrong_version():
    import pytest
    with pytest.raises(ValueError):
        S.validate_scene({"schema_version": 999, "meta": {}, "worlds": {}})
