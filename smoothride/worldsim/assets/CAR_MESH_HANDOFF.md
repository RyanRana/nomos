# Workstream: 3D car mesh + Newtonian physics state (→ Cesium 3D view)

Isolated under `smoothride/worldsim/` (untracked) — **no overlap** with the Cesium
or 2D-RL lanes. Three new files; nothing existing was edited.

## What this delivers
| file | role |
|---|---|
| `assets/car_mesh.py` | **trimesh** sedan generator → `meshes/car_body.glb` (Cesium), `car_body.obj` (MuJoCo visual), `car_hull.obj` (convex collider) |
| `scenes/car-v2/scene.xml` | car-v1's **unchanged** Newtonian dynamics + the body mesh as a non-colliding visual overlay |
| `physics_state.py` | `PhysicsState` (mirrors the 2D `State`) + `rollout_mujoco()` emitting the **same `tr` dict** the kinematic env produces |

Regenerate meshes: `python -m smoothride.worldsim.assets.car_mesh`
Validate (load mesh scene, drive under physics, check state): `python -m smoothride.worldsim.physics_state`

## The state is identical to 2D
2D: `tr = {pos (T,N,2), heading (T,N), speed (T,N), crashed, goals, ped}`.
`rollout_mujoco(...)` returns exactly those keys, so `demo/export_web._pack_world`
consumes a **physics** rollout with no change → same `trajectories.json`
(`cars[].lng/lat/hdg/spd`) the Cesium view already loads. Newtonian motion in, 2D
state out.

## Frame contract (so Cesium orients the model right)
Local **+X = forward**, +Y = left, +Z = up. `heading = atan2(dy,dx)`, heading 0 →
car points +X = **east**. trimesh bakes the z-up→glTF y-up correction into the
`.glb`, so the car stands upright; the entity's heading drives yaw.

## Cesium wiring (the Cesium lane applies this in `app.js` — kept out of here to avoid a merge conflict)
Copy `car_body.glb` next to the viewer (e.g. `demo/cesium/assets/car_body.glb`),
then in `addWorld()` replace the `point:` graphics with an oriented model. The
trajectory already carries `c.hdg[t]`:

```js
// orientation from heading (rad, CCW from east) — VelocityOrientation also works
const ori = new Cesium.SampledProperty(Cesium.Quaternion);
for (let t = 0; t < NF; t++) {
  const time = Cesium.JulianDate.addSeconds(startTime, t * DT, new Cesium.JulianDate());
  const hpr  = new Cesium.HeadingPitchRoll(Cesium.Math.PI_OVER_TWO - c.hdg[t], 0, 0); // +X-fwd → Cesium heading(=clockwise from north)
  const carto = Cesium.Cartesian3.fromDegrees(c.lng[t], c.lat[t], 0);
  ori.addSample(time, Cesium.Transforms.headingPitchRollQuaternion(carto, hpr));
}
viewer.entities.add({
  position: pos, orientation: ori,
  model: { uri: "./assets/car_body.glb", minimumPixelSize: 24, scale: 1.0,
           color, colorBlendMode: Cesium.ColorBlendMode.MIX, colorBlendAmount: 0.6,
           heightReference: Cesium.HeightReference.CLAMP_TO_GROUND },
});
```
Keep the `point` as a fallback for far zoom if desired (`model` + `point` coexist).
The `Math.PI_OVER_TWO - hdg` converts our east-CCW heading to Cesium's
north-clockwise heading; flip the sign if the model faces backwards.

## Status / caveats
- ✅ mesh watertight (488 v / 956 t); ✅ car-v2 loads in MuJoCo; ✅ drives 74.9 m
  under physics; ✅ `tr` matches the 2D schema. Preview frames: `meshes/preview_*.png`.
- Physics unchanged from car-v1 (box inertia + cylinder wheels). The mesh is
  visual-only (`contype=0`), so dynamics stay the validated, stable ones.
- `crashed`/`goals` in the physics `tr` are currently zeros — wire contact-based
  crash + goal-radius checks when the planner is connected (TODO marked in code).
- For the multi-car SF scene, `build_sf_scene.py` can reference the same mesh asset
  per car (add the `<asset><mesh>` once + a `class="visual"` geom in `_car_body`);
  left as a one-flag follow-up to keep this lane non-breaking.
```
