# Handoff: physics cars + 3D San Francisco on Antim/HUD worldsim (Newton)

This replaces the Isaac/WheeledLab demo path. Instead of building a task into Isaac
Sim (heavy image, nvcr.io, RTX gating), we run the physics on the **Antim Labs
"Gizmo" worldsim template** — [github.com/hud-evals/worldsim-template](https://github.com/hud-evals/worldsim-template)
— which runs **Newton physics** (Warp + mujoco-warp) over **MJCF scenes**. A
physics car is just MJCF + the template's generic `step(action)` tool. No Isaac.

> Why it's the right call: in worldsim, **"scenes are environments."** A car scene
> + a reward is a first-class env, exactly the pattern the template is built for —
> and the whole thing is MuJoCo models under the hood, so we author and validate
> with plain `mujoco` before ever touching a GPU.

---

## 1. The worldsim stack (what I read in the repo)

| piece | what it is |
|---|---|
| **Gizmo (Antim Labs)** | the scene engine. Generate scenes at **gizmo.antimlabs.com**, drop them under `scenes/<id>/` as `scene.xml` + `metadata.json`. |
| **Newton** | GPU physics engine (bundled wheel `wheels/newton-*.whl`), Warp compute + a **mujoco-warp** solver. It wraps a **MuJoCo model** (`solver.mj_model` / `mj_data`). |
| `sim/server.py` | the sim, served as a **FastMCP tool server** (an `mcp` capability). |
| `sim/host.py` | spawns the sim in its own process; exposes `mcp_url`. |
| `environment/env.py` | a HUD `Environment` whose `@env.template` tasks `reset` + grade by reading sim state over MCP. |
| `scenes/` | each subfolder = a live env (`tabletop-v1`, `franka-libero-v1`). |

### The API that matters for us (all MCP tools on the sim)

| tool | signature | use |
|---|---|---|
| `list_scenes()` | → `[{scene_id, format, description}]` | discovery (scans `scenes/`) |
| `reset(scene_id, settle_steps=500, ...)` | load + settle a scene | **call first** |
| `step(action)` | `action: list[float]`, **one per `<actuator>`, in XML order** | drive everything |
| `get_object_state(name)` | → `{position:{x,y,z}, ...}` | per-car pose readback |
| `get_joint_state(name)` | → `{position, ...}` | wheel/steer state |
| `get_state()` | full dump (gripper-centric; for cars use sensors + get_object_state) | |
| `render(camera, w, h)` | → PNG | offscreen frames for video |
| `get_scene_info()` | actuators / cameras / bodies | introspection |

`step` literally does `mj_data.ctrl[:] = action; mj_step(...)`. **That is the entire
control contract** — line up your action vector with the `<actuator>` order and you
drive the scene. Scene discovery also accepts a **`build_scene.py`** (format
`"python"`) instead of a static `scene.xml`, if you ever want reset-time generation.

---

## 2. What I built here (and validated)

All three are **MuJoCo-validated locally** (Newton uses the same MuJoCo model, so a
clean `mj_loadXML` + a physics rollout is a real correctness check):

| artifact | what | status |
|---|---|---|
| `scenes/car-v1/scene.xml` | one rear-wheel-drive Ackermann car, MJCF | ✅ compiles; drives +2.4 m/s under throttle |
| `build_sf_scene.py` | OSMnx road graph → `sf-city-v1/scene.xml` (roads + N cars, optional buildings) | ✅ 8 cars, 381 geoms, cars rest stable |
| `control_bridge.py` | setpoints → `[drive,drive,steer,steer]×N` (pure-pursuit + P throttle) | ✅ self-test: drove a car 25 m→8.4 m to a waypoint |

**Control contract** (per car, concatenated in spawn order):
`action = [drive_rl, drive_rr, steer_l, steer_r]` — drive = rear-wheel target rad/s
(× 0.33 m wheel radius = m/s), steer = front angle rad (±0.7). Roads are **visual
decals** (`contype=0`), cars roll on the flat ground plane (a raised road mesh
ejects the wheels — learned that the hard way).

---

## 3. Build-on-top plan (the actual next steps)

### Step 0 — stand up worldsim
```bash
git clone https://github.com/hud-evals/worldsim-template && cd worldsim-template
uv sync                      # installs the bundled Newton wheel
python scripts/check_setup.py   # boots the sim, grades one rollout (first reset compiles Warp ~1 min)
```
⚠ **Newton/Warp almost certainly wants CUDA.** The template's own VLA path serves
the policy on Modal/GPU. If `check_setup.py` fails on this Mac (no CUDA), run the
**sim on a Linux+NVIDIA box or a Modal GPU** — the same place the Isaac path would
have run, but with a far lighter image and no nvcr.io. Don't assume Apple-Silicon
CPU works until `check_setup.py` passes.

### Step 1 — install our scenes
```bash
# generate, writing straight into the template's scenes dir
python -m smoothride.worldsim.build_sf_scene --cars 20 --buildings \
    --out /path/to/worldsim-template/scenes/sf-city-v1
# car-v1 too (smoke):
cp -r smoothride/worldsim/scenes/car-v1 /path/to/worldsim-template/scenes/
```
`reset(scene_id="sf-city-v1")` should now load the city.

### Step 2 — render on Modal GPU (the turnkey path)
Newton needs CUDA, so the sim + render run on a Modal GPU. `render_modal.py` builds
the worldsim+Newton image, generates the SF scene, drives every car along a real
route (planner + control_bridge), and writes an mp4 to a Volume:
```bash
pip install -e ".[worldsim]" && modal token new
modal run smoothride/worldsim/render_modal.py --cars 24 --seconds 20
modal volume get smoothride-worldsim-out sf_physics_24cars.mp4 .
```
The planner + control loop are MuJoCo-validated locally (`control_bridge` /
`planner` self-tests); only the Newton runtime + image build are first-run-on-Modal.

### Step 2b — the raw control loop (what render_modal does inside)
```python
# pseudo, against the sim MCP (sim_tools.call(...) in env.py, or a FastMCP client)
reset(scene_id="sf-city-v1")
mc = MultiCarController(n_cars=N)               # control_bridge
for t in range(T):
    poses, yaws, speeds = [], [], []
    for i in range(N):
        s = get_object_state(f"c{i}_chassis")    # pose readback
        poses.append((s["position"]["x"], s["position"]["y"]))
        # yaw from cI_quat sensor (full vectors via sensors, not get_state's 1-dim dump)
    targets, target_speeds = plan(t)             # ← from our coordination layer
    step(mc.action(poses, yaws, speeds, targets, target_speeds))
    frame = render(camera="oblique", w=1280, h=720)   # → stitch to video
```

### Step 3 — feed it from SmoothRide's brain
`targets` / `target_speeds` come from either:
- **the trained coordination policy** — its (waypoint/velocity/heading) setpoints
  map straight onto `CarController` (that's why the interface was chosen), or
- **replay** our already-exported trajectories
  (`smoothride/demo/export_web` / `demo/isaac/export_setpoints`) as a per-car
  waypoint stream — zero new training, instant 3D hero shot.

### Step 4 — SF in 3D, better
- Roads: done (decal ribbons from real OSM geometry).
- **Buildings**: `--buildings` extrudes OSM footprints as boxes. Upgrade with real
  `building:levels` heights, or generate a richer block at **gizmo.antimlabs.com**
  and drop it in alongside.
- Cameras: `drone` (top-down) + `oblique` (3/4 skyline) are defined; add a
  chassis-mounted chase cam (see `car-v1`'s `<camera mode="trackcom">`) for a
  street-level shot.

---

## 4. Honest status & caveats

- **Not yet run on Newton** — everything here is validated with stock MuJoCo. The
  MJCF + control are correct; the open risk is purely Newton/Warp env setup
  (Step 0) and that it may require a GPU box.
- **Actuator gains are a tuning knob** — the car accelerates gently (heavy body,
  2 driven wheels). Bump `kv` on the drive actuators / lower chassis mass / raise
  `kp_speed` if you want snappier motion. Not a correctness issue.
- **Scale**: downtown SF is ~1.5 km; MuJoCo handles it, but hundreds of cars +
  building meshes will want the GPU solver (another reason Newton-on-GPU).
- **Relationship to the other viewers**: the deck.gl web map
  (`smoothride/demo/web`) stays the always-works 2D demo; this is the 3D physics
  upgrade. The Isaac path (`smoothride/demo/isaac`, `smoothride/lowlevel`) is
  **superseded** by this — keep it only if you specifically need WheeledLab's PhysX
  RC-car dynamics or sim-to-real transfer to physical cars.
