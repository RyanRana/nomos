"""SmoothRide × Antim/HUD worldsim (Newton physics) — the 3D physics-car path.

Replaces the Isaac/WheeledLab demo path with the Antim Labs "Gizmo" worldsim
template (Newton physics, MJCF scenes). A physics car is just MJCF + the generic
step(action) tool — no Isaac Lab, no nvcr.io, no RTX gating.

  scenes/car-v1/       a single Ackermann physics car (MJCF) — smoke scene
  build_sf_scene.py    OSMnx road network -> a 3D San Francisco MJCF scene + N cars
  control_bridge.py    setpoints -> [drive, steer] action vector (pure-pursuit + P)
  HANDOFF.md           the full API map + build-on-top plan

See HANDOFF.md. Validated with MuJoCo locally; runs on Newton inside the
worldsim-template (see its README — likely a CUDA box / Modal GPU).
"""
