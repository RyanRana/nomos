# Scaling realism — design notes

> "Hella cars? Highways, lanes, dynamics. Stop-and-go? Unruly pedestrians? SF
> hills and terrain in Isaac? We need to think all this through."

We do. The good news: almost every item below already has a home in the
architecture, because we split the problem into two layers. The discipline is to
put each kind of realism in the layer that can afford it.

## The governing principle: two layers absorb two kinds of realism

| | **Behavioral realism** | **Physical realism** |
|---|---|---|
| Where | Kinematic training env (JAX, Modal) | Isaac execution + frozen low-level controller |
| Owns | how many cars, lanes, highways, intersections, stop-and-go, pedestrians, routing, *timing* | tire/suspension/contact forces, real braking distance, **hills/terrain**, slope dynamics |
| Cost model | cheap, vectorized, thousands of agents | expensive, RTX, runs only the trained policy (no learning) |
| Why there | RL needs millions of fast steps; behavior is the thing being *learned* | physics is *executed*, not learned; the low-level controller already handles it |

The interface that lets them compose: the policy emits **setpoints** (target
velocity/heading), and the kinematic env uses the **car's real accel/brake/steer
limits**, so what's learned in 2D tracks when replayed through real dynamics in
Isaac. A few realism axes (notably **grade**) must appear in *both* layers — as a
scalar that shapes timing in training, and as actual terrain mesh in the demo.

Rule we hold to: **every realism axis we add must show up in an artifact (a
video).** If it can't be seen moving, it isn't real yet — it's a claim.

---

## 1. Hella cars (scale)

- **Bottleneck is O(N²).** Today neighbor-finding and collision are dense pairwise
  — fine at 20 cars, quadratic death at 1000. Fix: bin cars into a **uniform
  spatial grid / hash** and only compare within neighboring cells → O(N·k). This
  is the single change that unlocks scale.
- **Bigger world:** stitch several OSM bbox tiles (or one large grab) so there's
  road surface for the cars; the route pool scales linearly.
- **GPU (Modal):** the env is pure JAX — thousands of cars × many worlds is a
  GPU-memory question, not an architecture question.
- **v1 → next:** 20 dense cars now → spatial hash → 200–1000 cars as the "wow."

## 2. Highways

- OSMnx `network_type="drive"` **already pulls motorway edges**, and we already
  compute a per-edge `speed_kph`. Today we cap everyone at a single `v_max`; the
  fix is **per-edge speed limit** so highways run fast and surface streets slow.
- Highways add **merging/weaving at on/off-ramps** — a genuinely new coordination
  skill, and a great visual (zipper merges with no crashes). Ramp topology comes
  free from OSM.
- High speed stresses the kinematic↔physics gap; keep the kinematic model inside
  its validity envelope and let the low-level controller own the tire forces.

## 3. Lanes

- We already pull **`edge_lanes`** per segment. Today each car drives one
  right-offset centerline (2 implicit opposing lanes). The upgrade: give each car
  a **discrete lane index**, lateral offset = `(lane − (L−1)/2)·lane_width`, and a
  **lane-change action** (shift ±1 when the gap is clear).
- Payoff: real **overtaking**, opposing-flow separation, and structurally fewer
  crashes (the centerline head-on problem disappears — we already saw a lane
  *offset* cut crashes; full multi-lane generalizes it).

## 4. Dynamics

- **Training stays kinematic** — bicycle model + accel/jerk/brake limits + a grade
  term. That's the correct altitude for learning coordination and is what keeps it
  fast. We tune those limits to the demo car's real envelope.
- **True dynamics lives in Isaac** via the frozen WheeledLab-derived controller
  (rigid body, suspension, real braking). We deliberately do **not** put tire
  physics in the training loop — that was the whole point of the hierarchical split.

## 5. Stop-and-go & intersections (the theme's hard core)

- **Stop-and-go** needs explicit **longitudinal car-following**: an observation of
  the *lead vehicle's gap and closing speed*, plus a comfortable-braking model.
  An IDM-style prior can seed it; the RL refines it. Phantom-jam waves then emerge
  from dense following — and make a striking visual.
- **Intersections without lights** is the biggest *behavioral* gap right now — and
  it's literally the hackathon thesis ("function without traffic lights"). Cars
  must **negotiate right-of-way**: a learned yielding policy (centralized critic
  sees the whole junction) optionally shaped by a simple priority rule. This is
  where multi-agent RL earns its keep over scripted traffic.
- Today cars cross intersections obliviously; adding junction-aware obs + a
  yielding reward is high-priority.

## 6. Unruly pedestrians

- A **new non-vehicle agent class**: stochastic walkers that jaywalk (random walk +
  road crossings) — "unruly" = unpredictable, exactly the hard case.
- Cars observe them through the **same neighbor channel**, but hitting a pedestrian
  carries an **asymmetric, severe penalty** (a pedestrian is not a fender-bender).
- This is a **direct callback to the hotel-robot prior** (predict-and-avoid moving
  people) — and reuses the proximity/APF machinery we already built. v1 can be
  scripted motion; later, pedestrians get their own policy for true unpredictability.

## 7. SF hills & terrain (the Isaac layer)

- OSM has no reliable elevation, so we pull a **DEM** (USGS 3DEP / SRTM) and
  **drape the road network over it** → a z per node, a **grade per edge**. That
  gives `road-mesh-builder` a **2.5D hilly mesh** for Isaac.
- **WheeledLab's `elevation` task is exactly the slope-handling low-level skill** —
  this is why it's in our stack. We train the low-level controller on grade so it
  climbs/descends SF hills with correct dynamics.
- **Hills also touch the training layer:** add **per-edge grade** as (a) a speed/
  accel modifier (uphill slower, downhill longer braking distance) and (b) a scalar
  in the car's observation, so the *coordination* policy accounts for hill timing —
  not just the physics layer. This is the canonical "appears in both layers" case.

---

## What to build next (by ROI on the demo)

1. **Multi-lane + lane-change** — structurally kills crashes, enables overtaking. (behavioral, high ROI)
2. **Intersection yielding** — the "no traffic lights" headline; where MARL shines. (behavioral, on-thesis)
3. **Spatial hashing → 200–1000 cars** — the "hella cars" wow shot. (scale)
4. **Per-edge speed / highways** — cheap variety + merge visuals. (behavioral, cheap)
5. **Pedestrians** — strong narrative (hotel-robot callback), moderate effort. (new agent class)
6. **DEM terrain + WheeledLab elevation** — Isaac hill demo; heavier, demo-polish. (physical)

Each is an incremental change to one layer with a visible artifact at the end — no
big-bang rewrite. The architecture already has a slot for every one of them.
