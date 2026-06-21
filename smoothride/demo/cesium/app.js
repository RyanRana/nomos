// Real 3D San Francisco + the Nomos fleet driving on it.
// Map = Cesium World Terrain + OSM Buildings. Cars = procedural GLB models
// (worldsim/assets: sedan/suv/coupe), each given a RANDOM body + palette color,
// driven along the exported RL trajectories and oriented by per-frame heading.
const TOKEN = new URLSearchParams(location.search).get("ionToken")
  || (window.CESIUM_ION_TOKEN && window.CESIUM_ION_TOKEN !== "PASTE_YOUR_CESIUM_ION_TOKEN"
        ? window.CESIUM_ION_TOKEN : null);

const SF = { lon: -122.4090, lat: 37.7886 };
const TRAJ_URL = "../web/public/trajectories.json";
const AMBER = Cesium.Color.fromCssColorString("#f59e0b").withAlpha(0.5);

// Image pasted onto building facades. Drop a file next to index.html and set its
// name here (transparent PNG works). Override at runtime with ?mural=<url>.
const MURAL_IMAGE = new URLSearchParams(location.search).get("mural") || "./building-side.png";

// Facade images: each building is randomly skinned with one of these. URL-encode the
// folder name because it contains a space ("building sides").
const SIDE_IMAGES = [1, 2, 3, 4, 5].map((n) => `./building%20sides/side${n}.jpg`);

// How to apply them: "tile" = pictures stacked up the facade at a fixed size, kept
// undistorted (default). "single" = stretch one photo to fill the whole facade.
// Switch with ?skin=single.
const SKIN_MODE = new URLSearchParams(location.search).get("skin") === "single" ? "single" : "tile";

// Draw a labeled placeholder skin so murals render even before you supply a photo.
function placeholderMural() {
  const cv = document.createElement("canvas");
  cv.width = 512; cv.height = 768;
  const g = cv.getContext("2d");
  g.fillStyle = "#182030"; g.fillRect(0, 0, cv.width, cv.height);
  g.fillStyle = "#5a96d2";
  for (let y = 40; y < cv.height; y += 90)
    for (let x = 40; x < cv.width; x += 90) g.fillRect(x, y, 55, 60);
  g.strokeStyle = "#ffb428"; g.lineWidth = 8; g.strokeRect(4, 4, cv.width - 8, cv.height - 8);
  g.fillStyle = "#fff"; g.font = "bold 64px sans-serif"; g.textAlign = "center";
  g.fillText("MURAL", cv.width / 2, cv.height / 2);
  return cv;
}

// Resolve to the configured image if it loads, else the placeholder canvas.
function resolveMuralImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(url);
    img.onerror = () => { console.warn(`mural image "${url}" not found — using placeholder`); resolve(placeholderMural()); };
    img.src = url;
  });
}

// ---- the procedural fleet: 3 body GLBs × a color palette (worldsim/assets) ----
const BODIES = ["sedan", "suv", "coupe"];
const PALETTE = [
  "#c81f1f", "#2166c8", "#e6e6eb", "#17171c", "#a8b0b8",
  "#198050", "#edc724", "#f0731a", "#19999e", "#73192e",
].map((h) => Cesium.Color.fromCssColorString(h));

// deterministic per-car RNG -> a car keeps its body+color across frames/reloads
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function carLook(i) {
  const r = mulberry32(0x5eed + (i * 2654435761) | 0);
  return { body: BODIES[Math.floor(r() * BODIES.length)],
           color: PALETTE[Math.floor(r() * PALETTE.length)] };
}

function msg(html) {
  const el = document.getElementById("msg");
  el.innerHTML = html; el.setAttribute("data-show", "");
}
window.addEventListener("error", (e) => msg(`<div>Error: <code>${e.message}</code></div>`));
window.addEventListener("unhandledrejection",
  (e) => msg(`<div>Error: <code>${(e.reason && (e.reason.message || e.reason)) || e.reason}</code></div>`));

async function boot() {
  if (typeof Cesium === "undefined") return msg("<div>Cesium failed to load (CDN).</div>");
  if (TOKEN) Cesium.Ion.defaultAccessToken = TOKEN;

  const opts = {
    animation: true, timeline: true, geocoder: false, baseLayerPicker: false,
    homeButton: false, sceneModePicker: false, navigationHelpButton: false,
    fullscreenButton: false, infoBox: false, selectionIndicator: false,
  };
  if (TOKEN) opts.terrain = Cesium.Terrain.fromWorldTerrain();
  else opts.baseLayer = false;

  const viewer = new Cesium.Viewer("cesiumContainer", opts);
  viewer.scene.globe.depthTestAgainstTerrain = !!TOKEN;
  window.viewer = viewer; // exposed for the GIF capture harness

  let muralImg = null;
  if (TOKEN) {
    try {
      const osm = await Cesium.createOsmBuildingsAsync();
      viewer.scene.primitives.add(osm);
      window.osmBuildings = osm; // exposed for the GIF capture harness
      // Faster first paint: load coarser tiles, skip intermediate LODs instead of
      // streaming every level, and only refine to full detail once the camera
      // settles. Override the look with ?sse=<n> (lower = sharper but slower).
      const sse = Number(new URLSearchParams(location.search).get("sse")) || 20;
      osm.maximumScreenSpaceError = sse;          // detail target
      osm.skipLevelOfDetail = true;               // jump straight toward the target LOD
      osm.baseScreenSpaceError = 1024;
      osm.skipScreenSpaceErrorFactor = 16;
      osm.skipLevels = 1;
      // SOLID buildings: dynamic-SSE dithers/fades tiles while the camera moves,
      // which reads as low-opacity ghost buildings — turn it off so facades stay opaque.
      osm.dynamicScreenSpaceError = false;
      // Load NEARBY first + keep what's loaded so panning back doesn't re-stream (less lag):
      osm.foveatedScreenSpaceError = true;        // sharp near screen centre...
      osm.foveatedConeSize = 0.3;                 // ...cheaper at the edges
      osm.preloadFlightDestinations = true;       // prefetch where the camera is heading
      osm.cacheBytes = 1073741824;                // 1 GB tile cache (was tiny -> constant reload)
      osm.maximumCacheOverflowBytes = 1073741824;
      // Skin EVERY building, each randomly assigned one of the side images.
      try {
        const atlas = await buildImageAtlas(SIDE_IMAGES);
        // Square footprint (wall == floor) so the square atlas cell isn't re-stretched
        // on the wall -> pictures stay undistorted as they stack.
        skinBuildingsWithAtlas(osm, atlas, {
          mode: SKIN_MODE, center: [SF.lon, SF.lat], wallMeters: 16, floorMeters: 16,
        });
        console.log(`Skinned buildings: ${atlas.count} images, mode=${SKIN_MODE}.`);
      } catch (e) {
        console.warn("atlas skin failed, falling back to single placeholder:", e);
        muralImg = await resolveMuralImage(MURAL_IMAGE);
        skinBuildingsWithImage(osm, muralImg, { wallMeters: 14, floorMeters: 18 });
      }
      // Also allow click-to-place flat murals on specific walls.
      if (viewer.scene.pickPositionSupported) {
        muralImg = muralImg || await resolveMuralImage(MURAL_IMAGE);
        enableClickToPlaceMural(viewer, muralImg, { height: 120 });
      }
    } catch (e) { console.warn("OSM Buildings unavailable:", e); }
  } else {
    viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
      url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", maximumLevel: 19,
      credit: "© OpenStreetMap contributors",
    }));
  }

  // ---- the fleet (optional: map still works if trajectories aren't generated) ----
  let center = [SF.lon, SF.lat];
  try {
    // cache:no-store so a freshly regenerated (bigger) trajectories.json is never
    // served stale from the browser cache — otherwise you keep seeing the old fleet.
    const DATA = await (await fetch(TRAJ_URL, { cache: "no-store" })).json();
    center = DATA.meta.center || center;
    addFleet(viewer, DATA);
  } catch (e) {
    console.warn("no trajectories.json — showing the empty map:", e);
    msg(`<div>Map only — generate cars with
      <code>python -m smoothride.demo.export_web</code>, then reload.</div>`);
    setTimeout(() => document.getElementById("msg").removeAttribute("data-show"), 4000);
  }

  // Start zoomed into the downtown pocket. URL params let an external capture
  // harness frame a specific intersection/edge case:
  //   ?lon=&lat=&alt=&pitch=&heading=  -> camera   ?t=<frame>&pause=1 -> freeze time
  const q = new URLSearchParams(location.search);
  const camLon = parseFloat(q.get("lon")), camLat = parseFloat(q.get("lat"));
  const alt = parseFloat(q.get("alt")) || 300;
  const pitch = parseFloat(q.get("pitch")) || -32;
  const heading = parseFloat(q.get("heading")) || 0;
  const dest = (!isNaN(camLon) && !isNaN(camLat))
    ? Cesium.Cartesian3.fromDegrees(camLon, camLat, alt)
    : Cesium.Cartesian3.fromDegrees(center[0], center[1] - 0.0019, alt);
  viewer.camera.flyTo({
    destination: dest,
    orientation: { heading: Cesium.Math.toRadians(heading),
                   pitch: Cesium.Math.toRadians(pitch), roll: 0 },
    duration: q.has("lon") ? 0 : 1.5,
  });

  const tf = parseInt(q.get("t"), 10);
  if (!isNaN(tf) && viewer.clock) {
    viewer.clock.currentTime = Cesium.JulianDate.addSeconds(
      viewer.clock.startTime, tf * (window.__DT || 0.2), new Cesium.JulianDate());
    if (q.get("pause") === "1") viewer.clock.shouldAnimate = false;
  }
}

// Build the animation clock + both worlds: trained = 3D model fleet, untrained =
// faint "shadow world" points (gridlock), so the learning delta still reads.
function addFleet(viewer, DATA) {
  const NF = DATA.meta.n_steps, DT = DATA.meta.dt;
  window.__DT = DT;                 // for the ?t=<frame> capture param
  const start = Cesium.JulianDate.now();
  const stop = Cesium.JulianDate.addSeconds(start, NF * DT, new Cesium.JulianDate());
  Object.assign(viewer.clock, {
    startTime: start.clone(), stopTime: stop.clone(), currentTime: start.clone(),
    clockRange: Cesium.ClockRange.LOOP_STOP, multiplier: 2.0, shouldAnimate: true,
  });
  if (viewer.timeline) viewer.timeline.zoomTo(start, stop);

  const timeAt = (t) => Cesium.JulianDate.addSeconds(start, t * DT, new Cesium.JulianDate());

  // The 2D env continuously RESPAWNS a car (on goal/crash) at a new spot on its
  // route. That's a teleport: interpolating across it streaks the car over the
  // whole map and makes it pop in/out. So split each car's track into CONTINUOUS
  // trip segments — break wherever it jumps more than a car could plausibly move
  // in one step — and render each segment as its own entity that only exists
  // (availability) while on that drive. No streaks; spread spawns keep the field
  // full. JUMP scales off the speed limit so it adapts to the export's dt/stride.
  const JUMP = Math.max(25, (DATA.meta.vmax || 16) * DT * 5);
  function carto(c, t) { return Cesium.Cartesian3.fromDegrees(c.lng[t], c.lat[t], 0); }

  function segments(c) {
    const segs = [];
    let s = 0;
    for (let t = 1; t < NF; t++) {
      if (Cesium.Cartesian3.distance(carto(c, t - 1), carto(c, t)) > JUMP) {
        if (t - 1 > s) segs.push([s, t - 1]);
        s = t;
      }
    }
    if (NF - 1 > s) segs.push([s, NF - 1]);
    return segs;
  }

  // One persistent entity per car. To keep it ALWAYS on screen and never
  // teleporting, take the car's LONGEST continuous trip and HOLD position at both
  // ends (extrapolation) — so it drives its real route, then waits parked at a real
  // road spot instead of streaking away or blinking out on a respawn.
  function longestTrip(c) {
    const segs = segments(c);
    if (!segs.length) return null;
    return segs.reduce((a, b) => (b[1] - b[0] > a[1] - a[0] ? b : a));
  }
  function tripPos(c, t0, t1) {
    const pos = new Cesium.SampledPositionProperty();
    for (let t = t0; t <= t1; t++) pos.addSample(timeAt(t), carto(c, t));
    pos.setInterpolationOptions({ interpolationDegree: 1,
      interpolationAlgorithm: Cesium.LinearApproximation });
    pos.forwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
    pos.backwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
    return pos;
  }

  // Orientation from the exported heading, NOT velocity: VelocityOrientationProperty
  // goes undefined at zero speed, so a stopped/parked car snapped to a default facing
  // (the "turning sideways" glitch in stop-and-go). hdg is always defined. Cesium
  // heading = -hdg (our hdg is CCW-from-east; Cesium heading is CW-from-north about
  // the down axis) — verified the model's nose lands on +east with the orient probe.
  function tripOri(c, t0, t1) {
    const ori = new Cesium.SampledProperty(Cesium.Quaternion);
    for (let t = t0; t <= t1; t++) {
      const hpr = new Cesium.HeadingPitchRoll(-c.hdg[t], 0, 0);
      ori.addSample(timeAt(t), Cesium.Transforms.headingPitchRollQuaternion(carto(c, t), hpr));
    }
    ori.forwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
    ori.backwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
    return ori;
  }

  // Distance LOD so the city feels populated everywhere you pan WITHOUT lag and
  // WITHOUT cars popping in from nothing. Every car carries two graphics:
  //   * a cheap dot  — always present, so wherever you look there are cars;
  //   * the 3D model — only rendered within MODEL_FAR m of the camera (few dozen at
  //     a time -> cheap). They CROSS-FADE: as you approach, the dot fades out
  //     (translucencyByDistance) exactly as the model fades in, so it reads as
  //     lazy-loaded detail resolving, not a spawn. Cesium frustum-culls whatever is
  //     off-screen for free.
  const MODEL_FAR = 600;          // model drawn within this many metres of camera
  const carEntities = [];
  DATA.worlds.trained.cars.forEach((c, i) => {
    const trip = longestTrip(c);
    if (!trip) return;
    const look = carLook(i);
    carEntities.push(viewer.entities.add({
      position: tripPos(c, trip[0], trip[1]),
      orientation: tripOri(c, trip[0], trip[1]),
      point: {
        pixelSize: 7, color: look.color,
        outlineColor: Cesium.Color.BLACK.withAlpha(0.4), outlineWidth: 1,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        // invisible when close (model takes over), opaque far -> the placeholder
        translucencyByDistance: new Cesium.NearFarScalar(MODEL_FAR * 0.6, 0.0, MODEL_FAR, 1.0),
      },
      model: {
        uri: `./assets/${look.body}.glb`,
        minimumPixelSize: 24, maximumScale: 12, scale: 1.0,
        color: look.color, colorBlendMode: Cesium.ColorBlendMode.MIX, colorBlendAmount: 0.65,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0.0, MODEL_FAR),
      },
    }));
  });

  // ?track=<i> -> chase one car (handy for eyeballing orientation up close).
  const track = parseInt(new URLSearchParams(location.search).get("track"), 10);
  if (!isNaN(track) && carEntities[track]) viewer.trackedEntity = carEntities[track];

  // untrained world -> faint shadow points (the gridlock the policy fixes).
  (DATA.worlds.untrained ? DATA.worlds.untrained.cars : []).forEach((c) => {
    const trip = longestTrip(c);
    if (!trip) return;
    viewer.entities.add({
      position: tripPos(c, trip[0], trip[1]),
      point: { pixelSize: 7, color: AMBER, outlineColor: Cesium.Color.BLACK.withAlpha(0.4),
        outlineWidth: 1, heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY },
    });
  });

  setupHUD(viewer, DATA, start, DT, NF);
  addPedestrians(viewer, DATA, start, DT, NF);
}

// ---- pedestrians as simple 3D CHARACTERS (not dots) ----------------------
// Each ped is a little 3D figure — a body cylinder + a head sphere — that walks
// the sidewalks and stands on the terrain. Real world.peds trajectories are used
// when the export carries them; otherwise an ambient crowd is synthesized, anchored
// to real car routes, so the city reads as alive.

const PED_SHIRTS = ["#2a7de1", "#e14b4b", "#2bb673", "#e0a92b", "#7b5bd6", "#d94f8a", "#e7ecf3", "#1f9e9e"]
  .map((h) => Cesium.Color.fromCssColorString(h));
const PED_SKINS = ["#f0c9a8", "#e8b98c", "#c98a5b", "#a86a3c"]
  .map((h) => Cesium.Color.fromCssColorString(h));
function pedLook(i) {
  const r = mulberry32(0xa11ce + (i * 2654435761 | 0));
  return { shirt: PED_SHIRTS[Math.floor(r() * PED_SHIRTS.length)],
           skin: PED_SKINS[Math.floor(r() * PED_SKINS.length)] };
}

// Synthesize an ambient crowd: anchor each walker near a real car-route point (so
// they end up beside streets), nudge to the sidewalk, then random-walk slowly and
// stay within a small radius — a believable stroll, not RL agents.
function synthPeds(DATA, NF, DT) {
  const cars = DATA.worlds.trained.cars;
  if (!cars || !cars.length) return [];
  const n = Math.min(30, Math.max(16, Math.round(cars.length / 6)));
  const rng = mulberry32(0x9ed5eed);
  const peds = [];
  for (let i = 0; i < n; i++) {
    const c = cars[Math.floor(rng() * cars.length)];
    const t0 = Math.floor(rng() * Math.max(1, c.lng.length - 1));
    const lat0 = c.lat[t0], lng0 = c.lng[t0], cosl = Math.cos(lat0 * Math.PI / 180);
    const off = 6 + rng() * 9, a0 = rng() * Math.PI * 2;
    let x = Math.cos(a0) * off, y = Math.sin(a0) * off;
    let hd = rng() * Math.PI * 2; const spd = 0.7 + rng() * 0.9;
    const lng = [], lat = [];
    for (let t = 0; t < NF; t++) {
      hd += (rng() - 0.5) * 0.35;
      x += Math.cos(hd) * spd * DT; y += Math.sin(hd) * spd * DT;
      if (Math.hypot(x, y) > 32) hd += Math.PI;          // wander, but stay local
      lng.push(lng0 + x / (111320 * cosl));
      lat.push(lat0 + y / 111320);
    }
    peds.push({ lng, lat });
  }
  return peds;
}

function addPedestrians(viewer, DATA, start, DT, NF) {
  const w = DATA.worlds.trained;
  const peds = (w.peds && w.peds.length) ? w.peds : synthPeds(DATA, NF, DT);
  const timeAt = (t) => Cesium.JulianDate.addSeconds(start, t * DT, new Cesium.JulianDate());
  // A body part: the ped's (lng,lat) over time, lifted `h` metres above the ground
  // (RELATIVE_TO_GROUND so the figure stands on the terrain, not at sea level).
  function partPos(p, len, h) {
    const pos = new Cesium.SampledPositionProperty();
    for (let t = 0; t < len; t++)
      pos.addSample(timeAt(t), Cesium.Cartesian3.fromDegrees(p.lng[t], p.lat[t], h));
    pos.setInterpolationOptions({ interpolationDegree: 1, interpolationAlgorithm: Cesium.LinearApproximation });
    pos.forwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
    pos.backwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
    return pos;
  }
  peds.forEach((p, i) => {
    const len = Math.min(p.lng.length, NF);
    const look = pedLook(i);
    // body: a slightly tapered cylinder ~1.1 m tall (centre ~0.6 m up)
    viewer.entities.add({
      position: partPos(p, len, 0.6),
      cylinder: {
        length: 1.1, topRadius: 0.17, bottomRadius: 0.24,
        material: look.shirt,
        heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND,
      },
    });
    // head: a small sphere (~1.38 m up)
    viewer.entities.add({
      position: partPos(p, len, 1.38),
      ellipsoid: {
        radii: new Cesium.Cartesian3(0.16, 0.16, 0.19),
        material: look.skin,
        heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND,
      },
    });
  });
}

// Live tracker: recompute fleet metrics for the CURRENT frame on each clock tick.
//   Trips  = cumulative cars that have reached their destination (meta.trips_series,
//            else summed from per-car arr flags).
//   Cars   = fleet size.  Moving = cars with speed > 0.5 m/s.
//   Crashes= cars whose crash flag is set this frame.  Avg speed of the movers.
function setupHUD(viewer, DATA, start, dt, nf) {
  const cars = DATA.worlds.trained.cars;
  const meta = DATA.meta || {};
  const vmax = meta.vmax || 9;
  const n = cars.length;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set("m-cars", n);

  // ---- precompute per-frame series: moving / slowed / crashed / mean speed ----
  const mov = new Array(nf), jam = new Array(nf), cr = new Array(nf), ms = new Array(nf);
  for (let f = 0; f < nf; f++) {
    let m = 0, j = 0, c = 0, sp = 0, k = 0;
    for (const car of cars) {
      if (car.crash && car.crash[f]) { c++; continue; }
      const s = car.spd ? Math.max(0, car.spd[f]) : 0;
      if (s > 0.5) m++; else j++;
      sp += s; k++;
    }
    mov[f] = m; jam[f] = j; cr[f] = c; ms[f] = k ? sp / k : 0;
  }
  // trips: prefer the exported series, else accumulate from per-car arrival flags
  const trips = meta.trips_series || (() => {
    const a = new Array(nf);
    for (let f = 0; f < nf; f++) { let d = 0; for (const c of cars) if (c.arr && c.arr[f]) d++; a[f] = d; }
    return a;
  })();

  // ---- chart canvases (sized to their CSS box, DPR-aware) ----
  const ids = ["dc-trips", "dc-fleet", "dc-hist"];
  const ctx = {};
  function sizeCanvases() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    ids.forEach((id) => {
      const c = document.getElementById(id); if (!c) return;
      c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
      const x = c.getContext("2d"); x.setTransform(dpr, 0, 0, dpr, 0, 0); ctx[id] = x;
    });
  }
  sizeCanvases();
  window.addEventListener("resize", sizeCanvases);

  function drawLine(id, data, col, fill, f) {
    const x = ctx[id]; if (!x) return; const w = x.canvas.clientWidth, h = x.canvas.clientHeight;
    x.clearRect(0, 0, w, h);
    let mx = 1; for (const v of data) mx = Math.max(mx, v);
    x.strokeStyle = "rgba(255,255,255,.06)"; x.lineWidth = 1;
    for (let g = 0; g <= 2; g++) { const yy = 3 + g / 2 * (h - 9); x.beginPath(); x.moveTo(0, yy); x.lineTo(w, yy); x.stroke(); }
    const pt = (i) => [i / (nf - 1) * w, h - 3 - data[i] / mx * (h - 9)];
    if (fill) { x.beginPath(); for (let i = 0; i < nf; i++) { const [xx, yy] = pt(i); i ? x.lineTo(xx, yy) : x.moveTo(xx, yy); } x.lineTo(w, h); x.lineTo(0, h); x.closePath(); x.fillStyle = fill; x.fill(); }
    x.beginPath(); for (let i = 0; i < nf; i++) { const [xx, yy] = pt(i); i ? x.lineTo(xx, yy) : x.moveTo(xx, yy); } x.strokeStyle = col; x.lineWidth = 2; x.stroke();
    const px = f / (nf - 1) * w; x.strokeStyle = "rgba(52,211,153,.5)"; x.lineWidth = 1; x.beginPath(); x.moveTo(px, 0); x.lineTo(px, h); x.stroke();
  }
  function drawStack(id, f) {
    const x = ctx[id]; if (!x) return; const w = x.canvas.clientWidth, h = x.canvas.clientHeight;
    x.clearRect(0, 0, w, h);
    const mx = Math.max(1, n);
    const layer = (arr, base, col) => {
      x.beginPath();
      for (let i = 0; i < nf; i++) { const xx = i / (nf - 1) * w, yy = h - (base[i] + arr[i]) / mx * h; i ? x.lineTo(xx, yy) : x.moveTo(xx, yy); }
      for (let i = nf - 1; i >= 0; i--) { const xx = i / (nf - 1) * w, yy = h - base[i] / mx * h; x.lineTo(xx, yy); }
      x.closePath(); x.fillStyle = col; x.fill();
    };
    const b1 = mov.slice(), b2 = mov.map((m, i) => m + jam[i]);
    layer(mov, new Array(nf).fill(0), "rgba(52,211,153,.55)");
    layer(jam, b1, "rgba(245,158,11,.6)");
    layer(cr, b2, "rgba(239,68,68,.8)");
    const px = f / (nf - 1) * w; x.strokeStyle = "rgba(52,211,153,.5)"; x.lineWidth = 1; x.beginPath(); x.moveTo(px, 0); x.lineTo(px, h); x.stroke();
  }
  function drawHist(id, f) {
    const x = ctx[id]; if (!x) return; const w = x.canvas.clientWidth, h = x.canvas.clientHeight, B = 8, bins = new Array(B).fill(0);
    let cnt = 0;
    for (const car of cars) { if (car.crash && car.crash[f]) continue; const s = car.spd ? Math.max(0, car.spd[f]) : 0; bins[Math.min(B - 1, Math.floor(s / vmax * B))]++; cnt++; }
    const mx = Math.max(1, ...bins), bw = w / B;
    x.clearRect(0, 0, w, h);
    for (let i = 0; i < B; i++) { const bh = bins[i] / mx * (h - 3), xx = i * bw + 2; x.fillStyle = `hsl(${(i + 0.5) / B * 140},80%,52%)`; x.fillRect(xx, h - bh, bw - 4, bh); }
    set("dc-hn", cnt + " cars");
  }

  const update = () => {
    const f = Math.max(0, Math.min(nf - 1, Math.round(
      Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, start) / dt)));
    set("m-trips", trips[f]);
    set("m-moving", mov[f]);
    set("m-crashes", cr[f]);
    set("m-speed", (ms[f] * 2.23694).toFixed(0) + " mph");
    set("m-time", (f * dt).toFixed(0) + "s");
    const bar = (id, lab, v) => { const el = document.getElementById(id); if (el) el.style.width = (v / n * 100) + "%"; set(lab, v); };
    bar("fb-mov", "fbn-mov", mov[f]); bar("fb-jam", "fbn-jam", jam[f]); bar("fb-crash", "fbn-crash", cr[f]);
    drawLine("dc-trips", trips, "#34d399", "rgba(52,211,153,.12)", f);
    drawStack("dc-fleet", f);
    drawHist("dc-hist", f);
  };
  viewer.clock.onTick.addEventListener(update);
  update();
}

boot();
