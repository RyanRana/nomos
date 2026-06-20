// Real 3D San Francisco + the SmoothRide fleet driving on it.
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

// How to apply them: "single" = one photo fills each facade (default),
// "tile" = the busier repeating pattern. Switch with ?skin=tile.
const SKIN_MODE = new URLSearchParams(location.search).get("skin") === "tile" ? "tile" : "single";

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

  let muralImg = null;
  if (TOKEN) {
    try {
      const osm = await Cesium.createOsmBuildingsAsync();
      viewer.scene.primitives.add(osm);
      // Skin EVERY building, each randomly assigned one of the side images.
      try {
        const atlas = await buildImageAtlas(SIDE_IMAGES);
        skinBuildingsWithAtlas(osm, atlas, { mode: SKIN_MODE, center: [SF.lon, SF.lat] });
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

  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(center[0], center[1] - 0.0032, 420),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(-28), roll: 0 },
    duration: 1.5,
  });

}

// Build the animation clock + both worlds: trained = 3D model fleet, untrained =
// faint "shadow world" points (gridlock), so the learning delta still reads.
function addFleet(viewer, DATA) {
  const NF = DATA.meta.n_steps, DT = DATA.meta.dt;
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

  // trained world -> the random-colored 3D model fleet. Orientation is
  // velocity-derived (model +X is our forward axis) -> faces its travel direction.
  const carEntities = [];
  DATA.worlds.trained.cars.forEach((c, i) => {
    const trip = longestTrip(c);
    if (!trip) return;
    const look = carLook(i);
    const pos = tripPos(c, trip[0], trip[1]);
    carEntities.push(viewer.entities.add({
      position: pos, orientation: new Cesium.VelocityOrientationProperty(pos),
      model: {
        uri: `./assets/${look.body}.glb`,
        minimumPixelSize: 28, maximumScale: 12, scale: 1.0,
        color: look.color, colorBlendMode: Cesium.ColorBlendMode.MIX, colorBlendAmount: 0.65,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
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
}

boot();
