/* SmoothRide Cesium viewer — replays a scene.json (schema v1) of meshed cars on
   3D San Francisco. Terrain + OSM buildings when an ion token is present; flat
   ellipsoid + our extruded GeoJSON buildings otherwise. */

const CFG = window.SMOOTHRIDE_CONFIG || { cesiumIonToken: "" };
const WORLD = "trained";          // which world to animate
const CAR_L = 4.6, CAR_W = 2.0, CAR_H = 1.5;   // meters

async function main() {
  const hasToken = !!CFG.cesiumIonToken;
  if (hasToken) Cesium.Ion.defaultAccessToken = CFG.cesiumIonToken;

  const viewer = new Cesium.Viewer("cesiumContainer", {
    terrainProvider: hasToken
      ? await Cesium.createWorldTerrainAsync()
      : new Cesium.EllipsoidTerrainProvider(),
    // baseLayer:false => no default Cesium-ion imagery (which 401s without a
    // token). Without a token we drape free OpenStreetMap tiles instead, so the
    // real SF street grid is visible offline; with a token, ion world imagery.
    baseLayer: hasToken ? undefined : false,
    animation: true, timeline: true, baseLayerPicker: false, geocoder: false,
  });
  viewer.scene.globe.depthTestAgainstTerrain = true;
  window.viewer = viewer;   // handy for console debugging (single-viewer app)

  if (!hasToken) {
    viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
      url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      maximumLevel: 19, credit: "© OpenStreetMap contributors",
    }));
  }

  const scene = await (await fetch("public/scene.json")).json();
  if (scene.schema_version !== 1) throw new Error("unsupported schema " + scene.schema_version);
  const meta = scene.meta, world = scene.worlds[WORLD];

  // Buildings: prefer Cesium OSM Buildings (ion); else extrude our GeoJSON.
  if (hasToken) {
    try { viewer.scene.primitives.add(await Cesium.createOsmBuildingsAsync()); }
    catch (e) { console.warn("OSM Buildings failed, using GeoJSON", e); addGeoJsonBuildings(viewer, scene); }
  } else {
    addGeoJsonBuildings(viewer, scene);
  }

  drawRoads(viewer, scene.roads);

  // Time window for playback.
  const start = Cesium.JulianDate.now();
  const stop = Cesium.JulianDate.addSeconds(start, meta.dt * (meta.n_steps - 1), new Cesium.JulianDate());
  viewer.clock.startTime = start.clone();
  viewer.clock.stopTime = stop.clone();
  viewer.clock.currentTime = start.clone();
  viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
  viewer.clock.multiplier = 1.0;
  viewer.timeline.zoomTo(start, stop);

  world.cars.forEach((car) => addCar(viewer, car, start, meta));

  // HUD — trips and crashed both update LIVE at the current frame (start at 0),
  // not the end-of-run totals (which made "crashed" read non-zero before playback).
  document.getElementById("cars").textContent = world.summary.cars;
  document.getElementById("trips").textContent = world.trips_series[0];
  document.getElementById("crashed").textContent =
    world.cars.reduce((n, c) => n + (c.crash[0] || 0), 0);
  viewer.clock.onTick.addEventListener(() => {
    const frac = Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, start) / meta.dt;
    const f = Math.max(0, Math.min(world.trips_series.length - 1, Math.round(frac)));
    document.getElementById("trips").textContent = world.trips_series[f];
    document.getElementById("crashed").textContent =
      world.cars.reduce((n, c) => n + (c.crash[f] || 0), 0);
  });

  // Frame the city: fit the scene's bounding sphere with a fixed oblique tilt, so
  // the whole street grid fills the view at any bbox size (a fixed-altitude flyTo
  // leaves a small city as a speck under a horizon of the rest of the state).
  // bounds is [[wLon,sLat],[eLon,nLat]] (SW, NE).
  const [[wLon, sLat], [eLon, nLat]] = meta.bounds;
  const sphere = Cesium.BoundingSphere.fromPoints([
    Cesium.Cartesian3.fromDegrees(wLon, sLat),
    Cesium.Cartesian3.fromDegrees(eLon, nLat),
  ]);
  viewer.camera.flyToBoundingSphere(sphere, {
    offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-45), sphere.radius * 2.4),
    duration: 0,
  });
}

function sampledPosition(car, start, meta) {
  const p = new Cesium.SampledPositionProperty();
  for (let t = 0; t < car.lng.length; t++) {
    const when = Cesium.JulianDate.addSeconds(start, t * meta.dt, new Cesium.JulianDate());
    p.addSample(when, Cesium.Cartesian3.fromDegrees(car.lng[t], car.lat[t], (car.z[t] || 0) + CAR_H / 2));
  }
  return p;
}

function speedColor(spd, vmax) {
  const f = Math.max(0, Math.min(1, spd / (vmax || 16)));
  return Cesium.Color.fromHsl(0.33 * f, 0.9, 0.5);   // red(stopped) -> green(fast)
}

function frameIndex(car, start, time, meta) {
  const f = Cesium.JulianDate.secondsDifference(time, start) / meta.dt;
  return Math.max(0, Math.min(car.lng.length - 1, Math.round(f)));
}

function addCar(viewer, car, start, meta) {
  const pos = sampledPosition(car, start, meta);
  // Orient by the sim's OWN heading (car.hdg), not by velocity: hdg is the car's
  // true facing every step, so the box faces forward while driving and never
  // spins on a stop or a respawn-teleport (where velocity orientation goes wild).
  // Sim heading is radians CCW from east; Cesium heading is CW from north, hence
  // (pi/2 - hdg). Box dimensions.x (CAR_L, the length) lies along that heading.
  const orientation = new Cesium.CallbackProperty((time) => {
    const p = pos.getValue(time);
    if (!p) return undefined;
    // Box length (dimensions.x = body +X) must point along travel. Cesium heading
    // rotates about -Z (clockwise from East), and sim hdg is CCW from East, so the
    // angle that aligns +X with travel is -hdg (NOT pi/2-hdg, which faces sideways).
    const hdg = car.hdg[frameIndex(car, start, time, meta)];
    const hpr = new Cesium.HeadingPitchRoll(-hdg, 0, 0);
    return Cesium.Transforms.headingPitchRollQuaternion(p, hpr);
  }, false);
  viewer.entities.add({
    position: pos,
    orientation: orientation,
    box: {
      dimensions: new Cesium.Cartesian3(CAR_L, CAR_W, CAR_H),
      material: new Cesium.ColorMaterialProperty(new Cesium.CallbackProperty((time) => {
        const i = frameIndex(car, start, time, meta);
        return car.crash[i] ? Cesium.Color.DIMGRAY : speedColor(car.spd[i], meta.vmax);
      }, false)),
    },
  });
}

function drawRoads(viewer, roads) {
  roads.forEach((seg) => {
    viewer.entities.add({
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(
          [seg[0][0], seg[0][1], seg[0][2], seg[1][0], seg[1][1], seg[1][2]]),
        width: 3, material: Cesium.Color.fromCssColorString("#22d3ee").withAlpha(0.9),
        clampToGround: false,
      },
    });
  });
}

function addGeoJsonBuildings(viewer, scene) {
  const fc = scene.buildings;
  if (!fc || !fc.features) return;
  fc.features.forEach((ft) => {
    const ring = ft.geometry.coordinates[0];
    const flat = [];
    ring.forEach((p) => { flat.push(p[0], p[1]); });
    viewer.entities.add({
      polygon: {
        hierarchy: Cesium.Cartesian3.fromDegreesArray(flat),
        extrudedHeight: ft.properties.height || 8,
        material: Cesium.Color.fromCssColorString("#1a2230").withAlpha(0.9),
        outline: false,
      },
    });
  });
}

main().catch((e) => { console.error(e); alert("Viewer error: " + e.message); });
