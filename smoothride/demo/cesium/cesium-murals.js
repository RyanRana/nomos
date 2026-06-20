// cesium-murals.js
// Stick flat images onto the sides of Cesium OSM Buildings.
// Globals: addMuralToBuilding(viewer, opts), enableClickToPlaceMural(viewer, image, opts)

/**
 * Add a flat image (mural) onto a building facade, oriented by two base corners.
 *
 * @param {Cesium.Viewer} viewer
 * @param {object} opts
 * @param {[number, number]} opts.p1   - [lon, lat] of the bottom-LEFT corner
 * @param {[number, number]} opts.p2   - [lon, lat] of the bottom-RIGHT corner
 * @param {number} opts.base           - bottom height of the image (meters)
 * @param {number} opts.top            - top height of the image (meters)
 * @param {string} opts.image          - image url (PNG with alpha is fine)
 * @param {number} [opts.offset=0.8]   - push the wall this many meters out from the
 *                                        facade to avoid z-fighting flicker
 * @param {boolean} [opts.transparent=true]
 * @param {[number, number]} [opts.repeat=[1, 1]] - tile/stretch factor
 * @returns {Cesium.Entity}
 */
function addMuralToBuilding(viewer, opts) {
  const {
    p1, p2, base, top, image,
    offset = 0.8, transparent = true, repeat = [1, 1],
  } = opts;

  const [a, b] = offsetEdgeOutward(p1, p2, offset);

  return viewer.entities.add({
    wall: {
      positions: Cesium.Cartesian3.fromDegreesArrayHeights([
        a[0], a[1], base,
        b[0], b[1], base,
      ]),
      maximumHeights: [top, top],
      minimumHeights: [base, base],
      material: new Cesium.ImageMaterialProperty({
        image,
        transparent,
        repeat: new Cesium.Cartesian2(repeat[0], repeat[1]),
      }),
    },
  });
}

/**
 * Load several images and pack them into one square-ish atlas canvas.
 * @param {string[]} urls
 * @returns {Promise<{canvas: HTMLCanvasElement, cols: number, rows: number, count: number}>}
 */
async function buildImageAtlas(urls) {
  const imgs = await Promise.all(urls.map((u) => new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("failed to load " + u));
    im.src = u;
  })));

  const count = imgs.length;
  const cols = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / cols);
  const cell = 512; // each atlas cell, px

  const canvas = document.createElement("canvas");
  canvas.width = cols * cell;
  canvas.height = rows * cell;
  const g = canvas.getContext("2d");
  imgs.forEach((im, i) => {
    const cx = (i % cols) * cell;
    const cy = Math.floor(i / cols) * cell;
    g.drawImage(im, cx, cy, cell, cell); // stretch each into its cell
  });

  return { canvas, cols, rows, count };
}

/**
 * Skin EVERY building, randomly assigning each one a different image from an atlas.
 * A building's image is chosen by hashing its ground location, so it's consistent
 * across all that building's fragments and stable on zoom/pan.
 *
 * Two looks via opts.mode:
 *   "single" (default) - ONE image rendered to fill each facade (the photo reads as
 *                        the building's side). Use when images are flat facade photos.
 *   "tile"             - the image repeats as a pattern across the wall (busier design).
 *
 * Coordinates are computed in a fixed East-North-Up frame anchored at opts.center,
 * passed in as CPU-computed uniforms. This avoids the undocumented model matrices and
 * the "low precision" positionWC pitfalls, so heights and tiling are correct & stable.
 *
 * @param {Cesium.Cesium3DTileset} tileset
 * @param {{canvas: HTMLCanvasElement, cols: number, rows: number, count: number}} atlas
 * @param {object} [opts]
 * @param {[number, number]} [opts.center=[-122.409, 37.7886]] - lon/lat to anchor the ENU frame
 * @param {"single"|"tile"} [opts.mode="single"]
 * @param {number} [opts.fitWidth=38]   - single: image width spans this many meters
 * @param {number} [opts.fitHeight=60]  - single: image height spans this many meters (then clamps)
 * @param {number} [opts.wallMeters=14] - tile: repeat width, meters
 * @param {number} [opts.floorMeters=18]- tile: repeat height, meters
 * @param {number} [opts.groupMeters=45]- buildings within this grid cell share one image
 * @returns {Cesium.CustomShader}
 */
function skinBuildingsWithAtlas(tileset, atlas, opts = {}) {
  const {
    center = [-122.409, 37.7886],
    mode = "single",
    fitWidth = 38, fitHeight = 60,
    wallMeters = 14, floorMeters = 18,
    groupMeters = 45,
  } = opts;

  // Fixed local frame at the scene center (full CPU precision). The basis barely
  // changes across a city, so one frame is plenty for correct up/east/north.
  const origin = Cesium.Cartesian3.fromDegrees(center[0], center[1], 0);
  const enu = Cesium.Transforms.eastNorthUpToFixedFrame(origin);
  const col = (i) => {
    const c = Cesium.Matrix4.getColumn(enu, i, new Cesium.Cartesian4());
    return new Cesium.Cartesian3(c.x, c.y, c.z);
  };
  const east = col(0), north = col(1), up = col(2);

  const shader = new Cesium.CustomShader({
    lightingModel: Cesium.LightingModel.PBR,
    uniforms: {
      u_atlas: {
        type: Cesium.UniformType.SAMPLER_2D,
        value: new Cesium.TextureUniform({ url: atlas.canvas.toDataURL() }),
      },
      u_origin: { type: Cesium.UniformType.VEC3, value: origin },
      u_east: { type: Cesium.UniformType.VEC3, value: east },
      u_north: { type: Cesium.UniformType.VEC3, value: north },
      u_up: { type: Cesium.UniformType.VEC3, value: up },
      u_cols: { type: Cesium.UniformType.FLOAT, value: atlas.cols },
      u_rows: { type: Cesium.UniformType.FLOAT, value: atlas.rows },
      u_count: { type: Cesium.UniformType.FLOAT, value: atlas.count },
      u_mode: { type: Cesium.UniformType.FLOAT, value: mode === "tile" ? 1.0 : 0.0 },
      u_fitWidth: { type: Cesium.UniformType.FLOAT, value: fitWidth },
      u_fitHeight: { type: Cesium.UniformType.FLOAT, value: fitHeight },
      u_wallMeters: { type: Cesium.UniformType.FLOAT, value: wallMeters },
      u_floorMeters: { type: Cesium.UniformType.FLOAT, value: floorMeters },
      u_groupMeters: { type: Cesium.UniformType.FLOAT, value: groupMeters },
    },
    fragmentShaderText: `
      void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
        // Position in the fixed local ENU frame (meters from scene center).
        vec3 rel = fsInput.attributes.positionWC - u_origin;
        // World-space surface normal (eye normal -> world; czm_inverseView works here).
        vec3 nWC = normalize((czm_inverseView * vec4(fsInput.attributes.normalEC, 0.0)).xyz);
        if (abs(dot(nWC, u_up)) >= 0.5) return;             // skip roofs/ground

        vec3 tangentW = normalize(cross(u_up, nWC));        // horizontal along the wall
        float along  = dot(rel, tangentW);                  // meters along the wall
        float height = dot(rel, u_up);                       // meters above ground

        vec2 local;
        if (u_mode < 0.5) {
          // SINGLE: one photo fills the facade base-to-top.
          local = vec2(fract(along / u_fitWidth), clamp(height / u_fitHeight, 0.0, 1.0));
        } else {
          // TILE: repeat the image as a pattern.
          local = fract(vec2(along / u_wallMeters, height / u_floorMeters));
        }
        local = clamp(local, 0.003, 0.997);                 // stay inside the atlas cell

        // Pick this building's image by hashing its horizontal location.
        vec2 gcell = floor(vec2(dot(rel, u_east), dot(rel, u_north)) / u_groupMeters);
        float h = fract(sin(dot(gcell, vec2(12.9898, 78.233))) * 43758.5453);
        float idx = min(floor(h * u_count), u_count - 1.0);
        float ac = mod(idx, u_cols);
        float ar = floor(idx / u_cols);

        vec2 atlasUV = (vec2(ac, ar) + local) / vec2(u_cols, u_rows);
        material.diffuse = texture(u_atlas, atlasUV).rgb;
      }
    `,
  });

  tileset.customShader = shader;
  return shader;
}

/**
 * Skin EVERY building in a 3D Tileset with an image, in one GPU shader.
 *
 * OSM Buildings are untextured extruded footprints (no UVs), so we synthesize the
 * texture coordinates in the fragment shader from world position + normal:
 *   - classify each fragment as wall vs roof by how vertical its normal is
 *   - on walls, tile the image: horizontal axis along the wall, vertical axis = height
 * Result: all facades show the image at once, with zero extra geometry.
 *
 * @param {Cesium.Cesium3DTileset} tileset   - e.g. the OSM Buildings primitive
 * @param {string|HTMLCanvasElement} image
 * @param {object} [opts]
 * @param {number} [opts.wallMeters=14] - image width mapped across this many meters
 * @param {number} [opts.floorMeters=18]- image height mapped across this many meters
 * @param {boolean} [opts.skinRoofs=false]
 * @returns {Cesium.CustomShader}
 */
function skinBuildingsWithImage(tileset, image, opts = {}) {
  const { wallMeters = 14, floorMeters = 18, skinRoofs = false } = opts;
  const url = typeof image === "string" ? image : image.toDataURL();

  const shader = new Cesium.CustomShader({
    lightingModel: Cesium.LightingModel.PBR, // keep sun shading so depth still reads
    uniforms: {
      u_image: {
        type: Cesium.UniformType.SAMPLER_2D,
        value: new Cesium.TextureUniform({ url }),
      },
      u_wallMeters: { type: Cesium.UniformType.FLOAT, value: wallMeters },
      u_floorMeters: { type: Cesium.UniformType.FLOAT, value: floorMeters },
      u_skinRoofs: { type: Cesium.UniformType.FLOAT, value: skinRoofs ? 1.0 : 0.0 },
    },
    fragmentShaderText: `
      void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
        // Orientation from WORLD space (directions only -> precision-tolerant).
        vec3 posW = fsInput.attributes.positionWC;
        vec3 up   = normalize(posW);                        // geocentric up ~ local up
        vec3 nWC  = normalize((czm_inverseView * vec4(fsInput.attributes.normalEC, 0.0)).xyz);
        float verticality = abs(dot(nWC, up));              // ~0 wall, ~1 roof/ground

        // Texture coords from MODEL space: positionMC is small & precise, and is
        // independent of the camera -> the skin does NOT swim or change on zoom.
        vec3 posM = fsInput.attributes.positionMC;

        if (verticality < 0.5) {                            // a wall facade
          vec3 tangentW = normalize(cross(up, nWC));        // horizontal along the wall
          vec3 tM = normalize((czm_inverseModel * vec4(tangentW, 0.0)).xyz);
          vec3 uM = normalize((czm_inverseModel * vec4(up, 0.0)).xyz);
          float u = dot(posM, tM) / u_wallMeters;
          float v = dot(posM, uM) / u_floorMeters;          // height in "floors"
          material.diffuse = texture(u_image, fract(vec2(u, v))).rgb;
        } else if (u_skinRoofs > 0.5) {                     // optional: roofs too
          vec3 east  = normalize(cross(vec3(0.0, 0.0, 1.0), up));
          vec3 north = cross(up, east);
          vec3 eM = normalize((czm_inverseModel * vec4(east, 0.0)).xyz);
          vec3 nM = normalize((czm_inverseModel * vec4(north, 0.0)).xyz);
          float u = dot(posM, eM) / u_wallMeters;
          float v = dot(posM, nM) / u_wallMeters;
          material.diffuse = texture(u_image, fract(vec2(u, v))).rgb;
        }
      }
    `,
  });

  tileset.customShader = shader;
  return shader;
}

/**
 * Scatter N murals onto real buildings around a point and in the camera's view.
 * Heights are sampled from the live scene: each mural spans its building's south
 * facade (the face the demo camera looks at) and floats a few meters in front so
 * it's never buried inside the geometry.
 *
 * @param {Cesium.Viewer} viewer
 * @param {string|HTMLCanvasElement} image
 * @param {[number, number]} center      - [lon, lat] to scatter around
 * @param {number} [n=5]
 * @returns {Promise<Cesium.Entity[]>}
 */
async function addRandomMurals(viewer, image, center, n = 5) {
  const [clon, clat] = center;
  const mPerDegLat = 111320;
  const mPerDegLon = 111320 * Math.cos(Cesium.Math.toRadians(clat));
  const HALF = 11;   // half the mural width, meters (so ~22m wide)
  const FRONT = 7;   // float this far south, toward the camera, meters

  // Scatter candidate points north of the look point (i.e. into the view).
  const picks = [];
  for (let i = 0; i < n; i++) {
    picks.push([
      clon + (Math.random() - 0.5) * 0.004,
      clat + Math.random() * 0.003 + 0.0003,
    ]);
  }

  // Roof height: clamp a high point straight down onto buildings+terrain.
  const fromAbove = picks.map(([lo, la]) => Cesium.Cartesian3.fromDegrees(lo, la, 600));
  let roofs = [];
  try { roofs = await viewer.scene.clampToHeightMostDetailed(fromAbove); }
  catch (e) { console.warn("clampToHeight failed:", e); }

  // Ground height: sample the terrain under each point.
  let ground = picks.map(() => 0);
  try {
    const cartos = picks.map(([lo, la]) => Cesium.Cartographic.fromDegrees(lo, la));
    await Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, cartos);
    ground = cartos.map((c) => c.height || 0);
  } catch (e) { console.warn("sampleTerrain failed:", e); }

  const out = [];
  picks.forEach(([lo, la], i) => {
    const groundH = ground[i];
    const roofH = roofs[i] ? Cesium.Cartographic.fromCartesian(roofs[i]).height : groundH + 50;
    let base = groundH + 3;
    let top = roofH - 2;
    if (top - base < 25) top = base + 50; // street-level fallback: free-standing banner

    const latF = la - FRONT / mPerDegLat;   // nudge south, toward the camera
    const dLon = HALF / mPerDegLon;
    out.push(addMuralToBuilding(viewer, {
      p1: [lo - dLon, latF],
      p2: [lo + dLon, latF],
      base, top, image, offset: 0,         // already offset toward camera
    }));
  });

  console.log(`Placed ${out.length} murals around`, center);
  return out;
}

/**
 * Cover EVERY building in the camera's view with a mural. Lays a grid over the
 * visible area, clamps each grid point onto the scene, and where a point lands on
 * something tall enough to be a building (roof - ground > minHeight) it pastes a
 * facade mural there. Spacing ~one hit per typical building footprint.
 *
 * @param {Cesium.Viewer} viewer
 * @param {string|HTMLCanvasElement} image
 * @param {[number, number]} center        - [lon, lat] the view is centered on
 * @param {object} [opts]
 * @param {number} [opts.lonSpan=0.012]    - E-W extent of the grid, degrees
 * @param {number} [opts.latStart=-0.005]  - S edge relative to center, degrees
 * @param {number} [opts.latEnd=0.006]     - N edge relative to center, degrees
 * @param {number} [opts.step=0.00045]     - grid spacing (~50m), degrees
 * @param {number} [opts.minHeight=8]      - min roof-above-ground to count as a building
 * @returns {Promise<Cesium.Entity[]>}
 */
async function addMuralsAcrossView(viewer, image, center, opts = {}) {
  const [clon, clat] = center;
  const {
    lonSpan = 0.012, latStart = -0.005, latEnd = 0.006,
    step = 0.00045, minHeight = 8,
  } = opts;
  const mPerDegLat = 111320;
  const mPerDegLon = 111320 * Math.cos(Cesium.Math.toRadians(clat));

  const pts = [];
  for (let dlon = -lonSpan / 2; dlon <= lonSpan / 2; dlon += step)
    for (let dlat = latStart; dlat <= latEnd; dlat += step)
      pts.push([clon + dlon, clat + dlat]);

  const fromAbove = pts.map(([lo, la]) => Cesium.Cartesian3.fromDegrees(lo, la, 600));
  let roofs = [];
  try { roofs = await viewer.scene.clampToHeightMostDetailed(fromAbove); }
  catch (e) { console.warn("clampToHeight failed:", e); return []; }

  const cartos = pts.map(([lo, la]) => Cesium.Cartographic.fromDegrees(lo, la));
  try { await Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, cartos); }
  catch (e) { console.warn("sampleTerrain failed:", e); }

  const out = [];
  pts.forEach(([lo, la], i) => {
    if (!roofs[i]) return;
    const groundH = cartos[i].height || 0;
    const roofH = Cesium.Cartographic.fromCartesian(roofs[i]).height;
    if (roofH - groundH < minHeight) return; // bare ground, no building

    const base = groundH + 3;
    const top = roofH - 2;
    const latF = la - 7 / mPerDegLat;        // float south, toward the camera
    const dLon = 11 / mPerDegLon;
    out.push(addMuralToBuilding(viewer, {
      p1: [lo - dLon, latF],
      p2: [lo + dLon, latF],
      base, top, image, offset: 0,
    }));
  });

  console.log(`Placed ${out.length} murals across the view (${pts.length} grid points tested).`);
  return out;
}

/** Shift a 2-point edge sideways (perpendicular) by `meters`, in lon/lat space. */
function offsetEdgeOutward(p1, p2, meters) {
  const midLat = (p1[1] + p2[1]) / 2;
  const mPerDegLat = 111320;
  const mPerDegLon = 111320 * Math.cos(Cesium.Math.toRadians(midLat));

  const dx = (p2[0] - p1[0]) * mPerDegLon;
  const dy = (p2[1] - p1[1]) * mPerDegLat;
  const len = Math.hypot(dx, dy) || 1;

  const nxDeg = (-dy / len) * meters / mPerDegLon;
  const nyDeg = (dx / len) * meters / mPerDegLat;

  return [
    [p1[0] + nxDeg, p1[1] + nyDeg],
    [p2[0] + nxDeg, p2[1] + nyDeg],
  ];
}

/**
 * Interactive placement: click the bottom-LEFT corner, then the bottom-RIGHT
 * corner of a facade. A mural is added between them.
 * Returns a cleanup function that removes the click handler.
 *
 * @param {Cesium.Viewer} viewer
 * @param {string} image
 * @param {object} [opts]
 * @param {number} [opts.height=120]   - how tall the mural rises above the base
 * @param {(info:object)=>void} [opts.onPlace] - called with the placement coords
 */
function enableClickToPlaceMural(viewer, image, opts = {}) {
  const { height = 120, onPlace } = opts;
  const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
  let firstClick = null;

  const read = (clickPos) => {
    const cartesian = viewer.scene.pickPosition(clickPos);
    if (!Cesium.defined(cartesian)) return null;
    const c = Cesium.Cartographic.fromCartesian(cartesian);
    return {
      lon: Cesium.Math.toDegrees(c.longitude),
      lat: Cesium.Math.toDegrees(c.latitude),
      height: c.height,
    };
  };

  handler.setInputAction((click) => {
    const pt = read(click.position);
    if (!pt) { console.warn("Mural: no surface there — click on a building."); return; }

    if (!firstClick) {
      firstClick = pt;
      console.log("Mural corner 1 set — now click the other bottom corner.");
      return;
    }

    const base = Math.min(firstClick.height, pt.height);
    const info = {
      p1: [firstClick.lon, firstClick.lat],
      p2: [pt.lon, pt.lat],
      base, top: base + height,
    };
    addMuralToBuilding(viewer, { ...info, image });
    console.log("Mural placed:", info);
    if (onPlace) onPlace(info);
    firstClick = null;
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

  return () => handler.destroy();
}
