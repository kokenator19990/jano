// ================================================================
//  FaceLab Analytics — 3D Digital Clone System  v3.0
//  Three.js r162 · GLTF real heads · Bloom · Facial mesh overlay · Mouse parallax
// ================================================================

import * as THREE         from 'three';
import { OrbitControls }  from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader }     from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass }     from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass }from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass }     from 'three/addons/postprocessing/OutputPass.js';

// ─── PRESETS ─────────────────────────────────────────────────────
export const PRESETS = [
  { id: 0, label: 'Avatar A', sublabel: 'Femenino',  modelUrl: './models/head_female.glb' },
  { id: 1, label: 'Avatar B', sublabel: 'Masculino', modelUrl: './models/head_male.glb'   },
];

// All models normalised to this height (world units)
const TARGET_H = 2.2;

// ─── STATE ───────────────────────────────────────────────────────
let scene, camera, renderer, composer, controls, clock;
let head          = null;   // THREE.Group (model + landmark overlay)
let currentPreset = 0;
let raf           = null;
let loadingEl     = null;   // DOM overlay during GLB load
let loadSeq       = 0;      // cancels superseded loads on rapid preset switch

// Mouse parallax
let tRotY = 0, tRotX = 0;
const P_YAW   = 0.22;
const P_PITCH = 0.10;
const P_LERP  = 0.055;

const gltfLoader = new GLTFLoader();

// ─── PUBLIC API ──────────────────────────────────────────────────

export function init(canvas) {
  _teardown();

  const wrap = canvas.parentElement;
  const W = wrap.clientWidth  || 480;
  const H = wrap.clientHeight || 360;

  // ── Loading overlay (created once, persists across init calls) ─
  if (!loadingEl) {
    // Inject spin keyframe once
    if (!document.getElementById('_av_spin')) {
      const s = document.createElement('style');
      s.id = '_av_spin';
      s.textContent = '@keyframes _av_spin{to{transform:rotate(360deg)}}';
      document.head.appendChild(s);
    }
    loadingEl = document.createElement('div');
    Object.assign(loadingEl.style, {
      position: 'absolute', inset: '0', display: 'none',
      flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(5,16,26,0.88)', backdropFilter: 'blur(6px)',
      color: '#00cfe0', fontFamily: "'Space Grotesk',sans-serif",
      fontSize: '0.70rem', letterSpacing: '0.16em', gap: '14px',
      zIndex: '10', pointerEvents: 'none',
    });
    loadingEl.innerHTML = `
      <svg width="34" height="34" viewBox="0 0 34 34" fill="none"
           style="animation:_av_spin 0.9s linear infinite">
        <circle cx="17" cy="17" r="13" stroke="rgba(0,207,224,0.18)" stroke-width="2"/>
        <path d="M17 4 A13 13 0 0 1 30 17" stroke="#00cfe0" stroke-width="2.2"
              stroke-linecap="round"/>
      </svg>
      <span>CARGANDO MODELO</span>`;
    wrap.style.position = 'relative';
    wrap.appendChild(loadingEl);
  }

  // ── Scene ───────────────────────────────────────────────────────
  scene  = new THREE.Scene();
  clock  = new THREE.Clock();
  camera = new THREE.PerspectiveCamera(28, W / H, 0.05, 100);
  camera.position.set(0, 0, 5.6);

  // ── Renderer ────────────────────────────────────────────────────
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.toneMapping         = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled   = true;
  renderer.shadowMap.type      = THREE.PCFSoftShadowMap;

  // ── Post-processing chain: RenderPass → BloomPass → OutputPass ─
  //  OutputPass must be last — it applies tone mapping + sRGB gamma.
  //  Bloom threshold 3.50:
  //    • Skin linear lum ≈ 2.9  → below threshold → skin does NOT bloom ✓
  //    • Dot  linear lum ≈ 5.6  → above threshold → dots bloom ✓
  //    • Line linear lum ≈ 4.0  → above threshold → lines bloom ✓
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(W, H), 1.55, 0.50, 3.50));
  composer.addPass(new OutputPass());

  // ── Lights ──────────────────────────────────────────────────────
  _addLights();

  // ── Orbit controls ──────────────────────────────────────────────
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.065;
  controls.minDistance   = 2;
  controls.maxDistance   = 14;
  controls.target.set(0, 0, 0);
  controls.minPolarAngle = 0.06;
  controls.maxPolarAngle = Math.PI - 0.06;

  // ── Mouse parallax ──────────────────────────────────────────────
  document.addEventListener('mousemove', _onMouse);

  // ── Resize observer ─────────────────────────────────────────────
  new ResizeObserver(() => {
    const nW = wrap.clientWidth, nH = wrap.clientHeight;
    if (!nW || !nH) return;
    camera.aspect = nW / nH;
    camera.updateProjectionMatrix();
    renderer.setSize(nW, nH);
    composer.setSize(nW, nH);
  }).observe(wrap);

  renderLoop();
  loadPreset(0);
}

export function switchPreset(idx) {
  currentPreset = idx;
  loadPreset(idx);
}

export function applyPhoto(imgEl) {
  if (!head) return;

  // Identify the primary skin mesh (largest vertex count in the loaded model)
  let mainMesh = null, maxV = 0;
  head.traverse(obj => {
    if (!obj.isMesh) return;
    const n = obj.geometry?.attributes?.position?.count ?? 0;
    if (n > maxV) { maxV = n; mainMesh = obj; }
  });
  if (!mainMesh) return;

  // Build a square canvas texture with the uploaded photo centred on face
  const tc = document.createElement('canvas');
  tc.width = tc.height = 1024;
  const ctx = tc.getContext('2d');
  ctx.fillStyle = '#C8956C';
  ctx.fillRect(0, 0, 1024, 1024);

  const cx = 512, cy = 500, rx = 320, ry = 380;
  const sw = imgEl.naturalWidth  || imgEl.width  || 640;
  const sh = imgEl.naturalHeight || imgEl.height || 480;
  const sz = Math.min(sw, sh);
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.clip();
  ctx.drawImage(imgEl, (sw - sz) / 2, (sh - sz) / 2, sz, sz,
    cx - rx, cy - ry, rx * 2, ry * 2);
  ctx.restore();

  // Soft edge blend
  const grad = ctx.createRadialGradient(cx, cy, rx * 0.80, cx, cy, rx * 1.02);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, '#C8956C');
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx * 1.05, ry * 1.05, 0, 0, Math.PI * 2);
  ctx.clip();
  ctx.fillStyle = grad;
  ctx.fillRect(cx - rx * 1.1, cy - ry * 1.1, rx * 2.2, ry * 2.2);
  ctx.restore();

  const tex = new THREE.CanvasTexture(tc);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;

  const _apply = m => {
    m.map?.dispose();
    m.map = tex;
    m.color.setHex(0xffffff);
    m.needsUpdate = true;
  };
  Array.isArray(mainMesh.material)
    ? mainMesh.material.forEach(_apply)
    : _apply(mainMesh.material);
}

// ─── INTERNAL ────────────────────────────────────────────────────

function loadPreset(idx) {
  const seq    = ++loadSeq;
  const preset = PRESETS[idx];

  _showLoading(true);

  // Remove previous head immediately so the scene is clear during load
  if (head) { scene.remove(head); _disposeGroup(head); head = null; }

  gltfLoader.load(
    preset.modelUrl,

    gltf => {
      // Discard if a newer loadPreset() call was made while we were loading
      if (seq !== loadSeq) return;

      const model = gltf.scene;

      // ── Normalise: scale so bbox height = TARGET_H, centre at origin ─
      const b0 = new THREE.Box3().setFromObject(model);
      const s0 = b0.getSize(new THREE.Vector3());
      const c0 = b0.getCenter(new THREE.Vector3());
      const k  = TARGET_H / s0.y;

      model.scale.setScalar(k);
      model.position.set(-c0.x * k, -c0.y * k, -c0.z * k);
      model.updateWorldMatrix(true, true);   // flush matrices before bbox re-read

      // ── Shadows on all meshes ─────────────────────────────────────
      model.traverse(o => { if (o.isMesh) o.castShadow = o.receiveShadow = true; });

      // ── Recompute bbox of normalised model ────────────────────────
      const b1 = new THREE.Box3().setFromObject(model);
      const c1 = b1.getCenter(new THREE.Vector3());
      const s1 = b1.getSize(new THREE.Vector3());

      // ── Assemble head group: model + holographic landmark overlay ──
      head = new THREE.Group();
      head.add(model);
      head.add(_buildLandmarks(c1, s1.x * 0.5, s1.y * 0.5, s1.z * 0.5));
      scene.add(head);

      // ── Auto-frame camera ─────────────────────────────────────────
      //  distance so head fills ~80 % of viewport height at FOV 28°
      const fovR = camera.fov * (Math.PI / 180);
      const dist = (s1.y * 0.52) / Math.tan(fovR / 2);
      const tY   = c1.y + s1.y * 0.04;   // slight upward orbit pivot
      controls.target.set(c1.x, tY, c1.z);
      camera.position.set(c1.x, tY, c1.z + dist);
      camera.lookAt(controls.target);
      controls.update();

      _showLoading(false);
    },

    null,   // progress callback (unused)

    err => {
      if (seq === loadSeq) {
        console.error('[Avatar] Failed to load:', preset.modelUrl, err);
        _showLoading(false);
      }
    }
  );
}

function _showLoading(v) {
  if (loadingEl) loadingEl.style.display = v ? 'flex' : 'none';
}

function _onMouse(e) {
  tRotY =  ((e.clientX / window.innerWidth)  * 2 - 1) * P_YAW;
  tRotX = -((e.clientY / window.innerHeight) * 2 - 1) * P_PITCH * 0.55;
}

// ─── LIGHTING ────────────────────────────────────────────────────
function _addLights() {
  // Hemisphere ambient: sky (blue) + ground (warm)
  scene.add(new THREE.HemisphereLight(0x4A6880, 0x2A1810, 1.6));

  // Key: warm, upper-left-front
  const key = new THREE.DirectionalLight(0xFFF5EC, 4.2);
  key.position.set(-2, 3, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  scene.add(key);

  // Fill: cool, right-mid
  const fill = new THREE.DirectionalLight(0xCCDDFF, 1.4);
  fill.position.set(4, 0.5, 2);
  scene.add(fill);

  // Rim: blue-cool, back-top (separates head from background)
  const rim = new THREE.DirectionalLight(0x88AAFF, 3.0);
  rim.position.set(0, 2, -8);
  scene.add(rim);

  // Under-fill: slight warm bounce from below
  const under = new THREE.DirectionalLight(0xFFE8CC, 0.40);
  under.position.set(0, -5, 2);
  scene.add(under);
}

// ─── HOLOGRAPHIC FACIAL LANDMARK OVERLAY ─────────────────────────
//
//  35 anatomical landmarks positioned as fractions of the model's
//  normalised bounding-box half-extents:
//    world_x = cx + fx * hw
//    world_y = cy + fy * hh        (hh ≈ TARGET_H / 2 = 1.1)
//    world_z = cz + fz * hd + ZOFF (ZOFF prevents z-fighting with skin)
//
//  HDR colours make landmarks bloom selectively:
//    Dot  color.setRGB(0, 7,   8  ) → linear lum ≈ 5.6 > threshold 3.5 ✓
//    Line color.setRGB(0, 5,   5.5) → linear lum ≈ 4.0 > threshold 3.5 ✓
//    Skin                           → linear lum ≈ 2.9 < threshold 3.5 ✓
//
function _buildLandmarks(center, hw, hh, hd) {
  const { x: cx, y: cy, z: cz } = center;
  const ZOFF = 0.06;   // push overlay in front of skin to avoid z-fighting

  const pts = [
    // 0–2   Forehead
    [ 0.00,  0.80,  0.62], [-0.18,  0.76,  0.66], [ 0.18,  0.76,  0.66],
    // 3–5   L eyebrow
    [-0.46,  0.48,  0.74], [-0.28,  0.44,  0.82], [-0.14,  0.42,  0.86],
    // 6–8   R eyebrow
    [ 0.46,  0.48,  0.74], [ 0.28,  0.44,  0.82], [ 0.14,  0.42,  0.86],
    // 9–10  L eye corners
    [-0.36,  0.32,  0.84], [-0.18,  0.30,  0.88],
    // 11–12 R eye corners
    [ 0.36,  0.32,  0.84], [ 0.18,  0.30,  0.88],
    // 13–17 Nose
    [ 0.00,  0.20,  0.94], [ 0.00,  0.06,  0.96],
    [ 0.00, -0.04,  0.98], [-0.12, -0.11,  0.93], [ 0.12, -0.11,  0.93],
    // 18–21 Mouth
    [-0.20, -0.24,  0.88], [ 0.20, -0.24,  0.88],
    [ 0.00, -0.20,  0.93], [ 0.00, -0.32,  0.90],
    // 22–24 Chin
    [ 0.00, -0.56,  0.74], [-0.10, -0.50,  0.78], [ 0.10, -0.50,  0.78],
    // 25–28 Face outline L
    [-0.52,  0.40,  0.66], [-0.58,  0.14,  0.58],
    [-0.56, -0.10,  0.56], [-0.44, -0.42,  0.64],
    // 29–32 Face outline R
    [ 0.52,  0.40,  0.66], [ 0.58,  0.14,  0.58],
    [ 0.56, -0.10,  0.56], [ 0.44, -0.42,  0.64],
    // 33–34 Cheeks
    [-0.46,  0.06,  0.73], [ 0.46,  0.06,  0.73],
  ];

  // Shared CPU buffer (positions in world space)
  const buf = new Float32Array(pts.length * 3);
  pts.forEach(([fx, fy, fz], i) => {
    buf[i * 3]     = cx + fx * hw;
    buf[i * 3 + 1] = cy + fy * hh;
    buf[i * 3 + 2] = cz + fz * hd + ZOFF;
  });

  // ── Dots ───────────────────────────────────────────────────────
  const dGeo = new THREE.BufferGeometry();
  dGeo.setAttribute('position', new THREE.BufferAttribute(buf, 3));
  const dots = new THREE.Points(dGeo, new THREE.PointsMaterial({
    color:           new THREE.Color().setRGB(0, 7, 8),   // HDR cyan
    size:            0.045,
    sizeAttenuation: true,
    transparent:     true,
    opacity:         0.90,
    depthWrite:      false,
  }));

  // ── LineSegments (anatomical mesh) ─────────────────────────────
  const EDGES = [
    // Jaw contour L & R
    [25,26],[26,27],[27,28],[28,23],
    [29,30],[30,31],[31,32],[32,24],
    // Forehead arc L→brow→top→brow→R
    [25, 3],[ 3, 1],[ 1, 0],[ 0, 2],[ 2, 6],[ 6,29],
    // Eyebrows
    [ 3, 4],[ 4, 5],[ 6, 7],[ 7, 8],
    // Brow outer → eye outer
    [ 3, 9],[ 6,11],
    // Eye spans
    [ 9,10],[11,12],
    // Brow inner & eye inner → nose bridge top
    [ 5,13],[ 8,13],[10,13],[12,13],
    // Nose bridge
    [13,14],[14,15],
    // Nose tip → nostrils
    [15,16],[15,17],
    // Cheekbone
    [ 9,33],[11,34],[33,26],[34,30],
    // Cheek → mouth corner
    [33,18],[34,19],
    // Nostril → mouth corner
    [16,18],[17,19],
    // Upper lip
    [18,20],[20,19],
    // Lower lip
    [18,21],[21,19],
    // Mouth corner → chin
    [18,23],[19,24],
    // Chin
    [23,22],[24,22],
    // Jaw bottom → chin
    [28,22],[32,22],
    // Midline: forehead → nose, mouth → nose
    [ 0,13],[20,15],
  ];

  const lGeo = new THREE.BufferGeometry();
  lGeo.setAttribute('position', new THREE.BufferAttribute(buf.slice(), 3));
  lGeo.setIndex(new THREE.BufferAttribute(
    new Uint16Array(EDGES.flatMap(([a, b]) => [a, b])), 1,
  ));
  const lines = new THREE.LineSegments(lGeo, new THREE.LineBasicMaterial({
    color:       new THREE.Color().setRGB(0, 5, 5.5),   // HDR cyan
    transparent: true,
    opacity:     0.45,
    depthWrite:  false,
  }));

  const g = new THREE.Group();
  g.add(dots);
  g.add(lines);
  return g;
}

// ─── RENDER LOOP ─────────────────────────────────────────────────
function renderLoop() {
  raf = requestAnimationFrame(renderLoop);
  const t = clock.getElapsedTime();

  if (head) {
    // Exponential smoothing toward mouse-driven target angles
    head.rotation.y += (tRotY - head.rotation.y) * P_LERP;
    head.rotation.x += (tRotX - head.rotation.x) * P_LERP;
    // Gentle vertical float
    head.position.y = Math.sin(t * 0.55) * 0.015;
  }

  controls.update();
  composer.render();   // full post-processing chain
}

// ─── UTILITIES ───────────────────────────────────────────────────

function _teardown() {
  if (raf !== null)  { cancelAnimationFrame(raf); raf = null; }
  if (head)          { _disposeGroup(head); head = null; }
  if (composer)      { composer.renderTarget1?.dispose(); composer.renderTarget2?.dispose(); composer = null; }
  if (controls)      { controls.dispose(); controls = null; }
  if (renderer)      { renderer.dispose(); renderer = null; }
  document.removeEventListener('mousemove', _onMouse);
}

function _disposeGroup(group) {
  group.traverse(obj => {
    if (!obj.isMesh && !obj.isPoints && !obj.isLine) return;
    obj.geometry?.dispose();
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    mats.forEach(m => { m?.map?.dispose(); m?.normalMap?.dispose(); m?.roughnessMap?.dispose(); m?.dispose(); });
  });
}
