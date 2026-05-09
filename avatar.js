// ================================================================
//  FaceLab Analytics — 3D Digital Clone System  v3.2
//  Three.js r162 · GLTF heads · Projection Mapping · MediaPipe · Bloom
//
//  applyPhoto() pipeline:
//    1. Skin-tone crop   — instant, zero dependencies
//    2. Shader injection — paints photo onto 3D surface via projective UV
//    3. MediaPipe refine — async face-landmark crop replaces step 1 texture
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

const TARGET_H = 2.2;   // normalised head height (world units)

// ─── RENDERER STATE ──────────────────────────────────────────────
let scene, camera, renderer, composer, controls, clock;
let head          = null;
let currentPreset = 0;
let raf           = null;
let loadingEl     = null;
let loadSeq       = 0;

// Mouse parallax
let tRotY = 0, tRotX = 0;
const P_YAW   = 0.22;
const P_PITCH = 0.10;
const P_LERP  = 0.055;

// ─── PROJECTION STATE ────────────────────────────────────────────
//  Updated by loadPreset() and consumed by applyPhoto().
const _modelCenter = new THREE.Vector3();
const _modelSize   = new THREE.Vector3();
const _projPos     = new THREE.Vector3();   // projector world-space position
let   _projMat     = null;                  // projector viewProj matrix (Matrix4)
let   _faceMesh    = null;                  // cached largest mesh (skin)
let   _mpLandmarker = null;                 // MediaPipe FaceLandmarker (lazy)

const gltfLoader = new GLTFLoader();

// ─── PUBLIC API ──────────────────────────────────────────────────

export function init(canvas) {
  _teardown();

  const wrap = canvas.parentElement;
  const W = wrap.clientWidth  || 480;
  const H = wrap.clientHeight || 360;

  // ── Loading overlay ─────────────────────────────────────────────
  if (!loadingEl) {
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

  // ── Post-processing: RenderPass → BloomPass → OutputPass ────────
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(W, H), 1.55, 0.50, 3.50));
  composer.addPass(new OutputPass());

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

  document.addEventListener('mousemove', _onMouse);

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

// ─── PHOTO APPLICATION  ──────────────────────────────────────────
//
//  Three-phase pipeline:
//
//  Phase 1 (sync, instant): skin-tone crop → projection shader injected
//  Phase 2 (async ~1-5s):   MediaPipe landmark detection → precise crop
//                            replaces Phase 1 texture if face found
//
export function applyPhoto(imgEl) {
  if (!head) return;

  // Snap to frontal view so projection aligns with the face surface
  head.rotation.set(0, 0, 0);
  tRotY = 0; tRotX = 0;

  // Find (and cache) the primary skin mesh: largest vertex count in the model
  if (!_faceMesh) {
    let maxV = 0;
    head.traverse(o => {
      if (!o.isMesh) return;
      const n = o.geometry?.attributes?.position?.count ?? 0;
      if (n > maxV) { maxV = n; _faceMesh = o; }
    });
  }
  if (!_faceMesh) return;

  // Set up projector camera once per preset (after model bbox is known)
  if (!_projMat) _computeProjector();

  // Phase 1: immediate texture from skin-tone heuristic crop
  const canvas0 = _extractFaceFallback(imgEl);
  const tex0 = _buildTex(canvas0);
  _activateProjection(tex0);

  // Phase 2: async MediaPipe refinement (upgrades texture silently)
  _refineWithMediaPipe(imgEl).then(canvas1 => {
    if (!canvas1) return;
    const tex1 = _buildTex(canvas1);
    const old  = _applyNewTex(tex1);   // swap in better texture
    if (old) old.dispose();
  });
}

// ─── INTERNAL: LOAD PRESET ───────────────────────────────────────

function loadPreset(idx) {
  const seq    = ++loadSeq;
  const preset = PRESETS[idx];

  _showLoading(true);
  if (head) { scene.remove(head); _disposeGroup(head); head = null; }

  // Reset projection state on model change
  _faceMesh = null;
  _projMat  = null;

  gltfLoader.load(
    preset.modelUrl,

    gltf => {
      if (seq !== loadSeq) return;

      const model = gltf.scene;

      // ── Normalise to TARGET_H, centre at world origin ────────────
      const b0 = new THREE.Box3().setFromObject(model);
      const s0 = b0.getSize(new THREE.Vector3());
      const c0 = b0.getCenter(new THREE.Vector3());
      const k  = TARGET_H / s0.y;

      model.scale.setScalar(k);
      model.position.set(-c0.x * k, -c0.y * k, -c0.z * k);
      model.updateWorldMatrix(true, true);

      model.traverse(o => { if (o.isMesh) o.castShadow = o.receiveShadow = true; });

      // ── Normalised bbox ──────────────────────────────────────────
      const b1 = new THREE.Box3().setFromObject(model);
      const c1 = b1.getCenter(new THREE.Vector3());
      const s1 = b1.getSize(new THREE.Vector3());

      // Store for applyPhoto()
      _modelCenter.copy(c1);
      _modelSize.copy(s1);

      // ── Group ────────────────────────────────────────────────────
      head = new THREE.Group();
      head.add(model);
      head.add(_buildLandmarks(c1, s1.x * 0.5, s1.y * 0.5, s1.z * 0.5));
      scene.add(head);

      // ── Auto-frame camera ────────────────────────────────────────
      const fovR = camera.fov * (Math.PI / 180);
      const dist = (s1.y * 0.52) / Math.tan(fovR / 2);
      const tY   = c1.y + s1.y * 0.04;
      controls.target.set(c1.x, tY, c1.z);
      camera.position.set(c1.x, tY, c1.z + dist);
      camera.lookAt(controls.target);
      controls.update();

      _showLoading(false);
    },

    null,
    err => { if (seq === loadSeq) { console.error('[Avatar]', err); _showLoading(false); } }
  );
}

// ─── PROJECTION MAPPING ──────────────────────────────────────────
//
//  Sets up a virtual "projector" camera in front of the face and
//  injects a GLSL chunk into the model's existing MeshStandardMaterial
//  (via onBeforeCompile) that:
//    1. Computes each vertex's position in the projector's clip space
//    2. Back-face culls using the world-space normal dot projector dir
//    3. Blends the photo texture onto diffuseColor before PBR lighting
//
//  Because the blend happens BEFORE lighting, the projected face is
//  naturally shaded by the scene lights, giving a realistic appearance.
//

function _computeProjector() {
  // Projector at a canonical frontal position in front of the face.
  // Face forward (after normalization) is +Z. Face surface is at
  // approximately c.z + halfDepth * 0.55.
  const fZ = _modelCenter.z + _modelSize.z * 0.50;  // ≈ face surface z
  const pZ = fZ + 3.8;                               // 3.8 units in front

  _projPos.set(_modelCenter.x, _modelCenter.y + _modelSize.y * 0.02, pZ);

  const projCam = new THREE.PerspectiveCamera(27, 1, 0.1, 20);
  projCam.position.copy(_projPos);
  projCam.lookAt(_modelCenter.x, _modelCenter.y, fZ);
  projCam.updateMatrixWorld();

  // viewProj = projectionMatrix × viewMatrix (world → clip space)
  _projMat = new THREE.Matrix4()
    .multiplyMatrices(projCam.projectionMatrix, projCam.matrixWorldInverse);
}

// Vertex shader injection — adds projection varyings
const _VERT_PARS = `
  uniform mat4  uProjMatrix_fp;
  varying vec4  vProjCoord_fp;
  varying vec3  vWorldNorm_fp;
  varying vec3  vWorldPos_fp;
`;

const _VERT_MAIN = `
  // World-space position and normal for projection (uses 'transformed'
  // and 'objectNormal' which are already morph/skin processed at this point)
  vec4 _wpFP    = modelMatrix * vec4(transformed, 1.0);
  vWorldPos_fp  = _wpFP.xyz;
  vWorldNorm_fp = normalize(mat3(modelMatrix) * objectNormal);
  vProjCoord_fp = uProjMatrix_fp * _wpFP;
`;

// Fragment shader injection — blends photo into diffuseColor before lighting
const _FRAG_PARS = `
  uniform sampler2D uPhotoTex_fp;
  uniform float     uProjOn_fp;      // 0 = off (no photo), 1 = on
  uniform vec3      uProjPos_fp;     // projector world position
  varying vec4      vProjCoord_fp;
  varying vec3      vWorldNorm_fp;
  varying vec3      vWorldPos_fp;
`;

const _FRAG_INJECT = `
  // ── Photo projection blend (before PBR lighting) ──────────────
  if (uProjOn_fp > 0.5) {
    vec3  pndc  = vProjCoord_fp.xyz / vProjCoord_fp.w;
    vec2  puv   = pndc.xy * 0.5 + 0.5;

    bool  inF   = vProjCoord_fp.w > 0.0
                  && all(greaterThan(puv, vec2(0.0)))
                  && all(lessThan (puv, vec2(1.0)));

    // World-space facing: how directly does the surface face the projector?
    vec3  pdir  = normalize(uProjPos_fp - vWorldPos_fp);
    float face  = max(0.0, dot(normalize(vWorldNorm_fp), pdir));

    // Feather UV edges so photo doesn't have a hard border on the skin
    vec2  ef    = smoothstep(0.0,  0.15, puv) * smoothstep(1.0, 0.85, puv);

    float blend = inF
      ? smoothstep(0.12, 0.72, face) * ef.x * ef.y
      : 0.0;

    vec4  photo = texture2D(uPhotoTex_fp, puv);
    // Blend into diffuseColor so the photo is lit by scene lights
    diffuseColor.rgb = mix(diffuseColor.rgb, photo.rgb, blend * 0.90);
  }
  // ─────────────────────────────────────────────────────────────
`;

function _activateProjection(tex) {
  const mats = Array.isArray(_faceMesh.material)
    ? _faceMesh.material : [_faceMesh.material];

  mats.forEach(m => {
    if (m.userData._projShader) {
      // Shader already compiled: update uniforms in place (no recompile needed)
      m.userData._projShader.uniforms.uPhotoTex_fp.value = tex;
      m.userData._projShader.uniforms.uProjOn_fp.value   = 1.0;
      return;
    }

    // First call: inject shader and force recompile
    const projMat   = _projMat.clone();
    const projPos   = _projPos.clone();
    let   shaderRef = null;

    m.onBeforeCompile = shader => {
      shaderRef = shader;
      m.userData._projShader = shader;

      shader.uniforms.uProjMatrix_fp = { value: projMat  };
      shader.uniforms.uPhotoTex_fp   = { value: tex      };
      shader.uniforms.uProjPos_fp    = { value: projPos  };
      shader.uniforms.uProjOn_fp     = { value: 1.0      };

      // ── Vertex shader ──────────────────────────────────────────
      shader.vertexShader = shader.vertexShader
        .replace('#include <uv_pars_vertex>',
          `#include <uv_pars_vertex>\n${_VERT_PARS}`)
        .replace('#include <project_vertex>',
          `#include <project_vertex>\n${_VERT_MAIN}`);

      // ── Fragment shader ────────────────────────────────────────
      shader.fragmentShader =
        `${_FRAG_PARS}\n${shader.fragmentShader}`
          .replace('#include <map_fragment>',
            `#include <map_fragment>\n${_FRAG_INJECT}`);
    };

    // customProgramCacheKey ensures recompile if called again
    m.customProgramCacheKey = () => 'proj_v1';
    m.needsUpdate = true;
  });
}

function _applyNewTex(newTex) {
  const mats = Array.isArray(_faceMesh.material)
    ? _faceMesh.material : [_faceMesh.material];
  let old = null;
  mats.forEach(m => {
    const sh = m.userData._projShader;
    if (!sh) return;
    old = sh.uniforms.uPhotoTex_fp.value;
    sh.uniforms.uPhotoTex_fp.value = newTex;
  });
  return old;
}

function _buildTex(canvas) {
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ─── FACE EXTRACTION — SKIN-TONE HEURISTIC (no deps) ─────────────
//
//  Analyzes pixel data to locate skin-tone pixels, computes their
//  centroid, then crops a square region around that centroid.
//  Falls back to portrait-photo heuristic (upper-center) if detection
//  fails (< 4 % of pixels match skin model).
//
function _extractFaceFallback(imgEl) {
  const W = imgEl.naturalWidth  || imgEl.width  || 640;
  const H = imgEl.naturalHeight || imgEl.height || 480;

  // Downsample for fast pixel analysis (≤ 360 px wide)
  const scale = Math.min(1, 360 / W);
  const sw    = Math.round(W * scale);
  const sh    = Math.round(H * scale);

  const tmp = document.createElement('canvas');
  tmp.width = sw; tmp.height = sh;
  tmp.getContext('2d').drawImage(imgEl, 0, 0, sw, sh);
  const data = tmp.getContext('2d').getImageData(0, 0, sw, sh).data;

  let sx = 0, sy = 0, sc = 0;
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const i = (y * sw + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      // Simplified Kovač-Čajić RGB skin model
      if (r > 95 && g > 40 && b > 20 && r > g && r > b && (r - g) > 15) {
        sx += x; sy += y; sc++;
      }
    }
  }

  let cx, cy, sz;
  if (sc > sw * sh * 0.04) {        // 4 % threshold → skin found
    cx = (sx / sc) / scale;
    cy = (sy / sc) / scale;
    sz = Math.min(W, H) * 0.58;
  } else {                           // portrait fallback
    cx = W * 0.50;
    cy = H * 0.32;
    sz = Math.min(W, H) * 0.62;
  }

  return _cropToCanvas(imgEl, cx, cy, sz, W, H);
}

// ─── FACE EXTRACTION — MEDIAPIPE (async, lazy-loaded) ────────────
//
//  Loads MediaPipe FaceLandmarker on first use (~4 MB WASM + model).
//  Returns a precisely-cropped face canvas or null on failure.
//
async function _ensureMPLandmarker() {
  if (_mpLandmarker !== null) return _mpLandmarker;

  try {
    const mod = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
    );
    const { FaceLandmarker, FilesetResolver } = mod;

    const fs = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );

    _mpLandmarker = await FaceLandmarker.createFromOptions(fs, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/' +
          'face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'IMAGE',
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });

    console.info('[Avatar] MediaPipe FaceLandmarker ready');
  } catch (err) {
    console.warn('[Avatar] MediaPipe unavailable:', err.message);
    _mpLandmarker = false;   // sentinel: tried but failed — don't retry
  }

  return _mpLandmarker || null;
}

async function _refineWithMediaPipe(imgEl) {
  const fl = await _ensureMPLandmarker();
  if (!fl) return null;

  try {
    const result = fl.detect(imgEl);
    const lm = result?.faceLandmarks?.[0];
    if (!lm?.length) return null;

    const W = imgEl.naturalWidth  || imgEl.width  || 640;
    const H = imgEl.naturalHeight || imgEl.height || 480;

    // Bounding box from the 478 detected landmarks
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    lm.forEach(p => {
      if (p.x < x0) x0 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.x > x1) x1 = p.x;
      if (p.y > y1) y1 = p.y;
    });

    // Square crop: expand bbox 28% for context (hair, chin, ears)
    // Convert normalized [0,1] landmark coords to pixel space before computing size
    const cx      = (x0 + x1) / 2;
    const cy      = (y0 + y1) / 2;
    const halfPxX = (x1 - x0) / 2 * 1.28 * W;
    const halfPxY = (y1 - y0) / 2 * 1.28 * H;
    const sz      = Math.max(halfPxX, halfPxY) * 2;

    return _cropToCanvas(imgEl, cx * W, cy * H, sz, W, H);
  } catch (e) {
    console.warn('[Avatar] MediaPipe detect error:', e.message);
    return null;
  }
}

// ─── SHARED CROP UTILITY ─────────────────────────────────────────

function _cropToCanvas(imgEl, cx, cy, sz, W, H) {
  const out = document.createElement('canvas');
  out.width = out.height = 512;
  const ctx = out.getContext('2d');

  // Neutral skin fallback (shown at edges if crop overshoots image bounds)
  ctx.fillStyle = '#C29070';
  ctx.fillRect(0, 0, 512, 512);

  const sx = Math.max(0, cx - sz / 2);
  const sy = Math.max(0, cy - sz / 2);
  const sw = Math.min(sz, W - sx);
  const sh = Math.min(sz, H - sy);

  if (sw > 0 && sh > 0) {
    ctx.drawImage(imgEl, sx, sy, sw, sh, 0, 0, 512, 512);
  }

  return out;
}

// ─── MOUSE PARALLAX ──────────────────────────────────────────────
function _onMouse(e) {
  tRotY =  ((e.clientX / window.innerWidth)  * 2 - 1) * P_YAW;
  tRotX = -((e.clientY / window.innerHeight) * 2 - 1) * P_PITCH * 0.55;
}

// ─── LIGHTING ────────────────────────────────────────────────────
function _addLights() {
  scene.add(new THREE.HemisphereLight(0x4A6880, 0x2A1810, 1.6));

  const key = new THREE.DirectionalLight(0xFFF5EC, 4.2);
  key.position.set(-2, 3, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0xCCDDFF, 1.4);
  fill.position.set(4, 0.5, 2);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0x88AAFF, 3.0);
  rim.position.set(0, 2, -8);
  scene.add(rim);

  const under = new THREE.DirectionalLight(0xFFE8CC, 0.40);
  under.position.set(0, -5, 2);
  scene.add(under);
}

// ─── HOLOGRAPHIC LANDMARK OVERLAY ────────────────────────────────
//
//  35 anatomical points as fractional offsets of the model's bbox.
//  HDR cyan colors: lum > 3.5 → selective bloom; skin lum ≈ 2.9 → no bloom.
//
function _buildLandmarks(center, hw, hh, hd) {
  const { x: cx, y: cy, z: cz } = center;
  const ZOFF = 0.06;

  const pts = [
    [ 0.00,  0.80,  0.62], [-0.18,  0.76,  0.66], [ 0.18,  0.76,  0.66],
    [-0.46,  0.48,  0.74], [-0.28,  0.44,  0.82], [-0.14,  0.42,  0.86],
    [ 0.46,  0.48,  0.74], [ 0.28,  0.44,  0.82], [ 0.14,  0.42,  0.86],
    [-0.36,  0.32,  0.84], [-0.18,  0.30,  0.88],
    [ 0.36,  0.32,  0.84], [ 0.18,  0.30,  0.88],
    [ 0.00,  0.20,  0.94], [ 0.00,  0.06,  0.96],
    [ 0.00, -0.04,  0.98], [-0.12, -0.11,  0.93], [ 0.12, -0.11,  0.93],
    [-0.20, -0.24,  0.88], [ 0.20, -0.24,  0.88],
    [ 0.00, -0.20,  0.93], [ 0.00, -0.32,  0.90],
    [ 0.00, -0.56,  0.74], [-0.10, -0.50,  0.78], [ 0.10, -0.50,  0.78],
    [-0.52,  0.40,  0.66], [-0.58,  0.14,  0.58],
    [-0.56, -0.10,  0.56], [-0.44, -0.42,  0.64],
    [ 0.52,  0.40,  0.66], [ 0.58,  0.14,  0.58],
    [ 0.56, -0.10,  0.56], [ 0.44, -0.42,  0.64],
    [-0.46,  0.06,  0.73], [ 0.46,  0.06,  0.73],
  ];

  const buf = new Float32Array(pts.length * 3);
  pts.forEach(([fx, fy, fz], i) => {
    buf[i * 3]     = cx + fx * hw;
    buf[i * 3 + 1] = cy + fy * hh;
    buf[i * 3 + 2] = cz + fz * hd + ZOFF;
  });

  const dGeo = new THREE.BufferGeometry();
  dGeo.setAttribute('position', new THREE.BufferAttribute(buf, 3));
  const dots = new THREE.Points(dGeo, new THREE.PointsMaterial({
    color: new THREE.Color().setRGB(0, 7, 8),
    size: 0.045, sizeAttenuation: true,
    transparent: true, opacity: 0.90, depthWrite: false,
  }));

  const EDGES = [
    [25,26],[26,27],[27,28],[28,23],
    [29,30],[30,31],[31,32],[32,24],
    [25, 3],[ 3, 1],[ 1, 0],[ 0, 2],[ 2, 6],[ 6,29],
    [ 3, 4],[ 4, 5],[ 6, 7],[ 7, 8],
    [ 3, 9],[ 6,11],[ 9,10],[11,12],
    [ 5,13],[ 8,13],[10,13],[12,13],
    [13,14],[14,15],[15,16],[15,17],
    [ 9,33],[11,34],[33,26],[34,30],
    [33,18],[34,19],[16,18],[17,19],
    [18,20],[20,19],[18,21],[21,19],
    [18,23],[19,24],[23,22],[24,22],
    [28,22],[32,22],[ 0,13],[20,15],
  ];
  const lGeo = new THREE.BufferGeometry();
  lGeo.setAttribute('position', new THREE.BufferAttribute(buf.slice(), 3));
  lGeo.setIndex(new THREE.BufferAttribute(
    new Uint16Array(EDGES.flatMap(([a, b]) => [a, b])), 1,
  ));
  const lines = new THREE.LineSegments(lGeo, new THREE.LineBasicMaterial({
    color: new THREE.Color().setRGB(0, 5, 5.5),
    transparent: true, opacity: 0.45, depthWrite: false,
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
    head.rotation.y += (tRotY - head.rotation.y) * P_LERP;
    head.rotation.x += (tRotX - head.rotation.x) * P_LERP;
    head.position.y  = Math.sin(t * 0.55) * 0.015;
  }

  controls.update();
  composer.render();
}

// ─── UTILITIES ───────────────────────────────────────────────────

function _showLoading(v) {
  if (loadingEl) loadingEl.style.display = v ? 'flex' : 'none';
}

function _teardown() {
  if (raf !== null)  { cancelAnimationFrame(raf); raf = null; }
  if (head)          { _disposeGroup(head); head = null; }
  if (composer)      {
    composer.renderTarget1?.dispose();
    composer.renderTarget2?.dispose();
    composer = null;
  }
  if (controls)      { controls.dispose(); controls = null; }
  if (renderer)      { renderer.dispose(); renderer = null; }
  document.removeEventListener('mousemove', _onMouse);
  _faceMesh = null;
  _projMat  = null;
}

function _disposeGroup(group) {
  group.traverse(obj => {
    if (!obj.isMesh && !obj.isPoints && !obj.isLine) return;
    obj.geometry?.dispose();
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    mats.forEach(m => {
      m?.map?.dispose();
      m?.normalMap?.dispose();
      m?.roughnessMap?.dispose();
      // Dispose projected photo texture if present
      m?.userData?._projShader?.uniforms?.uPhotoTex_fp?.value?.dispose();
      m?.dispose();
    });
  });
}
