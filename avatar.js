// ================================================================
//  FaceLab Analytics — 3D Digital Clone System  v4.0
//  Three.js r162 · MediaPipe 468-pt mesh · Gemini AI Vision
//
//  Pipeline de reconstrucción facial:
//
//  FASE 0 (instantáneo):
//    MediaPipe FaceLandmarker → 468 landmarks 3D
//    → Three.js BufferGeometry con triangulación canónica
//    → Foto del paciente mapeada como textura UV
//    → Malla facial real del paciente encima del modelo 3D
//
//  FASE 1 (2-5 s, async):
//    Gemini Vision API → análisis de profundidades, proporciones,
//    tono de piel, forma de cara
//    → Enriquece la malla: Z-depth de nariz, mejillas, mentón
//    → Ajusta iluminación a tono de piel real
//    → Escala cabeza base según proporciones del paciente
//
//  FASE 2 (siempre visible):
//    Overlay holográfico de landmarks médicos con bloom selectivo
// ================================================================

import * as THREE          from 'three';
import { OrbitControls }   from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader }      from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer }  from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass }      from 'three/addons/postprocessing/OutputPass.js';
import { FACES }           from './geometry.js';   // 936 triangles × 3 = 2808 indices
import { analyzeWithGemini } from './gemini.js';

// ─── IMPORTAR CONFIG (API KEY) ────────────────────────────────
let _GEMINI_KEY = null;
try {
  const cfg = await import('./config.js');
  _GEMINI_KEY = cfg.GEMINI_API_KEY || null;
} catch (_) {
  // config.js no presente (deploy externo sin key)
}

// ─── PRESETS ─────────────────────────────────────────────────
export const PRESETS = [
  { id: 0, label: 'Avatar A', sublabel: 'Femenino',  modelUrl: './models/head_female.glb' },
  { id: 1, label: 'Avatar B', sublabel: 'Masculino', modelUrl: './models/head_male.glb'   },
];

const TARGET_H   = 2.2;
const FACE_W_RATIO = 0.88;   // cara ocupa ~88 % del ancho de la cabeza
const FACE_H_RATIO = 0.70;   // cara ocupa ~70 % del alto de la cabeza
const FACE_Z_RATIO = 0.44;   // superficie frontal de la cabeza

// ─── ESTADO DEL RENDERER ─────────────────────────────────────
let scene, camera, renderer, composer, controls, clock;
let head          = null;
let currentPreset = 0;
let raf           = null;
let loadingEl     = null;
let statusEl      = null;
let loadSeq       = 0;

let tRotY = 0, tRotX = 0;
const P_YAW   = 0.22;
const P_PITCH = 0.10;
const P_LERP  = 0.055;

// ─── ESTADO DEL MODELO 3D ────────────────────────────────────
const _modelCenter = new THREE.Vector3();
const _modelSize   = new THREE.Vector3();

// ─── ESTADO DE LA MALLA FACIAL ───────────────────────────────
let _faceMesh3D    = null;   // THREE.Mesh reconstruido de MediaPipe
let _mpLandmarker  = null;   // instancia de FaceLandmarker (lazy)
let _hemLight      = null;   // referencia a HemisphereLight para ajuste de tono

const gltfLoader = new GLTFLoader();

// ─── API PÚBLICA ─────────────────────────────────────────────

export function init(canvas) {
  _teardown();

  const wrap = canvas.parentElement;
  const W = wrap.clientWidth  || 480;
  const H = wrap.clientHeight || 360;

  _buildLoadingOverlay(wrap);
  _buildStatusBar(wrap);

  // Escena
  scene = new THREE.Scene();
  clock = new THREE.Clock();
  camera = new THREE.PerspectiveCamera(28, W / H, 0.05, 100);
  camera.position.set(0, 0, 5.6);

  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.toneMapping         = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled   = true;
  renderer.shadowMap.type      = THREE.PCFSoftShadowMap;

  // Post-processing
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(W, H), 1.45, 0.45, 3.50));
  composer.addPass(new OutputPass());

  _addLights();

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

// ─── PIPELINE PRINCIPAL DE FOTO ──────────────────────────────
//
//  FASE 0: MediaPipe → malla 3D inmediata
//  FASE 1: Gemini → profundidades, tono, proporciones (async)
//
export async function applyPhoto(imgEl) {
  if (!head) return;

  // Snap frontal para alinear proyección
  head.rotation.set(0, 0, 0);
  tRotY = 0; tRotX = 0;

  // Eliminar malla previa si existe
  _removeFaceMesh();

  // ── FASE 0: MediaPipe ────────────────────────────────────────
  _setStatus('Detectando rostro con MediaPipe...', true);

  const fl = await _ensureMPLandmarker();
  if (!fl) {
    _setStatus('MediaPipe no disponible — recarga la página', false);
    return;
  }

  let landmarks;
  try {
    const res = fl.detect(imgEl);
    landmarks = res?.faceLandmarks?.[0];
  } catch (e) {
    console.error('[Avatar] MediaPipe detect error:', e);
  }

  if (!landmarks?.length) {
    _setStatus('No se detectó rostro en la imagen', false);
    setTimeout(() => _setStatus(null), 3500);
    return;
  }

  _setStatus('Reconstruyendo malla facial 3D...', true);

  const faceMesh = _buildFaceMesh(landmarks, imgEl);
  head.add(faceMesh);
  _faceMesh3D = faceMesh;

  _setStatus('Malla facial 3D lista', false);
  setTimeout(() => _setStatus(null), 1800);

  // ── FASE 1: Gemini (no bloquea) ──────────────────────────────
  if (_GEMINI_KEY) {
    _setStatus('Analizando con Gemini AI...', false);
    analyzeWithGemini(imgEl, _GEMINI_KEY)
      .then(analysis => {
        if (!analysis) { _setStatus(null); return; }
        _applyGeminiEnhancement(analysis);
        _setStatus('Gemini: análisis aplicado', false);
        setTimeout(() => _setStatus(null), 2000);
      })
      .catch(e => {
        console.warn('[Avatar] Gemini failed:', e);
        _setStatus(null);
      });
  }
}

// ─── CONSTRUIR MALLA FACIAL 3D ───────────────────────────────
//
//  Convierte los 468 landmarks MediaPipe en una THREE.BufferGeometry
//  correctamente ubicada en el espacio 3D del modelo.
//
//  Vértices: escalados para que la cara llene el frente del modelo.
//  UVs:      coordenadas (x, y) del landmark en la foto — mapea
//            directamente los píxeles correctos de la foto a cada
//            vértice de la malla. Resultado: textura perfectamente
//            alineada con la geometría real del rostro.
//
function _buildFaceMesh(landmarks, imgEl) {
  const N = 468;

  // ── Bounding box de los landmarks en espacio normalizado ─────
  let lx0 = 1, ly0 = 1, lx1 = 0, ly1 = 0;
  for (let i = 0; i < N; i++) {
    const lm = landmarks[i];
    if (lm.x < lx0) lx0 = lm.x;
    if (lm.y < ly0) ly0 = lm.y;
    if (lm.x > lx1) lx1 = lm.x;
    if (lm.y > ly1) ly1 = lm.y;
  }

  const faceCxN = (lx0 + lx1) / 2;
  const faceCyN = (ly0 + ly1) / 2;
  const faceWN  = Math.max(lx1 - lx0, 0.01);
  const faceHN  = Math.max(ly1 - ly0, 0.01);

  // ── Escala para que la malla llene el frente del modelo ──────
  const scaleX = (_modelSize.x * FACE_W_RATIO) / faceWN;
  const scaleY = (_modelSize.y * FACE_H_RATIO) / faceHN;
  // Z: MediaPipe da profundidad relativa; la escalamos para que
  //    la variación de profundidad sea visible pero no exagerada
  const scaleZ = _modelSize.z * 0.32;

  // Centro de la cara en world space (ligeramente por encima del
  // centro del modelo, que incluye cuello)
  const wCX = _modelCenter.x;
  const wCY = _modelCenter.y + _modelSize.y * 0.06;
  const wCZ = _modelCenter.z + _modelSize.z * FACE_Z_RATIO;

  // ── Arrays de geometría ──────────────────────────────────────
  const positions = new Float32Array(N * 3);
  const uvs       = new Float32Array(N * 2);

  for (let i = 0; i < N; i++) {
    const lm = landmarks[i];

    // Posición 3D en world space
    positions[i * 3 + 0] = (lm.x - faceCxN) * scaleX + wCX;
    positions[i * 3 + 1] = -(lm.y - faceCyN) * scaleY + wCY;
    positions[i * 3 + 2] = lm.z * scaleZ + wCZ;

    // UV: coordenadas del landmark en la imagen original
    // Mapeo directo → cada vértice muestrea su píxel exacto en la foto
    uvs[i * 2 + 0] = lm.x;
    uvs[i * 2 + 1] = 1.0 - lm.y; // Y invertido (Three.js UV origin = bottom-left)
  }

  // ── Geometría ────────────────────────────────────────────────
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('uv',       new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(new THREE.BufferAttribute(new Uint16Array(FACES), 1));
  geo.computeVertexNormals();

  // ── Textura de la foto ───────────────────────────────────────
  const photoCanvas = document.createElement('canvas');
  photoCanvas.width  = imgEl.naturalWidth  || 640;
  photoCanvas.height = imgEl.naturalHeight || 480;
  photoCanvas.getContext('2d').drawImage(imgEl, 0, 0);

  const photoTex = new THREE.CanvasTexture(photoCanvas);
  photoTex.colorSpace  = THREE.SRGBColorSpace;
  photoTex.flipY       = false; // Ya invertimos Y en los UVs
  photoTex.minFilter   = THREE.LinearFilter;
  photoTex.magFilter   = THREE.LinearFilter;
  photoTex.needsUpdate = true;

  // ── Material PBR con textura de foto ─────────────────────────
  const mat = new THREE.MeshStandardMaterial({
    map:         photoTex,
    side:        THREE.FrontSide,
    roughness:   0.72,
    metalness:   0.0,
    envMapIntensity: 0.4,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = 'faceMesh3D';

  // Offset mínimo para evitar z-fighting con el modelo base
  mesh.renderOrder = 1;

  return mesh;
}

// ─── MEJORAS DE GEMINI ────────────────────────────────────────
//
//  Aplica el análisis de profundidades de Gemini sobre la malla
//  facial ya construida: desplaza vértices en Z para dar mayor
//  realismo a nariz, mejillas, mentón y arcos superciliares.
//
function _applyGeminiEnhancement(analysis) {
  if (!_faceMesh3D) return;

  const pos = _faceMesh3D.geometry.getAttribute('position');
  const N   = Math.min(468, pos.count);
  const dep  = analysis.depthFeatures || {};
  const prop = analysis.proportions   || {};
  const skin = analysis.skinAnalysis  || {};

  // ── 1. Profundidades por zona ────────────────────────────────
  //
  // MediaPipe landmark indices (aproximados, basados en el mapa canónico):
  //   Punta de nariz:    4, 1
  //   Dorso nariz:       6, 19, 94
  //   Alas nasales:      218, 438, 79, 309
  //   Arcos superciliares: 105, 107, 55, 8, 285, 336, 334
  //   Mejillas:          205, 425, 50, 280
  //   Labios:            13, 14, 317, 87, 146, 375
  //   Mentón:            152, 175, 199, 200
  //   Sienes:            127, 356, 454, 234

  const NOSE_BOOST   = (dep.noseProminence      ?? 0.5) * 0.09;
  const BROW_BOOST   = (dep.browRidge           ?? 0.3) * 0.035;
  const CHEEK_INDENT = (dep.cheekboneProminence ?? 0.5) * 0.025;
  const CHIN_BOOST   = (dep.chinProjection      ?? 0.4) * 0.030;
  const LIP_BOOST    = (dep.lipFullness         ?? 0.4) * 0.018;
  const EYE_DEPTH    = (dep.eyeSocketDepth      ?? 0.4) * 0.022;

  const bumps = [
    // [indices, deltaZ]
    [[4, 1, 19, 94],                           +NOSE_BOOST        ],
    [[218, 438, 79, 309, 49, 279],             +NOSE_BOOST * 0.55 ],
    [[105, 107, 55, 8, 285, 336, 334],         +BROW_BOOST        ],
    [[205, 425, 50, 280],                      +CHEEK_INDENT      ],
    [[152, 175, 199, 200, 32, 262],            +CHIN_BOOST        ],
    [[13, 14, 317, 87, 146, 375, 185, 409],   +LIP_BOOST         ],
    [[33, 7, 163, 246, 263, 362, 382, 466],   -EYE_DEPTH         ], // ojos se hunden
    [[127, 356, 454, 234, 162, 389],           -0.012             ], // sienes retroceden
  ];

  bumps.forEach(([indices, dz]) => {
    indices.forEach(idx => {
      if (idx < N) pos.setZ(idx, pos.getZ(idx) + dz);
    });
  });

  pos.needsUpdate = true;
  _faceMesh3D.geometry.computeVertexNormals();

  // ── 2. Ajuste de roughness por brillo de piel ────────────────
  const shininess = skin.shininess ?? 0.25;
  if (_faceMesh3D.material) {
    _faceMesh3D.material.roughness = THREE.MathUtils.clamp(0.55 + (1 - shininess) * 0.35, 0.55, 0.90);
    _faceMesh3D.material.needsUpdate = true;
  }

  // ── 3. Ajuste de iluminación ambiental al tono de piel ───────
  if (skin.primaryColor && _hemLight) {
    const skinColor  = new THREE.Color(skin.primaryColor);
    const warmth     = skin.warmth ?? 0.5;
    // Mezcla sutil: 18 % hacia el tono real del paciente
    _hemLight.color.lerp(skinColor, 0.18);
    // Si piel cálida → ajustar temperatura del key light
    const keyLight = scene.children.find(o => o.isDirectionalLight && o.position.z > 3);
    if (keyLight) {
      const warm = new THREE.Color(1.0, 0.97 + warmth * 0.04, 0.93 - warmth * 0.04);
      keyLight.color.lerp(warm, 0.12);
    }
  }

  // ── 4. Escala del modelo base (proporciones de Gemini) ───────
  if (prop.faceAspect && head) {
    // faceAspect = alto/ancho de cara. Más alto que 1.3 → cara más larga.
    const aspect     = THREE.MathUtils.clamp(prop.faceAspect, 0.85, 1.85);
    const targetY    = THREE.MathUtils.clamp(1.0 + (aspect - 1.3) * 0.10, 0.90, 1.10);
    head.scale.y     = THREE.MathUtils.lerp(head.scale.y, targetY, 0.55);
  }
  if (prop.jawWidth && head) {
    // jawWidth es fracción del ancho de la cara [0,1]. Referencia = 0.52.
    const jawRatio = THREE.MathUtils.clamp(prop.jawWidth / 0.52, 0.85, 1.18);
    head.scale.x   = THREE.MathUtils.lerp(head.scale.x, jawRatio, 0.40);
  }

  console.info('[Avatar] Gemini enhancement applied:', {
    noseBoost: NOSE_BOOST.toFixed(3),
    skinColor: skin.primaryColor,
    faceAspect: prop.faceAspect,
  });
}

// ─── CARGA DE PRESET ─────────────────────────────────────────

function loadPreset(idx) {
  const seq    = ++loadSeq;
  const preset = PRESETS[idx];

  _showLoading(true);
  if (head) { scene.remove(head); _disposeGroup(head); head = null; }
  _faceMesh3D = null;

  gltfLoader.load(
    preset.modelUrl,

    gltf => {
      if (seq !== loadSeq) return;

      const model = gltf.scene;

      // Normalizar a TARGET_H, centrar en origen
      const b0 = new THREE.Box3().setFromObject(model);
      const s0 = b0.getSize(new THREE.Vector3());
      const c0 = b0.getCenter(new THREE.Vector3());
      const k  = TARGET_H / s0.y;

      model.scale.setScalar(k);
      model.position.set(-c0.x * k, -c0.y * k, -c0.z * k);
      model.updateWorldMatrix(true, true);
      model.traverse(o => { if (o.isMesh) o.castShadow = o.receiveShadow = true; });

      // BBox normalizada
      const b1 = new THREE.Box3().setFromObject(model);
      const c1 = b1.getCenter(new THREE.Vector3());
      const s1 = b1.getSize(new THREE.Vector3());
      _modelCenter.copy(c1);
      _modelSize.copy(s1);

      head = new THREE.Group();
      head.add(model);
      head.add(_buildLandmarkOverlay(c1, s1.x * 0.5, s1.y * 0.5, s1.z * 0.5));
      scene.add(head);

      // Auto-frame cámara
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

// ─── MEDIAPIPE (LAZY) ─────────────────────────────────────────

async function _ensureMPLandmarker() {
  if (_mpLandmarker !== null) return _mpLandmarker || null;

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
    _mpLandmarker = false;
  }

  return _mpLandmarker || null;
}

// ─── HELPERS ─────────────────────────────────────────────────

function _removeFaceMesh() {
  if (!_faceMesh3D) return;
  if (head) head.remove(_faceMesh3D);
  _faceMesh3D.geometry?.dispose();
  const mat = _faceMesh3D.material;
  if (mat) {
    mat.map?.dispose();
    mat.dispose();
  }
  _faceMesh3D = null;
}

// ─── ILUMINACIÓN ─────────────────────────────────────────────

function _addLights() {
  _hemLight = new THREE.HemisphereLight(0x4A6880, 0x2A1810, 1.6);
  scene.add(_hemLight);

  // Key light (frontal cálido)
  const key = new THREE.DirectionalLight(0xFFF5EC, 4.2);
  key.position.set(-2, 3, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  scene.add(key);

  // Fill (lateral frío)
  const fill = new THREE.DirectionalLight(0xCCDDFF, 1.4);
  fill.position.set(4, 0.5, 2);
  scene.add(fill);

  // Rim (contraluz)
  const rim = new THREE.DirectionalLight(0x88AAFF, 3.0);
  rim.position.set(0, 2, -8);
  scene.add(rim);

  // Under (relleno inferior)
  const under = new THREE.DirectionalLight(0xFFE8CC, 0.40);
  under.position.set(0, -5, 2);
  scene.add(under);
}

// ─── OVERLAY HOLOGRÁFICO DE LANDMARKS ────────────────────────
//
//  35 puntos anatómicos + 44 aristas de estructura craneofacial.
//  Colores HDR (luminancia > 3.5) → bloom selectivo cyan.
//
function _buildLandmarkOverlay(center, hw, hh, hd) {
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

// ─── RENDER LOOP ─────────────────────────────────────────────

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

// ─── UI HELPERS ──────────────────────────────────────────────

function _buildLoadingOverlay(wrap) {
  if (loadingEl) return;
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

function _buildStatusBar(wrap) {
  if (statusEl) return;
  statusEl = document.createElement('div');
  Object.assign(statusEl.style, {
    position: 'absolute', bottom: '8px', left: '50%',
    transform: 'translateX(-50%)',
    background: 'rgba(0,10,20,0.82)', backdropFilter: 'blur(8px)',
    border: '1px solid rgba(0,207,224,0.25)',
    color: '#00cfe0', fontFamily: "'Space Grotesk',sans-serif",
    fontSize: '0.65rem', letterSpacing: '0.14em',
    padding: '5px 14px', borderRadius: '20px',
    display: 'none', alignItems: 'center', gap: '8px',
    zIndex: '20', pointerEvents: 'none',
    whiteSpace: 'nowrap',
    maxWidth: '90%',
  });
  wrap.appendChild(statusEl);
}

function _setStatus(msg, showSpinner = false) {
  if (!statusEl) return;
  if (!msg) {
    statusEl.style.display = 'none';
    return;
  }
  const spinHtml = showSpinner
    ? `<svg width="12" height="12" viewBox="0 0 12 12" fill="none"
          style="animation:_av_spin 0.9s linear infinite;flex-shrink:0">
         <path d="M6 1 A5 5 0 0 1 11 6" stroke="#00cfe0" stroke-width="1.8"
               stroke-linecap="round"/>
       </svg>`
    : `<span style="width:6px;height:6px;border-radius:50%;
          background:#00cfe0;box-shadow:0 0 6px #00cfe0;flex-shrink:0"></span>`;
  statusEl.innerHTML = `${spinHtml}<span>${msg}</span>`;
  statusEl.style.display = 'flex';
}

function _showLoading(v) {
  if (loadingEl) loadingEl.style.display = v ? 'flex' : 'none';
}

function _onMouse(e) {
  tRotY =  ((e.clientX / window.innerWidth)  * 2 - 1) * P_YAW;
  tRotX = -((e.clientY / window.innerHeight) * 2 - 1) * P_PITCH * 0.55;
}

// ─── TEARDOWN ────────────────────────────────────────────────

function _teardown() {
  if (raf !== null)  { cancelAnimationFrame(raf); raf = null; }
  if (head)          { _disposeGroup(head); head = null; }
  _faceMesh3D = null;
  if (composer) {
    composer.renderTarget1?.dispose();
    composer.renderTarget2?.dispose();
    composer = null;
  }
  if (controls) { controls.dispose(); controls = null; }
  if (renderer) { renderer.dispose(); renderer = null; }
  document.removeEventListener('mousemove', _onMouse);
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
      m?.dispose();
    });
  });
}
