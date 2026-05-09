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
import { KTX2Loader }      from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder }  from 'three/addons/libs/meshopt_decoder.module.js';
import { EffectComposer }  from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { BokehPass }       from 'three/addons/postprocessing/BokehPass.js';
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

// Contorno oval de la cara (36 índices de MediaPipe FaceLandmarker)
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109,
];

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

  // Loaders para texturas y mallas comprimidas
  const ktx2Loader = new KTX2Loader()
    .setTranscoderPath('https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/libs/basis/')
    .detectSupport(renderer);
  gltfLoader.setKTX2Loader(ktx2Loader);
  gltfLoader.setMeshoptDecoder(MeshoptDecoder);

  // Post-processing
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  
  // Bloom for glowing biometric nodes
  const bloomPass = new UnrealBloomPass(new THREE.Vector2(W, H), 1.8, 0.45, 1.5);
  composer.addPass(bloomPass);

  // Depth of Field (BokehPass) for cinematic photography look
  const bokehPass = new BokehPass(scene, camera, {
    focus: 5.6,
    aperture: 0.008,
    maxblur: 0.01,
    width: W, height: H
  });
  composer.addPass(bokehPass);

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

  _setStatus('Conformando malla a superficie 3D...', true);

  const faceMesh = _buildFaceMesh(landmarks, imgEl);
  head.add(faceMesh);
  _faceMesh3D = faceMesh;

  _setStatus('Malla facial 3D lista', false);
  setTimeout(() => _setStatus(null), 1800);

  // ── FASE 1: Gemini (no bloquea) ──────────────────────────────
  if (_GEMINI_KEY) {
    _setStatus('Analizando con Gemini AI...', false);
    // Capturamos la referencia actual para detectar si el usuario
    // cargó otra foto antes de que Gemini responda (race condition).
    const meshSnapshot = faceMesh;
    analyzeWithGemini(imgEl, _GEMINI_KEY)
      .then(analysis => {
        if (!analysis || _faceMesh3D !== meshSnapshot) { _setStatus(null); return; }
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
//  Pipeline de 2 pasos:
//
//  1. POSICIONES XY — escaladas desde los 468 landmarks de MediaPipe
//     para que la cara ocupe FACE_W/H_RATIO del frente del modelo.
//
//  2. POSICIÓN Z — raycasting: para cada vértice se dispara un rayo
function _buildFaceMesh(landmarks, imgEl) {
  const N = 468;

  // ── BBox de landmarks en espacio normalizado ──────────────
  let lx0 = 1, ly0 = 1, lx1 = 0, ly1 = 0;
  for (let i = 0; i < N; i++) {
    const lm = landmarks[i];
    if (lm.x < lx0) lx0 = lm.x; if (lm.y < ly0) ly0 = lm.y;
    if (lm.x > lx1) lx1 = lm.x; if (lm.y > ly1) ly1 = lm.y;
  }
  const faceCxN = (lx0 + lx1) / 2, faceCyN = (ly0 + ly1) / 2;
  const faceWN  = Math.max(lx1 - lx0, 0.01);
  const faceHN  = Math.max(ly1 - ly0, 0.01);

  // ── Escala XY ─────────────────────────────────────────────
  const scaleX = (_modelSize.x * FACE_W_RATIO) / faceWN;
  const scaleY = (_modelSize.y * FACE_H_RATIO) / faceHN;
  const uniformScale = (scaleX + scaleY) / 2;
  const scaleZ = uniformScale * 1.5; // Amplificador de profundidad (volumen)
  const wCX = _modelCenter.x;
  const wCY = _modelCenter.y + _modelSize.y * 0.06;

  // ── Calcular XYZ reales (Geometría Intrínseca) ───────────
  const positions = new Float32Array(N * 3);
  const uvs       = new Float32Array(N * 2);

  // Z base del rostro (frente de la cabeza)
  const baseZ = _modelCenter.z + _modelSize.z * 0.42;

  for (let i = 0; i < N; i++) {
    const lm = landmarks[i];
    positions[i * 3 + 0] = (lm.x - faceCxN) * scaleX + wCX;
    positions[i * 3 + 1] = -(lm.y - faceCyN) * scaleY + wCY;
    // La profundidad real de MediaPipe (lm.z negativo es hacia afuera)
    positions[i * 3 + 2] = baseZ + (-lm.z * scaleZ);

    uvs[i * 2 + 0] = lm.x;
    uvs[i * 2 + 1] = lm.y;
  }

  // ── Geometría ─────────────────────────────────────────────
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('uv',       new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(new THREE.BufferAttribute(new Uint16Array(FACES), 1));
  geo.computeVertexNormals();

  // ── Textura foto del paciente ──────────────────────────────
  const photoCanvas = document.createElement('canvas');
  photoCanvas.width  = imgEl.naturalWidth  || 640;
  photoCanvas.height = imgEl.naturalHeight || 480;
  photoCanvas.getContext('2d', { willReadFrequently: true }).drawImage(imgEl, 0, 0);
  const photoTex = new THREE.CanvasTexture(photoCanvas);
  photoTex.colorSpace  = THREE.SRGBColorSpace;
  photoTex.flipY       = false;
  photoTex.minFilter   = THREE.LinearFilter;
  photoTex.magFilter   = THREE.LinearFilter;
  photoTex.needsUpdate = true;

  // ── Alpha map (contorno oval difuminado) ────────────────────
  const alphaTex = _buildAlphaMap(landmarks);

  // ── Normal map (detalle superficial desde foto) ─────────────
  const normalTex = _buildNormalMap(imgEl);

  // ── Material PBR Físico (Piel Fotorrealista) ──────────────
  const mat = new THREE.MeshPhysicalMaterial({
    map:                 photoTex,
    alphaMap:            alphaTex,
    normalMap:           normalTex,
    normalScale:         new THREE.Vector2(1.2, 1.2), // Más relieve
    transparent:         true,   
    side:                THREE.FrontSide,
    roughness:           0.45, // Piel refleja luz natural
    metalness:           0.0,
    clearcoat:           0.3, // Capa brillante natural
    clearcoatRoughness:  0.25,
    sheen:               0.8, // Dispersión en bordes
    sheenColor:          new THREE.Color(0xffdcb3),
    envMapIntensity:     1.5,
    polygonOffset:       true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits:  -2,
  });

  // ── [Seamless Blend] Extraer color de piel y teñir cabeza base ──
  _blendBaseHeadSkin(imgEl, landmarks, wCX, wCY, scaleX, scaleY);

  const mesh = new THREE.Mesh(geo, mat);
  mesh.name        = 'faceMesh3D';
  mesh.renderOrder = 1;
  mesh.receiveShadow = true;
  scene.add(mesh);

  // ── Añadir UI Biométrica Holográfica ──
  const ui = _buildLandmarkOverlay(geo);
  if (ui) {
    ui.name = 'biometricUI';
    scene.add(ui);
  }

  return mesh;
}

// ─── BLEND DE CABEZA BASE ─────────────────────────────────────

function _blendBaseHeadSkin(imgEl, landmarks, wCX, wCY, scaleX, scaleY) {
  if (!head) return;

  const c = document.createElement('canvas');
  c.width = imgEl.naturalWidth || 640;
  c.height = imgEl.naturalHeight || 480;
  const ctx = c.getContext('2d');
  ctx.drawImage(imgEl, 0, 0, c.width, c.height);

  // Muestrear color promedio de nariz, frente y mejillas
  const skinPoints = [1, 10, 234, 454];
  let r = 0, g = 0, b = 0;
  let validPoints = 0;
  
  try {
    for (let idx of skinPoints) {
      const lm = landmarks[idx];
      const px = Math.floor(lm.x * c.width);
      const py = Math.floor(lm.y * c.height);
      const data = ctx.getImageData(px, py, 1, 1).data;
      r += data[0]; g += data[1]; b += data[2];
      validPoints++;
    }
  } catch (e) {
    console.warn('[Avatar] CORS error sampling skin color, using fallback', e);
  }

  let skinColor;
  if (validPoints > 0) {
    skinColor = new THREE.Color((r/validPoints)/255, (g/validPoints)/255, (b/validPoints)/255);
  } else {
    skinColor = new THREE.Color(0.85, 0.70, 0.60); // Beige fallback
  }
  
  head.traverse(o => {
    if (o.isMesh && o !== _faceMesh3D) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach(m => {
        if (!m) return;
        m.map = null; // Remover la textura "maniquí" azul/gris
        m.color.copy(skinColor); // Color piel promedio perfecto
        m.roughness = 0.65;
        m.needsUpdate = true;
      });
    }
  });
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

  // ── 3. Escala de normal map según prominencia 3D del paciente ─
  if (_faceMesh3D.material?.normalScale) {
    const np = dep.noseProminence      ?? 0.5;
    const br = dep.browRidge           ?? 0.3;
    const cp = dep.cheekboneProminence ?? 0.5;
    const ns = THREE.MathUtils.clamp((np + br + cp) / 3, 0.15, 0.75);
    _faceMesh3D.material.normalScale.set(ns, ns);
    _faceMesh3D.material.needsUpdate = true;
  }

  // ── 4. Ajuste de iluminación ambiental al tono de piel ───────
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

  // ── 5. Escala del modelo base (proporciones de Gemini) ───────
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
      // Opcional: _buildLandmarkOverlay(c1, s1.x * 0.5, s1.y * 0.5, s1.z * 0.5)
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
  if (_faceMesh3D) {
    if (head) head.remove(_faceMesh3D);
    scene.remove(_faceMesh3D);
    _faceMesh3D.geometry?.dispose();
    const mat = _faceMesh3D.material;
    if (mat) {
      mat.map?.dispose();
      mat.alphaMap?.dispose();
      mat.normalMap?.dispose();
      mat.dispose();
    }
    _faceMesh3D = null;
  }
  
  const ui = scene.getObjectByName('biometricUI');
  if (ui) {
    scene.remove(ui);
    ui.children.forEach(c => {
      c.geometry?.dispose();
      if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
      else c.material?.dispose();
    });
  }
}

// ─── ALPHA MAP (máscara oval con feathering) ──────────────────
//
//  Dibuja el contorno FACE_OVAL como polígono relleno en un
//  canvas 128 × 128, luego aplica box-blur separable para
//  suavizar los bordes. Three.js lee el canal R como alfa.
//
function _buildAlphaMap(landmarks) {
  const AW = 128, AH = 128;
  const c = document.createElement('canvas');
  c.width = AW; c.height = AH;
  const ctx = c.getContext('2d', { willReadFrequently: true });

  // Fondo negro (transparente)
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, AW, AH);

  // Polígono oval encogido (blanco) para que el blur termine antes del borde de la geometría
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  
  // Calcular centro de la cara para encoger hacia allí
  let cx = 0, cy = 0;
  FACE_OVAL.forEach(idx => { cx += landmarks[idx].x; cy += landmarks[idx].y; });
  cx /= FACE_OVAL.length;
  cy /= FACE_OVAL.length;

  FACE_OVAL.forEach((idx, i) => {
    const lm = landmarks[idx];
    // Pull inwards by 12% to guarantee feathering before the mesh ends
    const nx = cx + (lm.x - cx) * 0.88;
    const ny = cy + (lm.y - cy) * 0.88;
    
    const px = nx * AW;
    const py = ny * AH;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fill();

  // Box-blur separable agresivo → bordes muy difuminados para transición suave
  _boxBlurCanvasCtx(ctx, AW, AH, 12, 4);

  const tex = new THREE.CanvasTexture(c);
  tex.flipY     = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

// ─── NORMAL MAP (Sobel sobre foto en escala de grises) ────────
//
//  Reduce la foto a max 512 px, convierte a grises, aplica blur
//  y Sobel para obtener gradientes de superficie aproximados.
//  Resultado: mapa de normales tangentes que añade detalle 3D.
//
//  CRÍTICO: colorSpace = LinearSRGBColorSpace (no SRGB) para no
//  corromper los vectores normales con corrección de gamma.
//
function _buildNormalMap(imgEl) {
  const MAX = 512;
  const W0 = imgEl.naturalWidth  || imgEl.width  || 640;
  const H0 = imgEl.naturalHeight || imgEl.height || 480;
  const sc = Math.min(1, MAX / Math.max(W0, H0));
  const W  = Math.round(W0 * sc);
  const H  = Math.round(H0 * sc);

  // Dibujar foto reducida
  const srcC = document.createElement('canvas');
  srcC.width = W; srcC.height = H;
  srcC.getContext('2d', { willReadFrequently: true }).drawImage(imgEl, 0, 0, W, H);

  // Convertir a grises Float32
  const rgba = srcC.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, W, H).data;
  const gray = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    gray[i] = (0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]) / 255;
  }

  // Box-blur antes de Sobel (reduce ruido de piel/JPEG)
  const blurred = _boxBlurGray(gray, W, H, 3, 3);

  // Sobel → normal map
  const dstC = document.createElement('canvas');
  dstC.width = W; dstC.height = H;
  const dctx = dstC.getContext('2d', { willReadFrequently: true });
  const imgD = dctx.createImageData(W, H);
  const d    = imgD.data;
  const STR  = 3.0; // amplitud del gradiente

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - 1), x1 = Math.min(W - 1, x + 1);
      const y0 = Math.max(0, y - 1), y1 = Math.min(H - 1, y + 1);

      // Kernel Sobel 3 × 3
      const gx = (
        -blurred[y0 * W + x0] - 2 * blurred[y * W + x0] - blurred[y1 * W + x0]
        +blurred[y0 * W + x1] + 2 * blurred[y * W + x1] + blurred[y1 * W + x1]
      );
      const gy = (
        -blurred[y0 * W + x0] - 2 * blurred[y0 * W + x] - blurred[y0 * W + x1]
        +blurred[y1 * W + x0] + 2 * blurred[y1 * W + x] + blurred[y1 * W + x1]
      );

      // Normal tangente: nx=-Gx, ny=Gy, nz=1, normalizar
      const nx = -gx * STR;
      const ny =  gy * STR;
      const nz = 1.0;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      const idx = (y * W + x) * 4;
      d[idx]     = Math.round((nx / len * 0.5 + 0.5) * 255); // R
      d[idx + 1] = Math.round((ny / len * 0.5 + 0.5) * 255); // G
      d[idx + 2] = Math.round((nz / len * 0.5 + 0.5) * 255); // B
      d[idx + 3] = 255;
    }
  }
  dctx.putImageData(imgD, 0, 0);

  const tex = new THREE.CanvasTexture(dstC);
  tex.colorSpace = THREE.LinearSRGBColorSpace; // vectores, NO gamma
  tex.flipY      = false;
  tex.minFilter  = THREE.LinearFilter;
  tex.magFilter  = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

// ─── BOX BLUR — canvas RGBA (canal R como escala de grises) ───
function _boxBlurCanvasCtx(ctx, W, H, radius, passes) {
  for (let p = 0; p < passes; p++) {
    const img = ctx.getImageData(0, 0, W, H);
    const src = img.data;
    const hBuf = new Uint8ClampedArray(W * H);
    const vBuf = new Uint8ClampedArray(W * H);

    // Horizontal
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let s = 0, n = 0;
        for (let dx = -radius; dx <= radius; dx++) {
          s += src[(y * W + Math.min(W - 1, Math.max(0, x + dx))) * 4];
          n++;
        }
        hBuf[y * W + x] = s / n;
      }
    }
    // Vertical
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let s = 0, n = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          s += hBuf[Math.min(H - 1, Math.max(0, y + dy)) * W + x];
          n++;
        }
        vBuf[y * W + x] = s / n;
      }
    }
    for (let i = 0; i < W * H; i++) {
      const v = vBuf[i];
      src[i * 4] = src[i * 4 + 1] = src[i * 4 + 2] = src[i * 4 + 3] = v;
    }
    ctx.putImageData(img, 0, 0);
  }
}

// ─── BOX BLUR — array de grises Float32 ───────────────────────
function _boxBlurGray(src, W, H, radius, passes) {
  const a = src.slice();
  const t = new Float32Array(W * H);
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let s = 0, n = 0;
        for (let dx = -radius; dx <= radius; dx++) {
          s += a[y * W + Math.min(W - 1, Math.max(0, x + dx))]; n++;
        }
        t[y * W + x] = s / n;
      }
    }
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let s = 0, n = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          s += t[Math.min(H - 1, Math.max(0, y + dy)) * W + x]; n++;
        }
        a[y * W + x] = s / n;
      }
    }
  }
  return a;
}

// ─── ILUMINACIÓN ─────────────────────────────────────────────

function _addLights() {
  _hemLight = new THREE.HemisphereLight(0x4A6880, 0x2A1810, 1.0);
  scene.add(_hemLight);

  // Key light (frontal cálido cinematográfico)
  const key = new THREE.DirectionalLight(0xFFE8D6, 5.0);
  key.position.set(-3, 4, 6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  scene.add(key);

  // Fill (lateral frío)
  const fill = new THREE.DirectionalLight(0xAACCF5, 2.0);
  fill.position.set(4, 0, 3);
  scene.add(fill);

  // Rim Light Agresivo (contraluz espectacular Cyan/Frío)
  const rim = new THREE.DirectionalLight(0x66FFFF, 8.0);
  rim.position.set(5, 3, -5);
  scene.add(rim);

  // Under (relleno inferior)
  const under = new THREE.DirectionalLight(0xFFE8CC, 0.40);
  under.position.set(0, -5, 2);
  scene.add(under);
}

// ─── OVERLAY HOLOGRÁFICO DE LANDMARKS ────────────────────────
//
//  Genera una red biométrica estilo Sci-Fi (Cyan) usando 
//  los 468 nodos anatómicos proyectados por MediaPipe.
//
function _buildLandmarkOverlay(geo) {
  if (!geo) return null;

  const group = new THREE.Group();

  // 1. Nodos Biométricos (Cyan Bloom)
  const dotMat = new THREE.PointsMaterial({
    color: 0x00FFFF, 
    size: 0.015,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending, 
    depthWrite: false
  });
  const dots = new THREE.Points(geo, dotMat);
  group.add(dots);

  // 2. Wireframe / Red Geométrica
  const wireMat = new THREE.LineBasicMaterial({
    color: 0x00FFFF,
    transparent: true,
    opacity: 0.25,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  
  const wireGeo = new THREE.WireframeGeometry(geo);
  const wire = new THREE.LineSegments(wireGeo, wireMat);
  group.add(wire);

  // Pequeño offset Z para evitar Z-Fighting con la piel
  group.position.z += 0.005;

  return group;
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
      m?.alphaMap?.dispose();
      m?.normalMap?.dispose();
      m?.roughnessMap?.dispose();
      m?.dispose();
    });
  });
}
