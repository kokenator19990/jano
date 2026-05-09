// ================================================================
//  FaceLab Analytics — 3D Digital Clone System  v1.0
//  Three.js r162 · Procedural head mesh · Face texture mapping
// ================================================================

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ─── PRESETS ─────────────────────────────────────────────────────
export const PRESETS = [
  {
    id: 0,
    label: 'Avatar A',
    sublabel: 'Femenino',
    skinHex: '#F5C0A0',
    skinInt: 0xF5C0A0,
    hairInt: 0x180E08,
    eyeInt:  0x3C2010,
    lipInt:  0xC06872,
    hs: 1.28,   // height scale
    ws: 0.855,  // width scale
  },
  {
    id: 1,
    label: 'Avatar B',
    sublabel: 'Masculino',
    skinHex: '#CB8E68',
    skinInt: 0xCB8E68,
    hairInt: 0x080604,
    eyeInt:  0x1C1008,
    lipInt:  0x9A4838,
    hs: 1.22,
    ws: 0.935,
  },
];

// ─── STATE ───────────────────────────────────────────────────────
let scene, camera, renderer, controls, clock;
let head = null;
let currentPreset = 0;
let raf;

// ─── PUBLIC API ──────────────────────────────────────────────────

export function init(canvas) {
  const wrap = canvas.parentElement;
  const W = wrap.clientWidth;
  const H = wrap.clientHeight;

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(28, W / H, 0.1, 100);
  camera.position.set(0, 0.35, 5.8);
  camera.lookAt(0, 0.2, 0);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.18;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  addLights();

  head = buildHead(PRESETS[0]);
  scene.add(head);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.065;
  controls.minDistance = 2.5;
  controls.maxDistance = 10;
  controls.target.set(0, 0.25, 0);
  controls.minPolarAngle = 0.08;
  controls.maxPolarAngle = Math.PI - 0.08;

  clock = new THREE.Clock();
  renderLoop();

  new ResizeObserver(() => {
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (!W || !H) return;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
  }).observe(wrap);
}

export function switchPreset(index) {
  currentPreset = index;
  if (head) {
    scene.remove(head);
    disposeGroup(head);
  }
  head = buildHead(PRESETS[index]);
  scene.add(head);
}

export async function applyPhoto(imgEl) {
  const p = PRESETS[currentPreset];

  // Equirectangular texture (2:1) — face at U=0.25 → x=512
  const tc = document.createElement('canvas');
  tc.width = 2048; tc.height = 1024;
  const ctx = tc.getContext('2d');

  // Skin base fill
  ctx.fillStyle = p.skinHex;
  ctx.fillRect(0, 0, 2048, 1024);

  // Face ellipse centered at front of sphere (U=0.25, V≈0.44)
  const cx = 512, cy = 440, rx = 190, ry = 250;

  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.clip();

  const sw = imgEl.naturalWidth  || imgEl.width  || 640;
  const sh = imgEl.naturalHeight || imgEl.height || 480;
  const sz = Math.min(sw, sh);
  ctx.drawImage(imgEl, (sw-sz)/2, (sh-sz)/2, sz, sz,
    cx - rx, cy - ry, rx * 2, ry * 2);
  ctx.restore();

  // Soft edge blend — draw skin-colored ellipse at lower opacity on edges
  const edgeGrad = ctx.createRadialGradient(cx, cy, rx*0.78, cx, cy, rx*1.02);
  edgeGrad.addColorStop(0, 'rgba(0,0,0,0)');
  edgeGrad.addColorStop(1, p.skinHex);
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx*1.05, ry*1.05, 0, 0, Math.PI*2);
  ctx.clip();
  ctx.fillStyle = edgeGrad;
  ctx.fillRect(cx-rx*1.1, cy-ry*1.1, rx*2.2, ry*2.2);
  ctx.restore();

  const tex = new THREE.CanvasTexture(tc);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;

  head.traverse(obj => {
    if (obj.name === 'cranium') {
      if (obj.material.map) obj.material.map.dispose();
      obj.material.map = tex;
      obj.material.color.setHex(0xffffff);
      obj.material.needsUpdate = true;
    }
  });
}

// ─── LIGHTS ──────────────────────────────────────────────────────
function addLights() {
  scene.add(new THREE.AmbientLight(0x3A5870, 2.8));

  // Key — warm upper-left-front
  const key = new THREE.DirectionalLight(0xFFF0E0, 4.2);
  key.position.set(-2, 3.5, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  scene.add(key);

  // Fill — cool right
  const fill = new THREE.DirectionalLight(0xCCE4FF, 1.6);
  fill.position.set(4, 0.5, 2);
  scene.add(fill);

  // Rim — back halo (gives depth)
  const rim = new THREE.DirectionalLight(0xB8D4FF, 3.0);
  rim.position.set(0, 1.5, -7);
  scene.add(rim);

  // Under fill — slight warm
  const under = new THREE.DirectionalLight(0xFFE8CC, 0.55);
  under.position.set(0, -4, 2);
  scene.add(under);
}

// ─── HEAD BUILDER ────────────────────────────────────────────────
function buildHead(p) {
  const g = new THREE.Group();
  const { hs, ws } = p;
  const skinMat = (color = p.skinInt, rough = 0.70) =>
    new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0.018 });

  // ── Cranium ──────────────────────────────────────────────────
  const crGeo = new THREE.SphereGeometry(1, 96, 72);
  deformCranium(crGeo, p);
  crGeo.computeVertexNormals();
  const cranium = new THREE.Mesh(crGeo, skinMat());
  cranium.name = 'cranium';
  cranium.castShadow = cranium.receiveShadow = true;
  g.add(cranium);

  // ── Hair cap ─────────────────────────────────────────────────
  const hrGeo = new THREE.SphereGeometry(1.062, 72, 56, 0, Math.PI*2, 0, Math.PI*0.535);
  deformCranium(hrGeo, p);
  hrGeo.computeVertexNormals();
  const hair = new THREE.Mesh(hrGeo, skinMat(p.hairInt, 0.84));
  hair.castShadow = true;
  g.add(hair);

  // ── Eyes ─────────────────────────────────────────────────────
  [-1, 1].forEach(side => {
    const base = new THREE.Vector3(side * 0.245*ws, 0.17*hs, 0.875);

    // Sclera (white)
    const sclera = new THREE.Mesh(
      new THREE.SphereGeometry(0.085, 22, 18),
      skinMat(0xEEECEB, 0.22)
    );
    sclera.position.copy(base);
    g.add(sclera);

    // Iris
    const iris = new THREE.Mesh(
      new THREE.SphereGeometry(0.058, 18, 14),
      new THREE.MeshStandardMaterial({ color: p.eyeInt, roughness: 0.06, metalness: 0.55 })
    );
    iris.position.copy(base).addScaledVector(new THREE.Vector3(0,0,1), 0.06);
    g.add(iris);

    // Pupil
    const pupil = new THREE.Mesh(
      new THREE.SphereGeometry(0.030, 12, 10),
      skinMat(0x010101, 0.03)
    );
    pupil.position.copy(iris.position).addScaledVector(new THREE.Vector3(0,0,1), 0.040);
    g.add(pupil);

    // Eye shine
    const shine = new THREE.Mesh(
      new THREE.SphereGeometry(0.013, 6, 5),
      skinMat(0xFFFFFF, 0.0)
    );
    shine.position.set(
      pupil.position.x - 0.024*side,
      pupil.position.y + 0.022,
      pupil.position.z + 0.015
    );
    g.add(shine);

    // Upper eyelid
    const lid = new THREE.Mesh(
      new THREE.TorusGeometry(0.085, 0.016, 8, 26, Math.PI),
      skinMat(p.skinInt, 0.68)
    );
    lid.position.copy(base).addScaledVector(new THREE.Vector3(0,0,1), 0.014);
    lid.rotation.x = -0.14;
    lid.rotation.z = side * 0.06;
    g.add(lid);

    // Lower eyelid (thinner)
    const lLid = new THREE.Mesh(
      new THREE.TorusGeometry(0.085, 0.010, 6, 24, Math.PI),
      skinMat(p.skinInt, 0.70)
    );
    lLid.position.copy(base).addScaledVector(new THREE.Vector3(0,0,1), 0.010);
    lLid.rotation.x = Math.PI - 0.20;
    lLid.rotation.z = side * 0.04;
    g.add(lLid);
  });

  // ── Eyebrows ─────────────────────────────────────────────────
  [-1, 1].forEach(side => {
    const brow = new THREE.Mesh(
      new THREE.BoxGeometry(0.215, 0.026, 0.038),
      skinMat(p.hairInt, 0.88)
    );
    brow.position.set(side * 0.245*ws, 0.345*hs, 0.876);
    brow.rotation.z = side * -0.115;
    brow.rotation.y = side * 0.05;
    g.add(brow);
  });

  // ── Nose ─────────────────────────────────────────────────────
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.105, 20, 16), skinMat());
  nose.scale.set(1.04, 0.58, 0.50);
  nose.position.set(0, -0.085*hs, 0.972);
  g.add(nose);

  // Nose bridge
  const bridge = new THREE.Mesh(new THREE.SphereGeometry(0.052, 12, 10), skinMat());
  bridge.scale.set(0.7, 2.2, 0.6);
  bridge.position.set(0, 0.12*hs, 0.960);
  g.add(bridge);

  // Nostrils
  [-1, 1].forEach(side => {
    const n = new THREE.Mesh(new THREE.SphereGeometry(0.042, 12, 10), skinMat(0x0E0404, 0.96));
    n.scale.set(0.82, 0.65, 0.72);
    n.position.set(side * 0.108*ws, -0.180*hs, 0.910);
    g.add(n);
  });

  // ── Lips ─────────────────────────────────────────────────────
  const lipMat = skinMat(p.lipInt, 0.42);

  const uLip = new THREE.Mesh(new THREE.SphereGeometry(0.155, 20, 12), lipMat);
  uLip.scale.set(0.97, 0.36, 0.46);
  uLip.position.set(0, -0.320*hs, 0.904);
  g.add(uLip);

  const lLip = new THREE.Mesh(new THREE.SphereGeometry(0.155, 20, 12), lipMat);
  lLip.scale.set(0.93, 0.40, 0.44);
  lLip.position.set(0, -0.418*hs, 0.897);
  g.add(lLip);

  // Mouth crease (darker line)
  const crease = new THREE.Mesh(new THREE.SphereGeometry(0.145, 20, 8), skinMat(0x380606, 0.92));
  crease.scale.set(0.96, 0.14, 0.28);
  crease.position.set(0, -0.368*hs, 0.912);
  g.add(crease);

  // Philtrum (subtle indent above lip)
  const philt = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 8), skinMat(p.skinInt-0x0A0604, 0.72));
  philt.scale.set(0.6, 1.4, 0.4);
  philt.position.set(0, -0.225*hs, 0.95);
  g.add(philt);

  // ── Ears ─────────────────────────────────────────────────────
  [-1, 1].forEach(side => {
    const ear = new THREE.Mesh(new THREE.SphereGeometry(0.200, 18, 16), skinMat());
    ear.scale.set(0.40, 0.85, 0.44);
    ear.position.set(side * 0.898*ws, 0.04*hs, -0.02);
    ear.castShadow = true;
    g.add(ear);

    // Ear inner (concave — approximated with slightly smaller sphere)
    const earInner = new THREE.Mesh(new THREE.SphereGeometry(0.110, 12, 10), skinMat(0x0A0202, 0.96));
    earInner.scale.set(0.50, 0.68, 0.42);
    earInner.position.set(side * 0.895*ws, 0.03*hs, 0.038*side);
    g.add(earInner);

    // Ear lobe
    const lobe = new THREE.Mesh(new THREE.SphereGeometry(0.075, 10, 8), skinMat());
    lobe.position.set(side * 0.900*ws, -0.20*hs, -0.02);
    g.add(lobe);
  });

  // ── Neck ─────────────────────────────────────────────────────
  const neck = new THREE.Mesh(
    new THREE.CylinderGeometry(0.265, 0.308, 0.60, 30),
    skinMat()
  );
  neck.position.set(0, -1.40*hs, -0.04);
  neck.castShadow = true;
  g.add(neck);

  // Larynx (Adam's apple for male)
  if (p.id === 1) {
    const apple = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 10), skinMat());
    apple.scale.set(1.0, 0.9, 0.7);
    apple.position.set(0, -1.25*hs, 0.24);
    g.add(apple);
  }

  // ── Shoulders / Clothing ─────────────────────────────────────
  const shoul = new THREE.Mesh(
    new THREE.CylinderGeometry(0.56, 0.84, 0.32, 30),
    skinMat(0x18283C, 0.85)
  );
  shoul.position.set(0, -1.88*hs, -0.10);
  shoul.castShadow = true;
  g.add(shoul);

  // Collar
  const collar = new THREE.Mesh(
    new THREE.TorusGeometry(0.280, 0.050, 8, 30),
    skinMat(0x22344A, 0.80)
  );
  collar.position.set(0, -1.67*hs, 0.04);
  collar.rotation.x = -0.22;
  g.add(collar);

  // ── Landmark overlay ─────────────────────────────────────────
  g.add(buildLandmarks(p));

  return g;
}

function deformCranium(geo, { hs, ws }) {
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i);
    let y = pos.getY(i);
    let z = pos.getZ(i);

    // Face proportions
    y *= hs;
    x *= ws;

    // Flatten back of head
    if (z < 0) z *= 0.80;

    // Cheek bulge at y ≈ 0
    const yN = y / hs;
    const cheek = Math.exp(-((yN - 0.08) ** 2) / 0.13);
    x *= (1 + cheek * 0.055);

    // Jaw taper toward chin
    if (y < -hs * 0.40) {
      const t = Math.max(0, Math.min(1, (y + hs) / (hs * 0.60)));
      x *= (0.20 + t * 0.80);
      z *= Math.max(0.48, t * 0.90 + 0.10);
    }

    // Flatten crown
    if (y > hs * 0.72) {
      const t = (y - hs * 0.72) / (hs * 0.28);
      z *= (1.0 - t * 0.22);
    }

    pos.setXYZ(i, x, y, z);
  }
}

// ─── LANDMARK OVERLAY ────────────────────────────────────────────
function buildLandmarks({ hs, ws }) {
  const raw = [
    // Forehead
    [0,       0.81*hs,  0.62],
    [-0.16*ws,0.77*hs,  0.66],
    [ 0.16*ws,0.77*hs,  0.66],
    // L eyebrow
    [-0.40*ws,0.53*hs,  0.76],
    [-0.26*ws,0.48*hs,  0.84],
    [-0.13*ws,0.46*hs,  0.88],
    // R eyebrow
    [ 0.40*ws,0.53*hs,  0.76],
    [ 0.26*ws,0.48*hs,  0.84],
    [ 0.13*ws,0.46*hs,  0.88],
    // L eye corners
    [-0.33*ws,0.35*hs,  0.86],
    [-0.19*ws,0.33*hs,  0.90],
    // R eye corners
    [ 0.33*ws,0.35*hs,  0.86],
    [ 0.19*ws,0.33*hs,  0.90],
    // Nose bridge
    [0,  0.24*hs, 0.96],
    [0,  0.09*hs, 0.98],
    [0, -0.04*hs, 0.99],
    [-0.11*ws,-0.12*hs, 0.94],
    [ 0.11*ws,-0.12*hs, 0.94],
    // Mouth
    [-0.19*ws,-0.28*hs, 0.90],
    [ 0.19*ws,-0.28*hs, 0.90],
    [0, -0.24*hs, 0.95],
    [0, -0.35*hs, 0.92],
    // Chin
    [0, -0.58*hs, 0.76],
    [-0.10*ws,-0.52*hs, 0.80],
    [ 0.10*ws,-0.52*hs, 0.80],
    // Face outline L
    [-0.53*ws, 0.44*hs, 0.68],
    [-0.59*ws, 0.18*hs, 0.60],
    [-0.57*ws,-0.06*hs, 0.58],
    [-0.44*ws,-0.42*hs, 0.66],
    // Face outline R
    [ 0.53*ws, 0.44*hs, 0.68],
    [ 0.59*ws, 0.18*hs, 0.60],
    [ 0.57*ws,-0.06*hs, 0.58],
    [ 0.44*ws,-0.42*hs, 0.66],
    // Cheeks
    [-0.47*ws, 0.07*hs, 0.75],
    [ 0.47*ws, 0.07*hs, 0.75],
  ];

  const buf = new Float32Array(raw.length * 3);
  raw.forEach(([px, py, pz], i) => { buf[i*3]=px; buf[i*3+1]=py; buf[i*3+2]=pz; });

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(buf, 3));

  return new THREE.Points(geo, new THREE.PointsMaterial({
    color: 0x00D8F0,
    size: 0.054,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.88,
  }));
}

// ─── RENDER LOOP ─────────────────────────────────────────────────
function renderLoop() {
  raf = requestAnimationFrame(renderLoop);
  const t = clock.getElapsedTime();

  // Subtle idle animation
  if (head) {
    head.rotation.y = Math.sin(t * 0.38) * 0.042;
    head.position.y = Math.sin(t * 0.55) * 0.013;
  }

  controls.update();
  renderer.render(scene, camera);
}

// ─── UTILITY ─────────────────────────────────────────────────────
function disposeGroup(group) {
  group.traverse(obj => {
    if (!obj.isMesh) return;
    obj.geometry?.dispose();
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    mats.forEach(m => { m?.map?.dispose(); m?.dispose(); });
  });
}
