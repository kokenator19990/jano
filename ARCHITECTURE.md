# FaceLab Analytics — Arquitectura Técnica

> Versión actual: **v7.0**
> Deploy: https://jano-psi.vercel.app
> Repo: https://github.com/kokenator19990/jano
> Última actualización: 2026-05-09

---

## Índice

1. [Visión general](#1-visión-general)
2. [Stack tecnológico](#2-stack-tecnológico)
3. [Estructura de archivos](#3-estructura-de-archivos)
4. [Pipeline de reconstrucción facial](#4-pipeline-de-reconstrucción-facial)
5. [Módulos detallados](#5-módulos-detallados)
   - [avatar.js](#51-avatarjs)
   - [gemini.js](#52-geminijs)
   - [geometry.js](#53-geometryjs)
   - [index.html](#54-indexhtml)
   - [config.js](#55-configjs)
6. [Sistema de coordenadas y convenciones UV](#6-sistema-de-coordenadas-y-convenciones-uv)
7. [Gestión de memoria y ciclo de vida](#7-gestión-de-memoria-y-ciclo-de-vida)
8. [Iluminación y post-procesado](#8-iluminación-y-post-procesado)
9. [Historial de cambios por sesión](#9-historial-de-cambios-por-sesión)
10. [Problemas conocidos y limitaciones](#10-problemas-conocidos-y-limitaciones)
11. [Próximas mejoras sugeridas](#11-próximas-mejoras-sugeridas)

---

## 1. Visión general

FaceLab Analytics es una aplicación web de análisis facial que reconstruye en tiempo real un modelo 3D de la cara del usuario sobre una cabeza genérica. El objetivo es que al subir una foto, el maniquí base se transforme en un clon digital 3D realista de la persona, con overlay biométrico holográfico.

```
FOTO DEL USUARIO
      │
      ▼
┌─────────────────────────────────────────┐
│  FASE 0 — Inmediato (~500 ms)           │
│  MediaPipe 468 landmarks → malla 3D     │
│  Gradiente de profundidad radial         │
│  Subdivisión midpoint (936→3744 tris)   │
│  Foto mapeada como textura UV           │
│  Alpha map oval + Normal map Sobel      │
│  Skin blending del cráneo base          │
│  MeshPhysicalMaterial + RoomEnvironment │
└─────────────────────────────────────────┘
      │
      ▼ (async, no bloquea)
┌─────────────────────────────────────────┐
│  FASE 1 — IA (2-5 s)                   │
│  Gemini Vision API analiza la foto      │
│  → Desplaza vértices Z por zona         │
│  → Ajusta roughness al tono de piel     │
│  → Escala normalScale según prominencia │
│  → Adapta escala del modelo base        │
└─────────────────────────────────────────┘
      │
      ▼ (siempre activo)
┌─────────────────────────────────────────┐
│  FASE 2 — Overlay holográfico anatómico │
│  8 contornos anatómicos (LineSegments)  │
│  4 líneas craneométricas               │
│  22 nodos biométricos (Points)          │
│  Bloom selectivo cyan HDR               │
└─────────────────────────────────────────┘
```

---

## 2. Stack tecnológico

| Capa | Tecnología | Versión | Cómo se carga |
|---|---|---|---|
| Motor 3D | Three.js | r162 | ES importmap → jsDelivr CDN |
| Detección facial | MediaPipe FaceLandmarker | 0.10.14 | `import()` dinámico (lazy) |
| IA visual | Gemini Vision API | gemini-2.0-flash | Fetch directo |
| Triangulación | spite/FaceMeshFaceGeometry | — | `geometry.js` local |
| Post-procesado | UnrealBloomPass + BokehPass | r162 | jsDelivr CDN |
| Environment Map | RoomEnvironment + PMREMGenerator | r162 | jsDelivr CDN |
| Mallas comprimidas | KTX2Loader + MeshoptDecoder | r162 | jsDelivr CDN |
| Fuentes | Space Grotesk + Inter | — | Google Fonts CDN |
| Deploy | Vercel (static) | — | git push origin master |
| Sin build step | Vanilla ES modules | — | — |

**No hay bundler, no hay npm, no hay node_modules.** Todo se sirve como archivos estáticos.

---

## 3. Estructura de archivos

```
Jano/
├── index.html          UI completa + lógica de cámara/upload + importmap Three.js
├── avatar.js           Motor 3D — pipeline MediaPipe → malla → Gemini (1244 líneas, v7.0)
├── gemini.js           Cliente Gemini Vision API (analyzeWithGemini)
├── geometry.js         FACES[2808] — triangulación canónica MediaPipe (468 vértices)
├── config.js           GEMINI_API_KEY (gitignored — solo local)
├── armado.md           Documentación del proceso de depuración v4.0→v6.0
├── models/
│   ├── head_female.glb Modelo de cabeza femenina (normalizado a TARGET_H=2.2)
│   └── head_male.glb   Modelo de cabeza masculina
├── .gitignore          Excluye: config.js, node_modules, .vercel, .DS_Store, *.log
├── .vercel/            Configuración de proyecto Vercel (gitignored)
├── ARCHITECTURE.md     Este documento
└── .claude/
    └── handoffs/       Documentos de continuación de sesión
```

**Exports públicos de `avatar.js`:**
```javascript
export const PRESETS            // Array de presets disponibles
export function init(canvas)    // Inicializa escena Three.js + env map + postpro
export function switchPreset(idx) // Cambia modelo de cabeza
export async function applyPhoto(imgEl) // Pipeline principal
```

---

## 4. Pipeline de reconstrucción facial

### Fase 0 — `_buildFaceMesh(landmarks, imgEl)`

#### 4.1 Cálculo de posiciones XY

Se calcula el bounding box de los 468 landmarks normalizados `[0,1]` para centrar la cara:

```javascript
// Escala absoluta (independiente del tamaño del modelo)
TARGET_FACE_W = 1.75   // ancho de la cara en world units
TARGET_FACE_H = 2.25   // alto de la cara en world units
scaleX = TARGET_FACE_W / faceWidthNormalized
scaleY = TARGET_FACE_H / faceHeightNormalized

// Posición XY centrada en el modelo
positions[i*3+0] = (lm.x - faceCenterX) * scaleX + wCX     // wCX = 0
positions[i*3+1] = -(lm.y - faceCenterY) * scaleY + wCY    // wCY = 0.20
```

#### 4.2 Gradiente de profundidad radial (Z)

Cada vértice recibe profundidad Z basada en MediaPipe, atenuada hacia los bordes:

```javascript
// Distancia normalizada al centro del rostro
ndx = (lm.x - faceCxN) / (faceWN * 0.5)   // -1..+1
ndy = (lm.y - faceCyN) / (faceHN * 0.5)   // -1..+1
edgeDist = sqrt(ndx² + ndy²)

// Peso: 1.0 al centro, decae con factor 0.6
dw = max(0, 1.0 - edgeDist * 0.6)

// Z final
positions[i*3+2] = (-lm.z) * scaleZ * dw + baseZ   // baseZ = 0.28
```

**Valores del gradiente:**

| Posición | edgeDist | dw | Efecto |
|---|---|---|---|
| Nariz (centro) | 0.0 | 1.0 | Profundidad 3D completa |
| Ojos | ~0.4 | 0.76 | 76% de profundidad |
| Mejillas | ~0.6 | 0.64 | 64% de profundidad |
| Borde oval | ~1.0 | 0.40 | 40% |
| Esquinas extremas | ~1.4 | 0.16 | Casi plano (fundido con maniquí) |

**Por qué NO raycasting:** Ver sección 9, v7.0 — el raycasting contra la superficie curva del cráneo causa que los vértices de borde se envuelvan alrededor de la cabeza (efecto domo). El gradiente radial evita esto manteniendo los bordes planos.

#### 4.3 Subdivisión midpoint

Tras computar posiciones y UVs, la geometría se subdivide:

```
468 vértices + 936 triángulos (FACES canónico)
         │  _subdivideGeometry(geo)
         ▼
~1870 vértices + 3744 triángulos
```

Cada triángulo se divide en 4 sub-triángulos insertando vértices en el punto medio de cada arista. Los vértices originales (0-467) mantienen sus índices — esto es crítico para Gemini bumps y el overlay biométrico.

#### 4.4 UVs y coordenadas de textura

```javascript
uvs[i*2+0] = lm.x     // U
uvs[i*2+1] = lm.y     // V (directo, sin inversión)
```

Con `flipY = false` en todas las texturas (photo, alpha, normal), y UVs como `(lm.x, lm.y)`:
- El canvas se dibuja con `drawImage(imgEl, 0, 0)`
- El landmark en `(lm.x*W, lm.y*H)` del canvas corresponde a `UV(lm.x, lm.y)`
- Todas las texturas son mutuamente consistentes

#### 4.5 Alpha Map — difuminado de bordes

```
Contorno FACE_OVAL (36 índices MediaPipe) → canvas 128×128
   → polígono encogido 12% hacia el centro (pull inward)
   → relleno blanco sobre fondo negro
   → box-blur separable (radio=12, 4 pasadas) — agresivo
   → CanvasTexture con flipY=false
```

Three.js lee el **canal R** de `alphaMap` como transparencia. Requiere `transparent: true` en el material.

#### 4.6 Normal Map — relieve superficial desde foto

```
imgEl → canvas reducido (max 512px) → getImageData → Float32Array grises
   → box-blur separable (radio=3, 3 pasadas)
   → Sobel 3×3 → gx, gy
   → nx = -gx*STR,  ny = gy*STR,  nz = 1.0   (STR=3.0)
   → normalize → pack a RGB [0,255]
   → CanvasTexture con colorSpace=LinearSRGBColorSpace, flipY=false
```

**CRÍTICO:** `colorSpace = THREE.LinearSRGBColorSpace` (NO `SRGBColorSpace`).
Si se usa `SRGBColorSpace`, Three.js aplica corrección gamma a los vectores normales, deformando las direcciones.

#### 4.7 Material PBR (MeshPhysicalMaterial)

```javascript
MeshPhysicalMaterial({
  map:                 photoTex,          // foto del paciente (SRGB)
  alphaMap:            alphaTex,          // máscara oval difuminada
  normalMap:           normalTex,         // relieve Sobel (LinearSRGB)
  normalScale:         Vector2(1.0, 1.0), // ajustado por Gemini
  transparent:         true,              // requerido para alphaMap
  roughness:           0.52,              // piel realista
  metalness:           0.0,
  clearcoat:           0.12,              // capa oleosa sutil
  clearcoatRoughness:  0.45,
  sheen:               0.5,               // dispersión en bordes
  sheenRoughness:      0.4,
  sheenColor:          Color(0xffd4b0),   // tono cálido de piel
  specularIntensity:   0.35,              // highlights
  specularColor:       Color(0xffeedd),
  envMapIntensity:     0.8,               // reflejos del RoomEnvironment
  polygonOffset:       true,
  polygonOffsetFactor: -2,
  polygonOffsetUnits:  -2,                // evita z-fighting con modelo base
})
mesh.renderOrder = 1
```

#### 4.8 Skin blending — `_blendBaseHeadSkin()`

Extrae el color de piel de la foto muestreando 4 landmarks (nariz, frente, mejillas) y tiñe todos los materiales del modelo base GLB para que cuello, orejas y nuca coincidan con el tono del rostro.

```javascript
// Desactiva vertexColors y emisión (bloqueaban el color inyectado)
m.vertexColors = false
m.emissive.setHex(0x000000)
m.map = null              // remueve textura original
m.color.copy(skinColor)   // color promedio de la piel del paciente
```

---

### Fase 1 — `_applyGeminiEnhancement(analysis)`

Se ejecuta **async** — no bloquea el render de Fase 0.

#### 4.9 Bumps de profundidad por zona

Desplazamiento adicional en Z sobre los vértices originales (0-467):

| Zona | Índices MediaPipe | Delta Z |
|---|---|---|
| Punta de nariz | 4, 1, 19, 94 | +noseProminence × 0.09 |
| Alas nasales | 218, 438, 79, 309, 49, 279 | +noseProminence × 0.05 |
| Arcos superciliares | 105, 107, 55, 8, 285, 336, 334 | +browRidge × 0.035 |
| Mejillas | 205, 425, 50, 280 | +cheekboneProminence × 0.025 |
| Mentón | 152, 175, 199, 200, 32, 262 | +chinProjection × 0.030 |
| Labios | 13, 14, 317, 87, 146, 375, 185, 409 | +lipFullness × 0.018 |
| Cuencas oculares | 33, 7, 163, 246, 263, 362, 382, 466 | −eyeSocketDepth × 0.022 |
| Sienes | 127, 356, 454, 234, 162, 389 | −0.012 (fijo) |

**Nota:** Tras subdivisión, `pos.count > 468` pero `Math.min(468, pos.count) = 468`. Los bumps solo afectan vértices originales. Los midpoints entre un vértice bumped y uno no-bumped quedan ligeramente discontinuos, pero el efecto es imperceptible (~0.04 units max).

#### 4.10 Roughness, normalScale, iluminación, escala

```javascript
// Roughness dinámico por brillo de piel
roughness = clamp(0.55 + (1-shininess)*0.35, 0.55, 0.90)

// normalScale por prominencia 3D
ns = clamp((nose + brow + cheekbone) / 3, 0.15, 0.75)
normalScale.set(ns, ns)

// HemisphereLight lerp 18% hacia tono de piel
hemLight.color.lerp(skinColor, 0.18)

// Escala del modelo base
head.scale.y = lerp(current, 1.0 + (faceAspect-1.3)*0.10, 0.55)
head.scale.x = lerp(current, jawWidth/0.52, 0.40)
```

#### 4.11 Race condition guard

```javascript
const meshSnapshot = faceMesh
analyzeWithGemini(...).then(analysis => {
  if (_faceMesh3D !== meshSnapshot) return   // nueva foto → abort
  _applyGeminiEnhancement(analysis)
})
```

---

### Fase 2 — Overlay biométrico anatómico

#### 4.12 Contornos y nodos

**8 contornos anatómicos (BIO_PATHS):**
- Mandíbula/FACE_OVAL (36 segmentos)
- Ojo izquierdo (16 segmentos, cerrado)
- Ojo derecho (16 segmentos, cerrado)
- Ceja izquierda / derecha (9 segmentos cada una)
- Puente nasal (6 segmentos)
- Base nasal (6 segmentos)
- Labios exteriores (20 segmentos, cerrado)

**4 líneas craneométricas (BIO_MEASURES):**
- Bi-zigomática: 234→50→1→280→454
- Bi-temporal: 127→10→356
- Bi-cantal: 33→168→263
- Línea media vertical: 10→168→1→152

**22 nodos biométricos:** cardinales, esquinas oculares, nariz, comisuras, cejas, temporales, pómulos, mentón.

**Material:** Cyan (#00FFFF), AdditiveBlending, depthWrite=false, Z-offset=+0.006.

---

## 5. Módulos detallados

### 5.1 `avatar.js`

**Responsabilidad:** todo el motor 3D — escena, cámara, luces, environment map, modelos, malla facial, subdivisión, overlay.

#### API pública

```typescript
export const PRESETS: Array<{ id, label, sublabel, modelUrl }>
export function init(canvas: HTMLCanvasElement): void
export function switchPreset(idx: number): void
export async function applyPhoto(imgEl: HTMLImageElement): Promise<void>
```

#### Estado global (módulo-level)

```javascript
scene, camera, renderer, composer, controls, clock   // Three.js core
head            // THREE.Group (modelo + malla facial + overlay)
_faceMesh3D     // THREE.Mesh | null (malla facial activa)
_mpLandmarker   // FaceLandmarker | null | false (lazy init)
_hemLight       // HemisphereLight (referencia para ajuste de tono)
_modelCenter    // THREE.Vector3 (BBox center del modelo normalizado)
_modelSize      // THREE.Vector3 (BBox size del modelo normalizado)
_GEMINI_KEY     // string | null (de config.js, gitignored)
```

#### Funciones privadas

| Función | Propósito | Línea aprox |
|---|---|---|
| `_buildFaceMesh(landmarks, imgEl)` | Pipeline completo: posiciones → subdivisión → texturas → material → mesh | ~253 |
| `_blendBaseHeadSkin(imgEl, lm, ...)` | Extrae color de piel y tiñe cráneo/cuello del modelo base | ~369 |
| `_buildAlphaMap(landmarks)` | Canvas 128×128 → máscara oval encogida → blur → CanvasTexture | ~665 |
| `_buildNormalMap(imgEl)` | Foto → Sobel → normal map → CanvasTexture (LinearSRGB) | ~716 |
| `_boxBlurCanvasCtx(ctx, W, H, r, p)` | Box blur separable sobre canvas 2D (RGBA) | ~790 |
| `_boxBlurGray(src, W, H, r, p)` | Box blur separable sobre Float32Array grises | ~830 |
| `_conformToSurface(pos, N, lm, sZ)` | Raycasting Z a superficie (INACTIVA — causa domo) | ~857 |
| `_subdivideGeometry(geo)` | Midpoint subdivision: cada triángulo → 4 sub-triángulos | ~914 |
| `_applyGeminiEnhancement(analysis)` | Bumps Z, roughness, normalScale, escala (Fase 1) | ~429 |
| `_ensureMPLandmarker()` | Lazy-init de MediaPipe FaceLandmarker | ~595 |
| `_addLights()` | Hemisphere + Key + Fill + Rim + Under lights | ~978 |
| `_buildLandmarkOverlay(geo, pos)` | 8 contornos + 4 medidas + 22 nodos anatómicos | ~1005 |
| `_removeFaceMesh()` | Dispose completo (geo + map + alphaMap + normalMap + overlay) | ~632 |
| `_disposeGroup(group)` | Traverse y dispose recursivo | ~1230 |
| `loadPreset(idx)` | Carga GLB, normaliza a TARGET_H=2.2, sitúa cámara | ~537 |
| `renderLoop()` | RAF loop con bobbing y damping de rotación | ~1121 |

#### Constantes

```javascript
TARGET_H      = 2.2          // Altura normalizada del modelo (world units)
FACE_W_RATIO  = 0.52         // Referencia (no usada directamente; se usa TARGET_FACE_W)
FACE_H_RATIO  = 0.50         // Referencia
FACE_Z_RATIO  = 0.50         // Dead code
P_YAW         = 0.22         // Amplitud de rotación Y por mouse
P_PITCH       = 0.10         // Amplitud de rotación X por mouse
P_LERP        = 0.055        // Velocidad de interpolación de rotación

// Dentro de _buildFaceMesh:
TARGET_FACE_W = 1.75         // Ancho absoluto de la malla facial
TARGET_FACE_H = 2.25         // Alto absoluto
baseZ         = 0.28         // Profundidad base (frente del maniquí)
wCY           = 0.20         // Elevación del centro facial

FACE_OVAL = [10, 338, ...]   // 36 índices MediaPipe del contorno facial
```

---

### 5.2 `gemini.js`

**Responsabilidad:** comunicación con Gemini Vision API.

```typescript
export async function analyzeWithGemini(
  imgEl: HTMLImageElement,
  apiKey: string
): Promise<AnalysisResult | null>
```

**Modelo:** `gemini-2.0-flash`, temperatura 0.1, `responseMimeType: 'application/json'`.

**Estructura de respuesta:**

```typescript
{
  detected: boolean,
  quality: number,                   // [0,1]
  proportions: { faceAspect, jawWidth, ... },
  depthFeatures: {
    noseProminence, browRidge, cheekboneProminence,
    chinProjection, eyeSocketDepth, lipFullness, ...   // todos [0,1]
  },
  skinAnalysis: { primaryColor, shininess, warmth, ... },
  ...
}
```

---

### 5.3 `geometry.js`

```javascript
export const FACES   // Uint16 array de 2808 elementos (936 triángulos × 3)
```

Fuente: spite/FaceMeshFaceGeometry. Tras subdivisión → 3744 triángulos.

---

### 5.4 `index.html`

UI completa (~1619 líneas). Hero section con canvas 3D, controles de upload/cámara, presets Avatar A/B, métricas, features.

**Integración con avatar.js:**
```javascript
import { init, switchPreset, applyPhoto } from './avatar.js';
init(canvas)
// Upload → applyPhoto(img)
// Cámara → getUserMedia → drawImage → applyPhoto(img)
```

---

### 5.5 `config.js`

```javascript
// GITIGNORED — solo local
export const GEMINI_API_KEY = 'AIzaSy...';
```

En producción: no presente → Fase 1 desactivada silenciosamente.

---

## 6. Sistema de coordenadas y convenciones UV

### Espacio del modelo Three.js
- X: derecha
- Y: arriba
- Z: hacia la cámara (saliente)
- Cámara en `(0, 0, ~5.6)` mirando al origen

### MediaPipe FaceLandmarker
- `lm.x`: [0,1] izquierda→derecha
- `lm.y`: [0,1] arriba→abajo (**invertido** respecto a Three.js Y)
- `lm.z`: negativo = más cerca de cámara (**opuesto** a Three.js Z)

### Conversión MediaPipe → Three.js (v7.0)
```javascript
fx = lm.x - faceCxN          // centrar en 0
fy = -(lm.y - faceCyN)       // centrar en 0, negar Y
fz = -lm.z                   // negar Z (nariz → +Z)

position.x = fx * scaleX + wCX
position.y = fy * scaleY + wCY
position.z = fz * scaleZ * depthWeight + baseZ
```

### Convención UV (v7.0)
```
UV(lm.x, lm.y) con flipY=false
```
Las tres texturas (photo, alpha, normal) usan `flipY=false` y son mutuamente consistentes.

### Por qué `LinearSRGBColorSpace` para el normal map
Los valores RGB del normal map son vectores matemáticos. Corrección gamma los deforma. `LinearSRGBColorSpace` omite la corrección.

---

## 7. Gestión de memoria y ciclo de vida

### Texturas creadas por `applyPhoto`
1. `photoTex` (CanvasTexture) — foto del paciente
2. `alphaTex` (CanvasTexture) — máscara oval 128×128
3. `normalTex` (CanvasTexture) — normal map Sobel

### Geometrías
- Geometría original (468 verts) — dispuesta por `_subdivideGeometry`
- Geometría subdividida (~1870 verts) — dispuesta por `_removeFaceMesh`

### Liberación — `_removeFaceMesh()`
```javascript
mat.map?.dispose()       // photoTex → VRAM liberada
mat.alphaMap?.dispose()  // alphaTex → VRAM liberada
mat.normalMap?.dispose() // normalTex → VRAM liberada
mat.dispose()
geo.dispose()
// + overlay biométrico (lineGeo, nodeGeo, materiales)
```

### `_disposeGroup(group)` — traverse completo
Dispone geometrías, `map`, `alphaMap`, `normalMap`, `roughnessMap` de todos los meshes.

### Environment Map
Creado una vez en `init()` via `PMREMGenerator.fromScene(RoomEnvironment)`. Persiste en `scene.environment`. PMREMGenerator se libera tras generar la textura.

---

## 8. Iluminación y post-procesado

### Luces (configuradas en `_addLights`)

| Nombre | Tipo | Color | Intensidad | Posición |
|---|---|---|---|---|
| Ambiente | HemisphereLight | `#4A6880` / `#2A1810` | 1.0 | — |
| Key | DirectionalLight | `#FFE8D6` (cálido) | 2.5 | (-3, 4, 6) |
| Fill | DirectionalLight | `#AACCF5` (frío) | 1.2 | (4, 0, 3) |
| Rim | DirectionalLight | `#66FFFF` (cyan) | 4.0 | (5, 3, -5) |
| Under | DirectionalLight | `#FFE8CC` | 0.40 | (0, -5, 2) |

Key light: castShadow con mapSize 2048×2048, PCFSoftShadowMap.

### Environment Map

```javascript
const pmremGenerator = new THREE.PMREMGenerator(renderer);
scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
```

RoomEnvironment genera un estudio de iluminación procedural. El parámetro 0.04 controla el blur del env map. Todos los materiales PBR de la escena reciben reflejos automáticamente via `scene.environment`.

### Post-procesado — EffectComposer

```
RenderPass
  → UnrealBloomPass(strength=1.6, radius=0.45, threshold=3.0)
  → BokehPass(focus=5.6, aperture=0.008, maxblur=0.01)
  → OutputPass
```

- **Bloom:** threshold 3.0 → solo materiales HDR (overlay cyan). La piel no brilla.
- **BokehPass:** depth of field cinematográfico centrado en la distancia de la cabeza.

### Tone mapping

```javascript
renderer.toneMapping = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure = 1.15
```

### Render loop

```javascript
head.rotation.y += (tRotY - head.rotation.y) * 0.055   // damping (mouse)
head.rotation.x += (tRotX - head.rotation.x) * 0.055
head.position.y  = Math.sin(t * 0.55) * 0.015           // bobbing suave
controls.update()
composer.render()
```

---

## 9. Historial de cambios por sesión

### v3.x (sesiones anteriores)
- v3.0: primeros modelos GLTF head
- v3.1: projection mapping + MediaPipe detección
- v3.2: fix fórmula de proyección, frontal snap

### v4.0 — Rewrite completo (commit `b20126a`)
- Arquitectura nueva: Three.js ES modules + MediaPipe lazy-init + Gemini async
- `applyPhoto()` como API pública
- Pipeline en dos fases (Fase 0 inmediata + Fase 1 async)
- Overlay holográfico con bloom HDR selectivo

### v4.0 — Bug fix raycasting (commit `b0d8e58`)
- Face mesh aparecía como disco plano → raycasting Z-snap a superficie
- Negación de `lm.z` para convención correcta MediaPipe→Three.js

### v4.1 — Realismo + auditoría (commits `aee5668` y `9b6f553`)
- `_buildAlphaMap` — bordes suaves por contorno oval
- `_buildNormalMap` — relieve Sobel
- 3 bugs corregidos (memory leak, race condition, numeración)

### v6.0 — Fase cinematográfica (commits `58443e8` a `089aee0`)
- `MeshStandardMaterial` → `MeshPhysicalMaterial` (clearcoat, sheen)
- `_blendBaseHeadSkin()` — mimetización cromática del cráneo base
- Raycasting eliminado → posicionamiento absoluto (TARGET_FACE_W, baseZ)
- UV simplificado: `(lm.x, lm.y)` con `flipY=false`
- BokehPass para depth of field
- Overlay biométrico sparse (37 puntos + nearest-neighbor)
- Alpha map con oval encogido 12% y blur agresivo (r=12, 4 pasadas)
- Iluminación reconfigurada: rim cyan 4.0, key cálido 2.5

### v7.0 — PS5 quality (commits `8891041` y `cedf19d`)

**Nuevas funciones:**
- `_subdivideGeometry(geo)` — midpoint subdivision, 936→3744 triángulos
- `_conformToSurface(pos, N, lm, sZ)` — raycasting Z a superficie (CREADA PERO DESHABILITADA)

**Modificaciones:**
- **init():** añade `PMREMGenerator` + `RoomEnvironment` → `scene.environment` para reflejos PBR
- **_buildFaceMesh:** llama a `_subdivideGeometry` tras crear la geometría; gradiente de profundidad radial (`dw`) reemplaza posicionamiento Z uniforme
- **Material PBR:** roughness 0.52, clearcoat 0.12, sheen 0.5, specularIntensity 0.35, envMapIntensity 0.8
- **_buildLandmarkOverlay:** reescrito con 8 contornos anatómicos + 4 líneas craneométricas + 22 nodos (reemplaza nearest-neighbor sparse)

**Bug introducido y corregido:**
- `_conformToSurface` (raycasting) causaba que vértices de borde se envolvieran alrededor del cráneo (efecto domo/casco). Deshabilitado y reemplazado por gradiente radial.

---

## 10. Problemas conocidos y limitaciones

### Performance

| Issue | Severidad | Descripción |
|---|---|---|
| `_buildNormalMap` en main thread | Media | ~50-200ms en móviles. Solución: Web Worker |
| Subdivisión en main thread | Baja | <20ms para 936 triángulos. Aceptable |
| UnrealBloom + BokehPass en móvil | Media | Dos pasadas de post-procesado. Considerar desactivar en gama baja |

### Visuales

| Issue | Severidad | Descripción |
|---|---|---|
| Normal map basado en brillo ≠ geometría | Diseño | Sobel deriva normals del color/sombra, no geometría real |
| No hay corrección de iluminación | Media | La textura incluye sombras del ambiente real de la foto |
| Gemini bumps post-subdivisión | Baja | Midpoints entre vértice bumped y no-bumped tienen step menor |
| `_conformToSurface` genera domo | Documentado | Función existe pero no se invoca |

### Funcionales

| Issue | Severidad | Descripción |
|---|---|---|
| Gemini API key solo local | Alta | En Vercel sin config.js → Fase 1 deshabilitada |
| Sin retries en Gemini | Baja | No reintenta si la API falla |
| `currentPreset` dead code | Trivial | Variable declarada pero nunca leída |

---

## 11. Próximas mejoras sugeridas

### Alta prioridad

| Mejora | Técnica | Impacto |
|---|---|---|
| **Gemini en producción** | Endpoint proxy Vercel (serverless) con API key como env var | Fase 1 activa para todos |
| **Normal map a Web Worker** | `OffscreenCanvas` + `ImageBitmap` | Elimina freeze en móviles |

### Media prioridad

| Mejora | Técnica | Impacto |
|---|---|---|
| **Conformado Z adaptativo** | Raycasting SOLO en anillo FACE_OVAL (36 pts) → baseZ adaptativo; interiores interpolados | Mejor ajuste Z sin efecto domo |
| **Deformación de malla base** | Morph targets por proporciones Gemini | El maniquí se parece a la persona |
| **Corrección de iluminación** | Estimar dirección de luz en foto → alinear luces del escenario | Coherencia foto/3D |
| **Mobile optimización** | `navigator.hardwareConcurrency < 4` → sin Bloom/Bokeh | Experiencia en gama baja |

### Baja prioridad

| Mejora | Técnica | Impacto |
|---|---|---|
| **Estimación de albedo** | Separar textura difusa de sombras (intrinsic images) | Textura más limpia |
| **AR mode** | WebXR + hit-testing | Ver modelo en espacio real |
| **Export GLB** | `GLTFExporter` | Descargar cabeza personalizada |
| **Animaciones faciales** | MediaPipe FaceBlendshapes → blend shapes | Expresiones en tiempo real |

---

*Documentado el 2026-05-09. Versión v7.0.*
