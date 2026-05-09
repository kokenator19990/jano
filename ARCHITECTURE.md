# FaceLab Analytics — Arquitectura Técnica

> Versión actual: **v4.1**
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
   - [gemini.js](#52-geminiijs)
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

FaceLab Analytics es una aplicación web de análisis facial que reconstruye en tiempo real un modelo 3D de la cara del usuario sobre una cabeza genérica. El flujo tiene tres fases:

```
FOTO DEL USUARIO
      │
      ▼
┌─────────────────────────────────────────┐
│  FASE 0 — Inmediato (~500 ms)           │
│  MediaPipe 468 landmarks → malla 3D     │
│  Raycasting a superficie del modelo     │
│  Foto mapeada como textura UV           │
│  Alpha map oval + Normal map Sobel      │
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
│  FASE 2 — Overlay holográfico           │
│  35 puntos anatómicos (Points)          │
│  44 aristas craneofaciales (Lines)      │
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
| Post-procesado | UnrealBloomPass | r162 | jsDelivr CDN |
| Fuentes | Space Grotesk + Inter | — | Google Fonts CDN |
| Deploy | Vercel (static) | — | git push origin master |
| Sin build step | Vanilla ES modules | — | — |

**No hay bundler, no hay npm, no hay node_modules.** Todo se sirve como archivos estáticos.

---

## 3. Estructura de archivos

```
Jano/
├── index.html          UI completa + lógica de cámara/upload + importmap Three.js
├── avatar.js           Motor 3D — pipeline MediaPipe → malla → Gemini (API pública)
├── gemini.js           Cliente Gemini Vision API (analyzeWithGemini)
├── geometry.js         FACES[2808] — triangulación canónica MediaPipe (468 vértices)
├── config.js           GEMINI_API_KEY (gitignored — solo local)
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
export function init(canvas)    // Inicializa escena Three.js
export function switchPreset(idx) // Cambia modelo de cabeza
export async function applyPhoto(imgEl) // Pipeline principal
```

---

## 4. Pipeline de reconstrucción facial

### Fase 0 — `_buildFaceMesh(landmarks, imgEl)`

#### 4.1 Cálculo de posiciones XY

Se calcula el bounding box de los 468 landmarks normalizados `[0,1]` para centrar la cara:

```javascript
// Escala para que la cara ocupe FACE_W/H_RATIO del modelo
scaleX = (modelSize.x * 0.88) / faceWidthNormalized
scaleY = (modelSize.y * 0.70) / faceHeightNormalized

// Posición XY en espacio del modelo
positions[i*3+0] = (lm.x - faceCenterX) * scaleX + modelCenterX
positions[i*3+1] = -(lm.y - faceCenterY) * scaleY + modelCenterY  // eje Y invertido
```

#### 4.2 Raycasting Z (clave del realismo)

Cada vértice lanza un rayo desde `(x, y, RAY_Z)` en dirección `-Z` y se snappea a la superficie del modelo:

```
Para cada vértice i de 468:
  origen = (positions[i*3+0], positions[i*3+1], modelCenter.z + modelSize.z)
  rayo = dirección (0, 0, -1)
  impacto = Raycaster.intersectObjects(headMeshes)
  surfaceZ = impacto.point.z  ||  fallbackZ
  depthOffset = (-landmarks[i].z) * modelSize.z * 0.06   // signo negado
  positions[i*3+2] = surfaceZ + depthOffset + 0.003       // anti z-fight
```

**Nota crítica de convención Z:**
En MediaPipe, `lm.z < 0` = más cerca de la cámara = `+Z` en Three.js.
Por eso se niega: `(-lm.z) * escala` → nariz sobresale, cuencas oculares retroceden.

**Prerequisito:** `head.rotation.set(0,0,0)` + `updateWorldMatrix(true,true)` antes del raycasting, para que las matrices mundo coincidan con las posiciones XY calculadas.

#### 4.3 UVs y coordenadas de textura

```javascript
uvs[i*2+0] = lm.x          // U
uvs[i*2+1] = 1.0 - lm.y    // V (invertido respecto a imagen)
```

Con `flipY = false` en todas las texturas (photo, alpha, normal), la relación canvas→UV es:
- `UV(u, v)` → pixel de canvas `(u * W, (1-v) * H)`

Esto significa que para que la foto aparezca correcta, el canvas dibujado con `drawImage(imgEl, 0, 0)` (donde landmark `lm` está en `(lm.x*W, lm.y*H)`) es consistente con UV `(lm.x, 1-lm.y)`.

#### 4.4 Alpha Map — difuminado de bordes

```
Contorno FACE_OVAL (36 índices MediaPipe) → canvas 128×128
   → polígono relleno blanco sobre negro
   → box-blur separable (radio=4, 3 pasadas)
   → CanvasTexture con flipY=false
```

Three.js lee el **canal R** de `alphaMap` como transparencia. Requiere `transparent: true` en el material.

**FACE_OVAL indices:**
`[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]`

#### 4.5 Normal Map — relieve superficial desde foto

```
imgEl → canvas reducido (max 512px) → getImageData → Float32Array grises
   → box-blur separable (radio=3, 3 pasadas)
   → Sobel 3×3 → gx, gy
   → nx = -gx*STR,  ny = gy*STR,  nz = 1.0   (STR=3.0)
   → normalize → pack a RGB [0,255]
   → CanvasTexture con colorSpace=LinearSRGBColorSpace, flipY=false
```

**CRÍTICO:** `colorSpace = THREE.LinearSRGBColorSpace` (NO `SRGBColorSpace`).
Si se usa `SRGBColorSpace`, Three.js aplica corrección gamma a los vectores normales, deformando las direcciones y produciendo artefactos visuales.

El normal map empieza en `normalScale = (0.4, 0.4)`, ajustado dinámicamente por Gemini en la Fase 1.

#### 4.6 Material PBR

```javascript
MeshStandardMaterial({
  map:                 photoTex,         // foto del paciente (SRGB, flipY=false)
  alphaMap:            alphaTex,         // máscara oval (flipY=false)
  normalMap:           normalTex,        // relieve Sobel (LinearSRGB, flipY=false)
  normalScale:         Vector2(0.4, 0.4),
  transparent:         true,             // requerido para alphaMap
  roughness:           0.72,
  metalness:           0.0,
  envMapIntensity:     0.4,
  polygonOffset:       true,
  polygonOffsetFactor: -2,
  polygonOffsetUnits:  -2,              // evita z-fighting con el modelo base
})
mesh.renderOrder = 1
```

---

### Fase 1 — `_applyGeminiEnhancement(analysis)`

Aplica el análisis de Gemini sobre la malla ya construida. Se ejecuta **async** — no bloquea el render de Fase 0.

#### 4.7 Bumps de profundidad por zona

Desplazamiento adicional en Z sobre los vértices de 468, por grupos anatómicos:

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

#### 4.8 Roughness dinámico

```javascript
roughness = clamp(0.55 + (1 - shininess) * 0.35, 0.55, 0.90)
// shininess: [0,1] donde 1 = muy brillante (piel grasa/humeda)
```

#### 4.9 normalScale dinámico

```javascript
ns = clamp((noseProminence + browRidge + cheekboneProminence) / 3, 0.15, 0.75)
material.normalScale.set(ns, ns)
// Caras con rasgos prominentes → más relieve en el normal map
```

#### 4.10 Iluminación ambiental

La `HemisphereLight` se ajusta sutilmente (18%) al tono de piel primario de Gemini. La key light se calienta o enfría según el valor `warmth`.

#### 4.11 Escala del modelo base

```javascript
// faceAspect = alto/ancho. > 1.3 → cara más larga
targetY = clamp(1.0 + (faceAspect - 1.3) * 0.10, 0.90, 1.10)
head.scale.y = lerp(head.scale.y, targetY, 0.55)

// jawWidth normalizado contra referencia 0.52
jawRatio = clamp(jawWidth / 0.52, 0.85, 1.18)
head.scale.x = lerp(head.scale.x, jawRatio, 0.40)
```

#### 4.12 Race condition guard

```javascript
const meshSnapshot = faceMesh   // referencia antes de la llamada async
analyzeWithGemini(...).then(analysis => {
  if (!analysis || _faceMesh3D !== meshSnapshot) return  // nueva foto cargada → abort
  _applyGeminiEnhancement(analysis)
})
```

---

## 5. Módulos detallados

### 5.1 `avatar.js`

**Responsabilidad:** todo el motor 3D — escena, cámara, luces, modelos, malla facial.

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
head            // THREE.Group (modelo + overlay holográfico)
_faceMesh3D     // THREE.Mesh | null (malla facial activa)
_mpLandmarker   // FaceLandmarker | null | false (lazy init)
_hemLight       // HemisphereLight (referencia para ajuste de tono)
_modelCenter    // THREE.Vector3 (BBox center del modelo normalizado)
_modelSize      // THREE.Vector3 (BBox size del modelo normalizado)
_GEMINI_KEY     // string | null (de config.js, gitignored)
```

#### Funciones privadas

| Función | Propósito |
|---|---|
| `_buildFaceMesh(landmarks, imgEl)` | Construye THREE.Mesh con raycasting, 3 texturas, PBR |
| `_buildAlphaMap(landmarks)` | Canvas 128×128 → máscara oval → CanvasTexture |
| `_buildNormalMap(imgEl)` | Foto → Sobel → normal map → CanvasTexture |
| `_boxBlurCanvasCtx(ctx, W, H, r, p)` | Box blur separable sobre canvas 2D (RGBA) |
| `_boxBlurGray(src, W, H, r, p)` | Box blur separable sobre Float32Array grises |
| `_applyGeminiEnhancement(analysis)` | Desplazos Z, roughness, normalScale, escala |
| `_ensureMPLandmarker()` | Lazy-init de MediaPipe FaceLandmarker |
| `_addLights()` | Hem + Key + Fill + Rim + Under lights |
| `_buildLandmarkOverlay(c, hw, hh, hd)` | 35 puntos + 44 aristas holográficos |
| `_removeFaceMesh()` | Dispose completo (geo + map + alphaMap + normalMap) |
| `_disposeGroup(group)` | Traverse y dispose recursivo |
| `_buildLoadingOverlay(wrap)` | Spinner de carga del modelo |
| `_buildStatusBar(wrap)` | Barra de estado deslizante |
| `_setStatus(msg, showSpinner)` | Actualiza la barra de estado |
| `_teardown()` | Limpieza completa (RAF, composer, renderer, listeners) |
| `loadPreset(idx)` | Carga GLB, normaliza, sitúa cámara |
| `renderLoop()` | RAF loop con bobbing y damping de rotación |
| `_onMouse(e)` | Mouse tracking para rotación suave |
| `_showLoading(v)` | Muestra/oculta overlay de carga |

#### Constantes

```javascript
TARGET_H      = 2.2          // Altura normalizada del modelo (world units)
FACE_W_RATIO  = 0.88         // Cara ocupa 88% del ancho del modelo
FACE_H_RATIO  = 0.70         // Cara ocupa 70% del alto del modelo
FACE_Z_RATIO  = 0.44         // Referencia histórica (no usada actualmente)
P_YAW         = 0.22         // Amplitud de rotación Y por mouse
P_PITCH       = 0.10         // Amplitud de rotación X por mouse
P_LERP        = 0.055        // Velocidad de interpolación de rotación

FACE_OVAL = [10, 338, ...]   // 36 índices MediaPipe del contorno facial
```

---

### 5.2 `gemini.js`

**Responsabilidad:** comunicación con Gemini Vision API. Sin dependencias externas.

```typescript
export async function analyzeWithGemini(
  imgEl: HTMLImageElement,
  apiKey: string
): Promise<AnalysisResult | null>
```

**Flujo interno:**
1. `_imgToBase64(imgEl, maxSide=768)` → JPEG base64
2. `fetch(GEMINI_URL(key), body)` con prompt estructurado
3. Parse JSON de la respuesta
4. Retorna `null` si `detected === false` o en cualquier error

**Modelo:** `gemini-2.0-flash`
**Temperatura:** 0.1
**`responseMimeType`:** `'application/json'`
**Safety settings:** todas en `BLOCK_NONE` (contenido médico de análisis facial)

**Estructura de respuesta esperada:**

```typescript
{
  detected: boolean,
  quality: number,                   // [0,1]
  faceRegion: { x1, y1, x2, y2 },   // normalizado
  landmarks2D: {
    leftEye, rightEye, noseTip, noseBase,
    leftMouth, rightMouth, mouthCenter,
    leftEar, rightEar, chin, leftBrow, rightBrow, forehead
  },
  proportions: {
    eyeWidth, interEyeDistance, noseWidth, mouthWidth,
    faceWidth, faceHeight, foreheadRatio, noseLength,
    jawWidth, faceAspect                                   // = faceHeight/faceWidth
  },
  depthFeatures: {
    noseProminence, noseWidth3D, cheekboneProminence,
    chinProjection, browRidge, eyeSocketDepth,
    lipFullness, jawlineSharpness                          // todos en [0,1]
  },
  skinAnalysis: {
    primaryColor, foreheadColor, cheekColor, lipColor,    // hex RGB
    shininess, warmth, undertone
  },
  headShape: string,                 // 'oval' | 'round' | etc.
  hairColor: string,                 // hex RGB
  expression: string,
  orientation: { yaw, pitch, roll }  // grados
}
```

---

### 5.3 `geometry.js`

**Responsabilidad:** triangulación canónica de la malla facial de MediaPipe.

```javascript
export const FACES   // Uint16 array de 2808 elementos (936 triángulos × 3 vértices)
```

Fuente: [spite/FaceMeshFaceGeometry](https://github.com/spite/FaceMeshFaceGeometry)
Los índices corresponden exactamente a los 468 landmarks de MediaPipe FaceLandmarker.

---

### 5.4 `index.html`

**Responsabilidad:** UI completa (sin framework) + integración con `avatar.js`.

**Secciones del HTML:**
- Header con logo, navegación
- Hero section con `<canvas id="avatarCanvas">` + controles (Upload, Cámara, presets Avatar A/B)
- Sección de métricas (3 tarjetas)
- Sección de features (grid)
- Modal de cámara con stream de video y botón de captura
- `<canvas id="snapCanvas" style="display:none">` para capturas de cámara

**Integración con avatar.js (al final del HTML):**

```javascript
import { init, switchPreset, applyPhoto } from './avatar.js';

init(canvas)                    // inicializa escena

// Upload de archivo
fileInput → img.onload → applyPhoto(img) + URL.revokeObjectURL

// Cámara
getUserMedia({ facingMode: 'user', width: 640, height: 480 })
→ drawImage en snapCanvas
→ img.src = toDataURL('image/jpeg', 0.92)
→ applyPhoto(img)

// Preset switching
.av-preset-btn[data-preset] → switchPreset(Number(btn.dataset.preset))
```

**Design tokens CSS (`:root`):**

```css
--bg: #05101a          --cyan: #00cfe0
--bg-surface: #0a1828  --glow-sm / glow-md / glow-lg
--bg-card: #0d1f32     --font-display: 'Space Grotesk'
--font: 'Inter'
```

---

### 5.5 `config.js`

```javascript
// GITIGNORED — solo existe localmente
export const GEMINI_API_KEY = 'AIzaSy...';
```

En producción (Vercel), `config.js` no está presente → `_GEMINI_KEY = null` → Fase 1 (Gemini) queda desactivada silenciosamente. La Fase 0 (malla MediaPipe) siempre funciona.

**Para activar Gemini en producción:** agregar como variable de entorno en Vercel y leerla server-side (actualmente no implementado).

---

## 6. Sistema de coordenadas y convenciones UV

### Espacio del modelo Three.js
- X: derecha
- Y: arriba
- Z: hacia la cámara (saliente)
- Cámara en `(0, 0, 5.6)` mirando al origen

### MediaPipe FaceLandmarker
- `lm.x`: [0,1] izquierda→derecha (misma que Three.js X normalizado)
- `lm.y`: [0,1] arriba→abajo (**invertido** respecto a Three.js Y)
- `lm.z`: negativo = más cerca de cámara (**opuesto** a Three.js Z)

### Conversión MediaPipe → Three.js
```javascript
position.x =  (lm.x - faceCenter.x) * scaleX + modelCenter.x
position.y = -(lm.y - faceCenter.y) * scaleY + modelCenter.y   // negar Y
position.z = surfaceZ + (-lm.z) * depthScale                    // negar Z
```

### Convención UV → Canvas pixel (con `flipY=false`)
```
UV (u, v)  →  pixel (u * W,  (1-v) * H)
```
Esta relación significa que dibujar en el canvas en coordenadas de imagen `(lm.x * W, lm.y * H)` es consistente con la UV `(lm.x, 1-lm.y)`. Las tres texturas (photo, alpha, normal) usan `flipY=false` y son mutuamente consistentes.

### Por qué `LinearSRGBColorSpace` para el normal map
WebGL por defecto aplica corrección gamma (`sRGB → linear`) al leer texturas con `SRGBColorSpace`. Los valores RGB del normal map son vectores matemáticos, no colores perceptuales. Aplicarles gamma los convierte en vectores incorrectos (`(0.5→0.216, 0.5→0.216, 1.0→1.0)` en lugar de `(0, 0, 1)` para normal plano). `LinearSRGBColorSpace` omite la corrección gamma.

---

## 7. Gestión de memoria y ciclo de vida

### Creación de texturas (por `applyPhoto`)
Cada llamada a `applyPhoto` crea 3 objetos GPU:
1. `photoTex` (CanvasTexture) — foto del paciente
2. `alphaTex` (CanvasTexture) — máscara oval 128×128
3. `normalTex` (CanvasTexture) — normal map

### Liberación explícita — `_removeFaceMesh()`
Llamada al inicio de cada `applyPhoto` y desde `switchPreset` implícitamente via `_disposeGroup`.

```javascript
mat.map?.dispose()       // photoTex → VRAM liberada
mat.alphaMap?.dispose()  // alphaTex → VRAM liberada
mat.normalMap?.dispose() // normalTex → VRAM liberada
mat.dispose()
geo.dispose()
```

### `_disposeGroup(group)` — traverse completo
Aplicado cuando se cambia de preset. Dispone geometrías, `map`, `alphaMap`, `normalMap`, `roughnessMap` de todos los meshes del grupo.

### `_teardown()` — limpieza total
Llamada por `init()` si ya existía una instancia. Cancela RAF, destruye composer, renderer, controls, remueve event listeners.

### Ciclo completo de `applyPhoto → applyPhoto`
```
applyPhoto(foto1)
  → _removeFaceMesh()  [noop, nada existe]
  → _buildFaceMesh()   → crea geo + 3 texturas + mesh
  → head.add(mesh)     [mesh en escena]
  → Gemini async start

applyPhoto(foto2)
  → _removeFaceMesh()  → dispose de las 3 texturas + geo + mat
  → _buildFaceMesh()   → nuevas 3 texturas + mesh
  → Gemini async start  (meshSnapshot != null → guarda referencia)

[si Gemini de foto1 responde aquí]
  → _faceMesh3D !== meshSnapshot(foto1) → return early ✓
```

---

## 8. Iluminación y post-procesado

### Luces (configuradas en `_addLights`)

| Nombre | Tipo | Color | Intensidad | Posición |
|---|---|---|---|---|
| Ambiente | HemisphereLight | `#4A6880` / `#2A1810` | 1.6 | — |
| Key | DirectionalLight | `#FFF5EC` (cálido) | 4.2 | (-2, 3, 5) |
| Fill | DirectionalLight | `#CCDDFF` (frío) | 1.4 | (4, 0.5, 2) |
| Rim | DirectionalLight | `#88AAFF` | 3.0 | (0, 2, -8) |
| Under | DirectionalLight | `#FFE8CC` | 0.40 | (0, -5, 2) |

La `HemisphereLight` se ajusta dinámicamente en Fase 1 (18% lerp hacia el tono de piel de Gemini).
La Key light ajusta temperatura según `warmth` de Gemini (12% lerp).

### Post-procesado — EffectComposer

```
RenderPass → UnrealBloomPass(strength=1.45, radius=0.45, threshold=3.50) → OutputPass
```

El bloom threshold `3.50` es intencional: solo activan bloom los materiales HDR (colores > 1.0 en linear). Los puntos y líneas del overlay holográfico usan `setRGB(0, 7, 8)` y `setRGB(0, 5, 5.5)` — muy por encima del threshold → bloom selectivo cyan.

### Render loop

```javascript
head.rotation.y += (tRotY - head.rotation.y) * 0.055   // damping suave (mouse)
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
- **Arquitectura nueva:** Three.js ES modules + MediaPipe lazy-init + Gemini async
- `applyPhoto()` exportada como API pública
- Pipeline en dos fases (Fase 0 inmediata + Fase 1 async)
- Overlay holográfico con bloom HDR selectivo
- `gemini.js` y `config.js` separados

### v4.0 — Bug fix raycasting (commit `b0d8e58`)
**Problema:** face mesh aparecía como disco plano desconectado de la cabeza.
**Causa 1:** Z computado con fórmula plana `lm.z * scaleZ + wCZ` (sin conformar a superficie).
**Causa 2:** signo de Z incorrecto (nariz retrocedía, cuencas sobresalían).
**Solución:**
- Raycasting por vértice: rayo `(x, y, RAY_Z)` en dirección `(0,0,-1)` → snap a superficie
- Negación de `lm.z` para convención correcta MediaPipe→Three.js
- `polygonOffset: true, factor: -2, units: -2` para evitar z-fighting
- `head.rotation.set(0,0,0) + updateWorldMatrix()` antes del raycasting

### v4.1 — Realismo + auditoría (commits `aee5668` y `9b6f553`)

**Nuevas funciones:**
- `_buildAlphaMap(landmarks)` — bordes suaves por contorno oval
- `_buildNormalMap(imgEl)` — relieve superficial via Sobel
- `_boxBlurCanvasCtx(ctx, W, H, r, p)` — blur para canvas RGBA
- `_boxBlurGray(src, W, H, r, p)` — blur para Float32 grises
- `FACE_OVAL` constant (36 índices)

**Modificaciones:**
- `_buildFaceMesh`: integra `alphaMap` + `normalMap` + `transparent: true`
- `_applyGeminiEnhancement`: paso 3 nuevo → `normalScale` dinámico `[0.15–0.75]`
- `_removeFaceMesh`: añade `mat.alphaMap?.dispose()` y `mat.normalMap?.dispose()`
- `_disposeGroup`: añade `m?.alphaMap?.dispose()`
- `applyPhoto`: `meshSnapshot` guard para race condition Gemini

**Bugs corregidos:**
1. Memory leak: `alphaMap` no se liberaba al `switchPreset` con malla activa
2. Race condition: análisis Gemini de foto anterior aplicado a nueva malla
3. Numeración duplicada de secciones en `_applyGeminiEnhancement`

---

## 10. Problemas conocidos y limitaciones

### Performance
| Issue | Severidad | Descripción |
|---|---|---|
| `_buildNormalMap` en main thread | Media | ~50-200ms para imágenes 640×480. Visible freeze en móviles lentos. Solución: Web Worker |
| Raycasting 468×N | Baja | 468 intersectObjects en modelo ~2k polígonos. ~20-50ms. AABB culling ayuda |
| UnrealBloom en móvil | Media | Costoso en GPU móvil. Considerar deshabilitar o reducir resolución |

### Visuales
| Issue | Severidad | Descripción |
|---|---|---|
| Normal map basado en brillo ≠ geometría | Diseño | El Sobel deriva normals del color/sombra de la foto, no de la geometría real. Funciona pero no es físicamente correcto |
| No hay corrección de iluminación | Media | La textura incluye las sombras del ambiente real de la foto. El modelo 3D tiene iluminación virtual distinta → inconsistencia |

### Funcionales
| Issue | Severidad | Descripción |
|---|---|---|
| Gemini API key hardcodeada en config.js | Alta | Solo funciona localmente. En Vercel sin config.js → Fase 1 deshabilitada silenciosamente |
| Sin retries en Gemini | Baja | Si la API falla por rate limit, no se reintenta |
| `currentPreset` no se lee | Trivial | Variable dead code |
| `FACE_Z_RATIO` no se usa | Trivial | Constante dead code (reemplazada por raycasting) |

---

## 11. Próximas mejoras sugeridas

Ordenadas por impacto visual / complejidad:

### Alta prioridad

| Mejora | Técnica | Impacto |
|---|---|---|
| **Gemini en producción** | Variable de entorno Vercel o backend proxy | Fase 1 siempre activa para todos los usuarios |
| **Normal map a Web Worker** | `new Worker()` + `OffscreenCanvas` | Elimina freeze en móviles |
| **Corrección de iluminación** | Estimar dirección de luz en foto → ajustar luces del escenario | Coherencia foto/3D mucho mayor |

### Media prioridad

| Mejora | Técnica | Impacto |
|---|---|---|
| **Deformación de malla base** | Morph targets o blend shapes por proporción facial | El modelo 3D refleja la forma real de la cara |
| **Skinning de cuello** | Extender malla facial hacia el cuello con gradiente de alpha | Integración más suave con el modelo base |
| **Estimación de albedo** | Separar textura difusa de sombras (intrinsic images) | Textura más limpia bajo iluminación virtual |
| **Mobile optimización** | Deshabilitar UnrealBloom en `navigator.hardwareConcurrency < 4` | Mejor experiencia en gama baja |

### Baja prioridad

| Mejora | Técnica | Impacto |
|---|---|---|
| **Malla pura MediaPipe** | Reemplazar cabeza genérica con malla de 468 puntos extendida | Forma de cara 100% personalizada |
| **AR mode** | WebXR + hit-testing | Ver modelo en espacio real |
| **Export GLB** | `GLTFExporter` de Three.js | Descargar la cabeza personalizada |
| **Animaciones faciales** | Blend shapes de MediaPipe FaceBlendshapes | Expresiones en tiempo real |

---

*Documentado automáticamente el 2026-05-09.*
