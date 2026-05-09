// ============================================================
//  FaceLab Analytics — Gemini Vision API
//  Analiza una imagen de rostro y devuelve medidas 3D precisas
//  para enriquecer la reconstrucción de malla facial.
// ============================================================

const GEMINI_MODEL = 'gemini-1.5-pro';
const GEMINI_URL   = (key) =>
  `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${key}`;

const FACE_ANALYSIS_PROMPT = `
Analyze this face photo for 3D facial reconstruction.
Return ONLY valid JSON — no markdown, no explanation, no code fences.

{
  "detected": true,
  "quality": 0.95,
  "faceRegion": {
    "x1": 0.20, "y1": 0.10, "x2": 0.80, "y2": 0.90
  },
  "landmarks2D": {
    "leftEye":         {"x": 0.38, "y": 0.38},
    "rightEye":        {"x": 0.62, "y": 0.38},
    "noseTip":         {"x": 0.50, "y": 0.57},
    "noseBase":        {"x": 0.50, "y": 0.63},
    "leftMouth":       {"x": 0.40, "y": 0.72},
    "rightMouth":      {"x": 0.60, "y": 0.72},
    "mouthCenter":     {"x": 0.50, "y": 0.72},
    "leftEar":         {"x": 0.18, "y": 0.46},
    "rightEar":        {"x": 0.82, "y": 0.46},
    "chin":            {"x": 0.50, "y": 0.88},
    "leftBrow":        {"x": 0.35, "y": 0.30},
    "rightBrow":       {"x": 0.65, "y": 0.30},
    "forehead":        {"x": 0.50, "y": 0.14}
  },
  "proportions": {
    "eyeWidth":           0.08,
    "interEyeDistance":   0.24,
    "noseWidth":          0.13,
    "mouthWidth":         0.20,
    "faceWidth":          0.60,
    "faceHeight":         0.78,
    "foreheadRatio":      0.32,
    "noseLength":         0.19,
    "jawWidth":           0.52,
    "faceAspect":         1.30
  },
  "depthFeatures": {
    "noseProminence":     0.65,
    "noseWidth3D":        0.40,
    "cheekboneProminence":0.50,
    "chinProjection":     0.35,
    "browRidge":          0.30,
    "eyeSocketDepth":     0.45,
    "lipFullness":        0.45,
    "jawlineSharpness":   0.55
  },
  "skinAnalysis": {
    "primaryColor":   "#C09070",
    "foreheadColor":  "#C89878",
    "cheekColor":     "#B88868",
    "lipColor":       "#A06858",
    "shininess":      0.25,
    "warmth":         0.65,
    "undertone":      "warm"
  },
  "headShape":   "oval",
  "hairColor":   "#1A0A00",
  "expression":  "neutral",
  "orientation": {"yaw": 0, "pitch": 0, "roll": 0}
}

Rules:
- All 2D coordinates: normalized [0,1] from top-left corner of the full image.
- Depth values: [0,1] where 1 = maximum prominence/depth.
- Colors: hex RGB string.
- faceAspect = faceHeight / faceWidth.
- Return detected:false if no face is visible (don't fabricate data).
`.trim();

/**
 * Converts an HTMLImageElement to a base64 JPEG string.
 */
function _imgToBase64(imgEl, maxSide = 768) {
  const W = imgEl.naturalWidth  || imgEl.width  || 640;
  const H = imgEl.naturalHeight || imgEl.height || 480;
  const scale = Math.min(1, maxSide / Math.max(W, H));
  const sw = Math.round(W * scale);
  const sh = Math.round(H * scale);

  const c = document.createElement('canvas');
  c.width = sw; c.height = sh;
  c.getContext('2d').drawImage(imgEl, 0, 0, sw, sh);
  return c.toDataURL('image/jpeg', 0.88).split(',')[1];
}

/**
 * Sends the face photo to Gemini Vision and returns structured analysis.
 * Returns null on failure.
 *
 * @param {HTMLImageElement} imgEl
 * @param {string} apiKey
 * @returns {Promise<Object|null>}
 */
export async function analyzeWithGemini(imgEl, apiKey) {
  if (!apiKey || !imgEl) return null;

  const base64 = _imgToBase64(imgEl);

  let resp;
  try {
    resp = await fetch(GEMINI_URL(apiKey), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{
          parts: [
            { text: FACE_ANALYSIS_PROMPT },
            { inlineData: { mimeType: 'image/jpeg', data: base64 } },
          ],
        }],
        generationConfig: {
          temperature:      0.1,
          topP:             0.95,
          maxOutputTokens:  2048,
          responseMimeType: 'application/json',
        },
        safetySettings: [
          { category: 'HARM_CATEGORY_HARASSMENT',        threshold: 'BLOCK_NONE' },
          { category: 'HARM_CATEGORY_HATE_SPEECH',       threshold: 'BLOCK_NONE' },
          { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_NONE' },
          { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' },
        ],
      }),
    });
  } catch (e) {
    console.warn('[Gemini] Network error:', e.message);
    return null;
  }

  if (!resp.ok) {
    console.warn('[Gemini] HTTP error:', resp.status, await resp.text().catch(() => ''));
    return null;
  }

  let data;
  try {
    data = await resp.json();
  } catch (e) {
    console.warn('[Gemini] JSON parse error:', e.message);
    return null;
  }

  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) {
    console.warn('[Gemini] Empty response:', JSON.stringify(data).slice(0, 200));
    return null;
  }

  try {
    // Strip possible markdown code fences if model ignores responseMimeType
    const clean = text.replace(/^```(?:json)?\n?/m, '').replace(/\n?```$/m, '').trim();
    const parsed = JSON.parse(clean);
    if (!parsed.detected) {
      console.info('[Gemini] No face detected in image.');
      return null;
    }
    console.info('[Gemini] Analysis OK — quality:', parsed.quality);
    return parsed;
  } catch (e) {
    console.warn('[Gemini] Could not parse response JSON:', text.slice(0, 300));
    return null;
  }
}
