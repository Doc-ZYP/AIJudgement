const fileInput = document.getElementById("file");
const runBtn = document.getElementById("run");
const clearBtn = document.getElementById("clear");

const preview = document.getElementById("preview");
const pctx = preview.getContext("2d", { willReadFrequently: true });

const processed = document.getElementById("processed");
const gctx = processed.getContext("2d", { willReadFrequently: true });

const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const verdictEl = document.getElementById("verdict");

let originalImg = null;

// --- Config you can tune ---
const SCALES = [512, 256];
// 0..1 normalized crop rectangles in source image space
const CROPS = [
  { name: "center", x: 0.15, y: 0.15, w: 0.70, h: 0.70 },
  { name: "tl",     x: 0.00, y: 0.00, w: 0.70, h: 0.70 },
  { name: "tr",     x: 0.30, y: 0.00, w: 0.70, h: 0.70 },
  { name: "bl",     x: 0.00, y: 0.30, w: 0.70, h: 0.70 },
  { name: "br",     x: 0.30, y: 0.30, w: 0.70, h: 0.70 }
];

const MAG_PERCENTILE = 70; // ignore flat regions; keep top 30% gradients
const THRESHOLD = 0.18;    // still needs calibration for your use-case
const UNCERTAIN_MARGIN = 0.03;

function setStatus(msg) { statusEl.textContent = msg; }

function resetUI() {
  pctx.clearRect(0, 0, preview.width, preview.height);
  gctx.clearRect(0, 0, processed.width, processed.height);
  originalImg = null;
  runBtn.disabled = true;
  clearBtn.disabled = true;
  scoreEl.textContent = "—";
  verdictEl.textContent = "—";
  setStatus("Choose an image to begin.");
}
clearBtn.addEventListener("click", resetUI);

fileInput.addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;

  const url = URL.createObjectURL(f);
  const img = new Image();
  img.onload = () => {
    originalImg = img;

    // Draw a simple fitted preview (not used for analysis; analysis uses the original image)
    drawFitted(preview, pctx, img);

    // Clear processed
    gctx.clearRect(0, 0, processed.width, processed.height);

    runBtn.disabled = false;
    clearBtn.disabled = false;
    setStatus("Ready. Click Analyze.");
    URL.revokeObjectURL(url);
  };
  img.onerror = () => {
    setStatus("Could not load that image.");
    runBtn.disabled = true;
    clearBtn.disabled = false;
    URL.revokeObjectURL(url);
  };
  img.src = url;
});

runBtn.addEventListener("click", () => {
  if (!originalImg) return;

  setStatus("Analyzing multi-crop + multi-scale…");

  // Analyze all crops/scales
  const scores = [];
  let processedRendered = false;

  for (const size of SCALES) {
    for (const crop of CROPS) {
      const { score, gradMag } = analyzeCropAtScale(originalImg, crop, size, MAG_PERCENTILE);
      if (Number.isFinite(score)) scores.push(score);

      // Render processed view only once: center crop at 512
      if (!processedRendered && size === 512 && crop.name === "center" && gradMag) {
        renderGradientMagnitude(processed, gctx, gradMag, size, size);
        processedRendered = true;
      }
    }
  }

  if (scores.length === 0) {
    setStatus("Analysis failed (no valid pixels after masking). Try another image.");
    return;
  }

  // Combine across crops/scales
  const finalScore = mean(scores);

  scoreEl.textContent = finalScore.toFixed(4);

  let verdict;
  if (finalScore > THRESHOLD + UNCERTAIN_MARGIN) verdict = "Likely REAL (more coherent gradients)";
  else if (finalScore < THRESHOLD - UNCERTAIN_MARGIN) verdict = "Possibly AI / Synthetic (less coherent gradients)";
  else verdict = "Uncertain (near threshold — needs more evidence)";

  verdictEl.textContent = verdict;

  setStatus(
    `Combined ${scores.length} measurements (crops × scales). ` +
    `Mask kept pixels above the ${MAG_PERCENTILE}th percentile of gradient magnitude.`
  );
});

// ======================
// Analysis pipeline
// ======================

function analyzeCropAtScale(img, crop, size, magPercentile) {
  // Offscreen canvas to draw the crop at the target size
  const oc = document.createElement("canvas");
  oc.width = size; oc.height = size;
  const octx = oc.getContext("2d", { willReadFrequently: true });

  // Compute crop rectangle in source pixels
  const sx = Math.floor(crop.x * img.width);
  const sy = Math.floor(crop.y * img.height);
  const sw = Math.floor(crop.w * img.width);
  const sh = Math.floor(crop.h * img.height);

  // Draw crop scaled to size×size
  octx.drawImage(img, sx, sy, sw, sh, 0, 0, size, size);

  const imageData = octx.getImageData(0, 0, size, size);
  const data = imageData.data;

  // Luminance
  const L = new Float32Array(size * size);
  for (let i = 0, p = 0; i < L.length; i++, p += 4) {
    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;
    L[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  // Sobel gradients
  const { Gx, Gy } = sobelGradients(L, size, size);

  // Magnitude (for masking)
  const mag = new Float32Array(size * size);
  for (let i = 0; i < mag.length; i++) {
    const gx = Gx[i], gy = Gy[i];
    mag[i] = Math.hypot(gx, gy);
  }

  // Compute threshold magnitude at percentile
  const magThr = percentile(mag, magPercentile);

  // Covariance over masked pixels only
  const C = covariance2DMasked(Gx, Gy, mag, magThr);
  if (!C) return { score: NaN, gradMag: null };

  const { lambda1, lambda2 } = eigenvalues2x2(C);

  const eps = 1e-12;
  const score = (lambda1 - lambda2) / (lambda1 + lambda2 + eps);

  // Return mag only if you want to render processed (we do for center@512)
  return { score, gradMag: mag };
}

function sobelGradients(L, W, H) {
  const Gx = new Float32Array(W * H);
  const Gy = new Float32Array(W * H);

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;

      const a = L[(y - 1) * W + (x - 1)];
      const b = L[(y - 1) * W + (x)];
      const c = L[(y - 1) * W + (x + 1)];
      const d = L[(y) * W + (x - 1)];
      const f = L[(y) * W + (x + 1)];
      const g = L[(y + 1) * W + (x - 1)];
      const h = L[(y + 1) * W + (x)];
      const k = L[(y + 1) * W + (x + 1)];

      const gx = (-a + c) + (-2 * d + 2 * f) + (-g + k);
      const gy = (a + 2 * b + c) + (-g - 2 * h - k);

      Gx[i] = gx;
      Gy[i] = gy;
    }
  }
  return { Gx, Gy };
}

function covariance2DMasked(Gx, Gy, mag, thr) {
  const n = Gx.length;

  // Mean over masked pixels
  let mx = 0, my = 0, count = 0;
  for (let i = 0; i < n; i++) {
    if (mag[i] >= thr) {
      mx += Gx[i];
      my += Gy[i];
      count++;
    }
  }
  if (count < 500) return null; // too few pixels => unstable
  mx /= count; my /= count;

  // Covariance
  let cxx = 0, cxy = 0, cyy = 0;
  for (let i = 0; i < n; i++) {
    if (mag[i] >= thr) {
      const dx = Gx[i] - mx;
      const dy = Gy[i] - my;
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    }
  }
  cxx /= count; cxy /= count; cyy /= count;
  return { cxx, cxy, cyy };
}

function eigenvalues2x2({ cxx, cxy, cyy }) {
  const tr = cxx + cyy;
  const det = cxx * cyy - cxy * cxy;
  const disc = Math.max(0, tr * tr - 4 * det);
  const s = Math.sqrt(disc);
  const l1 = (tr + s) / 2;
  const l2 = (tr - s) / 2;
  return l1 >= l2 ? { lambda1: l1, lambda2: l2 } : { lambda1: l2, lambda2: l1 };
}

// Percentile via sorting (simple + reliable; OK for 512^2)
function percentile(arrFloat32, p) {
  const n = arrFloat32.length;
  const copy = new Array(n);
  for (let i = 0; i < n; i++) copy[i] = arrFloat32[i];
  copy.sort((a, b) => a - b);
  const idx = Math.min(n - 1, Math.max(0, Math.floor((p / 100) * (n - 1))));
  return copy[idx];
}

function mean(xs) {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

// Render gradient magnitude as a grayscale image (normalized)
function renderGradientMagnitude(canvas, ctx, mag, W, H) {
  const img = ctx.createImageData(W, H);

  // Robust normalization: use 99th percentile as white point
  const white = percentile(mag, 99) || 1e-6;

  for (let i = 0, p = 0; i < mag.length; i++, p += 4) {
    const v = Math.min(1, mag[i] / white);
    const g = Math.round(v * 255);
    img.data[p] = g;
    img.data[p + 1] = g;
    img.data[p + 2] = g;
    img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// Simple fitted preview draw (centered letterbox)
function drawFitted(canvas, ctx, img) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b0d10";
  ctx.fillRect(0, 0, W, H);

  const scale = Math.min(W / img.width, H / img.height);
  const dw = Math.round(img.width * scale);
  const dh = Math.round(img.height * scale);
  const dx = Math.round((W - dw) / 2);
  const dy = Math.round((H - dh) / 2);
  ctx.drawImage(img, dx, dy, dw, dh);
}

// start
resetUI();
