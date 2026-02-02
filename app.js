const fileInput = document.getElementById("file");
const runBtn = document.getElementById("run");
const clearBtn = document.getElementById("clear");

const canvas = document.getElementById("preview");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const verdictEl = document.getElementById("verdict");

let loadedImage = null;

// --- Utilities ---
function setStatus(msg) { statusEl.textContent = msg; }

function resetUI() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  loadedImage = null;
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
    // Fit image into 512x512 while preserving aspect ratio
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

    loadedImage = { img, dx, dy, dw, dh };
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
  if (!loadedImage) return;
  setStatus("Analyzing…");

  // Read pixels from the whole 512x512 canvas
  const W = canvas.width, H = canvas.height;
  const imageData = ctx.getImageData(0, 0, W, H);
  const data = imageData.data;

  // Convert to luminance L (float)
  const L = new Float32Array(W * H);
  for (let i = 0, p = 0; i < L.length; i++, p += 4) {
    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;
    // Rec.709 luminance
    L[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  // Compute Sobel gradients
  const { Gx, Gy } = sobelGradients(L, W, H);

  // Covariance of gradient vectors across pixels
  const C = covariance2D(Gx, Gy);

  // Eigenvalues of 2x2 covariance
  const { lambda1, lambda2 } = eigenvalues2x2(C);

  // Anisotropy ratio score in [0,1]
  const eps = 1e-12;
  const score = (lambda1 - lambda2) / (lambda1 + lambda2 + eps);

  // --- Heuristic threshold (NEEDS CALIBRATION) ---
  // Default guess: real photos tend to have higher anisotropy than diffusion samples.
  // You should replace this threshold after calibration.
  const threshold = 0.18;

  const verdict = score >= threshold
    ? "Likely REAL (more coherent gradients)"
    : "Possibly AI / Synthetic (more isotropic gradients)";

  scoreEl.textContent = score.toFixed(4);
  verdictEl.textContent = verdict;

  // Add some context
  setStatus(
    `Computed gradient covariance eigenvalues: λ1=${lambda1.toExponential(3)}, λ2=${lambda2.toExponential(3)}.`
  );
});

// --- Math / Image ops ---
function sobelGradients(L, W, H) {
  const Gx = new Float32Array(W * H);
  const Gy = new Float32Array(W * H);

  // Sobel kernels:
  // Gx = [-1 0 1; -2 0 2; -1 0 1]
  // Gy = [ 1 2 1;  0 0 0; -1 -2 -1]
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

function covariance2D(Gx, Gy) {
  // Compute mean
  const n = Gx.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) {
    mx += Gx[i];
    my += Gy[i];
  }
  mx /= n; my /= n;

  // Compute cov
  let cxx = 0, cxy = 0, cyy = 0;
  for (let i = 0; i < n; i++) {
    const dx = Gx[i] - mx;
    const dy = Gy[i] - my;
    cxx += dx * dx;
    cxy += dx * dy;
    cyy += dy * dy;
  }
  cxx /= n; cxy /= n; cyy /= n;

  return { cxx, cxy, cyy };
}

function eigenvalues2x2({ cxx, cxy, cyy }) {
  // For matrix [[cxx, cxy],[cxy, cyy]]
  // eigenvalues: (tr ± sqrt(tr^2 - 4 det))/2
  const tr = cxx + cyy;
  const det = cxx * cyy - cxy * cxy;
  const disc = Math.max(0, tr * tr - 4 * det);
  const s = Math.sqrt(disc);

  const l1 = (tr + s) / 2;
  const l2 = (tr - s) / 2;

  // Ensure lambda1 >= lambda2
  return l1 >= l2 ? { lambda1: l1, lambda2: l2 } : { lambda1: l2, lambda2: l1 };
}

// start
resetUI();
