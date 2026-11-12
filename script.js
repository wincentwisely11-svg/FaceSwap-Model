/* =====================================================================
   Face Swap (murni di browser). Alur:
   deteksi 468 landmark -> warp mesh sumber ke bentuk tujuan (per piksel, tanpa celah)
   -> mask oval wajah (dahi dikecilkan agar poni/rambut aman)
   -> pertahankan mata/mulut tujuan bila ekspresi terlalu berbeda
   -> koreksi pencahayaan dari kulit saja -> Poisson blending -> campur lembut
   ===================================================================== */
const MAX = 1280;
const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// indeks landmark MediaPipe
const OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
const LIPS_IN = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95];
const LIPS_OUT = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
const EYE_R = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7];
const EYE_L = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382];
const BROW_R = [70,63,105,66,107,55,65,52,53,46];
const BROW_L = [300,293,334,296,336,285,295,282,283,276];
const SKIN_REF = [50,280,205,425,117,346,6];

/* ---------- utilitas geometri & gambar ---------- */
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const pick = (pts, idx) => idx.map(i => pts[i]);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const centroid = p => [p.reduce((t, q) => t + q[0], 0) / p.length, p.reduce((t, q) => t + q[1], 0) / p.length];
const grow = (p, f) => { const c = centroid(p); return p.map(([x, y]) => [c[0] + (x - c[0]) * f, c[1] + (y - c[1]) * f]); };

// isi poligon (even-odd) pada pusat piksel -> Float32Array 0/1
function fillPoly(poly, w, h) {
  const m = new Float32Array(w * h);
  const ys = poly.map(p => p[1]);
  const yMin = Math.max(0, Math.floor(Math.min(...ys))), yMax = Math.min(h - 1, Math.ceil(Math.max(...ys)));
  for (let y = yMin; y <= yMax; y++) {
    const yc = y + 0.5, xs = [];
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [x1, y1] = poly[j], [x2, y2] = poly[i];
      if ((y1 <= yc && y2 > yc) || (y2 <= yc && y1 > yc)) xs.push(x1 + (yc - y1) / (y2 - y1) * (x2 - x1));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      for (let x = Math.max(0, Math.round(xs[k])); x < Math.min(w, Math.round(xs[k + 1])); x++) m[y * w + x] = 1;
    }
  }
  return m;
}

function boxPass(src, dst, len, r, step, lines, lstep) {
  for (let l = 0; l < lines; l++) {
    const o = l * lstep;
    let s = 0;
    for (let i = -r; i <= r; i++) s += src[o + clamp(i, 0, len - 1) * step];
    for (let i = 0; i < len; i++) {
      dst[o + i * step] = s / (2 * r + 1);
      s += src[o + Math.min(len - 1, i + r + 1) * step] - src[o + Math.max(0, i - r) * step];
    }
  }
}
function blur(a, w, h, r) { // box blur 2x ~ gaussian, in-place
  r = Math.round(r);
  if (r < 1) return a;
  const t = new Float32Array(a.length);
  for (let k = 0; k < 2; k++) { boxPass(a, t, w, r, 1, h, w); boxPass(t, a, h, r, w, w, 1); }
  return a;
}


// jarak ke tepi mask (dalam piksel*3, chamfer 3-4) untuk piksel di dalam mask
function distIn(mask, w, h) {
  const d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = mask[i] ? 1e9 : 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!d[i]) continue;
    let v = d[i];
    if (x > 0) v = Math.min(v, d[i - 1] + 3);
    if (y > 0) { v = Math.min(v, d[i - w] + 3); if (x > 0) v = Math.min(v, d[i - w - 1] + 4); if (x < w - 1) v = Math.min(v, d[i - w + 1] + 4); }
    d[i] = v;
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x;
    if (!d[i]) continue;
    let v = d[i];
    if (x < w - 1) v = Math.min(v, d[i + 1] + 3);
    if (y < h - 1) { v = Math.min(v, d[i + w] + 3); if (x < w - 1) v = Math.min(v, d[i + w + 1] + 4); if (x > 0) v = Math.min(v, d[i + w - 1] + 4); }
    d[i] = v;
  }
  return d;
}

// pertajam ringan (unsharp mask) untuk mengembalikan detail saat sumber diperbesar
function sharpen(L, w, h, r, amt) {
  if (amt <= 0.02) return;
  const N = w * h, a = new Float32Array(N);
  for (let k = 0; k < 3; k++) {
    for (let i = 0; i < N; i++) a[i] = L[i * 4 + k];
    const b = blur(a.slice(), w, h, r);
    for (let i = 0; i < N; i++) if (L[i * 4 + 3] > 250) L[i * 4 + k] = a[i] + amt * (a[i] - b[i]);
  }
}

// sudut kemiringan wajah: 0 = tegak, 180 = terbalik (dari vektor mulut -> mata)
function rollOf(P) {
  const m = (a, b) => [(P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2];
  const e = m(33, 263), o = m(61, 291);
  return Math.atan2(e[0] - o[0], o[1] - e[1]) * 180 / Math.PI;
}

/* ---------- triangulasi Delaunay (Bowyer-Watson) ---------- */
function inCircum(P, t, px, py) {
  const [a, b, c] = t.map(i => P[i]);
  const ax = a[0] - px, ay = a[1] - py, bx = b[0] - px, by = b[1] - py, cx = c[0] - px, cy = c[1] - py;
  const det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay) + (cx * cx + cy * cy) * (ax * by - bx * ay);
  const orient = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  return orient > 0 ? det > 0 : det < 0;
}
function delaunay(pts) {
  const n = pts.length;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const d = Math.max(x1 - x0, y1 - y0) * 20, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  const P = pts.concat([[mx - d, my - d], [mx, my + d], [mx + d, my - d]]);
  let tris = [[n, n + 1, n + 2]];
  for (let i = 0; i < n; i++) {
    const bad = [], keep = [], edges = new Map();
    for (const t of tris) (inCircum(P, t, P[i][0], P[i][1]) ? bad : keep).push(t);
    for (const t of bad) for (const [a, b] of [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]]) {
      const k = a < b ? a + "_" + b : b + "_" + a, e = edges.get(k);
      if (e) e.c++; else edges.set(k, { a, b, c: 1 });
    }
    for (const e of edges.values()) if (e.c === 1) keep.push([e.a, e.b, i]);
    tris = keep;
  }
  return tris.filter(t => t[0] < n && t[1] < n && t[2] < n);
}

/* ---------- warp: mesh tujuan dipetakan balik ke sumber (bilinear, tanpa celah) ---------- */
function warp(S, sp, tp, tris, w, h) {
  const L = new Uint8ClampedArray(w * h * 4), sd = S.data, sw = S.width, sh = S.height;
  for (const [ia, ib, ic] of tris) {
    const A = tp[ia], B = tp[ib], C = tp[ic];
    const den = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]);
    if (Math.abs(den) < 1e-9) continue;
    const xa = Math.max(0, Math.floor(Math.min(A[0], B[0], C[0]))), xb = Math.min(w - 1, Math.ceil(Math.max(A[0], B[0], C[0])));
    const ya = Math.max(0, Math.floor(Math.min(A[1], B[1], C[1]))), yb = Math.min(h - 1, Math.ceil(Math.max(A[1], B[1], C[1])));
    const SA = sp[ia], SB = sp[ib], SC = sp[ic];
    for (let y = ya; y <= yb; y++) {
      for (let x = xa; x <= xb; x++) {
        const px = x + 0.5, py = y + 0.5;
        const l1 = ((B[1] - C[1]) * (px - C[0]) + (C[0] - B[0]) * (py - C[1])) / den;
        const l2 = ((C[1] - A[1]) * (px - C[0]) + (A[0] - C[0]) * (py - C[1])) / den;
        const l3 = 1 - l1 - l2;
        if (l1 < -0.002 || l2 < -0.002 || l3 < -0.002) continue;
        const sx = clamp(l1 * SA[0] + l2 * SB[0] + l3 * SC[0] - 0.5, 0, sw - 1.001);
        const sy = clamp(l1 * SA[1] + l2 * SB[1] + l3 * SC[1] - 0.5, 0, sh - 1.001);
        const ix = sx | 0, iy = sy | 0, fx = sx - ix, fy = sy - iy;
        const i00 = (iy * sw + ix) * 4, i10 = i00 + 4, i01 = i00 + sw * 4, i11 = i01 + 4;
        const o = (y * w + x) * 4;
        for (let k = 0; k < 3; k++) {
          L[o + k] = (sd[i00 + k] * (1 - fx) + sd[i10 + k] * fx) * (1 - fy) + (sd[i01 + k] * (1 - fx) + sd[i11 + k] * fx) * fy;
        }
        L[o + 3] = 255;
      }
    }
  }
  return L;
}

/* ---------- koreksi pencahayaan lokal (hanya dari area kulit) ---------- */
function matchColor(L, T, skin, w, h, faceW) {
  const cell = Math.max(4, Math.round(faceW / 24));
  const gw = Math.ceil(w / cell), gh = Math.ceil(h / cell);
  const sl = new Float32Array(gw * gh * 3), sb = new Float32Array(gw * gh * 3), cn = new Float32Array(gw * gh);
  const tl = [0, 0, 0], tb = [0, 0, 0];
  let total = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x, i = p * 4;
    if (!skin[p] || L[i + 3] < 250) continue;
    const g = ((y / cell) | 0) * gw + ((x / cell) | 0);
    for (let k = 0; k < 3; k++) { sl[g * 3 + k] += L[i + k]; sb[g * 3 + k] += T[i + k]; tl[k] += L[i + k]; tb[k] += T[i + k]; }
    cn[g]++; total++;
  }
  if (!total) return;
  const lim = v => clamp(v, 0.6, 1.6);
  const gg = tl.map((v, k) => lim(tb[k] / (v || 1)));
  const R = 3, G = new Float32Array(gw * gh * 3);
  for (let gy = 0; gy < gh; gy++) for (let gx = 0; gx < gw; gx++) {
    const a = [0, 0, 0], b = [0, 0, 0];
    let c = 0;
    for (let yy = Math.max(0, gy - R); yy <= Math.min(gh - 1, gy + R); yy++) for (let xx = Math.max(0, gx - R); xx <= Math.min(gw - 1, gx + R); xx++) {
      const j = yy * gw + xx;
      if (!cn[j]) continue;
      for (let k = 0; k < 3; k++) { a[k] += sl[j * 3 + k]; b[k] += sb[j * 3 + k]; }
      c += cn[j];
    }
    for (let k = 0; k < 3; k++) G[(gy * gw + gx) * 3 + k] = c ? lim(b[k] / (a[k] || 1)) : gg[k];
  }
  for (let y = 0; y < h; y++) {
    const fy = y / cell - 0.5, y0 = Math.max(0, Math.floor(fy)), y1 = Math.min(gh - 1, y0 + 1), ty = clamp(fy - Math.floor(fy), 0, 1);
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      if (L[i + 3] === 0) continue;
      const fx = x / cell - 0.5, x0 = Math.max(0, Math.floor(fx)), x1 = Math.min(gw - 1, x0 + 1), tx = clamp(fx - Math.floor(fx), 0, 1);
      for (let k = 0; k < 3; k++) {
        const top = G[(y0 * gw + x0) * 3 + k] * (1 - tx) + G[(y0 * gw + x1) * 3 + k] * tx;
        const bot = G[(y1 * gw + x0) * 3 + k] * (1 - tx) + G[(y1 * gw + x1) * 3 + k] * tx;
        L[i + k] = Math.min(255, L[i + k] * (top * (1 - ty) + bot * ty));
      }
    }
  }
}

/* ---------- Poisson blending: offset harmonik di grid kasar, lalu campur lembut ---------- */
function poissonBlend(L, T, D, hard, alpha, w, h, s, good) {
  const lw = Math.ceil(w / s) + 2, lh = Math.ceil(h / s) + 2;
  const val = [0, 1, 2].map(() => new Float32Array(lw * lh)), inside = new Uint8Array(lw * lh);
  for (let j = 0; j < lh; j++) for (let i = 0; i < lw; i++) {
    const d = [0, 0, 0];
    let n = 0, m = 0, tot = 0;
    for (let yy = (j - 1) * s; yy < j * s; yy++) for (let xx = (i - 1) * s; xx < i * s; xx++) {
      const p = clamp(yy, 0, h - 1) * w + clamp(xx, 0, w - 1);
      tot++;
      if (hard[p] > 0.5) m++;
      if (L[p * 4 + 3] > 250 && good[p]) { for (let k = 0; k < 3; k++) d[k] += T[p * 4 + k] - L[p * 4 + k]; n++; }
    }
    const c = j * lw + i, edge = i === 0 || j === 0 || i === lw - 1 || j === lh - 1;
    inside[c] = (!edge && m * 2 > tot) ? 1 : 0;
    for (let k = 0; k < 3; k++) val[k][c] = n ? d[k] / n : 0;
  }
  const list = [], mean = [0, 0, 0];
  let mn = 0;
  for (let c = 0; c < lw * lh; c++) {
    if (!inside[c]) continue;
    list.push(c);
    for (const q of [c - 1, c + 1, c - lw, c + lw]) if (!inside[q]) { for (let k = 0; k < 3; k++) mean[k] += val[k][q]; mn++; }
  }
  if (!list.length) return;
  for (let k = 0; k < 3; k++) for (const c of list) val[k][c] = mn ? mean[k] / mn : 0;
  const omega = Math.min(1.95, 2 / (1 + Math.sin(Math.PI / Math.max(lw, lh))));
  for (let it = 0; it < 400; it++) for (let k = 0; k < 3; k++) {
    const v = val[k];
    for (let n = 0; n < list.length; n++) { const c = list[n]; v[c] += omega * ((v[c - 1] + v[c + 1] + v[c - lw] + v[c + lw]) / 4 - v[c]); }
  }
  const LIM = 70;
  for (let y = 0; y < h; y++) {
    const fy = y / s + 0.5, j0 = clamp(Math.floor(fy), 0, lh - 2), ty = clamp(fy - j0, 0, 1);
    for (let x = 0; x < w; x++) {
      const p = y * w + x, a = alpha[p];
      if (a < 0.01 || L[p * 4 + 3] < 250) continue;
      const fx = x / s + 0.5, i0 = clamp(Math.floor(fx), 0, lw - 2), tx = clamp(fx - i0, 0, 1);
      for (let k = 0; k < 3; k++) {
        const v = val[k];
        const top = v[j0 * lw + i0] * (1 - tx) + v[j0 * lw + i0 + 1] * tx;
        const bot = v[(j0 + 1) * lw + i0] * (1 - tx) + v[(j0 + 1) * lw + i0 + 1] * tx;
        const f = clamp(L[p * 4 + k] + clamp(top * (1 - ty) + bot * ty, -LIM, LIM), 0, 255);
        D[p * 4 + k] = T[p * 4 + k] * (1 - a) + f * a;
      }
    }
  }
}

/* ---------- inti: S & T = {data,width,height}; sp & tp = 468 titik piksel ---------- */
function blendFace(S, sp, T, tp) {
  const w = T.width, h = T.height, N = w * h, td = T.data;
  const xs = tp.map(p => p[0]), ys = tp.map(p => p[1]);
  const faceW = Math.max(...xs) - Math.min(...xs), faceH = Math.max(...ys) - Math.min(...ys);

  // 1) warp wajah sumber ke bentuk wajah tujuan
  const L = warp(S, sp, tp, delaunay(tp), w, h);
  const sxs = sp.map(p => p[0]), srcW = Math.max(...sxs) - Math.min(...sxs);
  sharpen(L, w, h, Math.max(1, Math.round(faceW * 0.008)), clamp((faceW / srcW - 1) * 0.8, 0, 0.7));

  // 2) mask: oval wajah (dahi dikecilkan halus), lalu memudar KE DALAM kontur wajah tujuan.
  //    Kontur asli (pipi, rahang, garis rambut) tetap utuh dan tajam; tidak ada bocoran ke latar.
  const oval = pick(tp, OVAL), c = centroid(oval);
  const base = fillPoly(oval.map(([x, y]) => {
    const t = clamp((c[1] - y) / (0.5 * faceH), 0, 1);
    return [c[0] + (x - c[0]) * (1 - 0.03 * t), c[1] + (y - c[1]) * (1 - 0.10 * t * t)];
  }), w, h);
  const dmap = distIn(base, w, h);
  const inset = Math.max(4, faceW * 0.022), feather = Math.max(4, faceW * 0.04);
  const hard = new Float32Array(N), alpha = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const d = dmap[i] / 3;
    if (d > inset) { hard[i] = 1; const t = clamp((d - inset) / feather, 0, 1); alpha[i] = t * t * (3 - 2 * t); }
  }

  // 3) pertahankan mata/mulut tujuan bila ekspresi sumber jauh lebih tertutup
  const open = (P, u, l, a, b) => dist(P[u], P[l]) / dist(P[a], P[b]);
  const hole = new Float32Array(N);
  const addHole = poly => { const m = fillPoly(poly, w, h); for (let i = 0; i < N; i++) if (m[i]) hole[i] = 1; };
  const mT = open(tp, 13, 14, 78, 308), mS = open(sp, 13, 14, 78, 308);
  if (mT > 0.06 && mS < 0.6 * mT) addHole(grow(pick(tp, LIPS_IN), 0.88));
  if (open(tp, 159, 145, 33, 133) > 0.12 && open(sp, 159, 145, 33, 133) < 0.6 * open(tp, 159, 145, 33, 133)) addHole(grow(pick(tp, EYE_R), 1.2));
  if (open(tp, 386, 374, 362, 263) > 0.12 && open(sp, 386, 374, 362, 263) < 0.6 * open(tp, 386, 374, 362, 263)) addHole(grow(pick(tp, EYE_L), 1.2));
  blur(hole, w, h, faceW * 0.012);

  // 4) rambut/poni di dahi tujuan: deteksi lewat chroma kulit, area itu tidak ditimpa
  const yc = (r, g, b) => [0.299 * r + 0.587 * g + 0.114 * b, 128 - 0.1687 * r - 0.3313 * g + 0.5 * b, 128 + 0.5 * r - 0.4187 * g - 0.0813 * b];
  const ref = [[], [], []];
  for (const i of SKIN_REF) {
    const [px, py] = [Math.round(tp[i][0]), Math.round(tp[i][1])], acc = [0, 0, 0];
    let n = 0;
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
      const x = clamp(px + dx, 0, w - 1), y = clamp(py + dy, 0, h - 1), o = (y * w + x) * 4;
      const v = yc(td[o], td[o + 1], td[o + 2]);
      for (let k = 0; k < 3; k++) acc[k] += v[k];
      n++;
    }
    for (let k = 0; k < 3; k++) ref[k].push(acc[k] / n);
  }
  const med = a => a.sort((p, q) => p - q)[a.length >> 1];
  const rY = med(ref[0]), rCb = med(ref[1]), rCr = med(ref[2]);
  const browTop = Math.min(...pick(tp, BROW_R.concat(BROW_L)).map(p => p[1])) - 0.03 * faceH;
  const occ = new Float32Array(N);
  for (let y = 0; y < Math.min(h, Math.ceil(browTop)); y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if (hard[p] < 0.5) continue;
    const [Y, Cb, Cr] = yc(td[p * 4], td[p * 4 + 1], td[p * 4 + 2]);
    if (Math.abs(Cb - rCb) + Math.abs(Cr - rCr) > 30 || Y < 0.5 * rY) occ[p] = 1;
  }
  blur(occ, w, h, faceW * 0.015);
  for (let i = 0; i < N; i++) alpha[i] *= (1 - hole[i]) * (1 - Math.min(1, occ[i] * 1.6));

  // 5) statistik warna hanya dari kulit (tanpa mata, alis, bibir)
  const feat = new Float32Array(N);
  for (const [P, f] of [[EYE_R, 1.4], [EYE_L, 1.4], [BROW_R, 1.3], [BROW_L, 1.3], [LIPS_OUT, 1.2]]) {
    const m = fillPoly(grow(pick(tp, P), f), w, h);
    for (let i = 0; i < N; i++) if (m[i]) feat[i] = 1;
  }
  const skin = new Uint8Array(N);
  for (let i = 0; i < N; i++) skin[i] = (hard[i] > 0.5 && !feat[i] && occ[i] < 0.2) ? 1 : 0;
  matchColor(L, td, skin, w, h, faceW);

  // 6) Poisson + campur lembut
  const D = new Uint8ClampedArray(td);
  const good = new Uint8Array(N);
  for (let i = 0; i < N; i++) good[i] = (occ[i] < 0.2 && hole[i] < 0.5) ? 1 : 0;
  poissonBlend(L, td, D, hard, alpha, w, h, faceW < 220 ? 2 : 4, good);
  return D;
}

if (typeof module !== "undefined") module.exports = { rollOf, distIn, blendFace, delaunay, fillPoly, blur, OVAL, LIPS_IN, EYE_R, EYE_L, BROW_R, BROW_L, LIPS_OUT, SKIN_REF };

/* =====================  UI & deteksi (browser)  ===================== */
if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const state = { src: null, dst: null }; // {c: canvas, det: undefined | null | {c, pts, rot}}
  let landmarker = null;
  const setStatus = t => { $("status").textContent = t; };
  const refresh = () => {
    $("btn-swap").disabled = !(landmarker && state.src && state.dst);
    $("btn-flip").disabled = !(state.src && state.dst);
  };

  async function initModel() {
    try {
      setStatus("Memuat model deteksi wajah (sekitar 4 MB)...");
      const { FaceLandmarker, FilesetResolver } = await import(MP + "/vision_bundle.mjs");
      const fileset = await FilesetResolver.forVisionTasks(MP + "/wasm");
      landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL }, runningMode: "IMAGE", numFaces: 5,
        minFaceDetectionConfidence: 0.3, minFacePresenceConfidence: 0.3
      });
      setStatus("Model siap. Pilih dua foto.");
      refresh();
    } catch (e) {
      console.error(e);
      setStatus("Gagal memuat model. Pastikan terhubung ke internet lalu muat ulang halaman.");
    }
  }

  const loadImage = file => new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = URL.createObjectURL(file);
  });

  async function handleFile(file, key) {
    if (!file) return;
    const img = await loadImage(file);
    const s = Math.min(1, MAX / Math.max(img.naturalWidth, img.naturalHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(img.naturalWidth * s);
    c.height = Math.round(img.naturalHeight * s);
    const g = c.getContext("2d");
    g.imageSmoothingQuality = "high";
    g.drawImage(img, 0, 0, c.width, c.height);
    state[key] = { c, det: undefined };
    const prev = $("prev-" + key);
    prev.src = c.toDataURL("image/jpeg", 0.85);
    prev.hidden = false;
    refresh();
  }
  $("file-src").addEventListener("change", e => handleFile(e.target.files[0], "src"));
  $("file-dst").addEventListener("change", e => handleFile(e.target.files[0], "dst"));
  $("btn-flip").addEventListener("click", () => {
    [state.src, state.dst] = [state.dst, state.src];
    const a = $("prev-src"), b = $("prev-dst");
    [a.src, b.src] = [b.src, a.src];
    setStatus("Peran foto ditukar.");
  });

  /* ---------- deteksi teliti: semua orientasi + penajaman ROI + pemindaian petak ---------- */
  function rotCanvas(c, deg) { // putar searah jarum jam (kelipatan 90)
    deg = ((deg % 360) + 360) % 360;
    if (!deg) return c;
    const r = document.createElement("canvas"), sw = deg === 90 || deg === 270;
    r.width = sw ? c.height : c.width;
    r.height = sw ? c.width : c.height;
    const g = r.getContext("2d");
    g.translate(r.width / 2, r.height / 2);
    g.rotate(deg * Math.PI / 180);
    g.drawImage(c, -c.width / 2, -c.height / 2);
    return r;
  }
  const area = P => { const x = P.map(p => p[0]), y = P.map(p => p[1]); return (Math.max(...x) - Math.min(...x)) * (Math.max(...y) - Math.min(...y)); };
  const bboxC = P => { const x = P.map(p => p[0]), y = P.map(p => p[1]); return [(Math.max(...x) + Math.min(...x)) / 2, (Math.max(...y) + Math.min(...y)) / 2]; };
  const detect = c => landmarker.detect(c).faceLandmarks.map(f => f.slice(0, 468).map(p => [p.x * c.width, p.y * c.height]));
  const bestOf = faces => faces.length ? faces.slice().sort((a, b) => area(b) - area(a))[0] : null;

  // deteksi pada potongan persegi (x,y,side) yang diskalakan ke `size` px; hasil dipetakan balik
  function cropDetect(c, x, y, side, size) {
    const k = document.createElement("canvas");
    k.width = k.height = size;
    const g = k.getContext("2d"), z = size / side;
    g.imageSmoothingQuality = "high";
    const sx = Math.max(0, x), sy = Math.max(0, y), ex = Math.min(c.width, x + side), ey = Math.min(c.height, y + side);
    if (ex <= sx || ey <= sy) return [];
    g.drawImage(c, sx, sy, ex - sx, ey - sy, (sx - x) * z, (sy - y) * z, (ex - sx) * z, (ey - sy) * z);
    return detect(k).map(P => P.map(([px, py]) => [px / z + x, py / z + y]));
  }

  function detectTiles(c) {
    const side = Math.min(c.width, c.height) * 0.7, out = [];
    const pos = len => [0, (len - side) / 2, len - side];
    for (const y of pos(c.height)) for (const x of pos(c.width)) out.push(...cropDetect(c, x, y, side, 768));
    return out;
  }

  // deteksi ulang pada potongan ketat di sekitar wajah -> landmark lebih presisi
  function refine(c, P) {
    const [cx, cy] = bboxC(P), xs = P.map(p => p[0]), ys = P.map(p => p[1]);
    const side = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 1.7;
    let best = null, bd = Infinity;
    for (const Q of cropDetect(c, cx - side / 2, cy - side / 2, side, 640)) {
      const [qx, qy] = bboxC(Q), d = Math.hypot(qx - cx, qy - cy);
      if (d < bd) { bd = d; best = Q; }
    }
    if (best) {
      const r = area(best) / area(P);
      if (bd < side * 0.15 && r > 0.7 && r < 1.4 && Math.abs(rollOf(best) - rollOf(P)) < 20) return best;
    }
    return P;
  }

  // cari wajah di orientasi mana pun (0/90/180/270); pilih yang paling tegak
  function findFace(c) {
    for (const pass of [detect, detectTiles]) {
      const cand = [];
      for (const rot of [0, 180, 90, 270]) {
        const rc = rotCanvas(c, rot), P = bestOf(pass(rc));
        if (!P) continue;
        const roll = Math.abs(rollOf(P));
        cand.push({ c: rc, pts: P, rot, roll });
        if (roll <= 30) break;
      }
      if (cand.length) {
        cand.sort((a, b) => a.roll - b.roll);
        const best = cand[0];
        best.pts = refine(best.c, best.pts);
        return best;
      }
    }
    return null;
  }

  const ensureDet = o => { if (o.det === undefined) o.det = findFace(o.c); return o.det; };
  const rotNote = (d, nama) => d.rot ? ` ${nama} terdeteksi terbalik/miring (${d.rot}°) dan dikoreksi otomatis.` : "";

  function runSwap() {
    const sd = ensureDet(state.src);
    if (!sd) throw new Error("Wajah tidak terdeteksi di Foto Sumber (semua orientasi sudah dicoba). Gunakan foto dengan wajah yang terlihat jelas.");
    const td = ensureDet(state.dst);
    if (!td) throw new Error("Wajah tidak terdeteksi di Foto Tujuan (semua orientasi sudah dicoba). Gunakan foto dengan wajah yang terlihat jelas.");

    // perkecil sumber bila wajahnya jauh lebih besar dari wajah tujuan (hindari aliasing)
    const wid = P => Math.max(...P.map(p => p[0])) - Math.min(...P.map(p => p[0]));
    let srcC = sd.c, sp = sd.pts;
    const ratio = wid(td.pts) * 1.15 / wid(sd.pts);
    if (ratio < 0.8) {
      srcC = document.createElement("canvas");
      srcC.width = Math.max(1, Math.round(sd.c.width * ratio));
      srcC.height = Math.max(1, Math.round(sd.c.height * ratio));
      const g = srcC.getContext("2d");
      g.imageSmoothingQuality = "high";
      g.drawImage(sd.c, 0, 0, srcC.width, srcC.height);
      sp = sd.pts.map(p => [p[0] * srcC.width / sd.c.width, p[1] * srcC.height / sd.c.height]);
    }
    const S = srcC.getContext("2d").getImageData(0, 0, srcC.width, srcC.height);

    // proses hanya area wajah tujuan (+ margin) pada orientasi tegak
    const dstC = td.c, dp = td.pts, W = dstC.width, H = dstC.height;
    const xs = dp.map(p => p[0]), ys = dp.map(p => p[1]);
    const m = 0.2 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    const x0 = Math.max(0, Math.floor(Math.min(...xs) - m)), y0 = Math.max(0, Math.floor(Math.min(...ys) - m));
    const x1 = Math.min(W, Math.ceil(Math.max(...xs) + m)), y1 = Math.min(H, Math.ceil(Math.max(...ys) + m));
    const T = dstC.getContext("2d").getImageData(x0, y0, x1 - x0, y1 - y0);
    const D = blendFace(S, sp, T, dp.map(p => [p[0] - x0, p[1] - y0]));

    const res = document.createElement("canvas");
    res.width = W; res.height = H;
    const rg = res.getContext("2d");
    rg.drawImage(dstC, 0, 0);
    rg.putImageData(new ImageData(D, T.width, T.height), x0, y0);

    // kembalikan ke orientasi asli foto tujuan
    const fin = rotCanvas(res, (360 - td.rot) % 360);
    const out = $("out");
    out.width = fin.width; out.height = fin.height;
    out.getContext("2d").drawImage(fin, 0, 0);
    out.hidden = false;
    out.toBlob(b => {
      const a = $("btn-dl");
      if (a.href) URL.revokeObjectURL(a.href);
      a.href = URL.createObjectURL(b);
      a.hidden = false;
    }, "image/png");
    return "Selesai." + rotNote(sd, "Foto sumber") + rotNote(td, "Foto tujuan");
  }

  $("btn-swap").addEventListener("click", async () => {
    $("btn-swap").disabled = true;
    setStatus("Mendeteksi wajah dan memproses...");
    await new Promise(r => setTimeout(r, 30));
    try {
      setStatus(runSwap());
    } catch (e) {
      console.error(e);
      setStatus(e.message || "Terjadi kesalahan saat memproses. Coba foto lain.");
    } finally {
      refresh();
    }
  });

  initModel();
}
