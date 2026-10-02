// Nova transformer engine, copied unchanged from the training site. Runs ONLY on the server.
/* ---------- Tensor helpers ---------- */
function T(rows, cols) { return { rows, cols, data: new Float32Array(rows * cols) }; }
function matmul(A, B) {
  const m = A.rows, k = A.cols, n = B.cols;
  const out = T(m, n);
  for (let i = 0; i < m; i++) {
    const ai = i * k, oi = i * n;
    for (let p = 0; p < k; p++) {
      const a = A.data[ai + p];
      if (a === 0) continue;
      const bp = p * n;
      for (let j = 0; j < n; j++) out.data[oi + j] += a * B.data[bp + j];
    }
  }
  return out;
}
function matmulT(A, B) {
  const m = A.rows, k = A.cols, n = B.rows;
  const out = T(m, n);
  for (let i = 0; i < m; i++) {
    const ao = i * k, oo = i * n;
    for (let j = 0; j < n; j++) {
      let s = 0; const bo = j * k;
      for (let p = 0; p < k; p++) s += A.data[ao + p] * B.data[bo + p];
      out.data[oo + j] = s;
    }
  }
  return out;
}
function transpose(A) {
  const out = T(A.cols, A.rows);
  for (let i = 0; i < A.rows; i++) for (let j = 0; j < A.cols; j++) out.data[j * A.rows + i] = A.data[i * A.cols + j];
  return out;
}
function addRowVec(A, v) { for (let i = 0; i < A.rows; i++) { const off = i * A.cols; for (let j = 0; j < A.cols; j++) A.data[off + j] += v[j]; } }
function addInto(dst, src) { for (let i = 0; i < dst.length; i++) dst[i] += src[i]; }
function sliceCols(A, start, len) { const out = T(A.rows, len); for (let i = 0; i < A.rows; i++) for (let j = 0; j < len; j++) out.data[i * len + j] = A.data[i * A.cols + start + j]; return out; }
function setCols(dst, start, src) { for (let i = 0; i < dst.rows; i++) for (let j = 0; j < src.cols; j++) dst.data[i * dst.cols + start + j] = src.data[i * src.cols + j]; }
function addCols(dst, start, src) { for (let i = 0; i < dst.rows; i++) for (let j = 0; j < src.cols; j++) dst.data[i * dst.cols + start + j] += src.data[i * src.cols + j]; }
function randn() { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

let PARAMS = [];
function newTensorParam(rows, cols, scale) {
  const p = T(rows, cols);
  for (let i = 0; i < p.data.length; i++) p.data[i] = randn() * scale;
  const grad = new Float32Array(p.data.length), m = new Float32Array(p.data.length), v = new Float32Array(p.data.length);
  PARAMS.push({ data: p.data, grad, m, v });
  p.grad = grad;
  return p;
}
function newVecParam(len, fillVal) {
  const data = new Float32Array(len);
  if (fillVal !== undefined) data.fill(fillVal);
  const grad = new Float32Array(len), m = new Float32Array(len), v = new Float32Array(len);
  PARAMS.push({ data, grad, m, v });
  return { data, grad };
}
function zeroGrads(paramList) { for (const p of paramList) p.grad.fill(0); }
function adamStep(paramList, lr, step, beta1, beta2, eps) {
  beta1 = beta1 ?? 0.9; beta2 = beta2 ?? 0.98; eps = eps ?? 1e-9;
  const b1t = 1 - Math.pow(beta1, step), b2t = 1 - Math.pow(beta2, step);
  for (const p of paramList) {
    const { data, grad, m, v } = p;
    for (let i = 0; i < data.length; i++) {
      const g = grad[i];
      m[i] = beta1 * m[i] + (1 - beta1) * g;
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;
      const mhat = m[i] / b1t, vhat = v[i] / b2t;
      data[i] -= lr * mhat / (Math.sqrt(vhat) + eps);
    }
  }
}

/* ---------- BPE Tokenizer ---------- */
class BPETokenizer {
  constructor() { this.specials = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']; this.vocab = {}; this.idToToken = []; this.merges = []; }
  _preTokenize(text) { return text.match(/[ \t]+|\n|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]/g) || []; }
  train(text, targetVocabSize, minFreq) {
    targetVocabSize = targetVocabSize || 1500; minFreq = minFreq || 2;
    const words = this._preTokenize(text);
    const wordFreq = new Map();
    for (const w of words) wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
    const seqs = new Map(); const baseChars = new Set();
    for (const w of wordFreq.keys()) { const arr = Array.from(w); seqs.set(w, arr); for (const c of arr) baseChars.add(c); }
    let vocabList = [...this.specials, ...Array.from(baseChars).sort()];
    const merges = [];
    const numMerges = Math.max(0, targetVocabSize - vocabList.length);
    for (let iter = 0; iter < numMerges; iter++) {
      const pairCounts = new Map();
      for (const [w, freq] of wordFreq) {
        const seq = seqs.get(w);
        for (let j = 0; j < seq.length - 1; j++) { const key = seq[j] + '\u0001' + seq[j + 1]; pairCounts.set(key, (pairCounts.get(key) || 0) + freq); }
      }
      let bestKey = null, bestCount = 0;
      for (const [k, c] of pairCounts) if (c > bestCount) { bestCount = c; bestKey = k; }
      if (!bestKey || bestCount < minFreq) break;
      const [a, b] = bestKey.split('\u0001');
      merges.push([a, b]);
      const merged = a + b;
      vocabList.push(merged);
      for (const [w, seq] of seqs) {
        const ns = []; let j = 0;
        while (j < seq.length) { if (j < seq.length - 1 && seq[j] === a && seq[j + 1] === b) { ns.push(merged); j += 2; } else { ns.push(seq[j]); j++; } }
        seqs.set(w, ns);
      }
    }
    this.merges = merges; this.idToToken = vocabList;
    this.vocab = {}; vocabList.forEach((t, i) => this.vocab[t] = i);
  }
  encode(text, addBos, addEos) {
    const ids = [];
    if (addBos) ids.push(this.vocab['<BOS>']);
    const words = this._preTokenize(text);
    for (const w of words) {
      let seq = Array.from(w);
      for (const [a, b] of this.merges) {
        const merged = a + b; const ns = []; let j = 0;
        while (j < seq.length) { if (j < seq.length - 1 && seq[j] === a && seq[j + 1] === b) { ns.push(merged); j += 2; } else { ns.push(seq[j]); j++; } }
        seq = ns;
      }
      for (const t of seq) ids.push(this.vocab[t] !== undefined ? this.vocab[t] : this.vocab['<UNK>']);
    }
    if (addEos) ids.push(this.vocab['<EOS>']);
    return ids;
  }
  decode(ids) { return ids.map(id => this.idToToken[id] || '').join(''); }
  get size() { return this.idToToken.length; }
  toJSON() { return { idToToken: this.idToToken, merges: this.merges, specials: this.specials }; }
  static fromJSON(obj) { const t = new BPETokenizer(); t.idToToken = obj.idToToken; t.merges = obj.merges; t.specials = obj.specials || t.specials; t.vocab = {}; t.idToToken.forEach((tok, i) => t.vocab[tok] = i); return t; }
}

/* ---------- LayerNorm / Linear / ReLU / Softmax ---------- */
function lnForward(X, gamma, beta, eps) {
  eps = eps || 1e-5;
  const Tr = X.rows, D = X.cols;
  const norm = T(Tr, D); const std = new Float32Array(Tr);
  for (let i = 0; i < Tr; i++) {
    let mean = 0; const off = i * D;
    for (let j = 0; j < D; j++) mean += X.data[off + j];
    mean /= D;
    let vr = 0; for (let j = 0; j < D; j++) { const d = X.data[off + j] - mean; vr += d * d; } vr /= D;
    const s = Math.sqrt(vr + eps); std[i] = s;
    for (let j = 0; j < D; j++) norm.data[off + j] = (X.data[off + j] - mean) / s;
  }
  const out = T(Tr, D);
  for (let i = 0; i < Tr; i++) { const off = i * D; for (let j = 0; j < D; j++) out.data[off + j] = norm.data[off + j] * gamma.data[j] + beta.data[j]; }
  return { out, cache: { norm, std, D } };
}
function lnBackward(dOut, cache, gamma) {
  const { norm, std, D } = cache; const Tr = dOut.rows;
  const dX = T(Tr, D); const dGamma = new Float32Array(D), dBeta = new Float32Array(D);
  for (let i = 0; i < Tr; i++) {
    const off = i * D; let meanDN = 0, meanDNnorm = 0; const dnorm = new Float32Array(D);
    for (let j = 0; j < D; j++) {
      const d = dOut.data[off + j];
      dGamma[j] += d * norm.data[off + j]; dBeta[j] += d;
      const dn = d * gamma.data[j]; dnorm[j] = dn; meanDN += dn; meanDNnorm += dn * norm.data[off + j];
    }
    meanDN /= D; meanDNnorm /= D; const s = std[i];
    for (let j = 0; j < D; j++) dX.data[off + j] = (dnorm[j] - meanDN - norm.data[off + j] * meanDNnorm) / s;
  }
  return { dX, dGamma, dBeta };
}
function linearForward(X, layer) { const Y = matmul(X, layer.W); addRowVec(Y, layer.b.data); return Y; }
function linearBackward(dY, X, layer) {
  const Xt = transpose(X); const gW = matmul(Xt, dY);
  addInto(layer.W.grad, gW.data);
  for (let i = 0; i < dY.rows; i++) { const off = i * dY.cols; for (let j = 0; j < dY.cols; j++) layer.b.grad[j] += dY.data[off + j]; }
  const Wt = transpose(layer.W);
  return matmul(dY, Wt);
}
function reluFwd(X) { const out = T(X.rows, X.cols); for (let i = 0; i < X.data.length; i++) out.data[i] = X.data[i] > 0 ? X.data[i] : 0; return out; }
function reluBwd(dOut, preAct) { const out = T(dOut.rows, dOut.cols); for (let i = 0; i < dOut.data.length; i++) out.data[i] = preAct.data[i] > 0 ? dOut.data[i] : 0; return out; }
function softmaxRow(row, out, off, len) {
  let mx = -Infinity; for (let j = 0; j < len; j++) mx = Math.max(mx, row[off + j]);
  let sum = 0; for (let j = 0; j < len; j++) { const e = Math.exp(row[off + j] - mx); out[off + j] = e; sum += e; }
  for (let j = 0; j < len; j++) out[off + j] /= sum;
}

/* ---------- Transformer model ---------- */
class NovaModel {
  constructor(vocabSize, cfg) {
    this.cfg = Object.assign({ dModel: 48, nHeads: 4, nLayers: 2, dFF: 192, ctxLen: 64 }, cfg);
    this.vocabSize = vocabSize;
    PARAMS = [];
    const d = this.cfg.dModel;
    this.wte = newTensorParam(vocabSize, d, 0.02);
    this.wpe = newTensorParam(this.cfg.ctxLen, d, 0.02);
    this.layers = [];
    for (let l = 0; l < this.cfg.nLayers; l++) {
      this.layers.push({
        ln1: { gamma: newVecParam(d, 1), beta: newVecParam(d, 0) },
        wq: { W: newTensorParam(d, d, Math.sqrt(1 / d)), b: newVecParam(d, 0) },
        wk: { W: newTensorParam(d, d, Math.sqrt(1 / d)), b: newVecParam(d, 0) },
        wv: { W: newTensorParam(d, d, Math.sqrt(1 / d)), b: newVecParam(d, 0) },
        wo: { W: newTensorParam(d, d, Math.sqrt(1 / d)), b: newVecParam(d, 0) },
        ln2: { gamma: newVecParam(d, 1), beta: newVecParam(d, 0) },
        w1: { W: newTensorParam(d, this.cfg.dFF, Math.sqrt(2 / d)), b: newVecParam(this.cfg.dFF, 0) },
        w2: { W: newTensorParam(this.cfg.dFF, d, Math.sqrt(2 / this.cfg.dFF)), b: newVecParam(d, 0) },
      });
    }
    this.lnf = { gamma: newVecParam(d, 1), beta: newVecParam(d, 0) };
    this.params = PARAMS;
    this.stepCount = 0; this.epochsDone = 0;
  }
  paramCount() { return this.params.reduce((s, p) => s + p.data.length, 0); }
  useParams() { PARAMS = this.params; }

  forward(ids, needCache) {
    const Tn = ids.length, d = this.cfg.dModel;
    const X0 = T(Tn, d);
    for (let t = 0; t < Tn; t++) {
      const tokOff = ids[t] * d, posOff = t * d, outOff = t * d;
      for (let k = 0; k < d; k++) X0.data[outOff + k] = this.wte.data[tokOff + k] + this.wpe.data[posOff + k];
    }
    let X = X0; const layerCaches = [];
    const H = this.cfg.nHeads, hd = d / H, scale = 1 / Math.sqrt(hd);
    for (const layer of this.layers) {
      const ln1 = lnForward(X, layer.ln1.gamma, layer.ln1.beta);
      const Xn = ln1.out;
      const Q = linearForward(Xn, layer.wq), K = linearForward(Xn, layer.wk), V = linearForward(Xn, layer.wv);
      const Ocat = T(Tn, d); const headCaches = [];
      for (let h = 0; h < H; h++) {
        const Qh = sliceCols(Q, h * hd, hd), Kh = sliceCols(K, h * hd, hd), Vh = sliceCols(V, h * hd, hd);
        const scores = matmulT(Qh, Kh);
        for (let i = 0; i < scores.data.length; i++) scores.data[i] *= scale;
        for (let i = 0; i < Tn; i++) for (let j = i + 1; j < Tn; j++) scores.data[i * Tn + j] = -1e9;
        const attnw = T(Tn, Tn);
        for (let i = 0; i < Tn; i++) softmaxRow(scores.data, attnw.data, i * Tn, Tn);
        const Oh = matmul(attnw, Vh);
        setCols(Ocat, h * hd, Oh);
        headCaches.push({ Qh, Kh, Vh, attnw });
      }
      const Y = linearForward(Ocat, layer.wo);
      const X2 = T(Tn, d); for (let i = 0; i < X2.data.length; i++) X2.data[i] = X.data[i] + Y.data[i];
      const ln2 = lnForward(X2, layer.ln2.gamma, layer.ln2.beta);
      const X2n = ln2.out;
      const FF1 = linearForward(X2n, layer.w1);
      const Hact = reluFwd(FF1);
      const FF2 = linearForward(Hact, layer.w2);
      const X3 = T(Tn, d); for (let i = 0; i < X3.data.length; i++) X3.data[i] = X2.data[i] + FF2.data[i];
      layerCaches.push({ X, Xn, ln1cache: ln1.cache, Q, K, V, headCaches, Ocat, Y, X2, ln2cache: ln2.cache, X2n, FF1, Hact, FF2 });
      X = X3;
    }
    const lnf = lnForward(X, this.lnf.gamma, this.lnf.beta);
    const Xf = lnf.out;
    const logits = matmulT(Xf, this.wte);
    if (!needCache) return { logits };
    return { logits, cache: { X0, layerCaches, Xf, lnfCache: lnf.cache, ids } };
  }

  backward(dLogits, cache) {
    const d = this.cfg.dModel, H = this.cfg.nHeads, hd = d / H, scale = 1 / Math.sqrt(hd);
    const Tn = cache.ids.length;
    const dXf = matmul(dLogits, this.wte);
    const dWteFromHead = matmul(transpose(dLogits), cache.Xf);
    addInto(this.wte.grad, dWteFromHead.data);
    const lnfB = lnBackward(dXf, cache.lnfCache, this.lnf.gamma);
    addInto(this.lnf.gamma.grad, lnfB.dGamma); addInto(this.lnf.beta.grad, lnfB.dBeta);
    let dX = lnfB.dX;
    for (let li = this.layers.length - 1; li >= 0; li--) {
      const layer = this.layers[li], c = cache.layerCaches[li];
      const dX3 = dX; const dX2_direct = dX3; const dFF2 = dX3;
      const dHact = linearBackward(dFF2, c.Hact, layer.w2);
      const dFF1 = reluBwd(dHact, c.FF1);
      const dX2n = linearBackward(dFF1, c.X2n, layer.w1);
      const ln2B = lnBackward(dX2n, c.ln2cache, layer.ln2.gamma);
      addInto(layer.ln2.gamma.grad, ln2B.dGamma); addInto(layer.ln2.beta.grad, ln2B.dBeta);
      const dX2 = T(Tn, d);
      for (let i = 0; i < dX2.data.length; i++) dX2.data[i] = dX2_direct.data[i] + ln2B.dX.data[i];
      const dX_direct = dX2; const dY = dX2;
      const dOcat = linearBackward(dY, c.Ocat, layer.wo);
      const dQ = T(Tn, d), dK = T(Tn, d), dV = T(Tn, d);
      for (let h = 0; h < H; h++) {
        const hc = c.headCaches[h];
        const dOh = sliceCols(dOcat, h * hd, hd);
        const dattnw = matmulT(dOh, hc.Vh);
        const dVh = matmul(transpose(hc.attnw), dOh);
        const dscores = T(Tn, Tn);
        for (let i = 0; i < Tn; i++) {
          let s = 0; const off = i * Tn;
          for (let j = 0; j < Tn; j++) s += dattnw.data[off + j] * hc.attnw.data[off + j];
          for (let j = 0; j < Tn; j++) dscores.data[off + j] = hc.attnw.data[off + j] * (dattnw.data[off + j] - s) * scale;
        }
        const dQh = matmul(dscores, hc.Kh);
        const dKh = matmul(transpose(dscores), hc.Qh);
        addCols(dQ, h * hd, dQh); addCols(dK, h * hd, dKh); addCols(dV, h * hd, dVh);
      }
      const dXn_q = linearBackward(dQ, c.Xn, layer.wq);
      const dXn_k = linearBackward(dK, c.Xn, layer.wk);
      const dXn_v = linearBackward(dV, c.Xn, layer.wv);
      const dXnTotal = T(Tn, d);
      for (let i = 0; i < dXnTotal.data.length; i++) dXnTotal.data[i] = dXn_q.data[i] + dXn_k.data[i] + dXn_v.data[i];
      const ln1B = lnBackward(dXnTotal, c.ln1cache, layer.ln1.gamma);
      addInto(layer.ln1.gamma.grad, ln1B.dGamma); addInto(layer.ln1.beta.grad, ln1B.dBeta);
      const dXtotal = T(Tn, d);
      for (let i = 0; i < dXtotal.data.length; i++) dXtotal.data[i] = dX_direct.data[i] + ln1B.dX.data[i];
      dX = dXtotal;
    }
    for (let t = 0; t < Tn; t++) {
      const tokOff = cache.ids[t] * d, posOff = t * d, srcOff = t * d;
      for (let k = 0; k < d; k++) { this.wte.grad[tokOff + k] += dX.data[srcOff + k]; this.wpe.grad[posOff + k] += dX.data[srcOff + k]; }
    }
  }

  trainStep(batchInputs, batchTargets, lr) {
    this.useParams(); zeroGrads(this.params);
    let totalLoss = 0, totalTokens = 0;
    for (let b = 0; b < batchInputs.length; b++) {
      const ids = batchInputs[b], targets = batchTargets[b];
      const { logits, cache } = this.forward(ids, true);
      const Tn = ids.length, V = this.vocabSize;
      const probs = new Float32Array(Tn * V);
      for (let i = 0; i < Tn; i++) softmaxRow(logits.data, probs, i * V, V);
      const dLogits = T(Tn, V);
      for (let i = 0; i < Tn; i++) {
        const off = i * V, tgt = targets[i];
        let p = probs[off + tgt]; if (p < 1e-9) p = 1e-9;
        totalLoss += -Math.log(p); totalTokens++;
        for (let j = 0; j < V; j++) dLogits.data[off + j] = probs[off + j];
        dLogits.data[off + tgt] -= 1;
      }
      for (let i = 0; i < dLogits.data.length; i++) dLogits.data[i] /= Tn;
      this.backward(dLogits, cache);
    }
    const bs = batchInputs.length;
    for (const p of this.params) for (let i = 0; i < p.grad.length; i++) p.grad[i] /= bs;
    this.stepCount++;
    adamStep(this.params, lr, this.stepCount);
    return totalLoss / totalTokens;
  }

  evalLoss(batchInputs, batchTargets) {
    this.useParams();
    let totalLoss = 0, totalTokens = 0;
    for (let b = 0; b < batchInputs.length; b++) {
      const ids = batchInputs[b], targets = batchTargets[b];
      const { logits } = this.forward(ids, false);
      const Tn = ids.length, V = this.vocabSize;
      for (let i = 0; i < Tn; i++) {
        const rowProbs = new Float32Array(V);
        softmaxRow(logits.data, rowProbs, i * V, V);
        let p = rowProbs[targets[i]]; if (p < 1e-9) p = 1e-9;
        totalLoss += -Math.log(p); totalTokens++;
      }
    }
    return totalTokens ? totalLoss / totalTokens : 0;
  }

  generate(promptIds, maxTokens, temperature, topK, topP, stopFlag) {
    this.useParams();
    let ids = promptIds.slice();
    const ctx = this.cfg.ctxLen;
    for (let step = 0; step < maxTokens; step++) {
      if (stopFlag && stopFlag.stop) break;
      const windowIds = ids.slice(Math.max(0, ids.length - ctx));
      const { logits } = this.forward(windowIds, false);
      const Tn = windowIds.length, V = this.vocabSize;
      const lastOff = (Tn - 1) * V;
      const scaled = new Float32Array(V);
      for (let j = 0; j < V; j++) scaled[j] = logits.data[lastOff + j] / Math.max(temperature, 1e-6);
      const probs = new Float32Array(V);
      softmaxRow(scaled, probs, 0, V);
      let idxs = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
      if (topK && topK > 0) idxs = idxs.slice(0, topK);
      if (topP && topP < 1) { let cum = 0; const kept = []; for (const i of idxs) { cum += probs[i]; kept.push(i); if (cum >= topP) break; } idxs = kept; }
      let sum = 0; for (const i of idxs) sum += probs[i];
      let r = Math.random() * sum, chosen = idxs[idxs.length - 1];
      for (const i of idxs) { r -= probs[i]; if (r <= 0) { chosen = i; break; } }
      ids.push(chosen);
      if (this.eosId !== undefined && chosen === this.eosId) break;
    }
    return ids;
  }

  serializeWeights() {
    const out = {};
    out.wte = Array.from(this.wte.data);
    out.wpe = Array.from(this.wpe.data);
    out.layers = this.layers.map(l => ({
      ln1: { gamma: Array.from(l.ln1.gamma.data), beta: Array.from(l.ln1.beta.data) },
      wq: { W: Array.from(l.wq.W.data), b: Array.from(l.wq.b.data) },
      wk: { W: Array.from(l.wk.W.data), b: Array.from(l.wk.b.data) },
      wv: { W: Array.from(l.wv.W.data), b: Array.from(l.wv.b.data) },
      wo: { W: Array.from(l.wo.W.data), b: Array.from(l.wo.b.data) },
      ln2: { gamma: Array.from(l.ln2.gamma.data), beta: Array.from(l.ln2.beta.data) },
      w1: { W: Array.from(l.w1.W.data), b: Array.from(l.w1.b.data) },
      w2: { W: Array.from(l.w2.W.data), b: Array.from(l.w2.b.data) },
    }));
    out.lnf = { gamma: Array.from(this.lnf.gamma.data), beta: Array.from(this.lnf.beta.data) };
    return out;
  }
  loadWeights(w) {
    this.wte.data.set(w.wte); this.wpe.data.set(w.wpe);
    w.layers.forEach((lw, i) => {
      const l = this.layers[i];
      l.ln1.gamma.data.set(lw.ln1.gamma); l.ln1.beta.data.set(lw.ln1.beta);
      l.wq.W.data.set(lw.wq.W); l.wq.b.data.set(lw.wq.b);
      l.wk.W.data.set(lw.wk.W); l.wk.b.data.set(lw.wk.b);
      l.wv.W.data.set(lw.wv.W); l.wv.b.data.set(lw.wv.b);
      l.wo.W.data.set(lw.wo.W); l.wo.b.data.set(lw.wo.b);
      l.ln2.gamma.data.set(lw.ln2.gamma); l.ln2.beta.data.set(lw.ln2.beta);
      l.w1.W.data.set(lw.w1.W); l.w1.b.data.set(lw.w1.b);
      l.w2.W.data.set(lw.w2.W); l.w2.b.data.set(lw.w2.b);
    });
    this.lnf.gamma.data.set(w.lnf.gamma); this.lnf.beta.data.set(w.lnf.beta);
  }
}


export { NovaModel, BPETokenizer };
