// verify_prefill.js - compare two prefill cross-validation exports.
// Each file was produced by running the engine with LLM_EXPORT_DIR=<dir>:
//   base: no --expert-backend  -> <dir>/prefill_export.bin
//   moe : --expert-backend     -> <dir>/prefill_export.bin
// Written once at context destruction. Format (PREFEXP1):
//   u32 n_embd_rows, u32 n_embd_dim
//   per row: u32 token_pos, float embd[n_embd_dim]          (LM head input, output tokens)
//   u32 n_hid_rows, u32 n_hid_dim
//   per row: u32 token_pos, float hidden[n_hid_dim]         (hidden state, all tokens)
//   u32 n_cache
//   per cache: u32 nl, name[nl], u32 n_layer
//     per layer: u32 il, u32 k_type, u64 k_nbytes, k bytes, u32 v_type, u64 v_nbytes, v bytes
// Reports per-token cosine / MAD / MSE for embd and hidden, the first diverging
// token+position, and per-(cache,layer) KV max byte diff.
// Usage: node tools/verify_prefill.js <std.bin> <moe.bin>
// PREFEXP2: dtype comes from prefill_meta.json (same dir) or the per-section
// GGML_TYPE marker (0=F32 1=F16); f16 rows are converted to float for compare.
const fs = require('fs');
const path = require('path');

function f16ToF32(h) {
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h & 0x7C00) >> 10;
    const f = h & 0x3FF;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1F) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

function parseFile(path) {
    const b = fs.readFileSync(path);
    let meta = null;
    try { meta = JSON.parse(fs.readFileSync(path.join(path.dirname(path), 'prefill_meta.json'), 'utf8')); } catch (e) {}
    const rdU32 = (o) => b.readUInt32LE(o);
    const rdU64 = (o) => b.readBigUInt64LE(o);
    const magic = b.toString('ascii', 0, 8);
    if (magic !== 'PREFEXP1' && magic !== 'PREFEXP2') throw new Error('bad magic');
    let off = 8;
    const metaSecs = meta && meta.files && meta.files['prefill_export_main.bin'] && meta.files['prefill_export_main.bin'].sections || [];
    const secEsz = (i, binDt) => {
        const md = metaSecs[i] && metaSecs[i].dtype;
        if (md === 'f16') return 2;
        if (md === 'f32') return 4;
        return binDt === 1 ? 2 : 4; // fallback: bin GGML_TYPE marker / default f32
    };
    function readRows(n, nDim, esz) {
        const rows = [];
        for (let i = 0; i < n; i++) {
            const pos = rdU32(off); off += 4;
            const data = new Float32Array(nDim);
            if (esz === 2) {
                for (let j = 0; j < nDim; j++) data[j] = f16ToF32(b.readUInt16LE(off + j * 2));
            } else {
                for (let j = 0; j < nDim; j++) data[j] = b.readFloatLE(off + j * 4);
            }
            off += nDim * esz;
            rows.push({ pos, data });
        }
        return rows;
    }
    const nRows = rdU32(off); off += 4;
    const nDim = rdU32(off); off += 4;
    let binDt = null;
    if (magic === 'PREFEXP2') { binDt = rdU32(off); off += 4; }
    const embd = readRows(nRows, nDim, secEsz(0, binDt));
    const nHRows = rdU32(off); off += 4;
    const nHDim = rdU32(off); off += 4;
    let binDtH = null;
    if (magic === 'PREFEXP2') { binDtH = rdU32(off); off += 4; }
    const hidden = readRows(nHRows, nHDim, secEsz(1, binDtH));
    const nCache = rdU32(off); off += 4;
    const caches = [];
    for (let ci = 0; ci < nCache; ci++) {
        const nl = rdU32(off); off += 4;
        const name = b.toString('utf8', off, off + nl); off += nl;
        const nLayer = rdU32(off); off += 4;
        const layers = [];
        for (let li = 0; li < nLayer; li++) {
            const il = rdU32(off); off += 4;
            const rdTensor = () => {
                off += 64; // 4x ne + 4x nb (int64 each): layout for per-token slicing
                const ty = rdU32(off); off += 4;
                const nn = rdU64(off); off += 8;
                const d = b.subarray(off, off + Number(nn)); off += Number(nn);
                return { ty, nn, d };
            };
            const tk = rdTensor();
            const tv = rdTensor();
            layers.push({ il, kType: tk.ty, kNb: tk.nn, k: tk.d, vType: tv.ty, vNb: tv.nn, v: tv.d });
        }
        caches.push({ name, layers });
    }
    return { embd, hidden, caches };
}

const A = parseFile(process.argv[2]);
const B = parseFile(process.argv[3]);
console.log(`rows: std embd=${A.embd.length} hidden=${A.hidden.length} | moe embd=${B.embd.length} hidden=${B.hidden.length}`);

const cos = (a, b) => {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return { c: dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-30), na, nb };
};
// Agree only when BOTH pass: cosine near 1 (direction) AND maxAbs small (magnitude).
// Cosine alone is scale-insensitive; a large uniform scale would still give cos=1.
const COS_TH = 1e-6;  // |1 - cos| gate
const ABS_TH = 1e-4;  // maxAbs gate (absolute)
function cmpRows(tag, ra, rb) {
    let first = null;
    for (let i = 0; i < Math.min(ra.length, rb.length); i++) {
        const a = ra[i], b = rb[i];
        if (a.pos !== b.pos) { console.log(`  ${tag} pos mismatch @${i}: std=${a.pos} moe=${b.pos}`); if (!first) first = { tag, idx: i }; continue; }
        let mad = 0, mse = 0, maxAbs = 0, maxIdx = -1;
        let dot = 0, na = 0, nb = 0;
        for (let j = 0; j < a.data.length; j++) {
            const d = Math.abs(a.data[j] - b.data[j]);
            mad += d; mse += d * d;
            if (d > maxAbs) { maxAbs = d; maxIdx = j; }
            dot += a.data[j] * b.data[j]; na += a.data[j] * a.data[j]; nb += b.data[j] * b.data[j];
        }
        mad /= a.data.length; mse /= a.data.length;
        const normA = Math.sqrt(na), normB = Math.sqrt(nb);
        // both all-zero vectors: cosine formula gives 0 (not 1), which would false-alarm
        let c = (normA < 1e-30 && normB < 1e-30) ? 1 : dot / (normA * normB + 1e-30);
        const cosBad = Math.abs(1 - c) > COS_TH;
        const absBad = maxAbs > ABS_TH;
        if (cosBad || absBad) {
            console.log(`  DIFF ${tag} @token#${a.pos} cos=${c.toFixed(10)} MAD=${mad.toExponential(3)} MSE=${mse.toExponential(3)} maxAbs=${maxAbs.toExponential(3)}@${maxIdx} (${cosBad ? 'cos' : ''}${cosBad && absBad ? '+' : ''}${absBad ? 'abs' : ''})`);
            if (!first) first = { tag, token: a.pos, cos: c, mad, mse, maxAbs, maxIdx, cosBad, absBad };
        }
    }
    return first;
}
const f1 = cmpRows('embd', A.embd, B.embd);
const f2 = cmpRows('hidden', A.hidden, B.hidden);
let kvFirst = null, kvDiff = 0;
for (let ci = 0; ci < Math.min(A.caches.length, B.caches.length); ci++) {
    const ca = A.caches[ci], cb = B.caches[ci];
    for (let li = 0; li < Math.min(ca.layers.length, cb.layers.length); li++) {
        const la = ca.layers[li], lb = cb.layers[li];
        for (const [tA, tB, dA, dB, tag] of [[la.kType, lb.kType, la.k, lb.k, 'k'], [la.vType, lb.vType, la.v, lb.v, 'v']]) {
            if (tA !== tB || dA.length !== dB.length) {
                console.log(`  KV DIFF ${ca.name} L${la.il} ${tag}: type/size std(${tA},${dA.length}) moe(${tB},${dB.length})`);
                kvDiff++; if (!kvFirst) kvFirst = `${ca.name} L${la.il} ${tag}`;
                continue;
            }
            let maxB = 0, maxI = -1, diffBytes = 0;
            for (let i = 0; i < dA.length; i++) {
                const d = Math.abs(dA[i] - dB[i]);
                if (d > maxB) { maxB = d; maxI = i; }
                if (d) diffBytes++;
            }
            if (diffBytes > 0) {
                console.log(`  KV DIFF ${ca.name} L${la.il} ${tag}: bytes=${diffBytes}/${dA.length} maxByte=${maxB}@${maxI}`);
                kvDiff++; if (!kvFirst) kvFirst = `${ca.name} L${la.il} ${tag}`;
            }
        }
    }
}
if (f1 || f2 || kvDiff) {
    console.log(`\nFIRST DIVERGENCE: embd=${f1 ? JSON.stringify(f1) : 'none'} hidden=${f2 ? JSON.stringify(f2) : 'none'} KV=${kvFirst || 'none'} (KV diffs=${kvDiff})`);
} else {
    console.log('\nRESULT: IDENTICAL (all aligned tokens + KV)');
}
