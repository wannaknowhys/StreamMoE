// kv_cos.js - per-token per-layer KV cache cos between two prefill exports.
// Usage: node kv_cos.js <a.bin> <b.bin> [out.txt]
//
// 2026-09 fix over the previous version: correct f16 decode (incl. subnormal /
// inf / NaN payloads) and transparent NaN cells. A cell is NaN when either
// side's row has no finite data (zero-length, all-NaN or misaligned cache) -
// downstream consumers must treat NaN as "not comparable", never as cos ~ 0.
// Output shape is unchanged: one `cache=` header line, then one line per token
//   <token>\t<layer-k cos>...   (a NaN cell prints as NaN)
const fs = require('fs');
function parse(p) {
    const b = fs.readFileSync(p);
    const rdu = (o) => b.readUInt32LE(o);
    const rd64 = (o) => Number(b.readBigInt64LE(o));
    const rdu64 = (o) => Number(b.readBigUInt64LE(o));
    let off = 8;
    const magic = b.toString('ascii', 0, 8);
    if (magic !== 'PREFEXP2' && magic !== 'PREFEXP1') throw new Error('bad magic ' + magic);
    const nRows = rdu(off); off += 4; const nDim = rdu(off); off += 4; const dt = rdu(off); off += 4;
    off += nRows * (4 + nDim * (dt ? 2 : 4));
    const hRows = rdu(off); off += 4; const hDim = rdu(off); off += 4; const hdt = rdu(off); off += 4;
    off += hRows * (4 + hDim * (hdt ? 2 : 4));
    const nCache = rdu(off); off += 4;
    const caches = [];
    for (let ci = 0; ci < nCache; ci++) {
        const nl = rdu(off); off += 4;
        const name = b.toString('utf8', off, off + nl); off += nl;
        const nLayer = rdu(off); off += 4;
        const layers = [];
        for (let li = 0; li < nLayer; li++) {
            const il = rdu(off); off += 4;
            const t = {};
            for (const key of ['k', 'v']) {
                const ne = []; for (let i = 0; i < 4; i++) { ne.push(rd64(off)); off += 8; }
                const nb = []; for (let i = 0; i < 4; i++) { nb.push(rd64(off)); off += 8; }
                const ty = rdu(off); off += 4; const nn = rdu64(off); off += 8;
                t[key] = { ne, nb, ty, nn, d: b.subarray(off, off + nn) };
                off += nn;
            }
            layers.push({ il, k: t.k, v: t.v });
        }
        caches.push({ name, layers });
    }
    return { magic, nRows, caches };
}
// correct f16 decode: returns NaN for NaN payloads, +/-Infinity for inf
function f16(h) {
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0x1f) return m ? NaN : s * Infinity;
    if (e === 0) return s * Math.pow(2, -14) * m / 1024;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
}
// row for token t: cache storage is token-contiguous, row stride = nb[1].
// Returns {out: Float64Array, invalid: count of NaN/Inf entries}.
function row(tensor, t) {
    const esz = tensor.ty === 1 ? 2 : 4;
    const start = t * tensor.nb[1];
    const n = Math.max(0, Math.floor(tensor.nb[1] / esz));
    const out = new Float64Array(n);
    let invalid = 0;
    if (tensor.ty === 1) {
        for (let i = 0; i < n; i++) {
            const v = f16(tensor.d.readUInt16LE(start + i * 2));
            if (v !== v || v === Infinity || v === -Infinity) invalid++;
            out[i] = v;
        }
    } else {
        for (let i = 0; i < n; i++) {
            const v = tensor.d.readFloatLE(start + i * 4);
            if (v !== v || v === Infinity || v === -Infinity) invalid++;
            out[i] = v;
        }
    }
    return { out, invalid };
}
function cos(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-30);
}
const a = parse(process.argv[2]);
const b = parse(process.argv[3]);
console.log('rows A=%d B=%d caches A=%d B=%d', a.nRows, b.nRows, a.caches.length, b.caches.length);
const out = [];
const T = Math.min(a.nRows, b.nRows);
for (let ci = 0; ci < a.caches.length; ci++) {
    const ca = a.caches[ci];
    const cb = b.caches[ci];
    if (!cb) continue;
    out.push('cache=' + ca.name + ' layers=' + ca.layers.map(l => l.il).join(','));
    for (let t = 0; t < T; t++) {
        const rowv = [t];
        for (let li = 0; li < ca.layers.length; li++) {
            const la = ca.layers[li], lb = cb.layers[li];
            if (!la.k || !lb.k || la.k.ty !== lb.k.ty) { rowv.push('NaN'); continue; }
            const ra = row(la.k, t), rb = row(lb.k, t);
            if (ra.invalid || rb.invalid) { rowv.push('NaN'); continue; }
            rowv.push(cos(ra.out, rb.out).toFixed(4));
        }
        out.push(rowv.join('\t'));
    }
}
const txt = out.join('\n');
const outp = process.argv[4];
if (outp) fs.writeFileSync(outp, txt);
else console.log(txt);
