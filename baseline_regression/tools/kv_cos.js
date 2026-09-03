// kv_cos.js - per-token per-layer KV cache cos between two prefill exports.
// Usage: node kv_cos.js <a.bin> <b.bin> [out.txt]
const fs = require('fs');
function parse(p) {
    const b = fs.readFileSync(p);
    const rdu = (o) => b.readUInt32LE(o);
    const rd64 = (o) => Number(b.readBigInt64LE(o));
    const rdu64 = (o) => Number(b.readBigUInt64LE(o));
    let off = 8;
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
                t[key] = { ne, nb, ty, nn, data: b.subarray(off, off + nn) };
                off += nn;
            }
            layers.push({ il, k: t.k, v: t.v });
        }
        caches.push({ name, layers });
    }
    return { nRows, caches };
}
const esz = (ty) => (ty === 1 ? 2 : 4); // f32=0, f16=1
function cosVec(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-30);
}
function rowVec(data, ty, nb, t) {
    const es = esz(ty);
    const off = t * nb[1];
    const n = Math.max(0, Math.floor((nb[1]) / es));
    const out = new Float32Array(n);
    if (ty === 1) { for (let i = 0; i < n; i++) { const h = data.readUInt16LE(off + i * 2); out[i] = ((h & 0x8000) ? -1 : 1) * ((h & 0x7C00) === 0 ? Math.pow(2, -14) * (h & 0x3FF) / 1024 : (h & 0x7C00) === 0x7C00 ? (h & 0x3FF) ? NaN : Infinity : Math.pow(2, (h & 0x7C00) / 1024 - 15) * (1 + (h & 0x3FF) / 1024)); } }
    else { for (let i = 0; i < n; i++) out[i] = data.readFloatLE(off + i * 4); }
    return out;
}
const a = parse(process.argv[2]);
const b = parse(process.argv[3]);
console.log('rows A=%d B=%d caches A=%d B=%d', a.nRows, b.nRows, a.caches.length, b.caches.length);
const out = [];
for (let ci = 0; ci < a.caches.length; ci++) {
    const ca = a.caches[ci], cb = b.caches[ci];
    const nL = Math.min(ca.layers.length, cb.layers.length);
    // header: cache name + layer ids
    out.push(`cache=${ca.name} layers=` + ca.layers.map(l => l.il).join(','));
    for (let t = 0; t < a.nRows; t++) {
        const row = [t];
        for (let li = 0; li < nL; li++) {
            const la = ca.layers[li], lb = cb.layers[li];
            const va = rowVec(la.k.data, la.k.ty, la.k.nb, t);
            const vb = rowVec(lb.k.data, lb.k.ty, lb.k.nb, t);
            row.push(cosVec(va, vb).toFixed(4));
        }
        out.push(row.join('\t'));
    }
}
const txt = out.join('\n');
const outp = process.argv[4];
if (outp) fs.writeFileSync(outp, txt);
else console.log(txt);
