// div_match.js - match expert-flip tokens vs high-divergence tokens between a
// baseline (known-good) export dir and a new export dir. Diagnostic for a
// DIVERGED baseline regression: if every high-divergence token also has an
// expert flip, the divergence is the known gemma-4 expert-flip noise (route
// boundary flips from ulp differences); if any high-div token has NO flip,
// that divergence is unexplained and deserves investigation.
// Usage: node tools/div_match.js <base_dir> <div_dir>
//   each dir needs: prefill_export_main.bin + prefill_meta.json + expert_history_main.bin
const fs = require('fs');
const path = require('path');

function f16ToF32(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h & 0x7C00) >> 10, f = h & 0x3FF;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1F) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
}
function parsePrefill(dir) {
    const b = fs.readFileSync(path.join(dir, 'prefill_export_main.bin'));
    let meta = null;
    try { meta = JSON.parse(fs.readFileSync(path.join(dir, 'prefill_meta.json'), 'utf8')); } catch (e) {}
    const rdU32 = (o) => b.readUInt32LE(o);
    const magic = b.toString('ascii', 0, 8);
    let off = 8;
    const metaSecs = meta && meta.files && meta.files['prefill_export_main.bin'] && meta.files['prefill_export_main.bin'].sections || [];
    const secEsz = (i, binDt) => {
        const md = metaSecs[i] && metaSecs[i].dtype;
        if (md === 'f16') return 2;
        if (md === 'f32') return 4;
        return binDt === 1 ? 2 : 4;
    };
    function readRows(n, nDim, esz) {
        const rows = [];
        for (let i = 0; i < n; i++) {
            const pos = rdU32(off); off += 4;
            const data = new Float32Array(nDim);
            if (esz === 2) { for (let j = 0; j < nDim; j++) data[j] = f16ToF32(b.readUInt16LE(off + j * 2)); }
            else { for (let j = 0; j < nDim; j++) data[j] = b.readFloatLE(off + j * 4); }
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
    return { embd, hidden };
}
function parseHistory(dir) {
    const b = fs.readFileSync(path.join(dir, 'expert_history_main.bin'));
    const n = b.readUInt32LE(8);
    const perToken = new Map(); // token -> Map(layer -> Set(expert))
    for (let i = 0; i < n; i++) {
        const o = 12 + i * 12;
        const l = b.readUInt32LE(o), t = b.readUInt32LE(o + 4), x = b.readUInt32LE(o + 8);
        if (!perToken.has(t)) perToken.set(t, new Map());
        const lm = perToken.get(t);
        if (!lm.has(l)) lm.set(l, new Set());
        lm.get(l).add(x);
    }
    return perToken;
}
const [da, db] = process.argv.slice(2);
if (!da || !db) { console.error('usage: node tools/div_match.js <base_dir> <div_dir>'); process.exit(2); }
const A = parsePrefill(da), B = parsePrefill(db);
const HA = parseHistory(da), HB = parseHistory(db);

const cos = (a, b) => {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-30);
};
const embdC = new Map(), hidC = new Map();
for (const ra of A.embd) { const rb = B.embd.find(r => r.pos === ra.pos); embdC.set(ra.pos, rb ? cos(ra.data, rb.data) : NaN); }
for (const ra of A.hidden) { const rb = B.hidden.find(r => r.pos === ra.pos); hidC.set(ra.pos, rb ? cos(ra.data, rb.data) : NaN); }

// per-token flip: layer count where expert set differs
const tokens = [...new Set([...HA.keys(), ...HB.keys()])].sort((x, y) => x - y);
const rows = [];
let flipped = 0;
for (const t of tokens) {
    const ma = HA.get(t) || new Map(), mb = HB.get(t) || new Map();
    let flips = 0, fl = [];
    const allL = new Set([...ma.keys(), ...mb.keys()]);
    for (const l of allL) {
        const sa = ma.get(l), sb = mb.get(l);
        const d = (!sa || !sb) ? true : (sa.size !== sb.size || [...sa].some(x => !sb.has(x)));
        if (d) { flips++; fl.push(l); }
    }
    if (flips > 0) flipped++;
    rows.push({ t, flips, fl, ec: embdC.get(t) ?? NaN, hc: hidC.get(t) ?? NaN });
}
const EMB_LO = 0.999;
const lo = rows.filter(r => r.ec < EMB_LO);
const mid = rows.filter(r => r.ec < 0.9999 && r.ec >= EMB_LO);
const hi = rows.filter(r => r.ec >= 0.9999);
const hitLo = lo.filter(r => r.flips > 0).length;
console.log(`tokens=${tokens.length} flipped=${flipped} (${(flipped / tokens.length * 100).toFixed(1)}%)`);
console.log(`embd cos buckets:  <0.999 n=${lo.length}  [0.999,0.9999) n=${mid.length}  >=0.9999 n=${hi.length}`);
console.log(`of the <0.999 high-divergence tokens, ${hitLo}/${lo.length} (${lo.length ? (hitLo / lo.length * 100).toFixed(1) : '-'}%) also have expert flips`);
console.log(`high-div tokens WITHOUT flip (unexplained by flips): ${lo.filter(r => r.flips === 0).map(r => r.t).join(' ') || 'none'}`);
console.log(`flipped tokens that are NOT high-div (embd>=0.9999): ${rows.filter(r => r.flips > 0 && r.ec >= 0.9999).length}`);
console.log('');
console.log('ranked by embd divergence (worst first):  t=token fl=#flipped_layers ec=embd_cos hc=hidden_cos');
console.log([...rows].sort((x, y) => x.ec - y.ec).slice(0, 25).map(r => `t=${r.t} fl=${String(r.flips).padStart(2)} L[${r.fl.join(',')}] ec=${r.ec.toFixed(6)} hc=${r.hc.toFixed(6)}`).join('\n'));
console.log('');
if (flipped === 0) {
    console.log('RESULT: no expert flips, no divergence (IDENTICAL)');
} else {
    console.log(`RESULT: ${lo.length - hitLo}/${lo.length} high-div tokens unexplained by expert flips (0 = divergence fully flip-explained)`);
}
