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
const fs = require('fs');

function parseFile(path) {
    const b = fs.readFileSync(path);
    const rdU32 = (o) => b.readUInt32LE(o);
    const rdU64 = (o) => b.readBigUInt64LE(o);
    if (b.toString('ascii', 0, 8) !== 'PREFEXP1') throw new Error('bad magic');
    let off = 8;
    const nRows = rdU32(off); off += 4;
    const nDim = rdU32(off); off += 4;
    const embd = []; // {pos, data}
    for (let i = 0; i < nRows; i++) {
        const pos = rdU32(off); off += 4;
        const data = new Float32Array(nDim);
        for (let j = 0; j < nDim; j++) { data[j] = b.readFloatLE(off + j * 4); }
        off += nDim * 4;
        embd.push({ pos, data });
    }
    const nHRows = rdU32(off); off += 4;
    const nHDim = rdU32(off); off += 4;
    const hidden = [];
    for (let i = 0; i < nHRows; i++) {
        const pos = rdU32(off); off += 4;
        const data = new Float32Array(nHDim);
        for (let j = 0; j < nHDim; j++) { data[j] = b.readFloatLE(off + j * 4); }
        off += nHDim * 4;
        hidden.push({ pos, data });
    }
    const nCache = rdU32(off); off += 4;
    const caches = [];
    for (let ci = 0; ci < nCache; ci++) {
        const nl = rdU32(off); off += 4;
        const name = b.toString('utf8', off, off + nl); off += nl;
        const nLayer = rdU32(off); off += 4;
        const layers = [];
        for (let li = 0; li < nLayer; li++) {
            const il = rdU32(off); off += 4;
            const kType = rdU32(off); off += 4; const kNb = rdU64(off); off += 8;
            const k = b.subarray(off, off + Number(kNb)); off += Number(kNb);
            const vType = rdU32(off); off += 4; const vNb = rdU64(off); off += 8;
            const v = b.subarray(off, off + Number(vNb)); off += Number(vNb);
            layers.push({ il, kType, kNb, k, vType, vNb, v });
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
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-30);
};
function cmpRows(tag, ra, rb) {
    let first = null;
    for (let i = 0; i < Math.min(ra.length, rb.length); i++) {
        const a = ra[i], b = rb[i];
        if (a.pos !== b.pos) { console.log(`  ${tag} pos mismatch @${i}: std=${a.pos} moe=${b.pos}`); if (!first) first = { tag, idx: i }; continue; }
        let mad = 0, mse = 0, maxAbs = 0, maxIdx = -1;
        for (let j = 0; j < a.data.length; j++) {
            const d = Math.abs(a.data[j] - b.data[j]);
            mad += d; mse += d * d;
            if (d > maxAbs) { maxAbs = d; maxIdx = j; }
        }
        mad /= a.data.length; mse /= a.data.length;
        const c = cos(a.data, b.data);
        if (Math.abs(1 - c) > 1e-9) {
            console.log(`  DIFF ${tag} @token#${a.pos} pos=${maxIdx} cos=${c.toFixed(10)} MAD=${mad.toExponential(3)} MSE=${mse.toExponential(3)} maxAbs=${maxAbs.toExponential(3)}`);
            if (!first) first = { tag, token: a.pos, cos: c, mad, mse, maxAbs, maxIdx };
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
