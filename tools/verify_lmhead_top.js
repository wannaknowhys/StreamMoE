// verify_lmhead_top.js - verify the model's output is at a high position:
// read the exported LM-head input (result_norm) from prefill_export*.bin,
// multiply by the model's output.weight (BF16, read from GGUF shard), get the
// full logits, and show the top-k tokens. If the sampled token ranks high here,
// the forward pass is correct / low-perplexity.
//
// usage:
//   node tools/verify_lmhead_top.js --prefill <prefill_export_main.bin> \
//       --model <shard containing output.weight> [--token <row idx>] [--k 10]
//
// output: top-k (token id, text, logit, softmax prob) + rank of the token that
// was actually sampled (pass --want <token id> to check its rank).
const fs = require('fs');
const path = require('path');

const MAGIC = 'GGUF';

function parseArgs() {
    const a = process.argv.slice(2);
    const p = { prefill: null, model: null, vocabModel: null, token: -1, k: 10, want: -1 };
    for (let i = 0; i < a.length; i++) {
        if (a[i] === '--prefill') p.prefill = a[++i];
        else if (a[i] === '--model') p.model = a[++i];
        else if (a[i] === '--vocab-model') p.vocabModel = a[++i];
        else if (a[i] === '--token') p.token = parseInt(a[++i]);
        else if (a[i] === '--k') p.k = parseInt(a[++i]);
        else if (a[i] === '--want') p.want = parseInt(a[++i]);
    }
    return p;
}

// infer shard 00001 (carries the tokenizer) from e.g. ...-00002-of-00005.gguf
function inferShard1(model) {
    const m = model.match(/-(\d{5})-of-(\d{5})\.gguf$/);
    if (!m) return null;
    return model.replace(`-${m[1]}-of-${m[2]}.gguf`, `-${String(1).padStart(5, '0')}-of-${m[2]}.gguf`);
}

// ---- GGUF low-level (from stream_moe_convert.js) ----
function alignUp(n, a) { return (n + a - 1) & ~(a - 1); }
class Reader {
    constructor(fd) { this.fd = fd; }
    read(offset, len) {
        const b = Buffer.alloc(len);
        let o = offset, got = 0;
        while (got < len) { const n = fs.readSync(this.fd, b, got, len - got, o); if (n <= 0) throw new Error('EOF'); got += n; o += n; }
        return b;
    }
}
function parseHeader(r, off) {
    const b = r.read(off, 24);
    if (b.toString('latin1', 0, 4) !== MAGIC) throw new Error('not a GGUF');
    const version = b.readUInt32LE(4);
    const n_tensors = Number(b.readBigUInt64LE(8));
    const n_kv = Number(b.readBigUInt64LE(16));
    return { version, alignment: 32, n_tensors, n_kv, off: off + 24 };
}
function readStr(r, off) {
    const b = r.read(off, 8); const len = Number(b.readBigUInt64LE(0));
    return { s: r.read(off + 8, len).toString('utf8'), off: off + 8 + len };
}
function readValue(r, off, t) {
    switch (t) {
        case 0: { const b = r.read(off, 1); return { v: b[0], off: off + 1 }; }
        case 1: { const b = r.read(off, 1); return { v: b.readInt8(0), off: off + 1 }; }
        case 2: { const b = r.read(off, 2); return { v: b.readUInt16LE(0), off: off + 2 }; }
        case 3: { const b = r.read(off, 2); return { v: b.readInt16LE(0), off: off + 2 }; }
        case 4: { const b = r.read(off, 4); return { v: b.readUInt32LE(0), off: off + 4 }; }
        case 5: { const b = r.read(off, 4); return { v: b.readInt32LE(0), off: off + 4 }; }
        case 6: { const b = r.read(off, 4); return { v: b.readFloatLE(0), off: off + 4 }; }
        case 7: { const b = r.read(off, 1); return { v: b[0] !== 0, off: off + 1 }; }
        case 8: { const s = readStr(r, off); return { v: s.s, off: s.off }; }
        case 9: {
            const h = r.read(off, 12); const et = h.readUInt32LE(0); const n = Number(h.readBigUInt64LE(4)); off += 12;
            const arr = [];
            for (let i = 0; i < n; i++) { const x = readValue(r, off, et); arr.push(x.v); off = x.off; }
            return { v: arr, off };
        }
        case 10: { const b = r.read(off, 8); return { v: b.readBigUInt64LE(0), off: off + 8 }; }
        case 11: { const b = r.read(off, 8); return { v: b.readBigInt64LE(0), off: off + 8 }; }
        case 12: { const b = r.read(off, 8); return { v: b.readDoubleLE(0), off: off + 8 }; }
        default: throw new Error('unknown value type ' + t);
    }
}
function readKV(r, off) {
    const k = readStr(r, off); off = k.off;
    const t = r.read(off, 4).readUInt32LE(0); off += 4;
    const x = readValue(r, off, t);
    return { k: k.s, v: x.v, off: x.off };
}
function parseGGUF(path) {
    const fd = fs.openSync(path, 'r');
    const r = new Reader(fd);
    const hdr = parseHeader(r, 0);
    const kvs = {};
    let off = hdr.off;
    for (let i = 0; i < hdr.n_kv; i++) {
        const x = readKV(r, off); off = x.off;
        kvs[x.k] = x.v;
    }
    const tensors = {};
    for (let i = 0; i < hdr.n_tensors; i++) {
        const nm = readStr(r, off); off = nm.off;
        const nd = r.read(off, 4).readUInt32LE(0); off += 4;
        const ne = [];
        for (let d = 0; d < nd; d++) { ne.push(Number(r.read(off, 8).readBigUInt64LE(0))); off += 8; }
        const type = r.read(off, 4).readUInt32LE(0); off += 4;
        const offset = Number(r.read(off, 8).readBigUInt64LE(0)); off += 8;
        tensors[nm.s] = { ne, type, offset };
    }
    const alignment = typeof kvs['general.alignment'] === 'number' ? kvs['general.alignment'] : 32;
    const dataOffset = alignUp(off, alignment);
    return { fd, r, kvs, tensors, dataOffset, alignment };
}

// ---- prefill_export.bin (PREFEXP1) ----
function readPrefill(path, wantToken) {
    const fd = fs.openSync(path, 'r');
    const r = new Reader(fd);
    let off = 0;
    const magic = r.read(off, 8); off += 8; // "PREFEXP1"
    const nRows = r.read(off, 4).readUInt32LE(0); off += 4;
    const nDim = r.read(off, 4).readUInt32LE(0); off += 4;
    const rows = [];
    for (let i = 0; i < nRows; i++) {
        const pos = r.read(off, 4).readUInt32LE(0); off += 4;
        const embd = new Float32Array(r.read(off, nDim * 4).buffer.slice(0)); off += nDim * 4;
        rows.push({ pos, embd });
    }
    fs.closeSync(fd);
    let idx = wantToken >= 0 ? wantToken : rows.length - 1;
    if (idx >= rows.length) idx = rows.length - 1;
    return { rows, idx };
}

// ---- BF16 -> float ----
function bf16toF32(buf, off) {
    const u16 = buf.readUInt16LE(off);
    const u32 = new Uint32Array(1); u32[0] = u16 << 16;
    return new Float32Array(u32.buffer)[0];
}

function main() {
    const p = parseArgs();
    if (!p.prefill || !p.model) {
        console.error('usage: node tools/verify_lmhead_top.js --prefill <prefill_export_main.bin> --model <shard> [--token N] [--k 10] [--want <tokenid>]');
        process.exit(1);
    }

    // LM-head input
    const pf = readPrefill(p.prefill, p.token);
    const embd = pf.rows[pf.idx].embd;
    console.log(`[lmhead] prefill rows=${pf.rows.length} dim=${embd.length} selected row=${pf.idx} (pos=${pf.rows[pf.idx].pos})`);

    // model GGUF (shard containing output.weight)
    const g = parseGGUF(p.model);
    const out = g.tensors['output.weight'];
    if (!out) { console.error(`[lmhead] no output.weight in ${p.model} (try the shard that has it)`); process.exit(1); }
    if (out.type !== 30) console.warn(`[lmhead] WARN output.weight type=${out.type} (expect 30=BF16)`);
    const [nEmb, nVocab] = out.ne; // [n_embd, n_vocab]
    if (nEmb !== embd.length) console.warn(`[lmhead] WARN dim mismatch: prefill dim=${embd.length} vs output n_embd=${nEmb}`);

    // tokenizer text (from --vocab-model or inferred shard1)
    const vPath = p.vocabModel || inferShard1(p.model);
    let tokens = [];
    if (vPath && fs.existsSync(vPath)) {
        try { tokens = parseGGUF(vPath).kvs['tokenizer.ggml.tokens'] || []; } catch (e) { tokens = []; }
    }
    const vocabN = tokens.length || nVocab;
    console.log(`[lmhead] vocab=${nVocab} tokenizer_tokens=${tokens.length} output.offset=${out.offset} (abs=${g.dataOffset + out.offset})`);

    // read output.weight (BF16) rows for the selected embedding, compute logits
    const fd = fs.openSync(p.model, 'r');
    const r = new Reader(fd);
    const rowBytes = nEmb * 2;
    const abs = g.dataOffset + out.offset;
    const logits = new Float32Array(nVocab);
    const buf = Buffer.alloc(rowBytes);
    for (let v = 0; v < nVocab; v++) {
        const rr = r.read(abs + v * rowBytes, rowBytes);
        let s = 0;
        for (let j = 0; j < nEmb; j++) s += embd[j] * bf16toF32(rr, j * 2);
        logits[v] = s;
    }
    fs.closeSync(fd);

    // softmax via logsumexp
    let mx = -Infinity;
    for (let v = 0; v < nVocab; v++) if (logits[v] > mx) mx = logits[v];
    let lse = 0;
    for (let v = 0; v < nVocab; v++) lse += Math.exp(logits[v] - mx);
    const logZ = mx + Math.log(lse);

    // top-k
    const order = Array.from({ length: nVocab }, (_, i) => i);
    const top = [];
    for (let v = 0; v < nVocab; v++) {
        const x = logits[v];
        if (top.length < p.k) { top.push({ v, x }); top.sort((a, b) => b.x - a.x); }
        else if (x > top[p.k - 1].x) { top[p.k - 1] = { v, x }; top.sort((a, b) => b.x - a.x); }
    }
    console.log(`\n=== top-${p.k} (from exported LM-head input) ===`);
    for (let i = 0; i < top.length; i++) {
        const t = top[i];
        const prob = Math.exp(t.x - logZ);
        const txt = tokens[t.v] != null ? JSON.stringify(tokens[t.v]) : '(no tokenizer)';
        console.log(`#${i + 1} token=${t.v} prob=${prob.toFixed(4)} ${txt}`);
    }

    if (p.want >= 0) {
        const wantLogit = logits[p.want];
        let rank = 1;
        for (let v = 0; v < nVocab; v++) if (logits[v] > wantLogit) rank++;
        console.log(`\n[lmhead] sampled token ${p.want} rank=${rank}/${nVocab} prob=${Math.exp(wantLogit - logZ).toFixed(4)}`);
    }
}

main();
