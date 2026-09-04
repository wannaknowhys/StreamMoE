// stream_moe_layout.js - single source of truth for converter layout logic.
//
// Model: the universal abstract description (docs/STREAMMOE_GGUF_FORMAT.md §9).
//   { arch, layout, nLayer, nExpert, nExpertUsed, incomplete, files, dense, expert }
//   dense:  [{ name, ne, type, size, srcs:[{fi,off,len,inOff}] }]   // in source order
//   expert: [{ name, ne, type, size, perExpert, branch, layer,
//              perExpertSrcs:[ [{fi,off,len,inOff}] x nExpert ] }]  // sorted (layer,ORDER)
//   src entries point into the source file(s): fi = index into files[], off = absolute
//   file offset, len = bytes, inOff = offset of the segment within its tensor/expertslice.
//
// Writers turn a Model into a stream of convertd commands (write_meta / copy /
// fill / close). v2-chunk sources are supported via multi-segment src lists
// (the physical bytes of one expert slice are scattered across N strip files).
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const EXE = path.join(__dirname, '..', 'temp', 'stream_moe_convertd.exe');
const ALIGN = 4096;
const ORDER = ['gate_up', 'gate', 'up', 'down'];

// ---------- convertd RPC (raw TCP, single persistent session per process) ----------
// convertd keeps open output handles (write_meta/copy/fill/close) in process
// memory, so one conversion flow must reuse ONE spawned child over ONE TCP
// connection. convertd prints "PORT n" to stderr; we connect, then send one
// JSON line per command and parse responses FIFO by line.
const net = require('net');
let g_child = null;
let g_socket = null;
let g_lineBuf = Buffer.alloc(0); // byte-accurate line accumulator (UTF-8 may span data events)
const g_pending = [];
let g_idleTimer = null;

function armIdle() {
    clearTimeout(g_idleTimer);
    g_idleTimer = setTimeout(() => {
        if (g_pending.length === 0 && g_socket && !g_socket.destroyed) {
            try { g_socket.end(); } catch (e) { /* ignore */ }
        }
    }, 300);
}

function connect(port, resolve, reject) {
    const s = net.connect(port, '127.0.0.1');
    s.on('connect', () => { g_socket = s; resolve(); });
    s.on('error', (e) => reject(e));
    s.on('close', () => { g_socket = null; });
    s.on('data', (d) => {
        g_lineBuf = Buffer.concat([g_lineBuf, d]);
        let idx;
        while ((idx = g_lineBuf.indexOf('\n')) >= 0) {
            const line = g_lineBuf.slice(0, idx).toString('utf8'); // full line => complete UTF-8
            g_lineBuf = g_lineBuf.slice(idx + 1);
            if (!line.trim()) continue;
            const p = g_pending.shift();
            if (p) {
                try { p.resolve(JSON.parse(line)); } catch (e) { p.reject(new Error('bad wrapper JSON: ' + line.slice(0, 200))); }
            }
        }
        armIdle();
    });
}

function ensureChild() {
    return new Promise((resolve, reject) => {
        if (g_socket && !g_socket.destroyed) return resolve();
        if (g_child) return; // spawn in flight, PORT pending on stderr
        g_child = spawn(EXE, ['0'], { stdio: ['ignore', 'ignore', 'pipe'] });
        let info = '';
        g_child.stderr.on('data', (d) => {
            info += d.toString();
            const m = info.match(/PORT (\d+)/);
            if (m) connect(Number(m[1]), resolve, reject);
        });
        g_child.on('error', (e) => reject(new Error('cannot spawn ' + EXE + ': ' + e.message)));
        g_child.on('exit', () => { g_child = null; g_socket = null; g_lineBuf = Buffer.alloc(0); });
    });
}

function call(cmd) {
    armIdle();
    return ensureChild().then(() => new Promise((resolve, reject) => {
        g_pending.push({ resolve, reject });
        g_socket.write(JSON.stringify(cmd) + '\n');
    }));
}

async function openFiles(paths) {
    const r = await call({ cmd: 'open', in: paths });
    if (!r.ok) throw new Error('open failed: ' + r.error);
    return r.files;
}

// ---------- metadata helpers ----------
const kv = (files, k) => {
    for (const f of files) { const x = f.meta.kv.find((e) => e.k === k); if (x) return x.v ?? x.arr; }
    return undefined;
};
const kvFile = (f, k) => { const x = f.meta.kv.find((e) => e.k === k); return x ? (x.v ?? x.arr) : undefined; };
const alignUp = (n, a = ALIGN) => Math.ceil(n / a) * a;

function branchOf(name) {
    if (name.includes('gate_up')) return 'gate_up';
    if (name.includes('ffn_gate_exps')) return 'gate';
    if (name.includes('ffn_up_exps')) return 'up';
    if (name.includes('down_exps')) return 'down';
    return name.split('.').pop() || name;
}
function layerOf(name) {
    const p = name.indexOf('blk.');
    if (p < 0) return -1;
    return parseInt(name.slice(p + 4), 10);
}

// ---------- Model construction (parse) ----------
function buildBlocks(files, nLayer, nExpert) {
    const sec = kv(files, 'stream_moe.expert_sections') || [];
    const nBlocks = sec.length / 3;
    const blocks = [];
    for (let i = 0; i < nBlocks; i++) blocks.push({ off: Number(sec[i * 3]), size: Number(sec[i * 3 + 1]), nsub: Number(sec[i * 3 + 2]) });
    return blocks;
}

function buildLayerBranches(files, nLayer) {
    const names = kv(files, 'stream_moe.expert_branch_names') || [];
    const sizes = kv(files, 'stream_moe.expert_branch_sizes') || [];
    const counts = kv(files, 'stream_moe.expert_branch_counts') || [];
    const branchAlign = Number(kv(files, 'stream_moe.branch_align') || 0) === 1;
    const layerBranches = [];
    let ni = 0, si = 0;
    for (let l = 0; l < nLayer; l++) {
        const cnt = Number(counts[l] || 0);
        const bs = [];
        let off = 0;
        for (let j = 0; j < cnt; j++) {
            const name = names[ni++];
            const perExpert = Number(sizes[si++]);
            bs.push({ name, tag: branchOf(name), perExpert, branchOff: off });
            // branch_align=1: each branch (tensor) slice starts 4K-aligned
            // (mirror computeV2Layout); legacy compact otherwise.
            off = branchAlign ? alignUp(off + perExpert, ALIGN) : off + perExpert;
        }
        layerBranches.push(bs);
    }
    return layerBranches;
}

// map a logical interval into source segments. kind='dense': rel relative to
// dense-section start; kind='block': rel relative to block blkIndex's start
// (block owns its own strip layout in v2-chunk). Single-file v2 => 1 segment.
function rangeToSegs(model, kind, blkIndex, rel, len) {
    const { files, incomplete } = model;
    if (!incomplete) {
        const dataOff = files[0].meta.data_offset;
        const abs = kind === 'block' ? model.blocks[blkIndex].off + rel : rel;
        return [{ fi: 0, off: dataOff + abs, len, inOff: 0 }];
    }
    const dataOff = files.map((f) => f.meta.data_offset);
    let counts, base;
    if (kind === 'dense') {
        counts = model.denseBlocks;
        base = files.map(() => 0);
    } else {
        counts = model.blockSlices[blkIndex];
        base = files.map((_, i) => (model.denseBlocks[i] + model.blockPrefix[blkIndex][i]) * ALIGN);
    }
    const cum = [];
    { let c = 0; for (const n of counts) { cum.push(c); c += n; } }
    const segs = [];
    let cur = Math.floor(rel / ALIGN);
    const end = Math.ceil((rel + len) / ALIGN);
    while (cur < end) {
        let fi = 0;
        while (fi < counts.length - 1 && cur >= cum[fi] + counts[fi]) fi++;
        const local = cur - cum[fi];
        const avail = Math.min(counts[fi] - local, end - cur);
        if (avail <= 0) throw new Error(`rangeToSegs: interval past strip coverage (${kind} blk ${blkIndex} rel ${rel} len ${len} counts [${counts}] cur ${cur}) - is every chunk file supplied?`);
        const segStart = Math.max(rel, cur * ALIGN);
        const segEnd = Math.min(rel + len, (cur + avail) * ALIGN);
        if (segEnd > segStart) {
            // base[fi] is the block strip's start inside the file data area;
            // within the strip the byte offset is local*ALIGN plus any
            // non-aligned remainder (segStart - cur*ALIGN) of the interval head.
            const localByte = local * ALIGN + (segStart - cur * ALIGN);
            segs.push({ fi, off: dataOff[fi] + base[fi] + localByte, len: segEnd - segStart, inOff: segStart - rel });
        }
        cur += avail;
    }
    return segs;
}

function buildModel(files) {
    const layout = kv(files, 'stream_moe.layout') || 'original';
    const arch = kv(files, 'general.architecture') || '';
    const nLayer = Number(kv(files, arch + '.block_count') ?? kv(files, 'block_count') ?? 0);
    const nExpert = Number(kv(files, arch + '.expert_count') ?? kv(files, 'expert_count') ?? 0);
    const nExpertUsed = Number(kv(files, arch + '.expert_used_count') ?? 0);
    const incomplete = Number(kv(files, 'stream_moe.incomplete') || 0) === 1;
    const model = { arch, layout, nLayer, nExpert, nExpertUsed, incomplete, files, dense: [], expert: [] };

    let blocks = null, layerBranches = null;
    if (layout === 'expert-blocks-v2') {
        blocks = buildBlocks(files, nLayer, nExpert);
        model.blocks = blocks;
        layerBranches = buildLayerBranches(files, nLayer);
        if (incomplete) {
            model.denseBlocks = files.map((f) => Number(kvFile(f, 'stream_moe.chunk_slices')[0]));
            model.blockSlices = blocks.map((_, b) => files.map((f) => Number(kvFile(f, 'stream_moe.chunk_slices')[1 + b])));
            model.blockPrefix = blocks.map((_, b) => files.map((_, i) => {
                let s = 0;
                for (let k = 0; k < b; k++) s += model.blockSlices[k][i];
                return s;
            }));
        }
    }

    // v2-chunk files each carry the full tensor_info table (data is the strip);
    // collect tensor metadata from file 0 only, segments map across all files.
    const tensorFiles = incomplete ? files.slice(0, 1) : files;
    for (let fi = 0; fi < tensorFiles.length; fi++) {
        const f = tensorFiles[fi];
        const dataOff = f.meta.data_offset;
        for (const t of f.meta.tensors) {
            const isScale = t.name.includes('.scale');
            const isExp = t.name.includes('_exps') && !isScale;
            if (isExp) {
                const perExpert = t.size / nExpert;
                const branch = branchOf(t.name);
                const layer = layerOf(t.name);
                const perExpertSrcs = [];
                if (layout === 'expert-blocks-v2') {
                    // v2: one block per expert; block (layer,e) holds that
                    // expert's branches concatenated. Branch offset within the
                    // block is branchOff (NOT e*perExpert - that is the
                    // contiguous-tensor layout of original/v1).
                    const br = layerBranches[layer].find((b) => b.tag === branch);
                    for (let e = 0; e < nExpert; e++) {
                        const rel = br.branchOff;
                        perExpertSrcs.push(rangeToSegs(model, 'block', layer * nExpert + e, rel, perExpert));
                    }
                } else {
                    for (let e = 0; e < nExpert; e++) perExpertSrcs.push([{ fi, off: dataOff + t.offset + e * perExpert, len: perExpert, inOff: 0 }]);
                }
                model.expert.push({ name: t.name, ne: t.ne, type: t.type, size: t.size, perExpert, branch, layer, perExpertSrcs });
            } else {
                model.dense.push({
                    name: t.name, ne: t.ne, type: t.type, size: t.size,
                    srcs: layout === 'expert-blocks-v2' ? rangeToSegs(model, 'dense', 0, t.offset, t.size) : [{ fi, off: dataOff + t.offset, len: t.size, inOff: 0 }],
                });
            }
        }
    }
    model.expert.sort((a, b) => a.layer - b.layer || ORDER.indexOf(a.branch) - ORDER.indexOf(b.branch));
    return model;
}

// ---------- split blocks across N strips (uniform or largest-remainder ratio) ----------
function splitBlocks(B, N, ratio) {
    const nslice = new Array(N).fill(0);
    if (!ratio || ratio.length === 0) {
        const base = Math.floor(B / N), rem = B % N;
        for (let i = 0; i < N; i++) nslice[i] = base + (i < rem ? 1 : 0);
    } else {
        const sum = ratio.reduce((a, b) => a + b, 0);
        const quota = ratio.map((r) => B * r / sum);
        let total = 0;
        for (let i = 0; i < N; i++) { nslice[i] = Math.floor(quota[i]); total += nslice[i]; }
        const diff = B - total;
        const order = [...Array(N).keys()].sort((a, b) => (quota[b] - nslice[b]) - (quota[a] - nslice[a]));
        for (let k = 0; k < diff; k++) nslice[order[k]]++;
    }
    if (nslice.reduce((a, b) => a + b, 0) !== B) throw new Error('split: strip sum != block count');
    return nslice;
}

// ---------- v2 layout computation (shared by writeV2 / writeV2chunk) ----------
// returns { denseEnd, blocks:[{off,size,raw}], layers, branchesByLayer, layerBlockIdx }
function computeV2Layout(model, offsets) {
    const nD = model.dense.length;
    let denseEnd = 0;
    for (let i = 0; i < nD; i++) denseEnd = Math.max(denseEnd, offsets[i] + model.dense[i].size);
    denseEnd = alignUp(denseEnd, ALIGN);
    const byLayer = new Map();
    for (const t of model.expert) {
        if (!byLayer.has(t.layer)) byLayer.set(t.layer, []);
        byLayer.get(t.layer).push(t);
    }
    const layers = [...byLayer.keys()].sort((a, b) => a - b);
    const branchesByLayer = new Map();
    const layerBranchSize = new Map();   // aligned cumulative end per layer (block size constant)
    // v2 block-internal layout: each branch (tensor) slice starts 4K-aligned
    // (branch_align=1, 2026-09) so every (expert,tensor) slice source offset is
    // DIO-able. Each block starts 4K-aligned; inside, branch starts advance by
    // alignUp(previous end, ALIGN). block size = aligned cumulative end (same
    // branch set for every expert of a layer, so constant within the layer).
    for (const l of layers) {
        const branches = ORDER.map((tag) => byLayer.get(l).find((t) => t.branch === tag)).filter(Boolean);
        let off = 0;
        for (const b of branches) {
            b.blockOff = off;
            off = alignUp(off + b.perExpert, ALIGN);   // aligned end of this branch
        }
        branchesByLayer.set(l, branches);
        layerBranchSize.set(l, off);
    }
    const blocks = [];
    const layerBlockIdx = new Map();
    let cur = denseEnd;
    for (const l of layers) {
        const branches = branchesByLayer.get(l);
        const raw = branches.reduce((a, b) => a + b.perExpert, 0);
        const size = layerBranchSize.get(l);
        layerBlockIdx.set(l, blocks.length);
        for (let e = 0; e < model.nExpert; e++) {
            blocks.push({ off: cur, size, raw });
            cur += size;
        }
    }
    return { denseEnd, blocks, layers, branchesByLayer, layerBlockIdx, nExpert: model.nExpert };
}

function layerOfBlock(layout, bi) {
    for (const l of layout.layers) {
        const start = layout.layerBlockIdx.get(l);
        if (bi >= start && bi < start + layout.nExpert) return l;
    }
    return -1;
}

// per-block [off,size,nsub] + branch layout KV for a computed v2 layout
function v2LayoutKV(layout) {
    const sec = [];
    for (const b of layout.blocks) sec.push(b.off, b.size, 0);
    for (let bi = 0; bi < layout.blocks.length; bi++) sec[bi * 3 + 2] = layout.branchesByLayer.get(layerOfBlock(layout, bi)).length;
    const bnames = [], bsizes = [], bcounts = [];
    for (const l of layout.layers) {
        const branches = layout.branchesByLayer.get(l);
        bcounts.push(branches.length);
        for (const b of branches) { bnames.push(b.name); bsizes.push(b.perExpert); }
    }
    return {
        'stream_moe.dense_section': [0, layout.denseEnd],
        'stream_moe.expert_sections': sec,
        'stream_moe.expert_branch_names': bnames,
        'stream_moe.expert_branch_sizes': bsizes,
        'stream_moe.expert_branch_counts': bcounts,
        'stream_moe.branch_align': 1,   // 2026-09: each branch slice inside a block starts 4K-aligned
    };
}

// map a logical interval over a linear "content" (sorted by logOff) to source
// segments. content item: { logOff, len, type:'dense', tensor } or
// { logOff, len, type:'exp', e, perExpertSrcs }. Each returned seg carries
// logOff = its offset within [start, start+len) (for target placement).
function contentToSegs(content, start, len) {
    const segs = [];
    for (const item of content) {
        const cStart = item.logOff, cEnd = item.logOff + item.len;
        if (cEnd <= start || cStart >= start + len) continue;
        const s = Math.max(start, cStart), e = Math.min(start + len, cEnd);
        const relStart = s - cStart, relEnd = e - cStart; // item-relative
        const list = item.type === 'dense' ? item.tensor.srcs : item.perExpertSrcs[item.e];
        for (const src of list) { // src: {fi,off,len,inOff}, inOff = src position inside the item
            const sStart = src.inOff, sEnd = src.inOff + src.len;
            if (sEnd <= relStart || sStart >= relEnd) continue;
            const ss = Math.max(relStart, sStart), se = Math.min(relEnd, sEnd);
            segs.push({ fi: src.fi, off: src.off + (ss - sStart), len: se - ss, logOff: s - start + (ss - relStart) });
        }
    }
    return segs;
}

function denseContent(model, offsets) {
    return model.dense.map((t, i) => ({ logOff: offsets[i], len: t.size, type: 'dense', tensor: t }));
}
function blockContentArr(layout, layer, e) {
    const out = [];
    const blkOff = layout.blocks[layout.layerBlockIdx.get(layer) + e].off;
    for (const b of layout.branchesByLayer.get(layer)) out.push({ logOff: blkOff + b.blockOff, len: b.perExpert, type: 'exp', e, perExpertSrcs: b.perExpertSrcs });
    return out;
}

// ---------- writers ----------
async function probeOffsets(model, outBase) {
    const tensors = [...model.dense, ...model.expert];
    const probePath = outBase + '.probe.gguf';
    const p = await call({
        cmd: 'write_meta', out: probePath, in: model.files.map((f) => f.path),
        skip_kv: ['split.', 'stream_moe.', 'general.alignment'],
        set_kv: { 'general.alignment': ALIGN, 'stream_moe.layout': 'expert-blocks-v2' },
        tensors: tensors.map((t) => ({ name: t.name, ne: t.ne, type: t.type })),
        alignment: ALIGN,
    });
    if (!p.ok) throw new Error('probe write_meta: ' + p.error);
    await call({ cmd: 'close', dst: probePath });
    try { fs.unlinkSync(probePath); } catch (e) { /* ignore */ }
    return { offsets: p.offsets, tensors };
}

function baseMetaCmd(model, out, setKv, tensors) {
    return {
        cmd: 'write_meta', out, in: model.files.map((f) => f.path),
        skip_kv: ['split.', 'stream_moe.', 'general.alignment'],
        set_kv: { 'general.alignment': ALIGN, 'stream_moe.layout': 'expert-blocks-v2', ...setKv },
        tensors: tensors.map((t) => ({ name: t.name, ne: t.ne, type: t.type })),
        alignment: ALIGN,
    };
}

async function writeV1(model, out) {
    const tensors = [...model.dense, ...model.expert];
    const meta = await call({
        cmd: 'write_meta', out, in: model.files.map((f) => f.path),
        skip_kv: ['split.', 'stream_moe.', 'general.alignment'],
        set_kv: { 'general.alignment': ALIGN, 'stream_moe.layout': 'sections-v1' },
        tensors: tensors.map((t) => ({ name: t.name, ne: t.ne, type: t.type, ...(t.perExpert ? { per_expert: t.perExpert } : {}) })),
        alignment: ALIGN,
    });
    if (!meta.ok) throw new Error('write_meta v1: ' + meta.error);
    const nD = model.dense.length;
    const ops = [];
    model.dense.forEach((t, i) => { for (const s of t.srcs) ops.push([s.fi, s.off, s.len, meta.offsets[i] + s.inOff]); });
    model.expert.forEach((t, i) => {
        // per-expert slices are 4K-padded in the output (convertd reflows the
        // expert section) so each slice starts 4K-aligned for straight DIO.
        const stride = alignUp(t.perExpert, ALIGN);
        t.perExpertSrcs.forEach((segs, e) => {
            for (const s of segs) ops.push([s.fi, s.off, s.len, meta.offsets[nD + i] + e * stride + s.inOff]);
        });
    });
    await call({ cmd: 'copy', src: model.files.map((f) => f.path), dst: out, ops }).then((r) => { if (!r.ok) throw new Error('copy v1: ' + r.error); });
    return await call({ cmd: 'close', dst: out });
}

async function writeV2(model, out) {
    const { offsets, tensors } = await probeOffsets(model, out);
    const layout = computeV2Layout(model, offsets);
    const meta = await call(baseMetaCmd(model, out, v2LayoutKV(layout), tensors));
    if (!meta.ok) throw new Error('write_meta v2: ' + meta.error);
    const nD = model.dense.length;
    const ops = [], fillOps = [];
    for (let i = 0; i < nD; i++) for (const s of model.dense[i].srcs) ops.push([s.fi, s.off, s.len, offsets[i] + s.inOff]);
    for (const l of layout.layers) {
        const branches = layout.branchesByLayer.get(l);
        const blkBase = layout.layerBlockIdx.get(l);
        for (let e = 0; e < model.nExpert; e++) {
            const blk = layout.blocks[blkBase + e];
            // copy each branch (tensor) slice to its 4K-aligned block-offset
            for (const b of branches) {
                for (const s of b.perExpertSrcs[e]) ops.push([s.fi, s.off, s.len, blk.off + b.blockOff + s.inOff]);
            }
            // fill every pad gap: between branches and after the last branch,
            // so a whole-aligned block span is fully in-file (DIO windows never
            // overrun); branch starts are 4K-aligned (branch_align=1).
            let gap = 0;
            for (const b of branches) {
                if (b.blockOff > gap) fillOps.push([blk.off + gap, b.blockOff - gap]);
                gap = b.blockOff + b.perExpert;
            }
            if (gap < blk.size) fillOps.push([blk.off + gap, blk.size - gap]);
        }
    }
    await call({ cmd: 'copy', src: model.files.map((f) => f.path), dst: out, ops }).then((r) => { if (!r.ok) throw new Error('copy v2: ' + r.error); });
    if (fillOps.length) await call({ cmd: 'fill', dst: out, ops: fillOps }).then((r) => { if (!r.ok) throw new Error('fill v2: ' + r.error); });
    return await call({ cmd: 'close', dst: out });
}

async function writeV2chunk(model, outBase, N, ratio) {
    const { offsets, tensors } = await probeOffsets(model, outBase);
    const layout = computeV2Layout(model, offsets);
    const { denseEnd, blocks, layers } = layout;
    const srcFiles = model.files.map((f) => f.path);

    const denseSlices = splitBlocks(denseEnd / ALIGN, N, ratio);
    const blkSlices = blocks.map((b) => splitBlocks(b.size / ALIGN, N, ratio));
    const blkPrefix = blocks.map((_, b) => {
        const p = [];
        for (let i = 0; i < N; i++) { let s = 0; for (let k = 0; k < b; k++) s += blkSlices[k][i]; p.push(s); }
        return p;
    });
    const stripStart = (slices, i) => slices.slice(0, i).reduce((a, b) => a + b, 0);

    const outs = [];
    for (let i = 0; i < N; i++) outs.push(`${outBase}${i + 1}.gguf`);

    const dc = denseContent(model, offsets);
    const blockContentOf = (b) => blockContentArr(layout, layerOfBlock(layout, b), b - layout.layerBlockIdx.get(layerOfBlock(layout, b)));

    for (let i = 0; i < N; i++) {
        const slices = [denseSlices[i], ...blocks.map((_, b) => blkSlices[b][i])];
        const meta = await call(baseMetaCmd(model, outs[i], {
            ...v2LayoutKV(layout),
            'stream_moe.chunk_no': i,
            'stream_moe.chunk_total': N,
            'stream_moe.incomplete': 1,
            'stream_moe.chunk_slices': slices,
        }, tensors));
        if (!meta.ok) throw new Error('v2chunk write_meta: ' + meta.error);

        const ops = [], fillOps = [];
        // dense strip
        const dStart = stripStart(denseSlices, i);
        for (const s of contentToSegs(dc, dStart * ALIGN, denseSlices[i] * ALIGN)) ops.push([s.fi, s.off, s.len, s.logOff]);
        // per-expert block strips (block order); fill each strip's tail padding
        // so the file's physical data area matches chunk_slices (merge reads
        // full aligned strips without running past EOF).
        for (let b = 0; b < blocks.length; b++) {
            const bStart = stripStart(blkSlices[b], i);
            const stripLen = blkSlices[b][i] * ALIGN;
            const fileBase = (denseSlices[i] + blkPrefix[b][i]) * ALIGN;
            const segs = contentToSegs(blockContentOf(b), blocks[b].off + bStart * ALIGN, stripLen);
            const covered = segs.map((s) => [s.logOff, s.logOff + s.len]).sort((a, c) => a[0] - c[0]);
            for (const s of segs) ops.push([s.fi, s.off, s.len, fileBase + s.logOff]);
            // fill the complement of branch data within the strip (inter-branch
            // 4K pads from branch_align=1 + tail) so the strip's physical area
            // matches chunk_slices and merge never overruns EOF.
            let cursor = 0;
            for (const [a, bEnd] of covered) {
                if (a > cursor) fillOps.push([fileBase + cursor, a - cursor]);
                cursor = Math.max(cursor, bEnd);
            }
            if (cursor < stripLen) fillOps.push([fileBase + cursor, stripLen - cursor]);
        }
        await call({ cmd: 'copy', src: srcFiles, dst: outs[i], ops }).then((r) => { if (!r.ok) throw new Error('copy v2chunk: ' + r.error); });
        if (fillOps.length) await call({ cmd: 'fill', dst: outs[i], ops: fillOps }).then((r) => { if (!r.ok) throw new Error('fill v2chunk: ' + r.error); });
        await call({ cmd: 'close', dst: outs[i] });
    }
    return outs;
}

module.exports = { call, openFiles, buildModel, writeV1, writeV2, writeV2chunk, kv, kvFile, branchOf, layerOf, splitBlocks, computeV2Layout, ALIGN };
