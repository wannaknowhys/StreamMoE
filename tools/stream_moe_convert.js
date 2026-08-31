// stream_moe_convert.js - converter CLI (5 sources x 3 targets).
//
// Any source -> any target: official multi-shard (-00001-of-00005) / original
// single file / v1 (sections-v1) / v2 (expert-blocks-v2) / v2-chunk (N strip
// files) all parse into the same Model; targets are v1 / v2 / v2-chunk.
//
// usage:
//   node tools/stream_moe_convert.js -m <model.gguf> -o <out.gguf> [--format v1|v2]
//   node tools/stream_moe_convert.js -m <model.gguf> -o <dir> --format v2chunk --chunks 5 [--ratio 8:9:9:7:9]
//   node tools/stream_moe_convert.js -m "c1.gguf;c2.gguf;..." -o <out.gguf> [--format v1|v2]
const fs = require('fs');
const path = require('path');
const { openFiles, buildModel, writeV1, writeV2, writeV2chunk, kv } = require('./stream_moe_layout');

function parseArgs() {
    const a = process.argv.slice(2);
    const p = { format: 'v1', model: null, out: null, chunks: 5, ratio: null };
    for (let i = 0; i < a.length; i++) {
        if (a[i] === '-m' || a[i] === '--model') p.model = a[++i];
        else if (a[i] === '-o' || a[i] === '--output') p.out = a[++i];
        else if (a[i] === '--format') p.format = a[++i];
        else if (a[i] === '--chunks') p.chunks = parseInt(a[++i], 10);
        else if (a[i] === '--ratio') p.ratio = a[++i];
    }
    if (!p.model || !p.out) {
        console.error('usage: node tools/stream_moe_convert.js -m <model> -o <out> [--format v1|v2|v2chunk] [--chunks N] [--ratio a:b:c]');
        process.exit(1);
    }
    if (!['v1', 'v2', 'v2chunk'].includes(p.format)) { console.error('--format must be v1|v2|v2chunk'); process.exit(1); }
    if (p.ratio) p.ratio = p.ratio.split(':').map((x) => parseInt(x, 10));
    return p;
}

// official multi-shard discovery: <base>-00001-of-00005.gguf ...
function discoverShards(files, modelArg) {
    if (modelArg.includes(';')) return modelArg.split(';').filter(Boolean);
    const split = kv(files, 'split.count');
    let n = split ? Number(split) : 1;
    // Fall back to the -00001-of-NNNNN filename pattern when split.count is
    // absent: some chunked sources carry no split KV (00001 is metadata-only,
    // tensors live spread across the later shards) and must still be opened.
    if (!split) {
        const m = /-(\d{5})-of-(\d{5})\.gguf$/.exec(modelArg);
        if (m) n = Number(m[2]);
    }
    if (n <= 1) return [modelArg];
    const base = modelArg.replace(/-\d{5}-of-\d{5}\.gguf$/, '');
    const out = [];
    for (let i = 1; i <= n; i++) out.push(`${base}-${String(i).padStart(5, '0')}-of-${String(n).padStart(5, '0')}.gguf`);
    return out;
}

async function main() {
    const p = parseArgs();
    let srcs = p.model.includes(';') ? p.model.split(';').filter(Boolean) : [p.model];
    let files = await openFiles(srcs);
    const shards = discoverShards(files, p.model);
    if (shards.length > files.length) {
        files = await openFiles(shards);
        srcs = shards;
    }
    const model = buildModel(files);
    console.log(`[parse] source layout=${model.layout} shards=${files.length} layers=${model.nLayer} experts=${model.nExpert} dense=${model.dense.length} expertTensors=${model.expert.length}`);

    if (p.format === 'v2chunk') {
        fs.mkdirSync(p.out, { recursive: true });
        const outBase = path.join(p.out, 'c');
        const outs = await writeV2chunk(model, outBase, p.chunks, p.ratio);
        console.log(`[v2chunk] ${outs.length} files -> ${p.out}`);
        for (const o of outs) console.log('  ' + o + '  ' + (fs.statSync(o).size));
        return;
    }
    const res = await (p.format === 'v2' ? writeV2(model, p.out) : writeV1(model, p.out));
    if (!res.ok) { console.error('write failed: ' + res.error); process.exit(1); }
    console.log(`[${p.format}] done -> ${p.out}  size=${res.size}`);
}

main().catch((e) => { console.error('[convert] ' + e.message); process.exit(1); });
