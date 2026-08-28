// stream_moe_convert.js - CLI front-end for the C/C++ GGUF converter wrapper
// (stream_moe_convertd). Discovers multi-shard siblings via the wrapper's `open`
// (split.count), then delegates the actual v1 conversion to the wrapper.
//
// usage: node tools/stream_moe_convert.js -m <model.gguf> -o <out.gguf> [--format v1]
const { spawn } = require('child_process');
const path = require('path');

const EXE = path.join(__dirname, '..', 'temp', 'stream_moe_convertd.exe');

function parseArgs() {
    const a = process.argv.slice(2);
    const p = { format: 'v1', model: null, out: null };
    for (let i = 0; i < a.length; i++) {
        if (a[i] === '-m' || a[i] === '--model') p.model = a[++i];
        else if (a[i] === '-o' || a[i] === '--output') p.out = a[++i];
        else if (a[i] === '--format') p.format = a[++i];
    }
    if (!p.model || !p.out) { console.error('usage: node tools/stream_moe_convert.js -m <model.gguf> -o <out.gguf> [--format v1]'); process.exit(1); }
    return p;
}

function call(cmd) {
    return new Promise((resolve, reject) => {
        const child = spawn(EXE, [], { stdio: ['pipe', 'pipe', 'pipe'] });
        let out = '';
        child.stdout.on('data', (d) => (out += d));
        child.stderr.on('data', (d) => process.stderr.write(d));
        child.on('close', () => {
            try { resolve(JSON.parse(out)); } catch (e) { reject(new Error('bad wrapper JSON: ' + out.slice(0, 200))); }
        });
        child.on('error', (e) => reject(new Error(`cannot spawn ${EXE}: ${e.message} (build it first)`)));
        child.stdin.write(JSON.stringify(cmd) + '\n');
        child.stdin.end();
    });
}

function discoverShards(mainPath) {
    return call({ cmd: 'open', path: mainPath }).then((j) => {
        if (!j.ok) throw new Error('open failed: ' + j.error);
        const split = j.meta.kv.find((k) => k.k === 'split.count');
        const n = split ? Number(split.v) : 1;
        if (n <= 1) return [mainPath];
        const base = mainPath.replace(/-\d{5}-of-\d{5}\.gguf$/, '');
        const paths = [];
        for (let i = 1; i <= n; i++) paths.push(`${base}-${String(i).padStart(5, '0')}-of-${String(n).padStart(5, '0')}.gguf`);
        return paths;
    });
}

async function main() {
    const p = parseArgs();
    const shards = await discoverShards(p.model);
    console.log(`[convert] shards=${shards.length}`);
    const res = await call({ cmd: 'convert', format: p.format, in: shards.join(';'), out: p.out });
    if (!res.ok) { console.error('[convert] FAILED: ' + res.error); process.exit(1); }
    console.log(`[convert] done: ${res.tensors} tensors (dense ${res.dense}, expert ${res.expert}) -> ${p.out}`);
}

main().catch((e) => { console.error('[convert] ' + e.message); process.exit(1); });
